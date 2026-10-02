"""Flatten/unflatten HGT weights and map coordinates to HGT parameter groups.

Groups: ``rel:<src__rel__dst>`` (relation_key/value/prior across layers),
``node:<type>`` (input projection + per-type q/k/v/out/norm/ffn), ``tactic_embedding``,
``classifier``. A relation absent from every batch of a client round gets grad None,
AdamW skips it, so its delta group is exactly zero ("inactive").

Precondition for "inactive => exactly zero delta": the local trainer must clear gradients with
``optimizer.zero_grad(set_to_none=True)``. A reused model with ``set_to_none=False`` keeps
zero/stale grad tensors, so AdamW would still apply weight decay and momentum to parameters
that no batch touched. Likewise, no EMA/weight averaging may modify parameters that received
no gradient.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch

_REL_RE = re.compile(r"^layers\.\d+\.relation_(?:key|value|prior)\.([^.]+)")
_RELATION_MARKERS = ("relation_key", "relation_value", "relation_prior")
_NODE_RE = re.compile(
    r"^(?:input_projection|layers\.\d+\.(?:query|key|value|out|norm_attn|norm_ffn|ffn))\.([^.]+)"
)


def group_of(name: str) -> str:
    match = _REL_RE.match(name)
    if match:
        return f"rel:{match.group(1)}"
    match = _NODE_RE.match(name)
    if match:
        return f"node:{match.group(1)}"
    if name.startswith("tactic_embedding"):
        return "tactic_embedding"
    if name.startswith("classifier"):
        return "classifier"
    return "other"


@dataclass
class ParamLayout:
    names: list[str]
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    offsets: list[int]
    numel: int
    groups: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "ParamLayout":
        names: list[str] = []
        shapes: list[torch.Size] = []
        dtypes: list[torch.dtype] = []
        offsets: list[int] = []
        groups: dict[str, list[tuple[int, int]]] = {}
        offset = 0
        for name, tensor in state_dict.items():
            if not torch.is_floating_point(tensor):
                raise ValueError(f"non-float state entry {name!r} is not supported by ParamLayout")
            group = group_of(name)
            if not group.startswith("rel:") and any(marker in name for marker in _RELATION_MARKERS):
                raise ValueError(
                    f"state_dict key {name!r} looks like a relation parameter but does not match "
                    "'layers.<i>.relation_*.<name>'; a wrapper prefix such as 'module.' (DDP) or "
                    "'_orig_mod.' (torch.compile) is probably present and must be stripped first"
                )
            n = int(tensor.numel())
            names.append(name)
            shapes.append(tensor.shape)
            dtypes.append(tensor.dtype)
            offsets.append(offset)
            groups.setdefault(group, []).append((offset, offset + n))
            offset += n
        return cls(names, shapes, dtypes, offsets, offset, groups)

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "ParamLayout":
        return cls.from_state_dict(model.state_dict())

    @property
    def group_names(self) -> list[str]:
        return list(self.groups.keys())

    def flatten(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for name, shape in zip(self.names, self.shapes):
            if name not in state_dict:
                raise ValueError(f"state_dict is missing key {name!r}")
            tensor = state_dict[name]
            if tensor.shape != shape:
                raise ValueError(
                    f"state_dict entry {name!r} has shape {tuple(tensor.shape)}, "
                    f"layout expects {tuple(shape)}"
                )
            parts.append(tensor.detach().reshape(-1).to("cpu", torch.float32))
        return torch.cat(parts)

    def unflatten(self, vec: torch.Tensor) -> dict[str, torch.Tensor]:
        if vec.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {tuple(vec.shape)}")
        if vec.numel() != self.numel:
            raise ValueError(f"vector has {vec.numel()} entries, layout expects {self.numel}")
        vec = vec.detach().to("cpu", torch.float32)
        out: dict[str, torch.Tensor] = {}
        for name, shape, dtype, start in zip(self.names, self.shapes, self.dtypes, self.offsets):
            n = math.prod(int(d) for d in shape)
            out[name] = vec[start:start + n].view(shape).to(dtype).clone()
        return out

    def group_coords(self, group_names: list[str]) -> torch.Tensor:
        parts = [torch.arange(a, b) for g in group_names for a, b in self.groups[g]]
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)

    def group_norms(self, deltas: torch.Tensor) -> torch.Tensor:
        cols = []
        for g in self.group_names:
            sq = torch.zeros(deltas.shape[0], dtype=torch.float64, device=deltas.device)
            for a, b in self.groups[g]:
                sq += deltas[:, a:b].to(torch.float64).pow(2).sum(dim=1)
            cols.append(sq.sqrt())
        return torch.stack(cols, dim=1).to(torch.float32)

    def active_groups(self, deltas: torch.Tensor) -> torch.Tensor:
        return self.group_norms(deltas) > 0
