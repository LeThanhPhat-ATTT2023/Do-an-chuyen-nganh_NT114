"""Architecture-agnostic model interface for the FL local trainer and evaluator.

A :class:`ModelAdapter` owns everything model-specific: building the network, turning a
client's flow ids into local training data, batching, the per-batch loss and full-graph /
full-table prediction. :class:`~graphslm_ids.offline.federated.local_trainer.LocalTrainer`
and the simulation only talk to this interface, so HGT and the flat baselines share them.

This module also holds the two flat baselines the HGT is compared against, trained with the
same FL protocol on the same data and client table: :class:`FlowMLP` on the standardized
flow features (``model.name: mlp``) and :class:`ByteCNN1D` on the ordered payload bytes of
each flow's first K packets (``model.name: cnn``). They use the same loss settings as HGT
(``client.loss_type`` / ``label_smoothing`` / ``focal_gamma`` and the client's class weights);
EACS and GCL are graph-model features and are ignored. Neither uses BatchNorm (GroupNorm
instead): every state_dict entry is a float parameter, so the parameter vector, the
aggregators and the attacks need no special case.

Adapters do not cache client data: ``client_data`` always builds; caching is the
simulation's job (``ExperimentContext.client_cache``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from graphslm_ids.offline.federated.config import DEFAULT_FL_CONFIG
from graphslm_ids.offline.training.train_hgt_flow_classifier import _compute_train_loss

if TYPE_CHECKING:
    from graphslm_ids.offline.federated.context import ExperimentContext
    from graphslm_ids.offline.training.noise_consensus import EACSController

MODEL_NAMES = ("hgt", "mlp", "cnn")
LOSS_TYPES = ("ce", "focal", "cb_focal")
FEATURE_CHUNK = 8192  # flows whose features are materialised at once (bounds peak memory)


@dataclass
class StepOutput:
    """One training batch: differentiable ``loss``, detached seed ``logits`` (n, C), ``labels`` (n,)."""

    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor


class ModelAdapter:
    name: str = ""

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def client_data(self, client_id: int, flow_ids: np.ndarray, seed: int) -> Any:
        raise NotImplementedError

    def client_eacs(self) -> EACSController | None:
        """A fresh per-client ``EACSController`` (``None`` when EACS is off or unsupported)."""
        return None

    def train_batches(self, data: Any, flow_ids: np.ndarray, labels: np.ndarray, *, seed: int) -> Iterable:
        raise NotImplementedError

    def forward_loss(
        self,
        model: nn.Module,
        batch: Any,
        *,
        class_weights: torch.Tensor | None,
        eacs: EACSController | None,
        round_idx: int,
    ) -> StepOutput | None:
        """``None`` means the batch has nothing to train on and is skipped."""
        raise NotImplementedError

    def predict(self, model: nn.Module, flow_ids: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """Global flow ids and float32 CPU logits ``(n, C)``, in the adapter's own order."""
        raise NotImplementedError

    def checkpoint_meta(self) -> dict[str, Any]:
        return {"model_name": self.name}


# -- flat baselines: networks -----------------------------------------------------------------


class FlowMLP(nn.Module):
    """MLP on the (standardized) flow feature vector: Linear + ReLU + Dropout per hidden size."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = int(in_dim)
        for hidden in hidden_dims:
            layers += [nn.Linear(width, int(hidden)), nn.ReLU(), nn.Dropout(float(dropout))]
            width = int(hidden)
        self.hidden = nn.Sequential(*layers)
        self.classifier = nn.Linear(width, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.hidden(x))


def pooled_length(input_len: int, kernel_size: int, pool: tuple[int, int], num_blocks: int) -> int:
    """Length left after ``num_blocks`` x (Conv1d(k, padding=k//2) -> MaxPool1d(*pool)); <= 0 if none."""
    length = int(input_len)
    for _ in range(num_blocks):
        length = length + 2 * (kernel_size // 2) - kernel_size + 1
        if length < pool[0]:
            return 0
        length = (length - pool[0]) // pool[1] + 1
    return length


class ByteCNN1D(nn.Module):
    """1D-CNN over the flattened ordered payload bytes of a flow's first K packets.

    ``num_blocks`` x ``Conv1d -> GroupNorm -> ReLU -> MaxPool1d``, a global max pool over the
    length, ``Dropout`` and a linear ``classifier``. GroupNorm instead of BatchNorm keeps
    every state entry a float parameter (no running stats, no batch-size dependence).
    Input ``(B, 1, input_len)``.
    """

    def __init__(
        self,
        input_len: int,
        num_classes: int,
        channels: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 7,
        num_groups: int = 8,
        pool: tuple[int, int] = (3, 2),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if pooled_length(input_len, kernel_size, pool, len(channels)) < 1:
            raise ValueError(
                f"input_len {input_len} is too short for {len(channels)} x (Conv1d k={kernel_size}, "
                f"MaxPool1d{tuple(pool)}); it must leave a length of at least 1"
            )
        blocks: list[nn.Module] = []
        in_ch = 1
        for ch in channels:
            blocks += [
                nn.Conv1d(in_ch, int(ch), int(kernel_size), padding=int(kernel_size) // 2),
                nn.GroupNorm(min(int(num_groups), int(ch)), int(ch)),
                nn.ReLU(),
                nn.MaxPool1d(*pool),
            ]
            in_ch = int(ch)
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(in_ch, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.features(x).amax(dim=-1)))


def _net_cfg(model_cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """``model.<key>`` over its defaults, so a config without the block still builds."""
    return {**DEFAULT_FL_CONFIG["model"][key], **(model_cfg.get(key) or {})}


def build_mlp(input_dim: int, num_classes: int, model_cfg: dict[str, Any]) -> FlowMLP:
    m = _net_cfg(model_cfg, "mlp")
    return FlowMLP(
        int(input_dim),
        int(num_classes),
        hidden_dims=tuple(int(h) for h in m["hidden_dims"]),
        dropout=float(m["dropout"]),
    )


def build_cnn(input_len: int, num_classes: int, model_cfg: dict[str, Any]) -> ByteCNN1D:
    m = _net_cfg(model_cfg, "cnn")
    return ByteCNN1D(
        int(input_len),
        int(num_classes),
        channels=tuple(int(c) for c in m["channels"]),
        kernel_size=int(m["kernel_size"]),
        num_groups=int(m["num_groups"]),
        dropout=float(m["dropout"]),
    )


# -- flat baselines: payload-byte input -------------------------------------------------------


def flow_packet_csr(owner: np.ndarray, num_flows: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR of each flow's packets: ``idx[ptr[f]:ptr[f + 1]]`` are flow ``f``'s packet ids in
    ascending order, which is the ``next_packet`` order (``graph_builder`` links rows
    ``i-1 -> i`` inside a flow). Packets with ``owner == -1`` belong to no flow and are skipped."""
    owner = np.asarray(owner, dtype=np.int64)
    packets = np.flatnonzero(owner >= 0)
    if packets.size and int(owner[packets].max()) >= num_flows:
        raise ValueError(f"packet owner {int(owner[packets].max())} is outside the {num_flows} flows")
    order = np.argsort(owner[packets], kind="stable")  # stable: ascending packet id inside a flow
    ptr = np.zeros(num_flows + 1, dtype=np.int64)
    np.cumsum(np.bincount(owner[packets], minlength=num_flows), out=ptr[1:])
    return ptr, packets[order]


def byte_columns(metadata: dict[str, Any] | None, prefix: str = "obyte_") -> np.ndarray:
    """Column indices of the ordered payload-byte features in ``packet_x``."""
    names = (metadata or {}).get("packet_feature_names")
    if not names:
        raise ValueError(
            "the graph metadata has no 'packet_feature_names'; the byte CNN cannot locate "
            f"the {prefix!r} payload-byte columns (rebuild the artifact with a current graph_builder)"
        )
    cols = [i for i, name in enumerate(names) if str(name).startswith(prefix)]
    if not cols:
        raise ValueError(f"no packet feature name starts with {prefix!r}")
    return np.asarray(cols, dtype=np.int64)


def flow_byte_matrix(
    flow_ids: np.ndarray,
    ptr: np.ndarray,
    idx: np.ndarray,
    packet_x: np.ndarray,
    byte_cols: np.ndarray,
    num_packets: int,
) -> np.ndarray:
    """``(n, num_packets * len(byte_cols))`` float32: each flow's byte columns of its first
    ``num_packets`` packets in packet order, zero-padded when the flow has fewer packets."""
    flow_ids = np.asarray(flow_ids, dtype=np.int64)
    byte_cols = np.asarray(byte_cols, dtype=np.int64)
    n = int(flow_ids.shape[0])
    out = np.zeros((n, int(num_packets), byte_cols.shape[0]), dtype=np.float32)
    starts = ptr[flow_ids]
    counts = np.minimum(ptr[flow_ids + 1] - starts, int(num_packets))
    total = int(counts.sum())
    if total:
        rows = np.repeat(np.arange(n), counts)
        slot = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        packets = idx[np.repeat(starts, counts) + slot]
        out[rows, slot] = packet_x[packets[:, None], byte_cols[None, :]]
    return out.reshape(n, int(num_packets) * byte_cols.shape[0])


# -- flat baselines: adapters -----------------------------------------------------------------


@dataclass
class TensorClientData:
    """A client's local rows: features ``x`` (n, D) and labels ``y`` (n,) int64."""

    x: torch.Tensor
    y: torch.Tensor


class _FlatAdapter(ModelAdapter):
    """Shared client-side recipe of the flat baselines; subclasses define the input.

    Subclasses set ``input_dim`` (width of one row) and ``store_dtype`` (client cache dtype)
    and implement :meth:`_features` and :meth:`build_model`.
    """

    input_dim: int
    store_dtype: torch.dtype = torch.float32

    def __init__(
        self,
        cfg: dict[str, Any],
        ctx: ExperimentContext,
        flow_stats: dict[str, list[float]],
        device: str | torch.device,
    ) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self.device = torch.device(device)
        self.flow_stats = {key: [float(v) for v in flow_stats[key]] for key in ("mean", "std")}
        client = {**DEFAULT_FL_CONFIG["client"], **cfg["client"]}
        self.loss_type = str(client["loss_type"]).lower()
        self.label_smoothing = float(client["label_smoothing"])
        self.focal_gamma = float(client["focal_gamma"])
        self.batch_size = int(client["batch_seed_flows"])
        self.eval_batch_size = int(cfg["eval"].get("batch_seed_flows", 512))

    def _features(self, flow_ids: np.ndarray) -> np.ndarray:
        """``(n, input_dim)`` float32 model input rows for ``flow_ids``."""
        raise NotImplementedError

    def _model_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def client_data(self, client_id: int, flow_ids: np.ndarray, seed: int) -> TensorClientData:
        ids = np.asarray(flow_ids, dtype=np.int64)
        x = torch.empty((ids.shape[0], self.input_dim), dtype=self.store_dtype)
        for start in range(0, ids.shape[0], FEATURE_CHUNK):
            x[start : start + FEATURE_CHUNK] = torch.from_numpy(self._features(ids[start : start + FEATURE_CHUNK]))
        return TensorClientData(x=x, y=torch.from_numpy(np.ascontiguousarray(self.ctx.labels[ids], dtype=np.int64)))

    def train_batches(self, data: TensorClientData, flow_ids: np.ndarray, labels: np.ndarray, *, seed: int):
        ids = torch.from_numpy(np.ascontiguousarray(flow_ids, dtype=np.int64))
        if ids.shape[0] != data.x.shape[0]:
            raise ValueError(f"client data has {data.x.shape[0]} rows but {ids.shape[0]} flow ids were given")
        return DataLoader(
            TensorDataset(data.x, data.y, ids),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
        )

    def forward_loss(
        self,
        model: nn.Module,
        batch: Any,
        *,
        class_weights: torch.Tensor | None,
        eacs: EACSController | None,
        round_idx: int,
    ) -> StepOutput | None:
        x, y, _ = batch
        if y.numel() == 0:
            return None
        x = self._model_input(x.to(self.device, dtype=torch.float32))
        y = y.to(self.device)
        logits = model(x).float()
        weight = None if class_weights is None else class_weights.to(self.device)
        loss = _compute_train_loss(
            logits,
            y,
            weight,
            loss_type=self.loss_type,
            label_smoothing=self.label_smoothing,
            focal_gamma=self.focal_gamma,
        )
        return StepOutput(loss=loss, logits=logits.detach(), labels=y)

    def predict(self, model: nn.Module, flow_ids: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        ids = np.array(flow_ids, dtype=np.int64)
        if ids.size == 0:
            return ids, torch.empty((0, self.ctx.num_classes), dtype=torch.float32)
        outs: list[torch.Tensor] = []
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for start in range(0, ids.shape[0], self.eval_batch_size):
                    x = torch.from_numpy(self._features(ids[start : start + self.eval_batch_size]))
                    outs.append(model(self._model_input(x.to(self.device))).float().cpu())
        finally:
            model.train(was_training)
        return ids, torch.cat(outs)


class MLPAdapter(_FlatAdapter):
    """:class:`FlowMLP` on ``flow_x`` standardized with the round-0 federated ``flow_stats``."""

    name = "mlp"

    def __init__(self, cfg, ctx, flow_stats, device) -> None:
        super().__init__(cfg, ctx, flow_stats, device)
        self.flow_x = ctx.artifact.node_features["flow"]
        self.input_dim = int(self.flow_x.shape[1])
        self.standardize = bool(cfg["data"].get("standardize_flow_features", True))
        self._mean = np.asarray(self.flow_stats["mean"], dtype=np.float32).reshape(1, -1)
        self._std = np.maximum(np.asarray(self.flow_stats["std"], dtype=np.float32).reshape(1, -1), 1e-6)
        if self._mean.shape[1] != self.input_dim or self._std.shape[1] != self.input_dim:
            raise ValueError(
                f"flow_stats have {self._mean.shape[1]} features but flow_x has {self.input_dim}"
            )

    def _features(self, flow_ids: np.ndarray) -> np.ndarray:
        x = np.asarray(self.flow_x[flow_ids], dtype=np.float32)
        return ((x - self._mean) / self._std).astype(np.float32, copy=False) if self.standardize else x

    def build_model(self) -> FlowMLP:
        return build_mlp(self.input_dim, self.ctx.num_classes, self.cfg["model"]).to(self.device)

    def checkpoint_meta(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "input_dim": self.input_dim,
            "label_names": dict(self.ctx.label_names),
            "flow_feature_stats": self.flow_stats,
        }


class CNNAdapter(_FlatAdapter):
    """:class:`ByteCNN1D` on the ordered payload bytes of each flow's first ``num_packets`` packets.

    Client rows are stored as float16 (bytes are ``byte / 255``) and cast to float32 per batch.
    """

    name = "cnn"
    store_dtype = torch.float16

    def __init__(self, cfg, ctx, flow_stats, device) -> None:
        super().__init__(cfg, ctx, flow_stats, device)
        m = _net_cfg(cfg["model"], "cnn")
        self.num_packets = int(m["num_packets"])
        if self.num_packets < 1:
            raise ValueError(f"model.cnn.num_packets must be >= 1, got {self.num_packets}")
        self.byte_cols = byte_columns(ctx.artifact.metadata, str(m["byte_prefix"]))
        self.packet_x = ctx.artifact.node_features["packet"]
        if int(self.byte_cols.max()) >= int(self.packet_x.shape[1]):
            raise ValueError(
                f"packet_feature_names lists column {int(self.byte_cols.max())} but packet_x has "
                f"only {int(self.packet_x.shape[1])} columns"
            )
        self.ptr, self.idx = flow_packet_csr(ctx.packet_owner, int(ctx.labels.shape[0]))
        self.input_dim = self.num_packets * int(self.byte_cols.shape[0])

    def _features(self, flow_ids: np.ndarray) -> np.ndarray:
        return flow_byte_matrix(flow_ids, self.ptr, self.idx, self.packet_x, self.byte_cols, self.num_packets)

    def _model_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)

    def build_model(self) -> ByteCNN1D:
        return build_cnn(self.input_dim, self.ctx.num_classes, self.cfg["model"]).to(self.device)

    def checkpoint_meta(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "input_len": self.input_dim,
            "num_packets": self.num_packets,
            "byte_columns": self.byte_cols.tolist(),
            "label_names": dict(self.ctx.label_names),
        }


def _model_name(value: Any) -> str:
    name = str(value or "hgt").lower()
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model.name {value!r}; expected one of {MODEL_NAMES}")
    return name


def build_adapter(
    cfg: dict[str, Any], ctx: Any, flow_stats: dict[str, list[float]], device: str | torch.device
) -> ModelAdapter:
    name = _model_name(cfg["model"].get("name", "hgt"))
    if name == "hgt":
        from graphslm_ids.offline.federated.hgt_adapter import HGTAdapter

        return HGTAdapter(cfg, ctx, flow_stats, device)
    if name == "mlp":
        return MLPAdapter(cfg, ctx, flow_stats, device)
    if name == "cnn":
        return CNNAdapter(cfg, ctx, flow_stats, device)
    raise AssertionError(name)  # pragma: no cover - MODEL_NAMES and this switch must agree


def rebuild_model(ckpt: dict[str, Any]) -> nn.Module:
    """Rebuild an untrained model from a checkpoint's ``checkpoint_meta`` keys, ``num_classes``
    and ``config["model"]``; the caller loads ``model_state_dict`` into it."""
    name = _model_name(ckpt.get("model_name", "hgt"))
    if name == "hgt":
        from graphslm_ids.offline.federated.hgt_adapter import build_hgt_model

        return build_hgt_model(
            node_input_dims=ckpt["node_input_dims"],
            edge_types=ckpt["edge_types"],
            num_classes=int(ckpt["num_classes"]),
            num_tactics=int(ckpt["num_tactics"]),
            model_cfg=ckpt["config"]["model"],
        )
    if name == "mlp":
        return build_mlp(int(ckpt["input_dim"]), int(ckpt["num_classes"]), ckpt["config"]["model"])
    if name == "cnn":
        return build_cnn(int(ckpt["input_len"]), int(ckpt["num_classes"]), ckpt["config"]["model"])
    raise AssertionError(name)  # pragma: no cover
