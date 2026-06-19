from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from graphslm_ids.models.hgt import EdgeKey, HeteroGraphTransformer


@dataclass
class HGTOutput:
    logits: Any
    edge_attention: dict[EdgeKey, Any]
    label_to_index: dict[str, int]


class HGTRuntime:
    """Runtime wrapper for the trained HGT flow classifier checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        graph_meta_json: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for HGTRuntime. Install requirements-ml.txt.") from exc

        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        self.checkpoint_path = Path(checkpoint_path)
        self.graph_meta_json = Path(graph_meta_json) if graph_meta_json is not None else None
        self.payload = self._load_checkpoint(self.checkpoint_path)
        self.edge_types = [_edge_key(edge) for edge in self.payload["edge_types"]]
        self.label_names = {
            int(key): str(value) for key, value in self.payload.get("label_names", {}).items()
        }
        if not self.label_names:
            self.label_names = {idx: str(idx) for idx in range(int(self.payload["num_classes"]))}
        self.label_to_index = {label: idx for idx, label in self.label_names.items()}
        self.flow_feature_stats = self.payload.get("flow_feature_stats") or {}

        model_config = (self.payload.get("config") or {}).get("model", {})
        node_input_dims = {str(k): int(v) for k, v in self.payload["node_input_dims"].items()}
        # Reconstruct the full node-type set from the checkpoint. The model
        # defaults to ["flow", "packet", "technique", "tactic"] when node_types is
        # omitted, which silently drops any extra projected type (e.g. v3 "host")
        # and breaks state_dict loading. Derive it from node_input_dims (+ the
        # id-only "tactic" node, which never carries an input dim).
        node_types = list(node_input_dims)
        if "tactic" not in node_types:
            node_types.append("tactic")
        self.model = HeteroGraphTransformer(
            node_input_dims=node_input_dims,
            node_types=node_types,
            edge_types=self.edge_types,
            num_classes=int(self.payload["num_classes"]),
            num_tactics=int(self.payload["num_tactics"]),
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            num_layers=int(model_config.get("num_layers", 2)),
            num_heads=int(model_config.get("num_heads", 4)),
            dropout=float(model_config.get("dropout", 0.0)),
            ffn_multiplier=int(model_config.get("ffn_multiplier", 2)),
        ).to(self.device)
        self.model.load_state_dict(self.payload["model_state_dict"])
        self.model.eval()

    @classmethod
    def from_model(
        cls,
        model: Any,
        label_to_index: dict[str, int],
        device: str = "cpu",
    ) -> "HGTRuntime":
        obj = cls.__new__(cls)
        import torch

        obj.torch = torch
        obj.device = torch.device(device)
        obj.model = model.to(obj.device)
        obj.model.eval()
        obj.edge_types = list(getattr(model, "edge_types", []))
        obj.label_to_index = dict(label_to_index)
        obj.label_names = {idx: label for label, idx in label_to_index.items()}
        obj.flow_feature_stats = {}
        obj.payload = {}
        obj.checkpoint_path = None
        obj.graph_meta_json = None
        return obj

    def infer(self, sub: Any) -> HGTOutput:
        torch = self.torch
        node_features = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in sub.node_features.items()
        }
        edge_index = self._edge_index_dict(sub.edge_index_dict)
        edge_weight = self._edge_weight_dict(sub.edge_weight_dict)
        with torch.no_grad():
            logits, attention = self.model(
                node_features=node_features,
                edge_index_dict=edge_index,
                edge_weight_dict=edge_weight,
                return_attention=True,
            )
        return HGTOutput(
            logits=logits[int(sub.seed_flow_local_idx)].detach().cpu(),
            edge_attention={key: value.detach().cpu() for key, value in attention.items()},
            label_to_index=dict(self.label_to_index),
        )

    def aggregate_packet_attention(self, sub: Any, attention: dict[EdgeKey, Any]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for edge_key, values in attention.items():
            src_type, _, dst_type = edge_key
            if src_type != "packet" and dst_type != "packet":
                continue
            edge_index = sub.edge_index_dict.get(edge_key)
            if edge_index is None:
                continue
            edge_np = _to_numpy(edge_index).astype(np.int64)
            value_np = _to_numpy(values).reshape(-1)
            if edge_np.shape[1] == 0 or value_np.size == 0:
                continue
            packet_indices = edge_np[0] if src_type == "packet" else edge_np[1]
            for idx, score in zip(packet_indices.tolist(), value_np.tolist()):
                packet_id = sub.packet_local_to_id.get(int(idx))
                if packet_id is None:
                    continue
                scores[packet_id] = max(scores.get(packet_id, 0.0), float(score))
        return scores

    def _load_checkpoint(self, path: Path) -> dict[str, Any]:
        torch = self.torch
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _edge_index_dict(self, edge_index_dict: dict[EdgeKey, Any]) -> dict[EdgeKey, Any]:
        torch = self.torch
        result: dict[EdgeKey, Any] = {
            edge_type: torch.empty((2, 0), dtype=torch.long, device=self.device)
            for edge_type in self.edge_types
        }
        for key, value in edge_index_dict.items():
            result[_edge_key(key)] = torch.as_tensor(value, dtype=torch.long, device=self.device)
        return result

    def _edge_weight_dict(self, edge_weight_dict: dict[EdgeKey, Any]) -> dict[EdgeKey, Any] | None:
        if not edge_weight_dict:
            return None
        torch = self.torch
        return {
            _edge_key(key): torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in edge_weight_dict.items()
        }


def _edge_key(value: Any) -> EdgeKey:
    if isinstance(value, tuple) and len(value) == 3:
        return (str(value[0]), str(value[1]), str(value[2]))
    if isinstance(value, list) and len(value) == 3:
        return (str(value[0]), str(value[1]), str(value[2]))
    if isinstance(value, str) and value.count("__") == 2:
        a, b, c = value.split("__")
        return (a, b, c)
    raise ValueError(f"Invalid edge key: {value!r}")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
