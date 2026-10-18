"""HGT behind the :class:`ModelAdapter` interface.

Lean client recipe (approved design): focal/CE with class-balanced weights from the client's
LOCAL labels, optional GCL auxiliary loss and EACS soft targets. No HGAA, TEW, DDP, EMA or
OOM retry. Each client trains on its own graph view (``build_client_artifact``), so a
relation that never appears in its batches gets grad ``None``, AdamW skips it, and its delta
is exactly zero. Prediction is experiment-only grading on the full graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from graphslm_ids.models.hgt import HeteroGraphTransformer
from graphslm_ids.offline.federated.client_graph import build_client_artifact
from graphslm_ids.offline.federated.config import DEFAULT_FL_CONFIG
from graphslm_ids.offline.federated.context import ExperimentContext, require_file
from graphslm_ids.offline.federated.models import ModelAdapter, StepOutput
from graphslm_ids.offline.training.neighbor_sampling import HeteroNeighborSampler, InMemoryNeighborBackend
from graphslm_ids.offline.training.noise_consensus import (
    EACSController,
    build_eacs_controller,
    neighbor_consensus,
    soft_target_focal_loss,
)
from graphslm_ids.offline.training.train_hgt_flow_classifier import (
    _compute_train_loss,
    _drop_edges,
    _set_tew_state,
    gcl_auxiliary_loss,
    load_class_technique_map,
    make_neighbor_loader,
    to_torch_batch,
)

BURST_KEYS = (("flow", "burst_neighbor", "flow"), ("flow", "rev_burst_neighbor", "flow"))
CONTAINS = ("flow", "contains", "packet")


@dataclass
class HGTClientData:
    backend: InMemoryNeighborBackend
    sampler: HeteroNeighborSampler


def build_hgt_model(
    *,
    node_input_dims: dict[str, int],
    edge_types: list[Any],
    num_classes: int,
    num_tactics: int,
    model_cfg: dict[str, Any],
) -> HeteroGraphTransformer:
    m = {**DEFAULT_FL_CONFIG["model"], **model_cfg}
    dims = {str(nt): int(d) for nt, d in node_input_dims.items()}
    return HeteroGraphTransformer(
        node_input_dims=dims,
        edge_types=[tuple(e) for e in edge_types],
        num_classes=int(num_classes),
        num_tactics=int(num_tactics),
        hidden_dim=int(m["hidden_dim"]),
        num_layers=int(m["num_layers"]),
        num_heads=int(m["num_heads"]),
        dropout=float(m["dropout"]),
        ffn_multiplier=int(m["ffn_multiplier"]),
        node_types=list(dims) + ["tactic"],
    )


def make_sampler(
    cfg: dict[str, Any], backend: InMemoryNeighborBackend, flow_stats: dict[str, list[float]], seed: int
) -> HeteroNeighborSampler:
    s = cfg["sampler"]
    return HeteroNeighborSampler(
        backend,
        hops=int(s.get("hops") or cfg["model"].get("num_layers", 2)),
        fanouts=dict(s.get("fanouts") or {}),
        reverse_fanouts=dict(s.get("reverse_fanouts") or {}),
        always_include_all_tactics=bool(s.get("always_include_all_tactics", True)),
        always_include_all_techniques=bool(s.get("always_include_all_techniques", True)),
        flow_feature_stats=flow_stats,
        standardize_flow_features=bool(cfg["data"].get("standardize_flow_features", True)),
        seed=int(seed),
    )


class HGTAdapter(ModelAdapter):
    name = "hgt"

    def __init__(
        self,
        cfg: dict[str, Any],
        ctx: ExperimentContext,
        flow_stats: dict[str, list[float]],
        device: str | torch.device,
    ) -> None:
        # The FL recipe has no TEW; clear the module flag a centralized run may have left on.
        _set_tew_state(False)
        self.cfg = cfg
        self.ctx = ctx
        self.flow_stats = {key: [float(v) for v in flow_stats[key]] for key in ("mean", "std")}
        self.device = torch.device(device)
        self.edge_types = list(ctx.backend.edge_types)
        self.use_sew = bool(cfg["data"].get("use_semantic_edge_weights", True))
        defaults = DEFAULT_FL_CONFIG["client"]
        client = {**defaults, **cfg["client"]}
        self.loss_type = str(client["loss_type"]).lower()
        self.label_smoothing = float(client["label_smoothing"])
        self.focal_gamma = float(client["focal_gamma"])
        self.drop_edge_prob = float(client["drop_edge_prob"] or 0.0)
        gcl = {**defaults["gcl"], **(client["gcl"] or {})}
        self.gcl_map = self._build_gcl_map(gcl)
        self.gcl_enabled = bool(self.gcl_map)
        self.gcl_weight = float(gcl["weight"])
        self.gcl_temperature = float(gcl["temperature"])
        self.gcl_n_negatives = int(gcl["n_negatives"])
        self._base_eacs = self._build_base_eacs({**defaults["eacs"], **(client["eacs"] or {})})
        workers = int(client["num_workers"])
        self._train_loader_cfg = {
            "train": {
                "batch_seed_flows": int(client["batch_seed_flows"]),
                "stratified_batch_sampling": bool(client["stratified_batch_sampling"]),
            },
            "dataloader": {
                "num_workers": workers,
                "pin_memory": bool(client["pin_memory"]),
                "persistent_workers": False,
                "prefetch_factor": 2,
            },
        }
        self._eval_loader_cfg = {
            "train": {
                "batch_seed_flows": int(cfg["eval"].get("batch_seed_flows", 512)),
                "stratified_batch_sampling": False,
            },
            "dataloader": {
                "num_workers": workers,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
            },
        }
        self._eval_seed = int(cfg["federated"].get("seed", 42)) + 7

    # -- construction helpers -------------------------------------------------------------

    def _label_mapping(self) -> dict[str, int]:
        return {name: idx for idx, name in self.ctx.label_names.items()}

    def _build_gcl_map(self, gcl: dict[str, Any]) -> dict[int, list[tuple[int, float]]]:
        if not gcl.get("enabled"):
            return {}
        csv_path = require_file(gcl.get("class_technique_map_csv"), "client.gcl.class_technique_map_csv")
        tech_to_idx = (self.ctx.backend.manifest or {}).get("technique_id_to_idx") or {}
        mapping = load_class_technique_map(
            csv_path=csv_path,
            label_mapping=self._label_mapping(),
            num_techniques=int(self.ctx.backend.num_techniques),
            technique_id_to_idx={str(k): int(v) for k, v in tech_to_idx.items()},
        )
        if not mapping:
            raise ValueError(
                f"client.gcl is enabled but {str(csv_path)!r} maps no class of this artifact to a technique"
            )
        return mapping

    def _build_base_eacs(self, e: dict[str, Any]) -> EACSController | None:
        if not e.get("enabled"):
            return None
        suspects = list(e["suspect_classes"] or [])
        unknown = sorted(set(suspects) - set(self.ctx.label_names.values()))
        if unknown:
            raise ValueError(f"client.eacs.suspect_classes has unknown classes {unknown}")
        anchor = None
        if e["anchor_mask_npy"]:
            anchor = np.load(require_file(e["anchor_mask_npy"], "client.eacs.anchor_mask_npy"))
        return build_eacs_controller(
            artifact=self.ctx.artifact,
            num_classes=self.ctx.num_classes,
            label_mapping=self._label_mapping(),
            suspect_classes=suspects,
            warmup_epochs=int(e["warmup_rounds"]),
            ema_decay=float(e["ema_decay"]),
            lambda_disambig=float(e["lambda_disambig"]),
            mitre_dir=str(e["mitre_dir"]),
            anchor_mask=anchor,
        )

    # -- ModelAdapter ---------------------------------------------------------------------

    def build_model(self) -> HeteroGraphTransformer:
        backend = self.ctx.backend
        return build_hgt_model(
            node_input_dims=dict(backend.feature_dims),
            edge_types=self.edge_types,
            num_classes=self.ctx.num_classes,
            num_tactics=int(backend.num_tactics),
            model_cfg=self.cfg["model"],
        ).to(self.device)

    def client_data(self, client_id: int, flow_ids: np.ndarray, seed: int) -> HGTClientData:
        view = build_client_artifact(self.ctx.artifact, np.asarray(flow_ids), owner=self.ctx.packet_owner)
        backend = InMemoryNeighborBackend(view)
        return HGTClientData(backend=backend, sampler=make_sampler(self.cfg, backend, self.flow_stats, seed))

    def client_eacs(self) -> EACSController | None:
        base = self._base_eacs
        if base is None:
            return None
        # Shared read-only masks, a fresh per-client EMA buffer: EACS state stays local.
        return EACSController(
            suspect_mask=base.suspect_mask,
            anchor_mask=base.anchor_mask,
            benign_class_id=base.benign_class_id,
            num_classes=base.num_classes,
            warmup_epochs=base.warmup_epochs,
            ema_decay=base.beta_buffer.decay,
            lambda_disambig=base.lambda_disambig,
        )

    def train_batches(self, data: HGTClientData, flow_ids: np.ndarray, labels: np.ndarray, *, seed: int):
        data.sampler.rng = np.random.default_rng(int(seed))
        loader, _ = make_neighbor_loader(
            np.asarray(flow_ids, dtype=np.int64),
            data.sampler,
            self._train_loader_cfg,
            shuffle=True,
            seed=int(seed),
            class_labels=np.asarray(labels, dtype=np.int64),
        )
        return loader

    def forward_loss(
        self,
        model: HeteroGraphTransformer,
        batch: Any,
        *,
        class_weights: torch.Tensor | None,
        eacs: EACSController | None,
        round_idx: int,
    ) -> StepOutput | None:
        nf, ei, ew, sm, sl = to_torch_batch(batch, self.edge_types, self.device, self.use_sew)
        if sl.numel() == 0:
            return None
        if self.drop_edge_prob > 0.0:
            ei, ew = _drop_edges(ei, ew, self.drop_edge_prob)
        if self.gcl_enabled:
            x_dict = model.encode(nf, ei, edge_weight_dict=ew)
            logits = model.classifier(x_dict["flow"])
        else:
            x_dict = None
            logits = model(nf, ei, edge_weight_dict=ew)
        seed_logits = logits[sm].float()
        weight = None if class_weights is None else class_weights.to(self.device)
        if eacs is not None:
            gids = torch.as_tensor(np.asarray(batch.seed_flow_ids, dtype=np.int64), device=self.device)
            consensus = self._consensus(logits, ei, sm, sl) if eacs.lambda_disambig < 1.0 else None
            targets = eacs.soft_targets(seed_logits, gids, sl, epoch=round_idx, consensus=consensus)
            loss = soft_target_focal_loss(
                seed_logits,
                targets,
                weight,
                loss_type=self.loss_type,
                label_smoothing=self.label_smoothing,
                focal_gamma=self.focal_gamma,
            )
        else:
            loss = _compute_train_loss(
                seed_logits,
                sl,
                weight,
                loss_type=self.loss_type,
                label_smoothing=self.label_smoothing,
                focal_gamma=self.focal_gamma,
            )
        if x_dict is not None:
            aux = self._gcl_loss(batch, nf, ei, sm, sl, x_dict)
            if aux is not None:
                loss = loss + self.gcl_weight * aux
        return StepOutput(loss=loss, logits=seed_logits.detach(), labels=sl)

    def predict(self, model: HeteroGraphTransformer, flow_ids: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        sampler = make_sampler(self.cfg, self.ctx.backend, self.flow_stats, self._eval_seed)
        loader, _ = make_neighbor_loader(
            np.asarray(flow_ids, dtype=np.int64),
            sampler,
            self._eval_loader_cfg,
            shuffle=False,
            seed=self._eval_seed,
        )
        ids: list[np.ndarray] = []
        outs: list[torch.Tensor] = []
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for batch in loader:
                    nf, ei, ew, sm, _ = to_torch_batch(batch, self.edge_types, self.device, self.use_sew)
                    logits = model(nf, ei, edge_weight_dict=ew)
                    ids.append(np.asarray(batch.seed_flow_ids, dtype=np.int64))
                    outs.append(logits[sm].float().cpu())
        finally:
            model.train(was_training)
        if not ids:
            return np.empty((0,), dtype=np.int64), torch.empty((0, self.ctx.num_classes), dtype=torch.float32)
        return np.concatenate(ids), torch.cat(outs)

    def checkpoint_meta(self) -> dict[str, Any]:
        backend = self.ctx.backend
        return {
            "model_name": self.name,
            "node_input_dims": {str(k): int(v) for k, v in backend.feature_dims.items()},
            "edge_types": [list(e) for e in self.edge_types],
            "num_tactics": int(backend.num_tactics),
            "label_names": dict(self.ctx.label_names),
            "flow_feature_stats": self.flow_stats,
        }

    # -- loss pieces ----------------------------------------------------------------------

    def _consensus(self, logits, ei, sm, sl) -> torch.Tensor | None:
        parts = [ei[k] for k in BURST_KEYS if k in ei and ei[k].numel() > 0]
        if not parts:
            return None
        seed_idx = torch.nonzero(sm, as_tuple=False).reshape(-1)
        probs = torch.softmax(logits.detach().float(), dim=1)
        return neighbor_consensus(probs, torch.cat(parts, dim=1), seed_idx, sl)

    def _gcl_loss(self, batch, nf, ei, sm, sl, x_dict) -> torch.Tensor | None:
        contains = ei.get(CONTAINS)
        if contains is None or contains.numel() == 0:
            return None
        flow_local, pkt_local = contains[0], contains[1]
        keep = sm[flow_local]
        if not bool(keep.any()):
            return None
        dense = torch.full((int(nf["flow"].shape[0]),), -1, dtype=torch.long, device=self.device)
        dense[torch.nonzero(sm, as_tuple=False).reshape(-1)] = sl
        pkt_class = dense[flow_local[keep]]
        valid = pkt_class >= 0
        if not bool(valid.any()):
            return None
        tech_lg = batch.local_to_global.get("technique")
        if tech_lg is None:
            return None
        if isinstance(tech_lg, torch.Tensor):
            tech_ids = tech_lg.detach().cpu().tolist()
        else:
            tech_ids = np.asarray(tech_lg).tolist()
        g2l = {int(g): i for i, g in enumerate(tech_ids)}
        local_map: dict[int, list[tuple[int, float]]] = {}
        for cls_idx, pairs in self.gcl_map.items():
            remapped = [(g2l[int(t)], float(w)) for t, w in pairs if int(t) in g2l]
            if remapped:
                local_map[int(cls_idx)] = remapped
        if not local_map:
            return None
        return gcl_auxiliary_loss(
            node_embeddings={"packet": x_dict["packet"], "technique": x_dict["technique"]},
            seed_packet_ids=pkt_local[keep][valid],
            seed_packet_flow_label_idx=pkt_class[valid],
            class_to_technique_idx=local_map,
            num_techniques=int(x_dict["technique"].shape[0]),
            temperature=self.gcl_temperature,
            n_negatives=self.gcl_n_negatives,
            device=self.device,
        )
