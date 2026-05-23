from __future__ import annotations

import argparse
from contextlib import nullcontext as _nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import random
import threading
import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from graphslm_ids.offline.training.hetero_graph_artifact import (
    load_graph_store_artifact,
    load_three_tier_graph_artifact,
)
from graphslm_ids.offline.training.neighbor_sampling import (
    FlowSeedDataset,
    HeteroNeighborSampler,
    InMemoryNeighborBackend,
    MiniBatchSubgraph,
    NeighborBackend,
    NeighborSamplingCollator,
    _make_worker_init_fn,
)
from graphslm_ids.offline.training.on_disk_graph_store import OnDiskHeteroGraphStore
from graphslm_ids.models.hgt import HeteroGraphTransformer
from graphslm_ids.utils.io import ensure_dir, read_yaml, write_json


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "source": "npz",
        "graph_npz": "data/processed/graph_artifact_3tier_t082_k5.npz",
        "graph_meta_json": "data/processed/graph_artifact_3tier_t082_k5.meta.json",
        "graph_store_root": "data/graph_store_v1",
        "read_sealed_only": True,
        "packet_feature": "semantic",
        "add_reverse_edges": True,
        "standardize_flow_features": True,
        "use_semantic_edge_weights": True,
        # v8.5 TEW — wire delta_time_seconds (next_packet edge attr) as edge_weight
        # so HGT layer's `scores += log(edge_weight)` becomes a temporal-locality
        # bias: small Δt amplifies attention (burst detection), large Δt suppresses.
        # Transform: edge_weight = 1.0 / (Δt + tew_epsilon).
        "use_temporal_edge_weights": False,
        "tew_epsilon": 1.0e-3,
    },
    "model": {
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "ffn_multiplier": 2,
    },
    "train": {
        "output_dir": "outputs/hgt_flow_classifier_t082_k5_l3_d01",
        "epochs": 150,
        "lr": 1e-3,
        "weight_decay": 5e-5,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "patience": 30,
        "class_weight": "balanced",
        "seed": 42,
        "device": "auto",
        "monitor": "val_macro_f1",
        "log_every": 1,
        "amp": True,
        "activation_checkpointing": False,
        "batch_seed_flows": 256,
        "grad_accum_steps": 1,
        "compile": False,
        "tf32": True,
        "scheduler": "cosine_annealing",
        "scheduler_pct_start": 0.05,
        "scheduler_eta_min": 1e-5,
        "grad_clip_norm": 1.0,
        "amp_dtype": "auto",
        # Loss function configurability
        "loss_type": "ce",           # 'ce' | 'focal' | 'cb_focal'
        "label_smoothing": 0.0,      # 0.0-0.2. 0.1 recommended.
        "focal_gamma": 2.0,          # focal loss focusing parameter
        # QUALITY v4: EMA model weights for smoother validation
        "ema_enabled": False,        # set true for +1-3% F1 on imbalanced data
        "ema_decay": 0.999,          # 0.999 = effective window ~1000 steps
        # DropEdge regularization (training-only graph augmentation)
        "drop_edge_prob": 0.0,       # 0.0-0.2. 0.1 typical for GNN regularization.
        # Optimizer numerics (BF16-stable defaults)
        "adamw_eps": 1.0e-6,
        "adamw_betas": (0.9, 0.95),
        # SAM (Sharpness-Aware Minimization, Foret et al. ICLR 2021).
        # Requires grad_accum_steps=1.
        "sam_enabled": False,
        "sam_rho": 0.05,
        # Skip val eval at epoch 1 (OneCycle warmup → near-random val anyway)
        "skip_val_first_epoch": False,
    },
    "sampler": {
        "hops": None,
        "fanouts": {
            "flow__contains__packet": 20,
            "packet__next_packet__packet": 4,
            "packet__matches_technique__technique": 5,
            "flow__matches_technique__technique": 5,
            "technique__belongs_to_tactic__tactic": 1,
        },
        "reverse_fanouts": {
            "rev_contains": 1,
            "rev_next_packet": 1,
            "rev_matches_technique": 0,
            "rev_belongs_to_tactic": 0,
        },
        "always_include_all_tactics": True,
        "always_include_all_techniques": True,
    },
    "dataloader": {
        "num_workers": -1,
        "prefetch_factor": 4,
        "pin_memory": True,
        "persistent_workers": True,
    },
}


def resolve_amp(config: dict, device: torch.device) -> tuple[bool, torch.dtype]:
    """Resolve mixed precision settings based on config and hardware capability.

    Returns ``(use_amp, amp_dtype)``.

    Only bfloat16 is supported (float16 overflows in this model). AMP is
    enabled only when:
      1. config["train"]["amp"] is True, AND
      2. the device natively supports bfloat16 (A100, A40, RTX 30xx+, etc.)

    The edge-softmax and attention norm layers in hgt.py contain explicit
    float32 guards that prevent the non-finite-gradient issue seen on T4
    (which lacks native bfloat16). On hardware with native bfloat16 these
    guards keep those ops in float32 while the rest of the forward/backward
    runs in bfloat16, giving speed + VRAM savings without numeric instability.

    Confirm a smoke run logs ``[diag] first optimizer step | grad_norm=<non-zero>``
    before committing to a full training run with AMP enabled.
    """
    if device.type != "cuda":
        return False, torch.float32

    want_amp = bool(config.get("train", {}).get("amp", False))
    if not want_amp:
        return False, torch.float32

    if not torch.cuda.is_bf16_supported():
        print(
            "[AMP] bfloat16 not supported on this device — falling back to FP32.",
            flush=True,
        )
        return False, torch.float32

    print("[AMP] bfloat16 enabled (native hardware support detected).", flush=True)
    return True, torch.bfloat16


def _wandb_init(config: dict, rank: int) -> bool:
    """Initialize a W&B run on rank-0. Returns True if active, False if skipped."""
    if rank != 0:
        return False
    wcfg = config.get("wandb", {}) or {}
    if not wcfg.get("project"):
        return False
    try:
        import wandb  # optional dependency
        run_name = wcfg.get("run_name") or (config.get("experiment") or {}).get("source_name")
        wandb.init(
            project=wcfg["project"],
            entity=wcfg.get("entity") or None,
            name=run_name or None,
            config=config,
            resume="allow",
        )
        return True
    except Exception:
        return False


def _wandb_log(entry: dict, use_wandb: bool) -> None:
    if not use_wandb:
        return
    try:
        import wandb
        flat: dict = {"epoch": entry["epoch"]}
        for split in ("train", "val"):
            for k, v in (entry.get(split) or {}).items():
                if isinstance(v, (int, float)):
                    flat[f"{split}/{k}"] = v
        wandb.log(flat, step=entry["epoch"])
    except Exception:
        pass


def _wandb_finish(use_wandb: bool) -> None:
    if not use_wandb:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


def _auto_dataloader_workers(num_workers_arg: int, n_gpus: int) -> int:
    """Resolve num_workers='-1' (auto) to a sensible CPU-count-based value.

    Default policy: reserve ~1 CPU per training thread, divide the rest evenly
    among GPUs so each device has its own prefetch fleet. Capped at 8 per GPU
    because neighbor sampling is memory-light but startup cost grows.
    """
    if num_workers_arg >= 0:
        return num_workers_arg
    cpu_n = os.cpu_count() or 1
    workers_per_gpu = max(1, min(cpu_n // max(n_gpus, 1) - 1, 8))
    return workers_per_gpu


def _log_epoch_diagnostics(
    epoch: int,
    elapsed_seconds: float,
    device: torch.device,
    rank: int = 0,
) -> None:
    """Emit one-line per-epoch diagnostic for Phase 1 speed-up tracking.

    Format: ``[diag] epoch=N | wall=X.Xs | peak_vram_gb=Y.YY``

    Peak VRAM is queried from ``torch.cuda.max_memory_allocated()`` then
    reset via ``reset_peak_memory_stats()`` so each epoch reports its OWN
    peak, not a monotonic high-water mark across the full run.

    Only emits on rank 0 — DDP non-rank-0 ranks stay silent.
    """
    if rank != 0:
        return
    parts = [f"epoch={epoch}", f"wall={elapsed_seconds:.1f}s"]
    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb = peak_bytes / (1024 ** 3)
        parts.append(f"peak_vram_gb={peak_gb:.2f}")
        torch.cuda.reset_peak_memory_stats(device)
    print(f"[diag] {' | '.join(parts)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact HGT-style flow classifier on the selected t082 three-tier graph artifact."
    )
    parser.add_argument("--config", default="configs/hgt.example.yaml")
    parser.add_argument("--graph-npz", default=None)
    parser.add_argument("--graph-meta-json", default=None)
    parser.add_argument("--graph-store-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision training/evaluation.")
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Trade extra compute for lower activation memory during HGT training.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--compile", action="store_true",
        help="Wrap HGT in torch.compile(mode='reduce-overhead'). Requires PyTorch >= 2.0.",
    )
    parser.add_argument(
        "--no-tf32", action="store_true",
        help="Disable TF32 matmul (default: on, ~2x speedup on Ampere+).",
    )
    parser.add_argument(
        "--no-scheduler", dest="no_scheduler", action="store_true",
        help="Disable OneCycleLR scheduler (use constant LR).",
    )
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if config_path.exists():
        loaded = read_yaml(config_path)
        if not isinstance(loaded, dict):
            raise ValueError("HGT config must be a YAML mapping.")
        config = deep_update(config, loaded)
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = deepcopy(config)
    if args.graph_npz is not None:
        config["data"]["graph_npz"] = args.graph_npz
    if args.graph_meta_json is not None:
        config["data"]["graph_meta_json"] = args.graph_meta_json
    if args.graph_store_root is not None:
        config["data"]["source"] = "graph_store"
        config["data"]["graph_store_root"] = args.graph_store_root
    if args.output_dir is not None:
        config["train"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.hidden_dim is not None:
        config["model"]["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        config["model"]["num_layers"] = args.num_layers
    if args.num_heads is not None:
        config["model"]["num_heads"] = args.num_heads
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.weight_decay is not None:
        config["train"]["weight_decay"] = args.weight_decay
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.amp:
        config["train"]["amp"] = True
    if args.activation_checkpointing:
        config["train"]["activation_checkpointing"] = True
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.compile:
        config["train"]["compile"] = True
    if args.no_tf32:
        config["train"]["tf32"] = False
    if getattr(args, "no_scheduler", False):
        config["train"]["scheduler"] = "none"
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _tune_cuda_backends(device: torch.device, enable_tf32: bool = True) -> None:
    """Enable TF32 matmul and cuDNN benchmark for sustained training throughput.

    No-op on CPU. TF32 trades ~3 mantissa bits for ~2x matmul speed on Ampere+
    and is the right default for graph transformer training where the loss
    landscape is robust to the precision drop.
    """
    if device.type != "cuda":
        return
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # cudnn.benchmark autotunes convolution algorithms; harmless for non-conv
    # workloads and a clear win when fanout/seed-batch shapes stabilise.
    torch.backends.cudnn.benchmark = True


# ── async checkpoint save ─────────────────────────────────────────────────────

_ckpt_save_thread: threading.Thread | None = None


def _save_checkpoint_bg(state: dict[str, Any], path: Path) -> None:
    """Save the checkpoint in a background thread; join the previous save first.

    Caller MUST CPU-clone any GPU tensors in ``state`` (especially the model
    state_dict) before invoking this, otherwise the background write races with
    the optimizer mutating the live CUDA storage.
    """
    global _ckpt_save_thread
    if _ckpt_save_thread is not None:
        _ckpt_save_thread.join()
    _ckpt_save_thread = threading.Thread(
        target=torch.save, args=(state, path), daemon=True,
    )
    _ckpt_save_thread.start()


def _join_checkpoint_bg() -> None:
    global _ckpt_save_thread
    if _ckpt_save_thread is not None:
        _ckpt_save_thread.join()
        _ckpt_save_thread = None


def setup_distributed() -> tuple[int, int, int]:
    """Return ``(rank, local_rank, world_size)`` and initialise NCCL/Gloo if launched via torchrun.

    Single-process (no ``LOCAL_RANK``) returns ``(0, 0, 1)`` so the same code
    runs unchanged on Kaggle / local CPU.  When launched via
    ``torchrun --nproc_per_node=N`` the function picks NCCL (GPU) or Gloo (CPU)
    backend automatically — same pattern as ``train_student_cnn.py``.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        # Gloo lets us smoke-test DDP on a CPU-only laptop / CI.
        dist.init_process_group(backend="gloo")
    return rank, local_rank, world_size


def teardown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _ddp_device(local_rank: int) -> torch.device:
    return torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")


def _maybe_compile(model: torch.nn.Module, enabled: bool, rank: int = 0) -> torch.nn.Module:
    """Wrap the plain HGT module in ``torch.compile`` when ``enabled`` is true.

    Must be called on the UN-wrapped module, BEFORE DDP — DDP then wraps the
    compiled module, which is the combination PyTorch supports. (The previous
    ``torch.compile(DDP(model))`` order silently broke gradient propagation
    on neighbor-sampled, variable-shape batches.)

    ``compile`` defaults to OFF in the configs: it trades a fragile dynamo +
    DDP + activation-checkpointing interaction for a modest speedup. Enable it
    only after a run has been confirmed to learn correctly without it.
    """
    if not enabled:
        return model
    try:
        compiled = torch.compile(model, mode="default", dynamic=True)
        if rank == 0:
            print("[torch.compile] HGT module compiled (mode='default', dynamic=True)", flush=True)
        return compiled  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover - depends on installed PyTorch
        if rank == 0:
            print(f"[torch.compile] skipped — {exc}", flush=True)
        return model


def stratified_split(
    labels: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label in sorted(np.unique(labels).tolist()):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)
        n = int(class_indices.shape[0])
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3 and test_ratio > 0:
            n_test = max(1, n_test)
        if n >= 3 and val_ratio > 0:
            n_val = max(1, n_val)
        # Ensure at least 1 training sample per class: reduce val first, then test.
        if n_test + n_val >= n:
            overflow = n_test + n_val - n + 1
            n_val = max(0, n_val - overflow)
        if n_test >= n:
            n_test = max(0, n - 1)

        test_indices.extend(class_indices[:n_test].tolist())
        val_indices.extend(class_indices[n_test : n_test + n_val].tolist())
        train_indices.extend(class_indices[n_test + n_val :].tolist())

    train = np.asarray(train_indices, dtype=np.int64)
    val = np.asarray(val_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test



def label_name_mapping(metadata: dict[str, Any], labels: np.ndarray) -> dict[int, str]:
    mapping = metadata.get("label_mapping", {})
    if isinstance(mapping, dict) and mapping:
        return {int(idx): str(name) for name, idx in mapping.items()}
    return {int(label): str(label) for label in sorted(np.unique(labels).tolist())}


def load_neighbor_backend(config: dict[str, Any]) -> NeighborBackend:
    source = str(config["data"].get("source", "npz")).lower()
    if source in {"on_disk_graph_store", "graph_store_csr", "graph_store"}:
        graph_store_root = Path(config["data"]["graph_store_root"])
        try:
            return OnDiskHeteroGraphStore(graph_store_root)
        except (FileNotFoundError, ValueError):
            if source != "graph_store":
                raise
            artifact = load_graph_store_artifact(
                graph_store_root=graph_store_root,
                packet_feature=str(config["data"]["packet_feature"]),
                add_reverse_edges=bool(config["data"]["add_reverse_edges"]),
                sealed_only=bool(config["data"].get("read_sealed_only", True)),
            )
            return InMemoryNeighborBackend(artifact)

    artifact = load_three_tier_graph_artifact(
        graph_npz=Path(config["data"]["graph_npz"]),
        graph_meta_json=Path(config["data"]["graph_meta_json"]),
        packet_feature=str(config["data"]["packet_feature"]),
        add_reverse_edges=bool(config["data"]["add_reverse_edges"]),
    )
    return InMemoryNeighborBackend(artifact)


def backend_splits(
    backend: NeighborBackend,
    labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_ids = getattr(backend, "split_ids", None)
    if callable(split_ids):
        train = split_ids("train")
        val = split_ids("val")
        test = split_ids("test")
        if train is not None and val is not None and test is not None:
            return (
                np.asarray(train, dtype=np.int64),
                np.asarray(val, dtype=np.int64),
                np.asarray(test, dtype=np.int64),
            )
    return stratified_split(
        labels=labels,
        val_ratio=float(config["train"]["val_ratio"]),
        test_ratio=float(config["train"]["test_ratio"]),
        seed=seed,
    )


def compute_flow_feature_stats_backend(
    backend: NeighborBackend,
    train_ids: np.ndarray,
    chunk_size: int = 65_536,
) -> dict[str, list[float]]:
    train_ids = np.asarray(train_ids, dtype=np.int64)
    dim = int(backend.feature_dims["flow"])
    if train_ids.size == 0:
        return {"mean": [0.0] * dim, "std": [1.0] * dim}

    total = np.zeros((dim,), dtype=np.float64)
    total_sq = np.zeros((dim,), dtype=np.float64)
    count = 0
    for start in range(0, int(train_ids.shape[0]), chunk_size):
        ids = train_ids[start : start + chunk_size]
        rows = backend.get_flow_features(ids).astype(np.float64)
        total += rows.sum(axis=0)
        total_sq += np.square(rows).sum(axis=0)
        count += int(rows.shape[0])
    mean = total / max(count, 1)
    var = np.maximum(total_sq / max(count, 1) - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-6)
    return {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def class_weights_from_backend(
    backend: NeighborBackend,
    train_idx: np.ndarray,
    num_classes: int,
    max_weight: float = float("inf"),
    weight_method: str = "inverse",
    cb_beta: float = 0.999,
    use_manifest_weights: bool = False,
) -> torch.Tensor:
    """Compute per-class loss weights from training labels.

    weight_method:
      - "inverse": w_c = N / (K * n_c), classic inverse frequency.
      - "cb": Class-Balanced (Cui et al. CVPR 2019), w_c = (1 - β) / (1 - β^n_c).
        Approximates effective number of samples; less aggressive on extreme-tail
        classes than inverse frequency, more stable under severe imbalance.

    Weights are normalized so mean = 1.0 (loss scale comparable to unweighted CE).

    Auto-adaptation: weights are always derived from the CURRENT split's label
    counts at runtime. ``use_manifest_weights`` (default False) must be set
    explicitly to honor manifest-baked weights — otherwise swapping/growing the
    dataset would silently reuse a stale distribution.
    """
    if use_manifest_weights:
        manifest_weights = (backend.manifest or {}).get("class_weights")
        if isinstance(manifest_weights, list) and len(manifest_weights) >= num_classes:
            weights = np.array(manifest_weights[:num_classes], dtype=np.float32)
            if max_weight < float("inf"):
                weights = np.minimum(weights, max_weight)
            return torch.from_numpy(weights)
    labels = backend.get_flow_labels(np.asarray(train_idx, dtype=np.int64))
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if weight_method == "cb":
        effective_num = 1.0 - np.power(cb_beta, np.maximum(counts, 1.0))
        weights = (1.0 - cb_beta) / np.maximum(effective_num, 1e-12)
    else:
        weights = np.zeros(num_classes, dtype=np.float64)
        nonzero = counts > 0
        weights[nonzero] = counts[nonzero].sum() / (float(num_classes) * counts[nonzero])
    nonzero_weights = weights[weights > 0]
    weight_mean = float(nonzero_weights.mean()) if nonzero_weights.size > 0 else 1.0
    if weight_mean > 0:
        weights = weights / weight_mean
    weights = weights.astype(np.float32)
    if max_weight < float("inf"):
        weights = np.minimum(weights, max_weight)
    return torch.from_numpy(weights)


def _compute_monitor_score(monitor: str, val_metrics: dict) -> float:
    """Map a monitor name → a scalar to MAXIMIZE (higher = better).

    Dataset-agnostic: reads only the per-epoch ``val_metrics`` dict, so it
    works unchanged for any num_classes / class distribution.

    Supported monitors:
      - "val_macro_f1": macro-averaged F1 (favors minority-class balance).
      - "val_accuracy": overall accuracy (favors majority classes).
      - "val_balanced": 0.5·(accuracy + macro_f1) — optimizes BOTH jointly, so
        checkpoint selection prefers a model strong on majority AND minority.
      - "val_loss": negative validation loss (lower loss → higher score).

    NaN propagates (e.g. epoch-1 val skipped) and the caller's isnan guard
    handles it.
    """
    if monitor == "val_macro_f1":
        return float(val_metrics["macro_f1"])
    if monitor == "val_accuracy":
        return float(val_metrics["accuracy"])
    if monitor == "val_balanced":
        return 0.5 * (float(val_metrics["accuracy"]) + float(val_metrics["macro_f1"]))
    if monitor == "val_loss":
        vl = val_metrics.get("loss")
        return -(float(vl) if vl is not None else float("inf"))
    raise ValueError(
        f"Unknown monitor metric {monitor!r}. Supported: "
        f"'val_macro_f1', 'val_accuracy', 'val_balanced', 'val_loss'."
    )


def _compute_train_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None,
    loss_type: str = "ce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Unified training loss: CE | Focal.

    - 'ce': F.cross_entropy with optional label_smoothing. Best for mild imbalance.
    - 'focal'/'cb_focal': FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t). Down-weights
      easy examples. Designed for severe imbalance. Paper default focal_gamma=2.0.

    Eval loss must stay plain CE (no smoothing) for raw loss comparability.
    """
    if loss_type in ("focal", "cb_focal"):
        # Focal loss (Lin et al. ICCV 2017): FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).
        # cb_focal is identical at the function level; the behavioral difference is
        # that cb_focal expects `weight` to come from Class-Balanced method (Cui et al. 2019).
        # Label smoothing: focal modulation is still keyed on TRUE-class probability,
        # but the per-sample CE component is replaced by smoothed CE
        #     (1-eps)·NLL_true + eps·(-mean(log_probs))
        # so configs that set label_smoothing>0 actually get regularization (prior
        # focal path silently dropped it — see BUG #1 in v8.x audit).
        log_probs = F.log_softmax(logits, dim=-1)
        targets = labels.unsqueeze(-1)
        log_p_t = log_probs.gather(dim=-1, index=targets).squeeze(-1)
        p_t = log_p_t.exp()
        focal_factor = (1.0 - p_t).pow(focal_gamma)
        if label_smoothing > 0.0:
            nll_true = -log_p_t
            nll_uniform = -log_probs.mean(dim=-1)
            base_loss = (1.0 - label_smoothing) * nll_true + label_smoothing * nll_uniform
        else:
            base_loss = -log_p_t
        loss = focal_factor * base_loss
        if weight is not None:
            alpha_t = weight.gather(dim=0, index=labels)
            loss = alpha_t * loss
        return loss.mean()
    return F.cross_entropy(
        logits, labels,
        weight=weight,
        label_smoothing=label_smoothing,
    )


def _drop_edges(
    edge_index: dict[tuple[str, str, str], torch.Tensor],
    edge_weight: dict[tuple[str, str, str], torch.Tensor] | None,
    drop_prob: float,
) -> tuple[
    dict[tuple[str, str, str], torch.Tensor],
    dict[tuple[str, str, str], torch.Tensor] | None,
]:
    """Random edge dropout (DropEdge) for graph regularization.

    Drops a fraction of edges uniformly at random per edge type. Only applied
    during training — eval/test paths see the full graph. Standard regularization
    for GNN that reduces over-smoothing and over-fitting on dense subgraphs.

    Returns new dicts; original tensors are not modified in-place.
    """
    if drop_prob <= 0.0:
        return edge_index, edge_weight
    new_ei: dict[tuple[str, str, str], torch.Tensor] = {}
    new_ew: dict[tuple[str, str, str], torch.Tensor] | None = (
        {} if edge_weight is not None else None
    )
    keep_prob = 1.0 - drop_prob
    for et, ei in edge_index.items():
        if ei.numel() == 0:
            new_ei[et] = ei
            if new_ew is not None and et in edge_weight:
                new_ew[et] = edge_weight[et]
            continue
        mask = torch.rand(ei.shape[1], device=ei.device) < keep_prob
        new_ei[et] = ei[:, mask]
        if new_ew is not None and et in edge_weight:
            new_ew[et] = edge_weight[et][mask]
    return new_ei, new_ew


class EMA:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of trainable parameters, updated after each optimizer
    step: shadow = decay * shadow + (1 - decay) * param. Validation uses the EMA
    weights which are smoother → +1-3% F1 on imbalanced classification typically.

    Usage:
        ema = EMA(model, decay=0.999)
        # training loop:
        for batch in loader:
            optimizer.step()
            ema.update(model)
        # before eval:
        ema.apply_shadow(model)
        val_metrics = evaluate(model)
        ema.restore(model)

    The shadow tensors live on the same device as model parameters. Memory
    overhead = 1× model size in params (plus another 1× during apply_shadow
    via backup). For 8M params @ fp32, ~64 MB — negligible on L40S 48GB.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Swap live params with shadow (EMA) params. Caller MUST call restore() later."""
        if self.backup:
            raise RuntimeError("EMA.apply_shadow() called twice without restore() — backup not empty.")
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Restore live params from backup. Must be called after apply_shadow()."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def _to_device(value: Any, device: torch.device, non_blocking: bool, dtype: Any = None) -> torch.Tensor:
    """Get a tensor on ``device``, handling numpy / pinned / unpinned inputs.

    The DataLoader can hand us either numpy arrays (legacy / single-process
    mode) or already-pinned torch tensors (``MiniBatchSubgraph.pin_memory()``
    path). When the input is already a pinned tensor, the ``.to(device,
    non_blocking=True)`` is a pure async H2D copy and overlaps with compute.
    """
    if isinstance(value, torch.Tensor):
        t = value
    else:
        arr = np.ascontiguousarray(value) if dtype is None else np.ascontiguousarray(value, dtype=dtype)
        t = torch.from_numpy(arr)
        if non_blocking and device.type == "cuda" and not t.is_pinned():
            t = t.pin_memory()
    return t.to(device, non_blocking=non_blocking)


# v8.5 TEW — module-level state read by ``to_torch_batch`` on every batch. A
# module singleton keeps the wiring surgical — no need to thread two extra
# kwargs through every call site of to_torch_batch.
_TEW_ENABLED: bool = False
_TEW_EPSILON: float = 1.0e-3


def _set_tew_state(enabled: bool, epsilon: float = 1.0e-3) -> None:
    """Configure temporal edge weights (TEW) used by :func:`to_torch_batch`.

    When ``enabled`` is True, the function wires next_packet edges' delta_time
    as edge_weight = 1/(Δt+epsilon); HGT will then add log(edge_weight) to
    attention scores, creating a temporal-locality bias.
    """
    global _TEW_ENABLED, _TEW_EPSILON
    _TEW_ENABLED = bool(enabled)
    _TEW_EPSILON = float(epsilon)


def to_torch_batch(
    batch: MiniBatchSubgraph,
    edge_types: list[tuple[str, str, str]],
    device: torch.device,
    use_semantic_edge_weights: bool,
    *,
    non_blocking: bool = True,
) -> tuple[
    dict[str, torch.Tensor],
    dict[tuple[str, str, str], torch.Tensor],
    dict[tuple[str, str, str], torch.Tensor] | None,
    torch.Tensor,
    torch.Tensor,
]:
    node_tensors = {
        key: _to_device(value, device, non_blocking)
        for key, value in batch.node_features.items()
    }
    edge_tensors: dict[tuple[str, str, str], torch.Tensor] = {}
    for edge_type in edge_types:
        value = batch.edge_index.get(edge_type)
        if value is None:
            value = np.empty((2, 0), dtype=np.int64)
        edge_tensors[edge_type] = _to_device(value, device, non_blocking, dtype=np.int64)

    edge_weights: dict[tuple[str, str, str], torch.Tensor] = {}
    if use_semantic_edge_weights:
        for edge_type in edge_types:
            if "matches_technique" not in edge_type[1]:
                continue
            value = batch.edge_attr.get(edge_type, np.empty((0,), dtype=np.float32))
            t = _to_device(value, device, non_blocking, dtype=np.float32)
            edge_weights[edge_type] = t.reshape(-1)

    # v8.5 TEW — wire next_packet Δt as temporal-locality bias. HGT layer adds
    # log(edge_weight) to attention scores; we pass 1/(Δt+ε) so:
    #   small Δt  → log(1/small)  = large positive → amplify (burst pattern)
    #   large Δt  → log(1/large)  = large negative → suppress (far in time)
    if _TEW_ENABLED:
        for edge_type in edge_types:
            if edge_type[1] != "next_packet":
                continue
            value = batch.edge_attr.get(edge_type)
            if value is None:
                continue
            t = _to_device(value, device, non_blocking, dtype=np.float32)
            delta_t = t.reshape(-1) if t.ndim <= 1 else t[:, 0]
            edge_weights[edge_type] = 1.0 / (delta_t.clamp_min(0.0) + _TEW_EPSILON)

    seed_mask = _to_device(batch.seed_mask, device, non_blocking, dtype=bool)
    seed_labels = _to_device(batch.seed_labels, device, non_blocking, dtype=np.int64)

    return node_tensors, edge_tensors, edge_weights or None, seed_mask, seed_labels


def _per_class_counts_tensor(
    pred: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return a ``(num_classes, 4)`` int64 tensor of ``[tp, fp, fn, support]`` per class.

    Designed so DDP eval can ``all_reduce(SUM)`` the result instead of gathering
    full per-rank prediction vectors (which is memory-heavy for big val splits).
    """
    pred = pred.long().reshape(-1)
    label = label.long().reshape(-1)
    device = pred.device
    cls = torch.arange(num_classes, device=device).view(-1, 1)
    pred_eq = pred.view(1, -1) == cls   # (C, N)
    lab_eq = label.view(1, -1) == cls   # (C, N)
    tp = (pred_eq & lab_eq).sum(dim=1)
    fp = (pred_eq & ~lab_eq).sum(dim=1)
    fn = (~pred_eq & lab_eq).sum(dim=1)
    support = lab_eq.sum(dim=1)
    return torch.stack([tp, fp, fn, support], dim=1).to(torch.int64)


def _metrics_from_counts(
    counts: np.ndarray,           # (num_classes, 4)
    label_names: dict[int, str],
    loss_sum: float | None,
    total_examples: int,
) -> dict[str, Any]:
    """Build the same metric dict shape as ``metrics_from_predictions`` from
    pre-aggregated per-class counts. Used by DDP eval after ``all_reduce``."""
    if counts.size == 0 or int(counts[:, 3].sum()) == 0:
        return {"count": 0, "loss": None, "accuracy": None, "macro_f1": None, "per_class": {}}
    tp = counts[:, 0].astype(np.int64)
    fp = counts[:, 1].astype(np.int64)
    fn = counts[:, 2].astype(np.int64)
    support = counts[:, 3].astype(np.int64)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    correct = 0
    for cid in range(counts.shape[0]):
        prec = tp[cid] / max(tp[cid] + fp[cid], 1)
        rec = tp[cid] / max(tp[cid] + fn[cid], 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
        if support[cid] > 0:
            f1_values.append(float(f1))
        per_class[label_names.get(cid, str(cid))] = {
            "support": int(support[cid]),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        correct += int(tp[cid])
    count = int(support.sum())
    return {
        "count": count,
        "loss": float(loss_sum / max(total_examples, 1)) if loss_sum is not None else None,
        "accuracy": float(correct / max(count, 1)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def metrics_from_predictions(
    pred_np: np.ndarray,
    label_np: np.ndarray,
    num_classes: int,
    label_names: dict[int, str],
    loss_sum: float | None = None,
) -> dict[str, Any]:
    if label_np.size == 0:
        return {"count": 0, "loss": None, "accuracy": None, "macro_f1": None, "per_class": {}}

    correct = int((pred_np == label_np).sum())
    count = int(label_np.shape[0])
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id in range(num_classes):
        tp = int(((pred_np == class_id) & (label_np == class_id)).sum())
        fp = int(((pred_np == class_id) & (label_np != class_id)).sum())
        fn = int(((pred_np != class_id) & (label_np == class_id)).sum())
        support = int((label_np == class_id).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if support > 0:
            f1_values.append(float(f1))
        per_class[label_names.get(class_id, str(class_id))] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return {
        "count": count,
        "loss": float(loss_sum / max(count, 1)) if loss_sum is not None else None,
        "accuracy": float(correct / max(count, 1)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def _interleave_by_class(
    flow_ids: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """Round-robin interleave flow_ids across classes for DDP stratified sampling.

    DistributedSampler cannot be combined with WeightedRandomSampler, so for
    multi-GPU training we pre-sort flow_ids into a class-interleaved order.
    DistributedSampler then shards and shuffles this pre-balanced sequence each
    epoch, giving each GPU approximately equal class representation per batch.
    """
    buckets = [
        flow_ids[class_labels == lbl].tolist()
        for lbl in sorted(np.unique(class_labels).tolist())
    ]
    result: list[int] = []
    max_len = max(len(b) for b in buckets)
    for i in range(max_len):
        for b in buckets:
            if i < len(b):
                result.append(b[i])
    return np.array(result, dtype=np.int64)


def make_neighbor_loader(
    flow_ids: np.ndarray,
    sampler: HeteroNeighborSampler,
    config: dict[str, Any],
    *,
    shuffle: bool,
    n_gpus: int = 1,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42,
    class_labels: np.ndarray | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a neighbor-sampling DataLoader.

    Returns ``(loader, dist_sampler)``. In DDP mode (``world_size > 1``) the
    DistributedSampler is set up to shard ``flow_ids`` across ranks; callers must
    invoke ``dist_sampler.set_epoch(epoch)`` each epoch to reshuffle. In
    single-process mode the second element is ``None``.

    When ``class_labels`` is provided and ``train.stratified_batch_sampling``
    is true in config, each training batch is class-balanced:
      - Single GPU: WeightedRandomSampler (exact per-batch balance).
      - Multi GPU: flow_ids pre-sorted by round-robin class interleaving;
        DistributedSampler shuffles the balanced sequence each epoch.
    """
    use_stratified = (
        class_labels is not None
        and shuffle
        and bool(config.get("train", {}).get("stratified_batch_sampling", False))
    )

    if use_stratified and world_size > 1:
        flow_ids = _interleave_by_class(flow_ids, class_labels)

    dl_cfg = config.get("dataloader") or {}
    num_workers = _auto_dataloader_workers(int(dl_cfg.get("num_workers", -1)), n_gpus)
    dataset = FlowSeedDataset(flow_ids)

    dist_sampler: DistributedSampler | None = None
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(config["train"].get("batch_seed_flows", 256)),
        "collate_fn": NeighborSamplingCollator(sampler),
        "num_workers": num_workers,
        "pin_memory": bool(dl_cfg.get("pin_memory", True)),
    }
    if world_size > 1:
        dist_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=bool(shuffle), seed=seed,
            drop_last=True,  # prevents padding duplicates that cause double-counting in all_reduce metrics
        )
        loader_kwargs["sampler"] = dist_sampler
        loader_kwargs["shuffle"] = False
    elif use_stratified:
        num_classes = int(class_labels.max()) + 1
        cls_counts = np.bincount(class_labels, minlength=num_classes).astype(np.float64)
        cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
        sample_weights = torch.from_numpy(
            (1.0 / cls_counts)[class_labels]
        ).float()
        loader_kwargs["sampler"] = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = bool(shuffle)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(dl_cfg.get("prefetch_factor", 4))
        loader_kwargs["persistent_workers"] = bool(dl_cfg.get("persistent_workers", True))
        loader_kwargs["worker_init_fn"] = _make_worker_init_fn(sampler, seed)
    return DataLoader(dataset, **loader_kwargs), dist_sampler


def evaluate_neighbor_sampling(
    *,
    model: HeteroGraphTransformer,
    loader: DataLoader,
    edge_types: list[tuple[str, str, str]],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype = torch.float16,
    use_semantic_edge_weights: bool,
    num_classes: int,
    label_names: dict[int, str],
    epoch: int = 0,
    split_name: str = "eval",
    is_ddp: bool = False,
    logit_adjustment: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Two paths:
      1. DDP (``is_ddp=True``): each rank evaluates its DistributedSampler shard,
         accumulates per-class counts on-device, ``all_reduce`` at the end.
      2. Single-device: straight loop.

    Metrics are ALWAYS computed on RAW logits (the true learning signal). When
    ``logit_adjustment`` is given, the post-hoc Menon-adjusted metrics are ALSO
    computed and attached under ``metrics["logit_adjusted"]``. We deliberately
    do NOT make the adjusted metrics primary: on an undertrained model the raw
    logits are near-flat, so subtracting τ·log(prior) forces argmax onto the
    rarest class for every sample (val_acc collapses to the rarest-class
    frequency). LA is only meaningful once raw logits actually discriminate,
    so it belongs as a post-hoc view, not the monitored/primary number.
    """
    model.eval()
    counts_acc = torch.zeros((num_classes, 4), dtype=torch.int64, device=device)
    counts_acc_adj = torch.zeros((num_classes, 4), dtype=torch.int64, device=device)
    loss_sum_t = torch.zeros(1, dtype=torch.float64, device=device)
    examples_t = torch.zeros(1, dtype=torch.int64, device=device)
    preds: list[np.ndarray] = []
    preds_adj: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loss_sum = 0.0
    total_batches = len(loader)
    progress_every = max(1, total_batches // 10) if total_batches else 0
    start = time.time()
    last_logged = 0
    # inference_mode is stricter than no_grad: PyTorch skips version-counter
    # bookkeeping on every output tensor, so eval throughput is ~5-10% higher
    # for graph-transformer workloads that produce many intermediate tensors.
    with torch.inference_mode():
        if is_ddp:
            for i, batch in enumerate(loader, start=1):
                node_features, edge_index, edge_weight, seed_mask, seed_labels = to_torch_batch(
                    batch, edge_types, device, use_semantic_edge_weights,
                )
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
                    seed_logits = logits[seed_mask]
                    loss = F.cross_entropy(seed_logits.float(), seed_labels, reduction="sum")
                loss_sum_t += loss.detach().to(torch.float64)
                examples_t += seed_labels.numel()
                _logits_raw = seed_logits.detach().float()
                counts_acc += _per_class_counts_tensor(_logits_raw.argmax(dim=1), seed_labels, num_classes)
                if logit_adjustment is not None:
                    _logits_adj = _logits_raw - logit_adjustment.to(_logits_raw.device)
                    counts_acc_adj += _per_class_counts_tensor(_logits_adj.argmax(dim=1), seed_labels, num_classes)
                if progress_every and (i - last_logged >= progress_every or i == total_batches):
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        pct = 100.0 * i / total_batches
                        elapsed = time.time() - start
                        print(
                            f"Epoch {epoch:03d} | {split_name:<5} {i:>5}/{total_batches} "
                            f"({pct:5.1f}%) | {elapsed:6.1f}s",
                            flush=True,
                        )
                    last_logged = i
            # All-reduce across ranks.
            dist.all_reduce(counts_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(examples_t, op=dist.ReduceOp.SUM)
            metrics = _metrics_from_counts(
                counts_acc.cpu().numpy(),
                label_names,
                float(loss_sum_t.item()),
                int(examples_t.item()),
            )
            if logit_adjustment is not None:
                dist.all_reduce(counts_acc_adj, op=dist.ReduceOp.SUM)
                metrics["logit_adjusted"] = _metrics_from_counts(
                    counts_acc_adj.cpu().numpy(),
                    label_names,
                    float(loss_sum_t.item()),
                    int(examples_t.item()),
                )
            return metrics
        else:
            for i, batch in enumerate(loader, start=1):
                node_features, edge_index, edge_weight, seed_mask, seed_labels = to_torch_batch(
                    batch,
                    edge_types,
                    device,
                    use_semantic_edge_weights,
                )
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
                    seed_logits = logits[seed_mask]
                    loss = F.cross_entropy(seed_logits.float(), seed_labels, reduction="sum")
                loss_sum += float(loss.item())
                _logits_raw = seed_logits.detach().float()
                preds.append(_logits_raw.argmax(dim=1).cpu().numpy())
                if logit_adjustment is not None:
                    _logits_adj = _logits_raw - logit_adjustment.to(_logits_raw.device)
                    preds_adj.append(_logits_adj.argmax(dim=1).cpu().numpy())
                labels.append(seed_labels.detach().cpu().numpy())
                if progress_every and (i - last_logged >= progress_every or i == total_batches):
                    pct = 100.0 * i / total_batches
                    elapsed = time.time() - start
                    print(
                        f"Epoch {epoch:03d} | {split_name:<5} {i:>5}/{total_batches} "
                        f"({pct:5.1f}%) | {elapsed:6.1f}s",
                        flush=True,
                    )
                    last_logged = i
    pred_np = np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)
    label_np = np.concatenate(labels) if labels else np.empty((0,), dtype=np.int64)
    metrics = metrics_from_predictions(pred_np, label_np, num_classes, label_names, loss_sum)
    if logit_adjustment is not None:
        adj_np = np.concatenate(preds_adj) if preds_adj else np.empty((0,), dtype=np.int64)
        metrics["logit_adjusted"] = metrics_from_predictions(
            adj_np, label_np, num_classes, label_names, loss_sum
        )
    return metrics


def train_neighbor_sampling(
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    *,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
) -> None:
    """Train the HGT classifier with neighbor sampling.

    Routing:
      * ``world_size > 1``      → DDP (torchrun): each rank holds one GPU/CPU,
        DistributedSampler shards seed flows, NCCL/Gloo all-reduces gradients.
      * Single-process, 1 device (GPU or CPU) → straight loop.
    """
    is_ddp = world_size > 1
    if is_ddp and rank == 0:
        print(f"[DDP] world_size={world_size} backend={dist.get_backend()}", flush=True)
    _tune_cuda_backends(device, enable_tf32=bool(config["train"].get("tf32", True)))

    backend = load_neighbor_backend(config)
    all_flow_ids = np.arange(int(backend.num_flows), dtype=np.int64)
    labels_np = backend.get_flow_labels(all_flow_ids)
    if labels_np.size == 0:
        raise ValueError("Cannot train HGT: graph has no flow labels.")
    unique_labels, label_counts = np.unique(labels_np, return_counts=True)
    if rank == 0:
        print(
            f"[diag] flow labels: unique={unique_labels.tolist()} "
            f"counts={label_counts.tolist()} "
            f"max={int(labels_np.max())} "
            f"num_classes will be={int(labels_np.max()) + 1}",
            flush=True,
        )
    if labels_np.max() == 0:
        raise ValueError(
            "Cannot train HGT: all flow labels are 0 (only 1 class detected). "
            "Check that get_flow_labels() returns correct attack/benign labels."
        )
    num_classes = int(labels_np.max()) + 1
    train_idx_np, val_idx_np, test_idx_np = backend_splits(backend, labels_np, config, seed)

    max_train_flows = int(config["train"].get("max_train_flows", 0))
    if max_train_flows > 0 and int(train_idx_np.shape[0]) > max_train_flows:
        rng_sub = np.random.default_rng(seed + rank)
        perm = rng_sub.permutation(train_idx_np.shape[0])
        full_n = int(train_idx_np.shape[0])
        train_idx_np = train_idx_np[perm[:max_train_flows]]
        if rank == 0:
            print(f"[subsample] Train: {max_train_flows:,} / {full_n:,} flows.", flush=True)

    flow_feature_stats: dict[str, list[float]] | None = None
    if bool(config["data"]["standardize_flow_features"]):
        manifest_stats = (backend.manifest or {}).get("flow_feature_stats")
        if isinstance(manifest_stats, dict) and "mean" in manifest_stats and "std" in manifest_stats:
            flow_feature_stats = {
                "mean": list(manifest_stats["mean"]),
                "std": list(manifest_stats["std"]),
            }
        else:
            flow_feature_stats = compute_flow_feature_stats_backend(backend, train_idx_np)

    sampler_cfg = dict(config.get("sampler") or {})
    sampler_hops = sampler_cfg.get("hops")
    if sampler_hops is None:
        sampler_hops = int(config["model"]["num_layers"])
    # Per-rank RNG seed so ranks don't all pick identical neighbour subsamples
    # (each rank already sees a different flow shard via DistributedSampler, but
    # diverging samplers help hide any residual correlation in random keys).
    sampler = HeteroNeighborSampler(
        backend,
        hops=int(sampler_hops),
        fanouts=dict(sampler_cfg.get("fanouts") or {}),
        reverse_fanouts=dict(sampler_cfg.get("reverse_fanouts") or {}),
        always_include_all_tactics=bool(sampler_cfg.get("always_include_all_tactics", True)),
        always_include_all_techniques=bool(sampler_cfg.get("always_include_all_techniques", True)),
        flow_feature_stats=flow_feature_stats,
        standardize_flow_features=bool(config["data"]["standardize_flow_features"]),
        seed=seed + rank,
    )
    train_labels_np = labels_np[train_idx_np]
    train_loader, train_dist_sampler = make_neighbor_loader(
        train_idx_np, sampler, config, shuffle=True,
        world_size=world_size, rank=rank, seed=seed,
        class_labels=train_labels_np,
    )
    val_loader, _ = make_neighbor_loader(
        val_idx_np, sampler, config, shuffle=False,
        world_size=world_size, rank=rank, seed=seed,
    )
    test_loader, _ = make_neighbor_loader(
        test_idx_np, sampler, config, shuffle=False,
        world_size=world_size, rank=rank, seed=seed,
    )

    edge_types = list(backend.edge_types)

    node_input_dims = {
        "flow": int(backend.feature_dims["flow"]),
        "packet": int(backend.feature_dims["packet"]),
        "technique": int(backend.feature_dims["technique"]),
    }
    raw_model = HeteroGraphTransformer(
        node_input_dims=node_input_dims,
        edge_types=edge_types,
        num_classes=num_classes,
        num_tactics=int(backend.num_tactics),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        num_heads=int(config["model"]["num_heads"]),
        dropout=float(config["model"]["dropout"]),
        ffn_multiplier=int(config["model"]["ffn_multiplier"]),
        activation_checkpointing=bool(config["train"].get("activation_checkpointing", False)),
    ).to(device)

    # Compile the plain module FIRST, then wrap with DDP — this is the order
    # PyTorch supports cleanly. torch.compile shares parameter storage with
    # raw_model (no copy), so raw_model stays the canonical handle for
    # state_dict save/load while ``core`` is what actually executes.
    # ``raw_model`` itself is never DDP/compile-wrapped, so checkpointing the
    # un-wrapped weights needs no _orig_mod / .module unwrapping.
    core = _maybe_compile(raw_model, bool(config["train"].get("compile", False)), rank=rank)

    if is_ddp:
        # find_unused_parameters=True is required: HGT carries per-relation
        # parameters (relation_key/value/prior) that only receive gradient when
        # at least one edge of that relation lands in the mini-batch. Neighbor
        # sampling can skip entire edge types on small batches, so DDP must walk
        # the autograd graph after backward to learn which params updated.
        #
        # gradient_as_bucket_view=True: DDP creates ``.grad`` views into its
        # internal flattened bucket buffer instead of a separate per-param grad
        # tensor. Saves ~one model-size of activation memory and shortens the
        # all-reduce path (no extra copy between .grad and the bucket).
        #
        # bucket_cap_mb=25: HGT is small (~10-40 MB of params), so the default
        # 25MB bucket already collapses into 1-2 buckets — keeping the default
        # avoids fragmenting the all-reduce across many tiny messages.
        device_ids = [local_rank] if device.type == "cuda" else None
        model: torch.nn.Module = DDP(
            core,
            device_ids=device_ids,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            bucket_cap_mb=int(config["train"].get("ddp_bucket_cap_mb", 25)),
        )
    else:
        model = core

    grad_accum_steps = max(1, int(config["train"].get("grad_accum_steps", 1)))
    _adamw_kwargs = dict(
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
        eps=float(config["train"].get("adamw_eps", 1e-6)),
        betas=tuple(config["train"].get("adamw_betas", (0.9, 0.95))),
        fused=(device.type == "cuda"),
    )
    sam_enabled = bool(config["train"].get("sam_enabled", False))
    sam_rho = float(config["train"].get("sam_rho", 0.05))
    if sam_enabled and is_ddp:
        if rank == 0:
            print("[SAM] DDP + SAM unsupported — disabling SAM, using AdamW.", flush=True)
        sam_enabled = False
    if sam_enabled and grad_accum_steps > 1:
        if rank == 0:
            print(
                f"[SAM] grad_accum_steps={grad_accum_steps} > 1 is incompatible — disabling SAM.",
                flush=True,
            )
        sam_enabled = False
    if sam_enabled:
        from graphslm_ids.offline.training.sam_optimizer import SAM
        optimizer = SAM(
            model.parameters(),
            base_optimizer=torch.optim.AdamW,
            rho=sam_rho,
            **_adamw_kwargs,
        )
        if rank == 0:
            print(f"[optimizer] SAM(AdamW) rho={sam_rho}", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **_adamw_kwargs)
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0))

    _scheduler_type = str(config["train"].get("scheduler", "onecycle")).lower()
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Scheduler steps once per optimizer step (every grad_accum_steps batches).
        # Use ceil, not floor: the trailing partial accumulation group at epoch end
        # still performs one optimizer+scheduler step, so floor() would undercount
        # by one step/epoch and slowly desync the LR curve over a long run.
        _optimizer_steps_per_epoch = max(
            1, -(-len(train_loader) // grad_accum_steps)
        )
        if _scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(config["train"]["lr"]),
                steps_per_epoch=_optimizer_steps_per_epoch,
                epochs=int(config["train"]["epochs"]),
                pct_start=float(config["train"].get("scheduler_pct_start", 0.05)),
            )
        elif _scheduler_type in {"cosine_annealing", "cosine"}:
            _total_cosine_steps = _optimizer_steps_per_epoch * int(config["train"]["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=_total_cosine_steps,
                eta_min=float(config["train"].get("scheduler_eta_min", 1e-5)),
            )
    # GradScaler calls optimizer.step() internally, so PyTorch's order check
    # (which uses optimizer._step_count, absent on AdamW) is a false positive.
    warnings.filterwarnings(
        "ignore",
        message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
        category=UserWarning,
    )
    # --- Class weight setup with optional DRW (Deferred Re-Weighting, Cao et al. 2019) ---
    # DRW trains with no class weight for the first drw_start_pct of epochs (better
    # feature learning), then activates weights for the remainder (refines rare classes).
    # drw_start_pct=0.0 → weight active from epoch 1 (no deferral).
    # drw_start_pct=1.0 → weight never applied (pure unweighted training).
    target_weight: torch.Tensor | None = None
    _weight_method = "off"
    _cb_beta_used: float | None = None
    if str(config["train"]["class_weight"]).lower() == "balanced":
        _max_w = float(config["train"].get("class_weight_cap", float("inf")))
        _weight_method = str(config["train"].get("class_weight_method", "inverse")).lower()
        _cb_beta_used = float(config["train"].get("cb_beta", 0.999))
        # Auto-adaptation: by default recompute class weights from the CURRENT
        # split's label counts. Reusing manifest-baked weights would silently
        # apply a previous dataset's distribution when the data is swapped/grown.
        _use_manifest_w = bool(config["train"].get("use_manifest_class_weights", False))
        target_weight = class_weights_from_backend(
            backend, train_idx_np, num_classes,
            max_weight=_max_w,
            weight_method=_weight_method,
            cb_beta=_cb_beta_used,
            use_manifest_weights=_use_manifest_w,
        ).to(device)
    drw_start_pct = float(config["train"].get("drw_start_pct", 0.0))
    drw_start_epoch = max(1, int(int(config["train"]["epochs"]) * drw_start_pct))
    # Active weight is selected per-epoch in the training loop based on DRW gate.
    weight = None

    # QUALITY v4: Configurable loss type for severe class imbalance.
    # - 'ce' (default): F.cross_entropy with optional label_smoothing (0.1 typical)
    # - 'focal': Focal Loss with focal_gamma (2.0 paper default). Down-weights
    #   easy examples → focus on hard misclassified minority class.
    loss_type = str(config["train"].get("loss_type", "ce")).lower()
    label_smoothing = float(config["train"].get("label_smoothing", 0.0))
    focal_gamma = float(config["train"].get("focal_gamma", 2.0))
    if loss_type not in {"ce", "focal", "cb_focal"}:
        raise ValueError(
            f"Unknown loss_type {loss_type!r}. Supported: 'ce', 'focal', 'cb_focal'."
        )
    if rank == 0:
        _drw_msg = (
            f"drw_start_epoch={drw_start_epoch}/{int(config['train']['epochs'])}"
            if target_weight is not None and drw_start_pct > 0.0
            else "drw=off"
        )
        print(
            f"[loss] type={loss_type} label_smoothing={label_smoothing} "
            f"focal_gamma={focal_gamma} "
            f"class_weight_method={_weight_method} "
            f"cb_beta={_cb_beta_used if _cb_beta_used is not None else 'n/a'} "
            f"{_drw_msg}",
            flush=True,
        )

    # Logit Adjustment (Menon et al. ICLR 2021): subtract τ·log(prior) from eval
    # logits to debias inference toward rare classes. τ=1.0 paper default. Set 0
    # to disable.
    logit_adjustment_tau = float(config["train"].get("logit_adjustment", 0.0))
    eval_logit_adjustment: torch.Tensor | None = None
    if logit_adjustment_tau > 0.0:
        _train_labels_for_prior = backend.get_flow_labels(
            np.asarray(train_idx_np, dtype=np.int64)
        )
        _prior_counts = np.bincount(_train_labels_for_prior, minlength=num_classes).astype(np.float64)
        _prior = np.maximum(_prior_counts / max(_prior_counts.sum(), 1.0), 1e-12)
        eval_logit_adjustment = torch.from_numpy(
            (logit_adjustment_tau * np.log(_prior)).astype(np.float32)
        ).to(device)
        if rank == 0:
            print(
                f"[logit_adjustment] tau={logit_adjustment_tau} "
                f"adj_min={eval_logit_adjustment.min().item():.3f} "
                f"adj_max={eval_logit_adjustment.max().item():.3f}",
                flush=True,
            )

    drop_edge_prob = float(config["train"].get("drop_edge_prob", 0.0))
    if rank == 0 and drop_edge_prob > 0.0:
        print(f"[drop_edge] prob={drop_edge_prob}", flush=True)

    # HGAA Phase 2 — Adaptive heterogeneous-graph augmentation. Gated entirely
    # by ``train.hgaa.enabled`` in config; when off the factory returns None
    # and the per-batch hook below is a no-op. Dataset-portable: tail classes
    # are auto-detected from the train labels (no hardcoded class IDs per
    # spec §1.5 DC-2).
    from graphslm_ids.offline.training.hgaa_augmentation import (
        build_hgaa_pipeline_from_config,
    )
    hgaa_train_labels_for_detect = backend.get_flow_labels(
        np.asarray(train_idx_np, dtype=np.int64)
    )
    # Build on all ranks — each rank's dataloader produces its own batches, so
    # each rank applies its own augmentation independently (seed offset by rank
    # to avoid identical RNG sequences across ranks).
    hgaa_pipeline = build_hgaa_pipeline_from_config(
        config=config,
        train_labels=hgaa_train_labels_for_detect,
        num_classes=num_classes,
        seed=int(config["train"].get("seed", 42)) + rank,
    )
    del hgaa_train_labels_for_detect  # free memory; not needed beyond detect

    skip_val_first_epoch = bool(config["train"].get("skip_val_first_epoch", False))

    # QUALITY v4: EMA setup. Shadow weights tracked on raw_model (unwrapped, no
    # DDP/compile). update() called after each successful optimizer step. eval
    # uses shadow via apply_shadow()/restore() bracket.
    ema_enabled = bool(config["train"].get("ema_enabled", False))
    ema_decay = float(config["train"].get("ema_decay", 0.999))
    ema = EMA(raw_model, decay=ema_decay) if ema_enabled else None
    if rank == 0 and ema_enabled:
        print(f"[ema] enabled — decay={ema_decay} (shadow tracks ~{int(1/(1-ema_decay))} steps)", flush=True)

    output_dir = ensure_dir(Path(config["train"]["output_dir"])) if rank == 0 else Path(config["train"]["output_dir"])
    best_checkpoint = output_dir / "hgt_flow_best.pt"
    label_names = label_name_mapping(backend.manifest, labels_np)
    monitor = str(config["train"]["monitor"])
    if monitor not in {"val_macro_f1", "val_accuracy", "val_balanced", "val_loss"}:
        raise ValueError(
            f"Unknown monitor metric {monitor!r}. Supported: "
            f"'val_macro_f1', 'val_accuracy', 'val_balanced', 'val_loss'."
        )
    log_every = max(1, int(config["train"]["log_every"]))
    use_amp, amp_dtype = resolve_amp(config, device)
    if rank == 0:
        if use_amp:
            print("[AMP] enabled — dtype=bfloat16 (native, no GradScaler)", flush=True)
        else:
            print(
                "[AMP] disabled — running in FP32 "
                "(AMP force-disabled in resolve_amp: bf16 froze training by skipping "
                "every optimizer step on non-finite gradients).",
                flush=True,
            )
    # GradScaler is permanently disabled: we only ever run bfloat16 (FP32 range,
    # no overflow) or pure FP32. Keeping a disabled scaler makes the
    # scaler.scale()/step()/update() calls below pure pass-throughs, so the
    # training loop needs no float16-specific branching.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    use_wandb = _wandb_init(config, rank)
    use_semantic_edge_weights = bool(config["data"]["use_semantic_edge_weights"])
    use_temporal_edge_weights = bool(config["data"].get("use_temporal_edge_weights", False))
    tew_epsilon = float(config["data"].get("tew_epsilon", 1.0e-3))
    _set_tew_state(enabled=use_temporal_edge_weights, epsilon=tew_epsilon)
    if rank == 0 and use_temporal_edge_weights:
        print(f"[TEW] temporal edge weights enabled — 1/(Δt+{tew_epsilon}) on next_packet edges", flush=True)

    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    train_batches_total = len(train_loader)
    train_progress_every = max(1, train_batches_total // 10) if train_batches_total else 0
    if rank == 0:
        print(
            f"[HGT] Starting training: epochs={int(config['train']['epochs'])} "
            f"train_batches/epoch={train_batches_total} val_batches={len(val_loader)} "
            f"test_batches={len(test_loader)} progress_every={train_progress_every}",
            flush=True,
        )

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        # DRW gate — activate class weights only after drw_start_epoch.
        if target_weight is not None and epoch >= drw_start_epoch:
            if weight is None and rank == 0 and drw_start_pct > 0.0:
                print(
                    f"[DRW] Activating class weights at epoch {epoch} "
                    f"(start_pct={drw_start_pct:.2f}).",
                    flush=True,
                )
            weight = target_weight
        else:
            weight = None
        if train_dist_sampler is not None:
            # Reshuffle the DistributedSampler so each epoch sees a different
            # rank-to-flow assignment — critical for SGD on sharded data.
            train_dist_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Train metric counts are accumulated on-device so the all-reduce at the
        # end of the epoch is a single small tensor instead of a giant concat.
        train_counts = torch.zeros((num_classes, 4), dtype=torch.int64, device=device)
        train_loss_sum_t = torch.zeros(1, dtype=torch.float64, device=device)
        train_examples_t = torch.zeros(1, dtype=torch.int64, device=device)
        pending_step = False
        skipped_steps = 0
        optimizer_steps = 0
        nonfinite_loss_count = 0
        # Running sums avoid building O(N_batches) Python lists and calling np.mean at epoch end.
        sampled_node_sum: dict[str, int] = {}
        sampled_node_cnt: dict[str, int] = {}
        sampled_edge_sum: dict[str, int] = {}
        sampled_edge_cnt: dict[str, int] = {}

        train_iter = iter(train_loader)
        step = 0
        batches_seen = 0
        last_logged_batches = 0
        epoch_start = time.time()
        while True:
            try:
                raw_batch = next(train_iter)
            except StopIteration:
                break
            # HGAA hook — may return raw_batch unchanged when disabled or when
            # the probabilistic gate didn't fire. Failure inside the pipeline is
            # logged and falls back to raw_batch so an augmentation bug cannot
            # abort training.
            if hgaa_pipeline is not None:
                raw_batch = hgaa_pipeline.maybe_augment(raw_batch)
            batch = raw_batch

            nf, ei, ew, sm, sl = to_torch_batch(
                batch, edge_types, device, use_semantic_edge_weights,
            )
            # In DDP, suppress the all-reduce on mid-accumulation backwards.
            # The trailing backward (or the only backward when accum=1) still
            # syncs gradients, so the optimizer step sees correct averages.
            sync_ctx = (
                model.no_sync()
                if is_ddp and (step + 1) % grad_accum_steps != 0
                else _nullcontext()
            )
            if drop_edge_prob > 0.0:
                ei, ew = _drop_edges(ei, ew, drop_edge_prob)
            with sync_ctx, torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(nf, ei, edge_weight_dict=ew)
                seed_logits = logits[sm].float()
                loss = _compute_train_loss(
                    seed_logits, sl, weight,
                    loss_type=loss_type,
                    label_smoothing=label_smoothing,
                    focal_gamma=focal_gamma,
                )
            batch_count = int(sl.numel())
            _loss_ok = bool(torch.isfinite(loss).item())
            if _loss_ok:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                nonfinite_loss_count += 1
            loss_val = float(loss.detach().item()) if _loss_ok else float("nan")
            if step == 0 and epoch == 1 and rank == 0:
                n_seeds_in_mask = int(sm.sum().item())
                print(
                    f"[diag] step=0 seeds_in_mask={n_seeds_in_mask} "
                    f"seed_labels_n={batch_count} loss={loss_val:.6f} "
                    f"logits_shape={tuple(seed_logits.shape)}",
                    flush=True,
                )
            if _loss_ok:
                train_loss_sum_t += loss_val * batch_count
                train_examples_t += batch_count
                pred = seed_logits.detach().float().argmax(dim=1)
                train_counts.add_(_per_class_counts_tensor(pred, sl, num_classes))
            for node_type, count in batch.stats.get("nodes", {}).items():
                c = int(count)
                sampled_node_sum[node_type] = sampled_node_sum.get(node_type, 0) + c
                sampled_node_cnt[node_type] = sampled_node_cnt.get(node_type, 0) + 1
            for edge_name, count in batch.stats.get("edges", {}).items():
                c = int(count)
                sampled_edge_sum[edge_name] = sampled_edge_sum.get(edge_name, 0) + c
                sampled_edge_cnt[edge_name] = sampled_edge_cnt.get(edge_name, 0) + 1

            pending_step = True
            if (step + 1) % grad_accum_steps == 0:
                if sam_enabled and _loss_ok:
                    # --- SAM two-step update ---
                    # Step 1: clip gradient, perturb weights toward sharpness direction.
                    scaler.unscale_(optimizer)
                    if grad_clip_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        _bad_grads = not total_norm.isfinite()
                    else:
                        _bad_grads = False
                        total_norm = torch.tensor(0.0)
                    if _bad_grads:
                        optimizer.zero_grad(set_to_none=True)
                        skipped_steps += 1
                        if rank == 0 and skipped_steps <= 5:
                            print(
                                f"[warn][SAM] first step skipped — non-finite grad norm "
                                f"(total_norm={float(total_norm):.3e}) at epoch {epoch} batch {step+1}.",
                                flush=True,
                            )
                    else:
                        optimizer.first_step(zero_grad=True)
                        # Step 2: second forward at perturbed weights (same batch).
                        with _nullcontext(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                            _logits_sam = model(nf, ei, edge_weight_dict=ew)
                            _seed_logits_sam = _logits_sam[sm].float()
                            _loss_sam = _compute_train_loss(
                                _seed_logits_sam, sl, weight,
                                loss_type=loss_type,
                                label_smoothing=label_smoothing,
                                focal_gamma=focal_gamma,
                            )
                        if torch.isfinite(_loss_sam):
                            _loss_sam.backward()
                            if grad_clip_norm > 0.0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                            optimizer.second_step(zero_grad=True)
                            optimizer_steps += 1
                            if scheduler is not None:
                                scheduler.step()
                            if ema is not None:
                                ema.update(raw_model)
                            if epoch == 1 and optimizer_steps == 1 and rank == 0:
                                _cur_lr = optimizer.param_groups[0]["lr"]
                                print(
                                    f"[diag] first optimizer step (SAM) | grad_norm={float(total_norm):.4e} "
                                    f"lr={_cur_lr:.3e}",
                                    flush=True,
                                )
                        else:
                            # Perturbed loss non-finite: restore weights, skip update.
                            optimizer.restore_weights()
                            skipped_steps += 1
                            if rank == 0 and skipped_steps <= 5:
                                print(
                                    f"[warn][SAM] second step skipped — non-finite perturbed loss "
                                    f"at epoch {epoch} batch {step+1}.",
                                    flush=True,
                                )
                else:
                    # Standard AdamW path (also used when sam_enabled but _loss_ok=False).
                    scaler.unscale_(optimizer)
                    if grad_clip_norm > 0.0:
                        # clip_grad_norm_ returns total_norm as a scalar tensor.
                        # When any grad is inf: total_norm=inf → clip_coef=0 → inf*0=NaN.
                        # Checking total_norm.isfinite() after the call is O(1) and catches
                        # this before the NaN gradients can be applied to weights.
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        _bad_grads = not total_norm.isfinite()
                    else:
                        _bad_grads = False
                    if _bad_grads:
                        # Discard NaN/inf gradients before step to avoid weight corruption.
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        skipped_steps += 1
                        # Loud warning: a silently-skipped step is invisible, and if EVERY
                        # step skips the model never learns (loss frozen at ln(num_classes)).
                        if rank == 0 and skipped_steps <= 5:
                            print(
                                f"[warn] optimizer step skipped — non-finite grad norm "
                                f"(total_norm={float(total_norm):.3e}) at epoch {epoch} batch {step+1}. "
                                f"Persistent skips freeze training; run in FP32 (amp: false) if this repeats.",
                                flush=True,
                            )
                    else:
                        _scale_before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        if scaler.get_scale() < _scale_before:
                            # scaler halved → step was skipped due to inf/NaN gradients
                            skipped_steps += 1
                            if rank == 0 and skipped_steps <= 5:
                                print(
                                    f"[AMP] step skipped (inf/NaN grad) at batch {step+1}, "
                                    f"scale {_scale_before:.0f}→{scaler.get_scale():.0f}",
                                    flush=True,
                                )
                        else:
                            optimizer_steps += 1
                            if scheduler is not None:
                                scheduler.step()
                            if ema is not None:
                                ema.update(raw_model)
                            # First-step learning-signal probe: a healthy run shows a
                            # non-trivial gradient norm here. A norm of ~0 means the
                            # backward graph is detached / weights will never move —
                            # the fingerprint of a frozen-loss run.
                            if epoch == 1 and optimizer_steps == 1 and rank == 0:
                                _gn = (
                                    float(total_norm) if grad_clip_norm > 0.0
                                    else float(
                                        torch.norm(
                                            torch.stack([
                                                p.grad.detach().norm()
                                                for p in model.parameters()
                                                if p.grad is not None
                                            ])
                                        )
                                    ) if any(p.grad is not None for p in model.parameters())
                                    else 0.0
                                )
                                _cur_lr = optimizer.param_groups[0]["lr"]
                                print(
                                    f"[diag] first optimizer step | grad_norm={_gn:.4e} "
                                    f"lr={_cur_lr:.3e}",
                                    flush=True,
                                )
                pending_step = False
            step += 1
            batches_seen += 1
            if rank == 0 and train_progress_every and (
                batches_seen - last_logged_batches >= train_progress_every
                or batches_seen >= train_batches_total
            ):
                running_loss = (
                    float(train_loss_sum_t.item()) / float(train_examples_t.item())
                    if int(train_examples_t.item()) else 0.0
                )
                pct = (
                    100.0 * batches_seen / train_batches_total
                    if train_batches_total else 0.0
                )
                elapsed = time.time() - epoch_start
                _cur_scale = scaler.get_scale() if scaler.is_enabled() else 0.0
                _skip_info = (
                    f" | skip={skipped_steps} ok={optimizer_steps} scale={_cur_scale:.0f}"
                    if scaler.is_enabled() else ""
                )
                print(
                    f"Epoch {epoch:03d} | train {batches_seen:>5}/{train_batches_total} "
                    f"({pct:5.1f}%) | loss={running_loss:.4f}{_skip_info} | {elapsed:6.1f}s",
                    flush=True,
                )
                last_logged_batches = batches_seen

        if pending_step:
            # Trailing micro-batches used model.no_sync() so their gradients were
            # never all-reduced across ranks. Without explicit sync here each rank
            # would step on its own local gradients, causing weight divergence.
            if is_ddp:
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                _bad_grads = not total_norm.isfinite()
            else:
                _bad_grads = False
            if _bad_grads:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                skipped_steps += 1
            else:
                _scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= _scale_before:
                    optimizer_steps += 1
                    if scheduler is not None:
                        scheduler.step()
                    if ema is not None:
                        ema.update(raw_model)

        # Fail fast: if no optimizer step succeeded this epoch, every gradient was
        # non-finite — the model is frozen. Aborting now saves hours vs. silently
        # producing a checkpoint that never learned.
        if optimizer_steps == 0:
            raise RuntimeError(
                f"Epoch {epoch}: 0 optimizer steps succeeded ({skipped_steps} skipped for "
                f"non-finite gradients) — the model cannot learn, weights stay frozen. "
                f"This is almost always AMP/bfloat16 numerical instability; train in FP32 "
                f"(amp: false). resolve_amp() already force-disables AMP — if you see this, "
                f"investigate the gradient path."
            )

        # All-reduce train metrics so every rank reports the global epoch view.
        if is_ddp:
            dist.all_reduce(train_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_loss_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_examples_t, op=dist.ReduceOp.SUM)

        train_examples = int(train_examples_t.item())
        train_metrics = _metrics_from_counts(
            train_counts.detach().cpu().numpy(),
            label_names,
            float(train_loss_sum_t.item()) if train_examples else None,
            train_examples,
        )
        # EMA shadow weights for eval. apply_shadow stays active until restore() below
        # so checkpoint save (if best) snapshots EMA weights from raw_model.state_dict().
        # During OneCycle warmup epoch 1, lr is near-zero → val is near-random → optionally
        # skip to save ~30 min on long runs.
        do_val = not (epoch == 1 and skip_val_first_epoch)
        if do_val and ema is not None:
            ema.apply_shadow(raw_model)
        if do_val:
            val_metrics = evaluate_neighbor_sampling(
                model=model,
                loader=val_loader,
                edge_types=edge_types,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                use_semantic_edge_weights=use_semantic_edge_weights,
                num_classes=num_classes,
                label_names=label_names,
                epoch=epoch,
                split_name="val",
                is_ddp=is_ddp,
                logit_adjustment=eval_logit_adjustment,
            )
        else:
            val_metrics = {
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "loss": None,
                "per_class": {},
                "count": 0,
            }
            if rank == 0:
                print(f"Epoch {epoch:03d} | val SKIPPED (skip_val_first_epoch=true, warmup)", flush=True)
        avg_nodes = {
            nt: sampled_node_sum[nt] / sampled_node_cnt[nt]
            for nt in sampled_node_sum
            if sampled_node_cnt.get(nt, 0) > 0
        }
        avg_edges = {
            en: sampled_edge_sum[en] / sampled_edge_cnt[en]
            for en in sampled_edge_sum
            if sampled_edge_cnt.get(en, 0) > 0
        }
        entry = {
            "epoch": epoch,
            "train": {key: value for key, value in train_metrics.items() if key != "per_class"},
            "val": {key: value for key, value in val_metrics.items() if key != "per_class"},
            "sampler": {
                "avg_subgraph_nodes": avg_nodes,
                "avg_subgraph_edges": avg_edges,
            },
        }
        history.append(entry)
        _log_epoch_diagnostics(
            epoch=epoch,
            elapsed_seconds=time.time() - epoch_start,
            device=device,
            rank=rank,
        )
        if hgaa_pipeline is not None and rank == 0:
            _hgaa_snap = hgaa_pipeline.stats_snapshot()
            print(
                f"[hgaa] epoch={epoch} considered={_hgaa_snap['considered']} "
                f"augmented={_hgaa_snap['augmented']} "
                f"aug_rate={_hgaa_snap['aug_rate']:.3f} "
                f"op_counts={_hgaa_snap['op_counts']}",
                flush=True,
            )
        _wandb_log(entry, use_wandb)

        monitor_score = _compute_monitor_score(monitor, val_metrics)
        if do_val and monitor_score > best_score and not math.isnan(monitor_score):
            best_score = monitor_score
            best_epoch = epoch
            epochs_without_improvement = 0
            if rank == 0:
                # Always save the unwrapped (un-DDP, un-compile) state_dict so
                # the runtime / inference path can load it without needing DDP
                # or torch.compile installed. raw_model is always the plain
                # HeteroGraphTransformer regardless of DDP/compile wrapping.
                # CPU-clone every tensor here BEFORE handing it off to the
                # background thread — the optimizer's next step would otherwise
                # race with the disk write on the same CUDA storage.
                cpu_state = {
                    k: v.detach().clone().cpu() for k, v in raw_model.state_dict().items()
                }
                _save_checkpoint_bg(
                    {
                        "model_state_dict": cpu_state,
                        "config": config,
                        "node_input_dims": node_input_dims,
                        "edge_types": [list(edge_key) for edge_key in edge_types],
                        "num_classes": num_classes,
                        "num_tactics": int(backend.num_tactics),
                        "label_names": label_names,
                        "flow_feature_stats": flow_feature_stats,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "test_metrics": None,
                    },
                    best_checkpoint,
                )
        elif do_val:
            epochs_without_improvement += 1

        # Restore live (optimizer-tracked) weights AFTER potential checkpoint save.
        # If val was skipped, apply_shadow wasn't called so nothing to restore.
        if do_val and ema is not None:
            ema.restore(raw_model)

        if rank == 0 and nonfinite_loss_count > 0:
            print(
                f"[warn] Epoch {epoch:03d}: {nonfinite_loss_count} batch(es) had a "
                f"non-finite (NaN/Inf) loss and were skipped — their gradients were "
                f"never applied. Persistent non-finite loss freezes training; "
                f"check AMP dtype and feature scaling.",
                flush=True,
            )
        if rank == 0 and (epoch == 1 or epoch % log_every == 0):
            _loss = float(train_metrics['loss']) if train_metrics['loss'] is not None else float('nan')
            _train_acc = train_metrics.get("accuracy")
            _train_acc_str = f"train_acc={_train_acc:.4f} " if _train_acc is not None else ""
            _la = val_metrics.get("logit_adjusted") if isinstance(val_metrics, dict) else None
            _la_str = (
                f" | LA: val_acc={_la['accuracy']:.4f} val_macro_f1={_la['macro_f1']:.4f}"
                if isinstance(_la, dict) else ""
            )
            print(
                f"Epoch {epoch:03d} | loss={_loss:.4f} {_train_acc_str}"
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}{_la_str} "
                f"avg_flow_nodes={avg_nodes.get('flow', 0.0):.1f} "
                f"avg_packet_nodes={avg_nodes.get('packet', 0.0):.1f}"
            )

        # Broadcast the early-stop decision from rank 0 so all ranks exit together.
        stop = epochs_without_improvement >= int(config["train"]["patience"])
        if is_ddp:
            stop_t = torch.tensor([1 if stop else 0], dtype=torch.int64, device=device)
            dist.broadcast(stop_t, src=0)
            stop = bool(stop_t.item())
        if stop:
            if rank == 0:
                print(f"Early stopping at epoch {epoch}.")
            break

    # Wait for the background checkpoint thread to flush AND for all ranks to
    # be ready — only then can rank 0 load the file back safely. In non-DDP
    # mode the barrier is a no-op.
    _join_checkpoint_bg()
    if is_ddp:
        dist.barrier()
    if rank == 0:
        best_payload = load_checkpoint(best_checkpoint, device)
        # Load best weights and eval test set once — avoids per-epoch test leakage.
        # raw_model is always the plain, un-wrapped HeteroGraphTransformer (DDP and
        # torch.compile wrap a separate ``core`` handle that shares its parameters).
        raw_model.load_state_dict(best_payload["model_state_dict"])
        state_holder = raw_model
        # In DDP mode, test_loader was built with DistributedSampler so rank 0 only
        # holds 1/world_size of the test indices. Rebuild without DistributedSampler
        # so rank 0 evaluates the complete test set.
        if is_ddp:
            test_loader_full, _ = make_neighbor_loader(
                test_idx_np, sampler, config, shuffle=False,
                n_gpus=1, world_size=1, rank=0, seed=seed,
            )
        else:
            test_loader_full = test_loader
        test_metrics = evaluate_neighbor_sampling(
            model=state_holder,
            loader=test_loader_full,
            edge_types=edge_types,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_semantic_edge_weights=use_semantic_edge_weights,
            num_classes=num_classes,
            label_names=label_names,
            split_name="test",
            is_ddp=False,
            logit_adjustment=eval_logit_adjustment,
        )
        device_str = f"{device} x{world_size}" if is_ddp else str(device)
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "device": device_str,
            "world_size": int(world_size),
            "best_checkpoint": str(best_checkpoint),
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "num_flows": int(backend.num_flows),
            "num_techniques": int(backend.num_techniques),
            "num_tactics": int(backend.num_tactics),
            "num_edge_types": int(len(edge_types)),
            "splits": {
                "train": int(train_idx_np.shape[0]),
                "val": int(val_idx_np.shape[0]),
                "test": int(test_idx_np.shape[0]),
            },
            "label_names": {str(key): value for key, value in label_names.items()},
            "best_val_metrics": best_payload["val_metrics"],
            "best_test_metrics": test_metrics,
            "history": history,
        }
        write_json(output_dir / "training_summary.json", summary)
        print(f"[OK] Best checkpoint: {best_checkpoint}")
        print(f"[OK] Training summary: {output_dir / 'training_summary.json'}")
    _wandb_finish(use_wandb)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(Path(args.config)), args)

    seed = int(config["train"]["seed"])
    set_seed(seed)

    rank, local_rank, world_size = setup_distributed()
    if world_size > 1:
        device = _ddp_device(local_rank)
    else:
        device = resolve_device(str(config["train"]["device"]))

    try:
        train_neighbor_sampling(
            config, seed, device,
            rank=rank, local_rank=local_rank, world_size=world_size,
        )
    finally:
        teardown_distributed()


if __name__ == "__main__":
    main()
