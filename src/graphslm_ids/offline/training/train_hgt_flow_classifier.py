from __future__ import annotations

import argparse
from contextlib import nullcontext as _nullcontext
from copy import deepcopy
from datetime import datetime, timezone
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
        "multi_gpu": True,
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
    """Resolve mixed precision. CURRENTLY FORCED TO FP32 — see below.

    Returns ``(use_amp, amp_dtype)``; ``use_amp`` is always ``False``.

    Why AMP is force-disabled:
      A Kaggle bfloat16 run of this HGT skipped EVERY optimizer step. The bf16
      backward through the custom edge-softmax / scatter-add message passing
      produced non-finite gradients, so ``clip_grad_norm_`` returned ``inf``,
      every step was discarded, and the loss froze at ``ln(num_classes)``.
      float16 is likewise unsafe (overflow). Until the bf16 message-passing
      backward is fixed and verified, training runs in pure FP32 — correct and
      only modestly slower for a model this small. The ``amp`` / ``amp_dtype``
      config keys are intentionally IGNORED here.

    To re-enable AMP after fixing the numerics: restore the hardware-aware
    logic (gate on ``config["train"]["amp"]`` + ``torch.cuda.is_bf16_supported``)
    and confirm a smoke run logs a non-zero first-step grad norm.
    """
    _ = (config, device)  # intentionally unused while AMP is force-disabled
    return False, torch.bfloat16


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
    parser.add_argument("--multi-gpu", dest="multi_gpu", action="store_true", default=None,
                        help="Use all available CUDA GPUs (neighbor_sampling mode only).")
    parser.add_argument("--no-multi-gpu", dest="multi_gpu", action="store_false",
                        help="Disable multi-GPU and force single GPU/CPU.")
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
    if args.multi_gpu is not None:
        config["train"]["multi_gpu"] = args.multi_gpu
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


def get_training_devices(primary: torch.device, multi_gpu: bool) -> list[torch.device]:
    """Return all available CUDA devices when multi_gpu=True, else just the primary device."""
    if primary.type != "cuda" or not multi_gpu:
        return [primary]
    n = torch.cuda.device_count()
    if n <= 1:
        return [primary]
    return [torch.device("cuda", i) for i in range(n)]


def _multi_gpu_train_step(
    model: HeteroGraphTransformer,
    batch_group: list[MiniBatchSubgraph],
    devices: list[torch.device],
    edge_types: list[tuple[str, str, str]],
    weight: torch.Tensor | None,
    scaler: torch.amp.GradScaler,
    use_semantic_edge_weights: bool,
    grad_accum_steps: int,
) -> tuple[float, int, list[np.ndarray], list[np.ndarray], dict[str, list[int]], dict[str, list[int]]]:
    """Forward-backward across multiple GPUs using parallel_apply, then sync gradients to primary model."""
    import torch.nn.parallel as P

    n = min(len(batch_group), len(devices))
    primary = devices[0]
    device_ids = [d.index for d in devices[:n]]

    replicas = P.replicate(model, device_ids, detach=False)

    inputs_args: list[tuple] = []
    inputs_kwargs: list[dict[str, Any]] = []
    masks: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch, device in zip(batch_group[:n], devices[:n]):
        nf, ei, ew, sm, sl = to_torch_batch(batch, edge_types, device, use_semantic_edge_weights)
        inputs_args.append((nf, ei))
        inputs_kwargs.append({"edge_weight_dict": ew})
        masks.append(sm)
        all_labels.append(sl)

    all_logits = P.parallel_apply(replicas, inputs_args, inputs_kwargs)

    preds: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    gpu_losses: list[torch.Tensor] = []
    loss_sum_f = 0.0
    examples = 0

    for logits, sm, sl, device in zip(all_logits, masks, all_labels, devices[:n]):
        seed_logits = logits[sm].float()
        w = weight.to(device) if weight is not None else None
        loss = F.cross_entropy(seed_logits, sl, weight=w)
        if not torch.isfinite(loss):
            continue
        gpu_losses.append(loss)
        preds.append(seed_logits.detach().argmax(dim=1).cpu().numpy())
        labels_out.append(sl.detach().cpu().numpy())
        batch_n = int(sl.numel())
        loss_sum_f += float(loss.item()) * batch_n
        examples += batch_n

    # Gather losses to primary device, scale by 1/(n_gpus * grad_accum), backward
    if gpu_losses:
        avg_loss = sum(lv.to(primary) for lv in gpu_losses) / (n * grad_accum_steps)
        scaler.scale(avg_loss).backward()

    # Aggregate replica gradients to the original model's parameters.
    # Use torch._foreach_add_ to fuse the per-parameter loops into a single
    # multi-tensor kernel (CUDA: 1 launch instead of N), and group the inbound
    # .to(primary) copies into one stream-overlapped pass per replica.
    replica_params = [list(r.parameters()) for r in replicas]
    model_params = list(model.parameters())
    n_params = len(model_params)

    foreach_available = hasattr(torch, "_foreach_add_")
    primary_grads_per_replica: list[list[torch.Tensor | None]] = []
    for rp_list in replica_params:
        replica_grads: list[torch.Tensor | None] = []
        for p_r in rp_list:
            if p_r.grad is None:
                replica_grads.append(None)
            elif p_r.device == primary:
                replica_grads.append(p_r.grad)
            else:
                replica_grads.append(p_r.grad.to(primary, non_blocking=True))
        primary_grads_per_replica.append(replica_grads)

    # Stage 1: ensure each model parameter has a grad tensor to accumulate into.
    for idx in range(n_params):
        if model_params[idx].grad is None:
            for replica_grads in primary_grads_per_replica:
                if replica_grads[idx] is not None:
                    model_params[idx].grad = torch.zeros_like(replica_grads[idx])
                    break

    # Stage 2: bulk add each replica's grads into the model's grads.
    for replica_grads in primary_grads_per_replica:
        tgt: list[torch.Tensor] = []
        src: list[torch.Tensor] = []
        for idx in range(n_params):
            g = replica_grads[idx]
            if g is None or model_params[idx].grad is None:
                continue
            tgt.append(model_params[idx].grad)
            src.append(g)
        if not tgt:
            continue
        if foreach_available:
            torch._foreach_add_(tgt, src)
        else:  # pragma: no cover - PyTorch < 1.10 fallback
            for t, s in zip(tgt, src):
                t.add_(s)

    node_stats: dict[str, list[int]] = {}
    edge_stats: dict[str, list[int]] = {}
    for batch in batch_group[:n]:
        for nt, cnt in batch.stats.get("nodes", {}).items():
            node_stats.setdefault(nt, []).append(int(cnt))
        for et, cnt in batch.stats.get("edges", {}).items():
            edge_stats.setdefault(et, []).append(int(cnt))

    return loss_sum_f, examples, preds, labels_out, node_stats, edge_stats


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
) -> torch.Tensor:
    manifest_weights = (backend.manifest or {}).get("class_weights")
    if isinstance(manifest_weights, list) and len(manifest_weights) >= num_classes:
        weights = np.array(manifest_weights[:num_classes], dtype=np.float32)
        if max_weight < float("inf"):
            weights = np.minimum(weights, max_weight)
        return torch.from_numpy(weights)
    labels = backend.get_flow_labels(np.asarray(train_idx, dtype=np.int64))
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (float(num_classes) * counts[nonzero])
    if max_weight < float("inf"):
        weights = np.minimum(weights, max_weight)
    return torch.from_numpy(weights)


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


def _multi_gpu_eval_step(
    model: HeteroGraphTransformer,
    batch_group: list[MiniBatchSubgraph],
    devices: list[torch.device],
    edge_types: list[tuple[str, str, str]],
    use_semantic_edge_weights: bool,
    use_amp: bool,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
    """Forward replicas across GPUs in parallel; no backward, no grad."""
    import torch.nn.parallel as P

    n = min(len(batch_group), len(devices))
    device_ids = [d.index for d in devices[:n]]
    replicas = P.replicate(model, device_ids, detach=True)

    inputs_args: list[tuple] = []
    inputs_kwargs: list[dict[str, Any]] = []
    masks: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch, device in zip(batch_group[:n], devices[:n]):
        nf, ei, ew, sm, sl = to_torch_batch(batch, edge_types, device, use_semantic_edge_weights)
        inputs_args.append((nf, ei))
        inputs_kwargs.append({"edge_weight_dict": ew})
        masks.append(sm)
        all_labels.append(sl)

    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        all_logits = P.parallel_apply(replicas, inputs_args, inputs_kwargs)

    loss_sum_f = 0.0
    preds: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    for logits, sm, sl in zip(all_logits, masks, all_labels):
        seed_logits = logits[sm].float()
        loss = F.cross_entropy(seed_logits, sl, reduction="sum")
        loss_sum_f += float(loss.item())
        preds.append(seed_logits.argmax(dim=1).cpu().numpy())
        labels_out.append(sl.detach().cpu().numpy())
    return loss_sum_f, preds, labels_out


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
    devices: list[torch.device] | None = None,
    is_ddp: bool = False,
) -> dict[str, Any]:
    """Three paths:
      1. DDP (``is_ddp=True``): each rank evaluates its DistributedSampler shard,
         accumulates per-class counts on-device, ``all_reduce`` at the end.
      2. Legacy multi-GPU (``devices`` given, no DDP): replicate the model and
         parallel_apply batches across GPUs — kept for Kaggle notebooks.
      3. Single-device: straight loop.
    """
    model.eval()
    counts_acc = torch.zeros((num_classes, 4), dtype=torch.int64, device=device)
    loss_sum_t = torch.zeros(1, dtype=torch.float64, device=device)
    examples_t = torch.zeros(1, dtype=torch.int64, device=device)
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loss_sum = 0.0
    total_batches = len(loader)
    progress_every = max(1, total_batches // 10) if total_batches else 0
    start = time.time()
    last_logged = 0
    n_gpus = len(devices) if devices else 1
    batches_seen = 0
    last_logged_batches = 0
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
                pred = seed_logits.detach().float().argmax(dim=1)
                counts_acc += _per_class_counts_tensor(pred, seed_labels, num_classes)
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
            return _metrics_from_counts(
                counts_acc.cpu().numpy(),
                label_names,
                float(loss_sum_t.item()),
                int(examples_t.item()),
            )
        elif n_gpus > 1 and devices is not None:
            it = iter(loader)
            while True:
                batch_group: list[MiniBatchSubgraph] = []
                for _ in range(n_gpus):
                    try:
                        batch_group.append(next(it))
                    except StopIteration:
                        break
                if not batch_group:
                    break
                if len(batch_group) > 1:
                    ls, bp, bl = _multi_gpu_eval_step(
                        model, batch_group, devices, edge_types,
                        use_semantic_edge_weights, use_amp, amp_dtype,
                    )
                    loss_sum += ls
                    preds.extend(bp)
                    labels.extend(bl)
                else:
                    batch = batch_group[0]
                    node_features, edge_index, edge_weight, seed_mask, seed_labels = to_torch_batch(
                        batch, edge_types, device, use_semantic_edge_weights,
                    )
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
                        seed_logits = logits[seed_mask]
                        loss = F.cross_entropy(seed_logits.float(), seed_labels, reduction="sum")
                    loss_sum += float(loss.item())
                    preds.append(seed_logits.detach().float().argmax(dim=1).cpu().numpy())
                    labels.append(seed_labels.detach().cpu().numpy())
                batches_seen += len(batch_group)
                if progress_every and (
                    batches_seen - last_logged_batches >= progress_every
                    or batches_seen >= total_batches
                ):
                    pct = 100.0 * batches_seen / max(total_batches, 1)
                    elapsed = time.time() - start
                    print(
                        f"Epoch {epoch:03d} | {split_name:<5} {batches_seen:>5}/{total_batches} "
                        f"({pct:5.1f}%) | {elapsed:6.1f}s",
                        flush=True,
                    )
                    last_logged_batches = batches_seen
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
                preds.append(seed_logits.detach().float().argmax(dim=1).cpu().numpy())
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
    return metrics_from_predictions(pred_np, label_np, num_classes, label_names, loss_sum)


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
      * Single-process, multi GPU visible AND ``train.multi_gpu`` → legacy
        ``replicate + parallel_apply + foreach_add_`` fallback (Kaggle notebooks
        without torchrun).
      * Single-process, 1 device (GPU or CPU) → straight loop.
    """
    is_ddp = world_size > 1
    multi_gpu = bool(config["train"].get("multi_gpu", True))
    # In DDP each rank owns exactly one device; the legacy multi-GPU primitive
    # is disabled so we don't double-replicate.
    if is_ddp:
        devices = [device]
        n_gpus = 1
        if rank == 0:
            print(f"[DDP] world_size={world_size} backend={dist.get_backend()}", flush=True)
    else:
        devices = get_training_devices(device, multi_gpu)
        n_gpus = len(devices)
        if n_gpus > 1:
            print(f"[Multi-GPU legacy] Using {n_gpus} GPUs: {', '.join(str(d) for d in devices)}")
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
        train_idx_np, sampler, config, shuffle=True, n_gpus=n_gpus,
        world_size=world_size, rank=rank, seed=seed,
        class_labels=train_labels_np,
    )
    val_loader, _ = make_neighbor_loader(
        val_idx_np, sampler, config, shuffle=False, n_gpus=n_gpus,
        world_size=world_size, rank=rank, seed=seed,
    )
    test_loader, _ = make_neighbor_loader(
        test_idx_np, sampler, config, shuffle=False, n_gpus=n_gpus,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    grad_accum_steps = max(1, int(config["train"].get("grad_accum_steps", 1)))
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
    weight = None
    if str(config["train"]["class_weight"]).lower() == "balanced":
        _max_w = float(config["train"].get("class_weight_cap", float("inf")))
        weight = class_weights_from_backend(backend, train_idx_np, num_classes, max_weight=_max_w).to(device)

    output_dir = ensure_dir(Path(config["train"]["output_dir"])) if rank == 0 else Path(config["train"]["output_dir"])
    best_checkpoint = output_dir / "hgt_flow_best.pt"
    label_names = label_name_mapping(backend.manifest, labels_np)
    monitor = str(config["train"]["monitor"])
    if monitor not in {"val_macro_f1", "val_loss"}:
        raise ValueError(f"Unknown monitor metric {monitor!r}. Supported: 'val_macro_f1', 'val_loss'.")
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
            # In DDP mode each rank takes one batch at a time (DDP handles
            # cross-rank parallelism through all-reduce). In legacy mode we
            # still gather one batch per visible GPU.
            group_size = 1 if is_ddp else n_gpus
            batch_group: list[MiniBatchSubgraph] = []
            for _ in range(group_size):
                try:
                    batch_group.append(next(train_iter))
                except StopIteration:
                    break
            if not batch_group:
                break

            if (not is_ddp) and n_gpus > 1 and len(batch_group) > 1:
                # Legacy single-process multi-GPU: replicate + parallel_apply.
                ls, ex, bp, bl, ns, es = _multi_gpu_train_step(
                    model, batch_group, devices, edge_types, weight,
                    scaler, use_semantic_edge_weights, grad_accum_steps,
                )
                train_loss_sum_t += float(ls)
                train_examples_t += int(ex)
                for p_np, l_np in zip(bp, bl):
                    local_counts = _per_class_counts_tensor(
                        torch.from_numpy(p_np), torch.from_numpy(l_np), num_classes,
                    ).to(device)
                    train_counts.add_(local_counts)
                for nt, cnts in ns.items():
                    sampled_node_sum[nt] = sampled_node_sum.get(nt, 0) + sum(cnts)
                    sampled_node_cnt[nt] = sampled_node_cnt.get(nt, 0) + len(cnts)
                for et, cnts in es.items():
                    sampled_edge_sum[et] = sampled_edge_sum.get(et, 0) + sum(cnts)
                    sampled_edge_cnt[et] = sampled_edge_cnt.get(et, 0) + len(cnts)
            else:
                # DDP or single-device: one forward + backward per step.
                batch = batch_group[0]
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
                with sync_ctx, torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(nf, ei, edge_weight_dict=ew)
                    seed_logits = logits[sm].float()
                    loss = F.cross_entropy(seed_logits, sl, weight=weight)
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
            batches_seen += len(batch_group)
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
            devices=devices if (n_gpus > 1 and not is_ddp) else None,
            is_ddp=is_ddp,
        )
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
        _wandb_log(entry, use_wandb)

        _val_loss = val_metrics["loss"]
        monitor_score = float(
            val_metrics["macro_f1"] if monitor == "val_macro_f1"
            else -(_val_loss if _val_loss is not None else float("inf"))
        )
        if monitor_score > best_score:
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
        else:
            epochs_without_improvement += 1

        if rank == 0 and nonfinite_loss_count > 0:
            print(
                f"[warn] Epoch {epoch:03d}: {nonfinite_loss_count} batch(es) had a "
                f"non-finite (NaN/Inf) loss and were skipped — their gradients were "
                f"never applied. Persistent non-finite loss freezes training; "
                f"check AMP dtype and feature scaling.",
                flush=True,
            )
        if rank == 0 and (epoch == 1 or epoch % log_every == 0):
            print(
                f"Epoch {epoch:03d} | "
                f"loss={float(train_metrics['loss']) if train_metrics['loss'] is not None else float('nan'):.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
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
        )
        device_str = (
            f"{device} x{world_size}" if is_ddp else
            str(device) + (f" x{n_gpus}" if n_gpus > 1 else "")
        )
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
