"""Train compact 1D-CNN student via embedding distillation.

Launch modes
------------
Single GPU / CPU:
    python train_student_cnn.py --payload-npy ... --teacher-npy ...

Multi-GPU (auto-scales to all GPUs on the node):
    torchrun --nproc_per_node=N train_student_cnn.py --payload-npy ... --teacher-npy ...

The script detects the DDP environment automatically via the LOCAL_RANK env var
that torchrun injects.  All flags work identically in both modes.
"""
from __future__ import annotations

import argparse
import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.models.student_cnn import Student1DCNN
from graphslm_ids.utils.io import ensure_dir, write_json


# ── distributed ───────────────────────────────────────────────────────────────

def setup_distributed() -> tuple[int, int, int]:
    """Return (rank, local_rank, world_size).  Init NCCL if torchrun set LOCAL_RANK."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def teardown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ── dataset ───────────────────────────────────────────────────────────────────

class DistillationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Pairs payload bytes with teacher embeddings; both mmap'd to avoid OOM."""

    def __init__(self, payload_npy: Path, teacher_npy: Path) -> None:
        self.payload = np.load(payload_npy, mmap_mode="r")
        self.teacher = np.load(teacher_npy, mmap_mode="r")

        if self.payload.ndim != 2 or self.teacher.ndim != 2:
            raise ValueError("Payload and teacher must be 2-D arrays.")

        if self.payload.shape[0] > self.teacher.shape[0]:
            import warnings
            warnings.warn(
                f"Payload rows ({self.payload.shape[0]:,}) > teacher rows "
                f"({self.teacher.shape[0]:,}). Truncating payload to match."
            )
            self.payload = self.payload[: self.teacher.shape[0]]
        elif self.payload.shape[0] < self.teacher.shape[0]:
            raise ValueError(
                f"Teacher rows ({self.teacher.shape[0]:,}) > payload rows "
                f"({self.payload.shape[0]:,}). Rebuild teacher from current payload."
            )

        self.embedding_dim = int(self.teacher.shape[1])

    def __len__(self) -> int:
        return int(self.payload.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.payload[idx], dtype=np.float32))
        y = torch.from_numpy(np.asarray(self.teacher[idx], dtype=np.float32))
        return x, y


# ── loss ──────────────────────────────────────────────────────────────────────

def distillation_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    mse_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse    = F.mse_loss(student_out, teacher_out)
    cosine = 1.0 - F.cosine_similarity(student_out, teacher_out, dim=1).mean()
    total  = mse_weight * mse + (1.0 - mse_weight) * cosine
    return total, mse, cosine


# ── epoch loop ────────────────────────────────────────────────────────────────

def run_epoch(
    *,
    model:      torch.nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    optimizer:  torch.optim.Optimizer | None,
    scheduler:  torch.optim.lr_scheduler.LRScheduler | None,
    scaler:     torch.cuda.amp.GradScaler,
    mse_weight: float,
    use_amp:    bool,
    rank:       int,
    world_size: int,
    is_ddp:     bool,
    raw_params: list[torch.nn.Parameter],
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    # Accumulate weighted sums on-device; all_reduce once after the loop.
    acc_loss   = torch.zeros(1, device=device)
    acc_mse    = torch.zeros(1, device=device)
    acc_cosine = torch.zeros(1, device=device)
    acc_n      = torch.zeros(1, device=device)

    bar = tqdm(loader, desc="train" if is_train else "valid",
               leave=False, disable=rank != 0)

    with torch.set_grad_enabled(is_train):
        for payload, teacher in bar:
            payload = payload.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type,
                                 dtype=torch.float16, enabled=use_amp):
                output = model(payload)
                loss, mse, cosine = distillation_loss(output, teacher, mse_weight)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            n = payload.shape[0]
            acc_loss   += loss.detach() * n
            acc_mse    += mse.detach()  * n
            acc_cosine += cosine.detach() * n
            acc_n      += n

    if is_ddp:
        for t in (acc_loss, acc_mse, acc_cosine, acc_n):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    n = max(float(acc_n.item()), 1.0)
    return {
        "loss":   float(acc_loss.item()   / n),
        "mse":    float(acc_mse.item()    / n),
        "cosine": float(acc_cosine.item() / n),
    }


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--payload-npy",  required=True)
    p.add_argument("--teacher-npy",  required=True)
    p.add_argument("--output-dir",   default="outputs/student_cnn")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument(
        "--batch-size", type=int, default=1024,
        help="Per-GPU batch size. Effective batch = batch_size × world_size.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Peak LR for OneCycleLR. Auto-scaled by world_size (linear scaling rule).",
    )
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--val-ratio",     type=float, default=0.1)
    p.add_argument("--mse-weight",    type=float, default=0.7)
    p.add_argument("--patience",      type=int,   default=6)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument(
        "--num-workers", type=int, default=-1,
        help="DataLoader workers per GPU. -1 = auto (cpu_count // world_size, capped at 8).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--compile", action="store_true",
        help="Apply torch.compile(mode='reduce-overhead'). Requires PyTorch >= 2.0.",
    )
    return p.parse_args()


# ── seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    rank, local_rank, world_size = setup_distributed()
    is_ddp = world_size > 1

    device  = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    use_amp = device.type == "cuda"

    # All ranks use the same seed so the train/val split is identical everywhere.
    set_seed(args.seed)

    # ── dataset / split ───────────────────────────────────────────────────────
    dataset = DistillationDataset(Path(args.payload_npy), Path(args.teacher_npy))
    total   = len(dataset)
    val_n   = max(1, int(total * args.val_ratio))
    train_n = total - val_n
    if train_n < 1:
        raise ValueError("Dataset too small for the requested val_ratio.")

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=generator)

    # ── DataLoader workers ────────────────────────────────────────────────────
    if args.num_workers < 0:
        cpu_n = os.cpu_count() or 1
        num_workers = max(1, min(cpu_n // max(world_size, 1), 8))
    else:
        num_workers = args.num_workers

    loader_kw = dict(
        pin_memory        = use_amp,
        num_workers       = num_workers,
        persistent_workers= num_workers > 0,
        prefetch_factor   = 2 if num_workers > 0 else None,
    )

    if is_ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler, drop_last=True,  **loader_kw)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  sampler=val_sampler,   drop_last=False, **loader_kw)
    else:
        train_sampler = None
        train_loader  = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True,  drop_last=True,  **loader_kw)
        val_loader    = DataLoader(val_ds,   batch_size=args.batch_size,
                                   shuffle=False, drop_last=False, **loader_kw)

    # ── model ─────────────────────────────────────────────────────────────────
    model = Student1DCNN(embedding_dim=dataset.embedding_dim, dropout=args.dropout).to(device)

    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            if rank == 0:
                print("[torch.compile] compiled with mode='reduce-overhead'")
        except Exception as exc:
            if rank == 0:
                print(f"[torch.compile] skipped — {exc}")

    if is_ddp:
        # find_unused_parameters=False: all params are always used → no autograd hook overhead.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # raw_params: the leaf parameters for grad-clipping (unwrap DDP / compiled wrappers).
    raw_model  = model.module if is_ddp else model
    raw_params = list(raw_model.parameters())

    # ── optimizer / scheduler / scaler ────────────────────────────────────────
    # Linear scaling rule: larger effective batch → proportionally larger LR.
    peak_lr   = args.lr * world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = peak_lr,
        steps_per_epoch = len(train_loader),
        epochs          = args.epochs,
        pct_start       = 0.3,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── output dir (rank 0 only) ───────────────────────────────────────────────
    output_dir     = ensure_dir(Path(args.output_dir)) if rank == 0 else Path(args.output_dir)
    best_ckpt_path = output_dir / "student_cnn_best.pt"

    if rank == 0:
        print(
            f"[train] device={device}  world_size={world_size}  "
            f"effective_batch={args.batch_size * world_size:,}  "
            f"peak_lr={peak_lr:.2e}  num_workers={num_workers}  "
            f"AMP={'on' if use_amp else 'off'}  "
            f"compile={'on' if args.compile else 'off'}  "
            f"samples={total:,}"
        )

    # ── training loop ─────────────────────────────────────────────────────────
    history:                  list[dict[str, float]] = []
    best_val                = float("inf")
    epochs_no_improve       = 0

    epoch_kw = dict(
        model=model, device=device,
        scaler=scaler, mse_weight=args.mse_weight,
        use_amp=use_amp, rank=rank, world_size=world_size,
        is_ddp=is_ddp, raw_params=raw_params,
    )

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_m = run_epoch(loader=train_loader, optimizer=optimizer,
                            scheduler=scheduler, **epoch_kw)
        val_m   = run_epoch(loader=val_loader,   optimizer=None,
                            scheduler=None,      **epoch_kw)

        if rank == 0:
            entry = {
                "epoch":        float(epoch),
                "train_loss":   train_m["loss"],
                "train_mse":    train_m["mse"],
                "train_cosine": train_m["cosine"],
                "val_loss":     val_m["loss"],
                "val_mse":      val_m["mse"],
                "val_cosine":   val_m["cosine"],
            }
            history.append(entry)
            print(
                f"Epoch {epoch:03d}  "
                f"train_loss={entry['train_loss']:.6f}  "
                f"val_loss={entry['val_loss']:.6f}  "
                f"val_cosine={entry['val_cosine']:.6f}"
            )

            if entry["val_loss"] < best_val:
                best_val        = entry["val_loss"]
                epochs_no_improve = 0
                # Always save raw (un-DDP, un-compiled) state dict for portability.
                save_model = raw_model
                if hasattr(save_model, "_orig_mod"):   # unwrap torch.compile
                    save_model = save_model._orig_mod
                torch.save(
                    {
                        "model_state_dict": save_model.state_dict(),
                        "embedding_dim":    dataset.embedding_dim,
                        "epoch":            epoch,
                        "val_loss":         best_val,
                    },
                    best_ckpt_path,
                )
            else:
                epochs_no_improve += 1

        # Broadcast the early-stop decision from rank 0 so all ranks exit together.
        if is_ddp:
            stop = torch.tensor(int(epochs_no_improve >= args.patience), device=device)
            dist.broadcast(stop, src=0)
            if stop.item():
                if rank == 0:
                    print("Early stopping triggered.")
                break
        elif epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    if rank == 0:
        summary = {
            "created_at_utc":    datetime.now(timezone.utc).isoformat(),
            "payload_npy":       args.payload_npy,
            "teacher_npy":       args.teacher_npy,
            "device":            str(device),
            "world_size":        world_size,
            "effective_batch":   args.batch_size * world_size,
            "peak_lr":           peak_lr,
            "samples_total":     total,
            "samples_train":     train_n,
            "samples_val":       val_n,
            "embedding_dim":     dataset.embedding_dim,
            "best_val_loss":     float(best_val),
            "best_checkpoint":   str(best_ckpt_path),
            "history":           history,
        }
        write_json(output_dir / "training_summary.json", summary)
        print(f"[OK] checkpoint  → {best_ckpt_path}")
        print(f"[OK] summary     → {output_dir / 'training_summary.json'}")

    teardown_distributed()


if __name__ == "__main__":
    main()
