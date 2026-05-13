from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.models.student_cnn import Student1DCNN
from graphslm_ids.utils.io import ensure_dir, write_json


class PayloadDataset(Dataset[torch.Tensor]):
    """Wraps a mmap'd numpy array for use with multi-worker DataLoader."""

    def __init__(self, matrix: np.ndarray, total_rows: int) -> None:
        self.matrix = matrix
        self.n = total_rows

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.matrix[idx], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export student 1D-CNN embeddings for all payload rows."
    )
    parser.add_argument("--payload-npy", required=True, help="Path to payload_256.npy")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to student checkpoint containing model_state_dict.",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/student_embeddings.npy",
        help="Output path for student embedding matrix (.npy).",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for rows to export (debugging only).",
    )
    parser.add_argument(
        "--l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2 normalize output embeddings for cosine similarity use.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:0 style values.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers for CPU prefetch. 0 = single-process (safe with mmap input). -1 = auto (cpu_count // 2, max 4).",
    )
    parser.add_argument(
        "--fp16-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save embeddings as float16 to halve disk usage (~29GB vs ~59GB for 19M rows).",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Apply torch.compile(mode='reduce-overhead') for faster inference. Requires PyTorch >= 2.0.",
    )
    return parser.parse_args()


def _auto_scale(device: torch.device, num_workers_arg: int) -> tuple[int, int]:
    """Return (num_workers, infer_threads) scaled to the device."""
    cpu_count = os.cpu_count() or 1
    if device.type == "cuda":
        workers = num_workers_arg if num_workers_arg >= 0 else min(cpu_count // 2, 8)
        infer_threads = max(4, cpu_count // 2)
    else:
        if num_workers_arg >= 0:
            workers = num_workers_arg
            infer_threads = max(1, cpu_count - workers)
        else:
            # Reserve ~1/4 cores for data loading, rest for PyTorch ops
            workers = min(max(1, cpu_count // 4), 4)
            infer_threads = max(1, cpu_count - workers)
    return workers, infer_threads


def main() -> None:
    import shutil

    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    num_workers, infer_threads = _auto_scale(device, args.num_workers)

    if device.type == "cpu":
        torch.set_num_threads(infer_threads)
        torch.set_num_interop_threads(max(1, infer_threads // 4))
        print(f"[cpu] {os.cpu_count()} cores → {infer_threads} infer threads + {num_workers} data workers")
    else:
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"[gpu] {gpu_name} ({gpu_mem:.1f} GB) — {num_workers} data workers")

    payload_path = Path(args.payload_npy)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)

    payload_matrix = np.load(payload_path, mmap_mode="r")
    if payload_matrix.ndim != 2:
        raise ValueError("Payload matrix must be 2D.")

    total_rows = int(payload_matrix.shape[0])
    if args.max_rows is not None:
        total_rows = min(total_rows, int(args.max_rows))

    if total_rows <= 0:
        raise ValueError("No payload rows to export.")

    out_dtype = np.float16 if args.fp16_output else np.float32
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing model_state_dict.")

    embedding_dim = int(checkpoint.get("embedding_dim", 768))
    out_bytes = total_rows * embedding_dim * np.dtype(out_dtype).itemsize
    try:
        free_bytes = shutil.disk_usage(str(output_path.parent)).free
        if out_bytes > free_bytes:
            raise RuntimeError(
                f"Not enough disk space: need {out_bytes / 1e9:.1f} GB, "
                f"have {free_bytes / 1e9:.1f} GB free at {output_path.parent}. "
                f"Try --fp16-output to halve the output size."
            )
        print(f"[disk] output size {out_bytes / 1e9:.2f} GB, free {free_bytes / 1e9:.1f} GB — OK")
    except RuntimeError:
        raise
    except Exception:
        pass

    model = Student1DCNN(embedding_dim=embedding_dim, dropout=args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _compile_active = False
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            _compile_active = True
            print("[torch.compile] wrapped with mode='reduce-overhead' (JIT on first batch)")
        except Exception as exc:
            print(f"[torch.compile] skipped at wrap stage — {exc}")

    use_amp = device.type == "cuda"
    loader = DataLoader(
        PayloadDataset(payload_matrix, total_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )

    embeddings = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=out_dtype,
        shape=(total_rows, embedding_dim),
    )

    start = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Export student embeddings"):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                try:
                    output = model(batch)
                except Exception as exc:
                    if _compile_active:
                        print(f"\n[torch.compile] JIT error ({type(exc).__name__}), falling back to eager")
                        model = getattr(model, "_orig_mod", model)
                        _compile_active = False
                        output = model(batch)
                    else:
                        raise
                if args.l2_normalize:
                    output = F.normalize(output, p=2, dim=1)
            end = start + len(batch)
            embeddings[start:end] = output.float().detach().cpu().numpy().astype(out_dtype)
            start = end

    embeddings.flush()

    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_val_loss = checkpoint.get("val_loss")
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload_npy": str(payload_path),
        "checkpoint": str(checkpoint_path),
        "output_path": str(output_path),
        "device": str(device),
        "rows": int(total_rows),
        "embedding_dim": int(embedding_dim),
        "batch_size": int(args.batch_size),
        "num_workers": int(num_workers),
        "infer_threads": int(infer_threads),
        "dropout": float(args.dropout),
        "l2_normalize": bool(args.l2_normalize),
        "amp": bool(use_amp),
        "compile": bool(args.compile),
        "fp16_output": bool(args.fp16_output),
        "output_dtype": str(out_dtype),
        "checkpoint_epoch": int(checkpoint_epoch) if checkpoint_epoch is not None else None,
        "checkpoint_val_loss": float(checkpoint_val_loss)
        if checkpoint_val_loss is not None
        else None,
    }
    write_json(output_path.with_suffix(".meta.json"), meta)

    print(f"[OK] Student embeddings saved: {output_path}")
    print(f"[OK] Metadata saved: {output_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
