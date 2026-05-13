from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.utils.io import write_json


def payload_row_to_hex_text(
    row: np.ndarray,
    drop_padding: bool = False,
    valid_length: int | None = None,
) -> str:
    """Convert uint8 payload vector to whitespace-separated hex tokens.

    Uses bytes.hex(' ') (C-level) instead of a Python loop — ~50x faster.
    """
    data = np.asarray(row, dtype=np.uint8)

    if valid_length is not None:
        clipped = max(0, min(int(valid_length), int(data.shape[0])))
        data = data[:clipped]
    elif drop_padding:
        nz = np.nonzero(data)[0]
        data = data[: nz[-1] + 1] if nz.size > 0 else data[:0]

    if data.size == 0:
        return "00"

    # C-level hex conversion — Python 3.8+
    return data.tobytes().hex(" ")


class PayloadDataset(Dataset):
    """Wraps the mmap payload matrix so DataLoader workers can preprocess in parallel."""

    def __init__(
        self,
        payload_matrix: np.ndarray,
        payload_lengths: np.ndarray | None,
        total_rows: int,
        drop_padding: bool,
    ) -> None:
        self.payload_matrix = payload_matrix
        self.payload_lengths = payload_lengths
        self.total_rows = total_rows
        self.drop_padding = drop_padding

    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(self, idx: int) -> str:
        row = np.asarray(self.payload_matrix[idx], dtype=np.uint8)
        if self.payload_lengths is not None:
            return payload_row_to_hex_text(row, valid_length=int(self.payload_lengths[idx]))
        return payload_row_to_hex_text(row, drop_padding=self.drop_padding)


class _TeacherForwardWrapper(torch.nn.Module):
    """Thin wrapper so DataParallel can scatter positional tensor args correctly.

    Mean-pooling and L2-normalisation are done INSIDE the forward so that
    DataParallel only gathers a (shard, hidden_size) tensor instead of the full
    (shard, seq_len, hidden_size), cutting inter-GPU transfer by ~seq_len times.
    """

    def __init__(self, backbone: torch.nn.Module, l2_normalize: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.l2_normalize = l2_normalize

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if self.l2_normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled


def _make_collate_fn(tokenizer, max_length: int):
    """Returns a collate_fn that tokenizes a list of hex strings."""

    def collate_fn(texts: list[str]):
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    return collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create teacher embedding targets from payload vectors using a transformer encoder."
    )
    parser.add_argument("--payload-npy", required=True, help="Path to payload_256.npy")
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional metadata CSV with payload_len_raw column.",
    )
    parser.add_argument("--output-path", required=True, help="Output path for teacher_targets.npy")
    parser.add_argument("--model-name", default="ehsanaghaei/SecureBERT")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Per-GPU batch size. 256 works well for T4 16GB with fp16.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--drop-padding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--l2-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-dtype",
        default="float16",
        choices=["float32", "float16"],
        help="float16 halves disk usage with negligible cosine-similarity loss.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for parallel CPU hex-conversion + tokenization.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        default=False,
        help="Disable automatic mixed precision (fp16 inference on GPU).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Apply torch.compile() to the model (PyTorch >= 2.0, ~10-20%% extra speedup).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload_path = Path(args.payload_npy)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload_matrix = np.load(payload_path, mmap_mode="r")
    if payload_matrix.ndim != 2:
        raise ValueError("Expected a 2D payload matrix.")

    total_rows = int(payload_matrix.shape[0])
    if args.max_rows is not None:
        total_rows = min(total_rows, args.max_rows)
    if total_rows == 0:
        raise ValueError("Input payload matrix is empty.")

    payload_lengths: np.ndarray | None = None
    if args.metadata_csv:
        metadata = pd.read_csv(args.metadata_csv)
        if "payload_len_raw" not in metadata.columns:
            raise ValueError("metadata-csv must contain payload_len_raw column.")
        if metadata.shape[0] < total_rows:
            raise ValueError("metadata-csv has fewer rows than payload matrix.")
        payload_lengths = np.asarray(
            metadata["payload_len_raw"].iloc[:total_rows], dtype=np.int64
        )

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    effective_batch = args.batch_size * max(1, n_gpus)
    use_amp = not args.no_amp and device.type == "cuda"

    print(
        f"[Setup] device={device}  n_gpus={n_gpus}  "
        f"effective_batch={effective_batch}  amp={use_amp}  "
        f"num_workers={args.num_workers}",
        flush=True,
    )

    # --- Model ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name).to(device)
    backbone.eval()

    teacher: torch.nn.Module = _TeacherForwardWrapper(backbone, l2_normalize=args.l2_normalize)
    if n_gpus > 1:
        print(f"[Multi-GPU] DataParallel across {n_gpus} GPUs.", flush=True)
        teacher = torch.nn.DataParallel(teacher)

    if args.compile:
        if hasattr(torch, "compile"):
            print("[Compile] Applying torch.compile()...", flush=True)
            teacher = torch.compile(teacher)
        else:
            print("[Compile] torch.compile not available (PyTorch < 2.0), skipping.", flush=True)

    hidden_size = int(backbone.config.hidden_size)
    save_dtype = np.dtype(args.save_dtype)

    # --- Disk check ---
    required_bytes = total_rows * hidden_size * save_dtype.itemsize
    disk = shutil.disk_usage(output_path.parent)
    print(
        f"[Disk] Required: {required_bytes / 1e9:.2f} GB | "
        f"Free: {disk.free / 1e9:.2f} GB | "
        f"dtype={save_dtype}",
        flush=True,
    )
    if required_bytes > disk.free * 0.95:
        safe_rows = int(disk.free * 0.90 / hidden_size / save_dtype.itemsize)
        raise RuntimeError(
            f"Not enough disk space: need {required_bytes / 1e9:.1f} GB, "
            f"only {disk.free / 1e9:.1f} GB free.\n"
            f"Options:\n"
            f"  --max-rows {safe_rows:,}   (use a subset)\n"
            f"  --save-dtype float16       (half the size)\n"
            f"  Both together for maximum fit."
        )

    # --- Output memmap ---
    print(f"[Memmap] Creating {output_path} ({required_bytes / 1e9:.2f} GB)...", flush=True)
    teacher_targets = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=save_dtype,
        shape=(total_rows, hidden_size),
    )
    print(f"[Memmap] Ready. Encoding {total_rows:,} rows...", flush=True)

    # --- DataLoader: parallel CPU preprocessing overlaps with GPU compute ---
    dataset = PayloadDataset(payload_matrix, payload_lengths, total_rows, args.drop_padding)
    loader = DataLoader(
        dataset,
        batch_size=effective_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=_make_collate_fn(tokenizer, args.max_length),
        persistent_workers=(args.num_workers > 0),
        shuffle=False,
    )

    # --- Encoding loop ---
    n_batches = (total_rows + effective_batch - 1) // effective_batch
    row_offset = 0

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for input_ids, attention_mask in tqdm(
            loader,
            desc="Encoding",
            total=n_batches,
            file=sys.stdout,
            dynamic_ncols=True,
        ):
            bsz = input_ids.size(0)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            pooled = teacher(input_ids, attention_mask)

            # float() cast handles fp16 autocast output before numpy conversion
            teacher_targets[row_offset : row_offset + bsz] = (
                pooled.detach().float().cpu().numpy().astype(save_dtype)
            )
            row_offset += bsz

    teacher_targets.flush()

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload_source": str(payload_path),
        "metadata_csv": args.metadata_csv,
        "teacher_model": args.model_name,
        "rows": total_rows,
        "embedding_dim": hidden_size,
        "batch_size_per_gpu": args.batch_size,
        "effective_batch_size": effective_batch,
        "num_gpus": max(1, n_gpus),
        "device": str(device),
        "max_length": args.max_length,
        "drop_padding": bool(args.drop_padding),
        "l2_normalize": bool(args.l2_normalize),
        "save_dtype": str(save_dtype),
        "output_path": str(output_path),
        "amp": use_amp,
        "num_workers": args.num_workers,
        "compiled": args.compile,
    }
    write_json(output_path.with_suffix(".meta.json"), meta)

    print(f"[OK] Teacher targets saved: {output_path}")
    print(f"[OK] Metadata: {output_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
