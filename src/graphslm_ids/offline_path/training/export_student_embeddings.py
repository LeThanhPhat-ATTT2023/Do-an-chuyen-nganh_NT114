from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.models.student_cnn import Student1DCNN
from graphslm_ids.utils.io import ensure_dir, write_json


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing model_state_dict.")

    embedding_dim = int(checkpoint.get("embedding_dim", 768))
    model = Student1DCNN(embedding_dim=embedding_dim, dropout=args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    embeddings = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, embedding_dim),
    )

    with torch.no_grad():
        for start in tqdm(range(0, total_rows, args.batch_size), desc="Export student embeddings"):
            end = min(start + args.batch_size, total_rows)
            batch_np = np.asarray(payload_matrix[start:end], dtype=np.float32)
            batch_tensor = torch.from_numpy(batch_np).to(device)

            output = model(batch_tensor)
            if args.l2_normalize:
                output = F.normalize(output, p=2, dim=1)

            embeddings[start:end] = output.detach().cpu().numpy().astype(np.float32)

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
        "dropout": float(args.dropout),
        "l2_normalize": bool(args.l2_normalize),
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
