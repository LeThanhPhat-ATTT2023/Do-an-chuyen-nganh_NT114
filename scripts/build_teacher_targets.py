from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.utils.io import write_json


def payload_row_to_hex_text(
    row: np.ndarray,
    drop_padding: bool = False,
    valid_length: int | None = None,
) -> str:
    """Convert uint8 payload vectors into whitespace-separated hex tokens."""
    data = np.asarray(row, dtype=np.uint8)

    if valid_length is not None:
        clipped_length = max(0, min(int(valid_length), int(data.shape[0])))
        data = data[:clipped_length]
    elif drop_padding:
        nonzero_indices = np.nonzero(data)[0]
        if nonzero_indices.size > 0:
            data = data[: nonzero_indices[-1] + 1]
        else:
            data = data[:0]

    if data.size == 0:
        return "00"

    return " ".join(f"{int(byte):02x}" for byte in data.tolist())


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply attention-mask-aware mean pooling for sentence embeddings."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create teacher embedding targets from payload vectors using a transformer encoder."
    )
    parser.add_argument("--payload-npy", required=True, help="Path to payload_256.npy")
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional metadata CSV with payload_len_raw column from extract_payload_dataset.py.",
    )
    parser.add_argument("--output-path", required=True, help="Output path for teacher_targets.npy")
    parser.add_argument(
        "--model-name",
        default="ehsanaghaei/SecureBERT",
        help="HuggingFace model id for teacher embedding extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick debugging.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:0 style values.",
    )
    parser.add_argument(
        "--drop-padding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop trailing zero bytes when metadata CSV is not provided.",
    )
    parser.add_argument(
        "--l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2 normalize embeddings for cosine similarity usage.",
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
        payload_lengths = np.asarray(metadata["payload_len_raw"].iloc[:total_rows], dtype=np.int64)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    teacher_targets = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, hidden_size),
    )

    with torch.no_grad():
        for start in tqdm(range(0, total_rows, args.batch_size), desc="Encoding batches"):
            end = min(start + args.batch_size, total_rows)
            batch_rows = np.asarray(payload_matrix[start:end], dtype=np.uint8)
            if payload_lengths is not None:
                batch_lengths = payload_lengths[start:end]
                texts = [
                    payload_row_to_hex_text(
                        row,
                        drop_padding=args.drop_padding,
                        valid_length=int(length),
                    )
                    for row, length in zip(batch_rows, batch_lengths)
                ]
            else:
                texts = [payload_row_to_hex_text(row, drop_padding=args.drop_padding) for row in batch_rows]

            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            if args.l2_normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            teacher_targets[start:end] = pooled.detach().cpu().numpy().astype(np.float32)

    teacher_targets.flush()

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload_source": str(payload_path),
        "metadata_csv": args.metadata_csv,
        "teacher_model": args.model_name,
        "rows": total_rows,
        "embedding_dim": hidden_size,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "drop_padding": bool(args.drop_padding),
        "l2_normalize": bool(args.l2_normalize),
        "output_path": str(output_path),
    }
    write_json(output_path.with_suffix(".meta.json"), meta)

    print(f"[OK] Teacher targets saved: {output_path}")
    print(f"[OK] Metadata saved: {output_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
