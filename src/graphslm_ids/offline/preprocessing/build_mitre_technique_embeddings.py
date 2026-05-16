from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from graphslm_ids.utils.io import ensure_dir, read_json, write_json


def compact_text(value: object) -> str:
    text = str(value or "")
    return " ".join(text.replace("\n", " ").split())


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class _MitreForwardWrapper(torch.nn.Module):
    """Pools inside forward so DataParallel only gathers (B, hidden) per shard."""

    def __init__(self, backbone: torch.nn.Module, l2_normalize: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.l2_normalize = l2_normalize

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        if self.l2_normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MITRE technique embeddings from MITRE technique CSV text."
    )
    parser.add_argument(
        "--techniques-csv",
        default="data/mitre/mitre_techniques.csv",
        help="Path to MITRE techniques CSV.",
    )
    parser.add_argument(
        "--output-path",
        default="data/mitre/mitre_techniques_embeddings.npy",
        help="Output path for MITRE technique embedding matrix (.npy).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional HuggingFace model id. If omitted, read from teacher meta or fallback default.",
    )
    parser.add_argument(
        "--teacher-meta-json",
        default="data/processed/teacher_targets.meta.json",
        help="Optional teacher meta JSON to auto-detect teacher model.",
    )
    parser.add_argument(
        "--text-fields",
        default="technique_id,name,description,tactics",
        help="Comma-separated fields from techniques CSV to compose embedding text.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2 normalize embeddings for cosine similarity use.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:0 style values.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        default=False,
        help="Disable fp16 autocast inference on GPU (default: enabled).",
    )
    return parser.parse_args()


def resolve_model_name(model_name: str | None, teacher_meta_json: Path) -> tuple[str, str]:
    if model_name:
        return model_name, "cli"

    if teacher_meta_json.exists():
        payload = read_json(teacher_meta_json)
        teacher_model = str(payload.get("teacher_model", "")).strip()
        if teacher_model:
            return teacher_model, "teacher_meta_json"

    return "ehsanaghaei/SecureBERT", "default"


def build_technique_text(row: dict[str, object], fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        if field not in row:
            continue
        value = compact_text(row.get(field, ""))
        if value:
            parts.append(f"{field}: {value}")

    if parts:
        return " ; ".join(parts)

    fallback = compact_text(row.get("technique_id", ""))
    return fallback if fallback else "unknown-technique"


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    techniques_csv = Path(args.techniques_csv)
    output_path = Path(args.output_path)
    teacher_meta_json = Path(args.teacher_meta_json)
    ensure_dir(output_path.parent)

    techniques_df = pd.read_csv(techniques_csv)
    if techniques_df.empty:
        raise ValueError("MITRE techniques CSV is empty.")
    if "technique_id" not in techniques_df.columns:
        raise ValueError("MITRE techniques CSV must contain technique_id column.")

    text_fields = [field.strip() for field in str(args.text_fields).split(",") if field.strip()]
    if not text_fields:
        raise ValueError("text-fields cannot be empty.")

    records = techniques_df.to_dict(orient="records")
    texts = [build_technique_text(row, text_fields) for row in records]

    model_name, model_source = resolve_model_name(args.model_name, teacher_meta_json)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name).to(device)
    backbone.eval()

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_amp = (not args.no_amp) and device.type == "cuda"
    effective_batch = args.batch_size * max(1, n_gpus)

    encoder: torch.nn.Module = _MitreForwardWrapper(backbone, l2_normalize=args.l2_normalize)
    if n_gpus > 1:
        print(f"[Multi-GPU] DataParallel across {n_gpus} GPUs — effective batch {effective_batch}")
        encoder = torch.nn.DataParallel(encoder)

    num_rows = len(texts)
    embedding_dim = int(backbone.config.hidden_size)
    output_array = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.float32,
        shape=(num_rows, embedding_dim),
    )

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for start in tqdm(range(0, num_rows, effective_batch), desc="Encode MITRE techniques"):
            end = min(start + effective_batch, num_rows)
            batch_texts = texts[start:end]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True)

            pooled = encoder(input_ids, attention_mask)
            output_array[start:end] = pooled.detach().float().cpu().numpy().astype(np.float32)

    output_array.flush()

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "techniques_csv": str(techniques_csv),
        "output_path": str(output_path),
        "rows": int(num_rows),
        "embedding_dim": int(embedding_dim),
        "text_fields": text_fields,
        "model_name": model_name,
        "model_name_source": model_source,
        "teacher_meta_json": str(teacher_meta_json),
        "batch_size_per_gpu": int(args.batch_size),
        "effective_batch_size": int(effective_batch),
        "num_gpus": int(max(1, n_gpus)),
        "max_length": int(args.max_length),
        "l2_normalize": bool(args.l2_normalize),
        "device": str(device),
        "amp": bool(use_amp),
    }
    write_json(output_path.with_suffix(".meta.json"), meta)

    print(f"[OK] MITRE technique embeddings saved: {output_path}")
    print(f"[OK] Metadata saved: {output_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
