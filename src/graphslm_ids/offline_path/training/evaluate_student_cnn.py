from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.models.student_cnn import Student1DCNN
from graphslm_ids.utils.io import ensure_dir, write_json


class DistillationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Pairs payload vectors with teacher embeddings and exposes sample indices."""

    def __init__(self, payload_npy: Path, teacher_npy: Path) -> None:
        self.payload = np.load(payload_npy, mmap_mode="r")
        self.teacher = np.load(teacher_npy, mmap_mode="r")

        if self.payload.ndim != 2:
            raise ValueError("Payload matrix must be 2D.")
        if self.teacher.ndim != 2:
            raise ValueError("Teacher matrix must be 2D.")
        if self.payload.shape[0] != self.teacher.shape[0]:
            raise ValueError("Payload and teacher rows must match.")

        self.embedding_dim = int(self.teacher.shape[1])

    def __len__(self) -> int:
        return int(self.payload.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        # torch.tensor creates a writable copy and avoids mmap read-only warnings.
        payload = torch.tensor(self.payload[idx], dtype=torch.float32)
        teacher = torch.tensor(self.teacher[idx], dtype=torch.float32)
        return payload, teacher, idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate distilled student embeddings against teacher targets.")
    parser.add_argument("--payload-npy", required=True)
    parser.add_argument("--teacher-npy", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-path", default="outputs/student_cnn/evaluation_summary.json")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_labels(metadata_csv: Path) -> list[str]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "label" not in (reader.fieldnames or []):
            raise ValueError("metadata.csv must contain a 'label' column.")

        labels: list[str] = []
        for row in reader:
            label = row.get("label")
            labels.append(label if label else "UNKNOWN")

    return labels


def finalize_overall(count: int, mse_sum: float, cosine_sim_sum: float, cosine_dist_sum: float) -> dict[str, float | int]:
    if count <= 0:
        return {
            "count": 0,
            "mse_mean": float("nan"),
            "cosine_similarity_mean": float("nan"),
            "cosine_distance_mean": float("nan"),
        }

    return {
        "count": int(count),
        "mse_mean": float(mse_sum / count),
        "cosine_similarity_mean": float(cosine_sim_sum / count),
        "cosine_distance_mean": float(cosine_dist_sum / count),
    }


def finalize_per_label(
    raw: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for label in sorted(raw.keys()):
        bucket = raw[label]
        count = int(bucket["count"])
        if count <= 0:
            continue

        result[label] = {
            "count": count,
            "mse_mean": float(float(bucket["mse_sum"]) / count),
            "cosine_similarity_mean": float(float(bucket["cosine_sim_sum"]) / count),
            "cosine_distance_mean": float(float(bucket["cosine_dist_sum"]) / count),
        }

    return result


def top_k_worst_labels(
    per_label: dict[str, dict[str, float | int]],
    k: int = 5,
) -> list[dict[str, float | int | str]]:
    ranked = sorted(
        per_label.items(),
        key=lambda item: (float(item[1]["cosine_similarity_mean"]), item[0]),
    )

    output: list[dict[str, float | int | str]] = []
    for label, metrics in ranked[:k]:
        output.append(
            {
                "label": label,
                "count": int(metrics["count"]),
                "mse_mean": float(metrics["mse_mean"]),
                "cosine_similarity_mean": float(metrics["cosine_similarity_mean"]),
            }
        )
    return output


def evaluate_split(
    model: Student1DCNN,
    dataset: DistillationDataset,
    indices: list[int],
    labels: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    split_name: str,
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    total_count = 0
    mse_sum = 0.0
    cosine_sim_sum = 0.0
    cosine_dist_sum = 0.0

    per_label_raw: dict[str, dict[str, float | int]] = {}

    model.eval()
    iterator = tqdm(loader, desc=f"eval-{split_name}", leave=False)
    with torch.no_grad():
        for payload, teacher, sample_idx in iterator:
            payload = payload.to(device)
            teacher = teacher.to(device)

            output = model(payload)

            mse_item = F.mse_loss(output, teacher, reduction="none").mean(dim=1)
            cosine_sim_item = F.cosine_similarity(output, teacher, dim=1)
            cosine_dist_item = 1.0 - cosine_sim_item

            batch_size_actual = int(payload.shape[0])
            total_count += batch_size_actual
            mse_sum += float(mse_item.sum().item())
            cosine_sim_sum += float(cosine_sim_item.sum().item())
            cosine_dist_sum += float(cosine_dist_item.sum().item())

            mse_values = mse_item.cpu().tolist()
            cosine_sim_values = cosine_sim_item.cpu().tolist()
            cosine_dist_values = cosine_dist_item.cpu().tolist()
            sample_indices = sample_idx.cpu().tolist()

            for i, dataset_idx in enumerate(sample_indices):
                label = labels[int(dataset_idx)]
                bucket = per_label_raw.setdefault(
                    label,
                    {
                        "count": 0,
                        "mse_sum": 0.0,
                        "cosine_sim_sum": 0.0,
                        "cosine_dist_sum": 0.0,
                    },
                )

                bucket["count"] = int(bucket["count"]) + 1
                bucket["mse_sum"] = float(bucket["mse_sum"]) + float(mse_values[i])
                bucket["cosine_sim_sum"] = float(bucket["cosine_sim_sum"]) + float(cosine_sim_values[i])
                bucket["cosine_dist_sum"] = float(bucket["cosine_dist_sum"]) + float(cosine_dist_values[i])

    return finalize_overall(total_count, mse_sum, cosine_sim_sum, cosine_dist_sum), finalize_per_label(per_label_raw)


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset = DistillationDataset(Path(args.payload_npy), Path(args.teacher_npy))
    labels = load_labels(Path(args.metadata_csv))
    if len(labels) != len(dataset):
        raise ValueError("metadata.csv row count must match payload/teacher row count.")

    total_samples = len(dataset)
    if total_samples < 2:
        raise ValueError("Need at least 2 samples to evaluate train/val splits.")

    val_size = max(1, int(total_samples * args.val_ratio))
    train_size = total_samples - val_size
    if train_size < 1:
        raise ValueError("Invalid split. Reduce val_ratio.")

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    train_indices = list(train_subset.indices)
    val_indices = list(val_subset.indices)
    all_indices = list(range(total_samples))

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise ValueError("Checkpoint does not contain model_state_dict.")

    embedding_dim = int(checkpoint.get("embedding_dim", dataset.embedding_dim))
    model = Student1DCNN(embedding_dim=embedding_dim, dropout=args.dropout).to(device)
    model.load_state_dict(model_state)

    all_overall, all_per_label = evaluate_split(
        model,
        dataset,
        all_indices,
        labels,
        device,
        args.batch_size,
        args.num_workers,
        split_name="all",
    )
    train_overall, train_per_label = evaluate_split(
        model,
        dataset,
        train_indices,
        labels,
        device,
        args.batch_size,
        args.num_workers,
        split_name="train",
    )
    val_overall, val_per_label = evaluate_split(
        model,
        dataset,
        val_indices,
        labels,
        device,
        args.batch_size,
        args.num_workers,
        split_name="val",
    )

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "payload_npy": args.payload_npy,
        "teacher_npy": args.teacher_npy,
        "metadata_csv": args.metadata_csv,
        "checkpoint": args.checkpoint,
        "samples_total": total_samples,
        "samples_train": train_size,
        "samples_val": val_size,
        "embedding_dim": embedding_dim,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", float("nan"))),
        "splits": {
            "all": {
                "overall": all_overall,
                "per_label": all_per_label,
                "per_label_worst_cosine_similarity_top5": top_k_worst_labels(all_per_label, k=5),
            },
            "train": {
                "overall": train_overall,
                "per_label": train_per_label,
                "per_label_worst_cosine_similarity_top5": top_k_worst_labels(train_per_label, k=5),
            },
            "val": {
                "overall": val_overall,
                "per_label": val_per_label,
                "per_label_worst_cosine_similarity_top5": top_k_worst_labels(val_per_label, k=5),
            },
        },
    }
    write_json(output_path, report)

    print(
        "[EVAL] split=val "
        f"count={val_overall['count']} "
        f"mse_mean={val_overall['mse_mean']:.6f} "
        f"cosine_similarity_mean={val_overall['cosine_similarity_mean']:.6f}"
    )
    print(f"[OK] Evaluation summary: {output_path}")


if __name__ == "__main__":
    main()
