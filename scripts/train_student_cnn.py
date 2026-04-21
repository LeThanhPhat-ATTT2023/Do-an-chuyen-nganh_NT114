from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.models.student_cnn import Student1DCNN
from graphslm_ids.utils.io import ensure_dir, write_json


class DistillationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Pairs payload vectors with teacher embeddings."""

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.payload[idx], dtype=np.float32))
        y = torch.from_numpy(np.asarray(self.teacher[idx], dtype=np.float32))
        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distillation_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    mse_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(student_out, teacher_out)
    cosine = 1.0 - F.cosine_similarity(student_out, teacher_out, dim=1).mean()
    total = mse_weight * mse + (1.0 - mse_weight) * cosine
    return total, mse, cosine


def run_epoch(
    model: Student1DCNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    mse_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_items = 0

    iterator = tqdm(loader, desc="train" if is_train else "valid", leave=False)
    for payload, teacher in iterator:
        payload = payload.to(device)
        teacher = teacher.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        output = model(payload)
        loss, mse, cosine = distillation_loss(output, teacher, mse_weight=mse_weight)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = int(payload.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_mse += float(mse.item()) * batch_size
        total_cosine += float(cosine.item()) * batch_size
        total_items += batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "mse": total_mse / max(total_items, 1),
        "cosine": total_cosine / max(total_items, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train compact 1D-CNN student using embedding distillation.")
    parser.add_argument("--payload-npy", required=True)
    parser.add_argument("--teacher-npy", required=True)
    parser.add_argument("--output-dir", default="outputs/student_cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--mse-weight", type=float, default=0.7)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset = DistillationDataset(Path(args.payload_npy), Path(args.teacher_npy))
    total_samples = len(dataset)
    if total_samples < 10:
        raise ValueError("Dataset too small. Need at least 10 samples for train/val split.")

    val_size = max(1, int(total_samples * args.val_ratio))
    train_size = total_samples - val_size
    if train_size < 1:
        raise ValueError("Invalid split. Increase dataset size or reduce val_ratio.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = Student1DCNN(embedding_dim=dataset.embedding_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = ensure_dir(Path(args.output_dir))
    best_ckpt_path = output_dir / "student_cnn_best.pt"

    history: list[dict[str, float]] = []
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, mse_weight=args.mse_weight)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None, mse_weight=args.mse_weight)

        entry = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_mse": float(train_metrics["mse"]),
            "train_cosine": float(train_metrics["cosine"]),
            "val_loss": float(val_metrics["loss"]),
            "val_mse": float(val_metrics["mse"]),
            "val_cosine": float(val_metrics["cosine"]),
        }
        history.append(entry)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={entry['train_loss']:.6f} "
            f"val_loss={entry['val_loss']:.6f} "
            f"val_cosine={entry['val_cosine']:.6f}"
        )

        if entry["val_loss"] < best_val:
            best_val = entry["val_loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": dataset.embedding_dim,
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload_npy": args.payload_npy,
        "teacher_npy": args.teacher_npy,
        "device": str(device),
        "samples_total": total_samples,
        "samples_train": train_size,
        "samples_val": val_size,
        "embedding_dim": dataset.embedding_dim,
        "best_val_loss": float(best_val),
        "best_checkpoint": str(best_ckpt_path),
        "history": history,
    }
    write_json(output_dir / "training_summary.json", summary)

    print(f"[OK] Best checkpoint: {best_ckpt_path}")
    print(f"[OK] Training summary: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
