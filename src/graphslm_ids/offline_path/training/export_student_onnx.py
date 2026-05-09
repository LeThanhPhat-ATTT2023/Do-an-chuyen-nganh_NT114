from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.models.student_cnn import Student1DCNN
from graphslm_ids.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained student 1D-CNN to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-path", default="outputs/student_cnn/student_cnn.onnx")
    parser.add_argument("--input-length", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dynamo-exporter", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing model_state_dict.")

    embedding_dim = int(checkpoint.get("embedding_dim", 768))

    model = Student1DCNN(embedding_dim=embedding_dim, dropout=args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)

    dummy_payload = torch.zeros(1, args.input_length, dtype=torch.float32, device=device)

    export_mode = "dynamo" if args.dynamo_exporter else "legacy"
    export_kwargs = {
        "input_names": ["payload"],
        "output_names": ["embedding"],
        "dynamic_axes": {
            "payload": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        "opset_version": args.opset,
        "do_constant_folding": True,
        "dynamo": args.dynamo_exporter,
    }

    try:
        torch.onnx.export(model, dummy_payload, output_path, **export_kwargs)
    except Exception as exc:
        if args.dynamo_exporter:
            print(f"[WARN] Dynamo exporter failed ({exc.__class__.__name__}); retrying with legacy exporter.")
            export_mode = "legacy_fallback"
            export_kwargs["dynamo"] = False
            torch.onnx.export(model, dummy_payload, output_path, **export_kwargs)
        else:
            raise

    if args.verify:
        import onnx

        graph = onnx.load(output_path)
        onnx.checker.check_model(graph)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "output_path": str(output_path),
        "export_mode": export_mode,
        "device": str(device),
        "embedding_dim": embedding_dim,
        "input_length": int(args.input_length),
        "opset": int(args.opset),
        "verified": bool(args.verify),
    }
    write_json(output_path.with_suffix(".meta.json"), summary)

    print(f"[OK] ONNX exported: {output_path}")
    print(f"[OK] Export summary: {output_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
