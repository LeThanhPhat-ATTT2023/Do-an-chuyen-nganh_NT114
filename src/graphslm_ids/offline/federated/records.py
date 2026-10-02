"""JSON-safe conversion and small file writers for FL run records."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | Path, payload: Any) -> None:
    """Write ``payload`` as indented JSON, creating parent dirs; replace the file atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    # Serialize before touching the disk so a bad payload never leaves a partial file behind.
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def append_jsonl(path: str | Path, record: Any) -> None:
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_jsonable(record)) + "\n")
