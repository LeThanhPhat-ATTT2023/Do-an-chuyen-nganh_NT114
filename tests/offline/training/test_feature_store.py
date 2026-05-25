from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")


def _load_downcast():
    script = Path(__file__).resolve().parents[3] / "scripts" / "tools" / "downcast_packet_x.py"
    spec = importlib.util.spec_from_file_location("downcast_packet_x_tool", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.downcast_packet_x


def test_downcast_packet_x_rewrites_npz_to_float16(tmp_path):
    downcast_packet_x = _load_downcast()

    src = tmp_path / "graph.npz"
    np.savez(
        str(src),
        flow_x=np.zeros((2, 4), dtype=np.float32),
        flow_y=np.array([0, 1], dtype=np.int64),
        packet_x=(np.arange(3 * 2323).reshape(3, 2323) % 100).astype(np.float32),
        host_x=np.zeros((1, 4), dtype=np.float32),
        technique_x=np.zeros((5, 8), dtype=np.float32),
    )
    out = tmp_path / "graph_fp16.npz"
    downcast_packet_x(src, out)

    with np.load(out) as loaded:
        assert loaded["packet_x"].dtype == np.float16
        assert loaded["flow_x"].dtype == np.float32
        assert loaded["flow_y"].tolist() == [0, 1]
