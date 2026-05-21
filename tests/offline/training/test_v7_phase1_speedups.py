"""Phase 1 speed-up verification tests.

These tests exercise the diagnostic logging additions in Phase 1 without
requiring an actual GPU run — they use a mock training loop.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import torch


def test_peak_vram_logged_at_epoch_end_when_cuda():
    """When training runs on CUDA, the epoch-end log must include peak VRAM in GB."""
    from graphslm_ids.offline.training.train_hgt_flow_classifier import (
        _log_epoch_diagnostics,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        _log_epoch_diagnostics(
            epoch=3,
            elapsed_seconds=42.5,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            rank=0,
        )
    out = buf.getvalue()
    assert "epoch=3" in out
    assert "wall=" in out
    if torch.cuda.is_available():
        assert "peak_vram_gb=" in out


def test_peak_vram_skipped_on_cpu():
    """CPU runs must not crash and should omit VRAM field."""
    from graphslm_ids.offline.training.train_hgt_flow_classifier import (
        _log_epoch_diagnostics,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        _log_epoch_diagnostics(
            epoch=1,
            elapsed_seconds=10.0,
            device=torch.device("cpu"),
            rank=0,
        )
    out = buf.getvalue()
    assert "epoch=1" in out
    assert "peak_vram_gb=" not in out


def test_log_only_emitted_on_rank_zero():
    """In DDP runs, non-rank-0 ranks must stay silent to avoid log spam."""
    from graphslm_ids.offline.training.train_hgt_flow_classifier import (
        _log_epoch_diagnostics,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        _log_epoch_diagnostics(
            epoch=1,
            elapsed_seconds=10.0,
            device=torch.device("cpu"),
            rank=2,
        )
    assert buf.getvalue() == ""
