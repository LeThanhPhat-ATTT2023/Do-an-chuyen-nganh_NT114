"""Resource auto-detection for the preprocessing pipeline.

Auto-scales multiprocessing worker counts based on available RAM and CPU
count so the same codebase runs safely across machines:

  Local dev  (16 GB RAM, 16 CPUs) — conservative caps to avoid OOM
  AWS g6e.2xlarge (64 GB RAM, 8 CPUs) — uses all 8 CPUs for PCAP stages
  Larger cloud instances — scales up automatically

Two worker regimes
------------------
auto_pcap_workers(n_tasks)
    PCAP-loading stages (Stage 1 + 2).  RAM-limited: each worker loads an
    entire PCAP into memory (~1.5–3 GB for DDoS captures).

auto_compute_workers(n_tasks)
    CPU-compute stages (packet_x + evidence pool, Stage 6).  Not RAM-limited
    per-worker; capped only by logical CPU count.
"""
from __future__ import annotations

import logging
import os

_LOG = logging.getLogger(__name__)

# Conservative peak-RAM estimate per PCAP-loading worker (dpkt parse buffer
# + resulting DataFrame).  DDoS captures can hit ~1.95 GB on disk → ~2.5 GB
# in Python objects.
_DEFAULT_PER_PCAP_GB: float = 2.5

# RAM reserved for the main process, downstream stages, and the OS.
# 6 GB covers the packets_df + feats_df + OS page cache on a 64 GB machine.
_MAIN_PROCESS_RESERVE_GB: float = 6.0


def _available_ram_gb() -> float:
    """Return available system RAM in GB.

    Detection order (most → least accurate):
      1. Linux ``/proc/meminfo`` MemAvailable  — excludes disk cache cleanly
      2. ``psutil`` virtual_memory().available  — cross-platform
      3. Fallback: ``cpu_count × 4 GB``         — underestimates but safe
    """
    # ── 1. Linux /proc/meminfo ────────────────────────────────────────────────
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, OSError, ValueError):
        pass

    # ── 2. psutil (optional dependency) ──────────────────────────────────────
    try:
        import psutil  # type: ignore[import]
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass

    # ── 3. Last-resort fallback ───────────────────────────────────────────────
    cpus = os.cpu_count() or 4
    _LOG.warning(
        "[resource] Cannot read available RAM; falling back to %d GB estimate "
        "(cpu_count × 4).  Install psutil for accurate detection.",
        cpus * 4,
    )
    return float(cpus * 4)


def auto_pcap_workers(
    n_tasks: int,
    per_pcap_gb: float = _DEFAULT_PER_PCAP_GB,
    reserve_gb: float = _MAIN_PROCESS_RESERVE_GB,
    override: int | None = None,
) -> int:
    """Recommend worker count for PCAP-loading stages (Stage 1 + Stage 2).

    Keeps ``n_workers × per_pcap_gb + reserve_gb ≤ available_ram`` so the
    pipeline never OOMs regardless of machine size.  Also caps at
    ``os.cpu_count()`` and ``n_tasks``.

    Args:
        n_tasks:      Number of PCAP tasks (files or class directories).
        per_pcap_gb:  Peak RAM per worker; default 2.5 GB.
        reserve_gb:   RAM to keep free for main process + OS; default 6 GB.
        override:     If set, skip auto-detection and return this value
                      (clamped to [1, n_tasks]).  Supplied by ``--n-workers``.

    Returns:
        Recommended worker count (≥ 1).

    Examples::

        # 16 GB machine, 13 tasks → avail≈10 GB, usable≈4 GB → ram_cap=1
        # but cpu_count=16 → min(13,16,1) = 1  (safe on 16 GB)
        # 64 GB machine, 13 tasks → avail≈58 GB, usable≈52 GB → ram_cap=20
        # cpu_count=8 → min(13,8,20) = 8  (uses all 8 CPUs on g6e.2xlarge)
    """
    if override is not None:
        n = max(1, min(n_tasks, override))
        _LOG.info("[resource] PCAP workers: override=%d → %d", override, n)
        return n

    n_cpu = os.cpu_count() or 1
    avail_gb = _available_ram_gb()
    usable_gb = max(0.0, avail_gb - reserve_gb)
    ram_cap = max(1, int(usable_gb / per_pcap_gb))
    n = min(n_tasks, n_cpu, ram_cap)
    _LOG.info(
        "[resource] PCAP workers: avail_ram=%.1f GB, usable=%.1f GB, "
        "per_worker=%.1f GB → ram_cap=%d | cpu_count=%d | tasks=%d → chosen=%d",
        avail_gb, usable_gb, per_pcap_gb, ram_cap, n_cpu, n_tasks, n,
    )
    return n


def auto_compute_workers(n_tasks: int, override: int | None = None) -> int:
    """Recommend worker count for CPU-compute stages (packet_x, evidence pool).

    These workers compute features from already-loaded data so their per-worker
    RAM footprint is small and bounded.  Use all logical CPUs, capped at
    ``n_tasks``.

    Args:
        n_tasks:  Number of parallel work units (chunks or batches).
        override: If set, skip auto-detection.  Supplied by ``--n-compute-workers``.

    Returns:
        Recommended worker count (≥ 1).
    """
    n_cpu = os.cpu_count() or 1
    if override is not None:
        n = max(1, min(n_tasks, override))
        _LOG.info("[resource] compute workers: override=%d → %d", override, n)
        return n
    n = min(n_tasks, n_cpu)
    _LOG.info(
        "[resource] compute workers: cpu_count=%d, tasks=%d → chosen=%d",
        n_cpu, n_tasks, n,
    )
    return n
