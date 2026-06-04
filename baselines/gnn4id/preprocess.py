#!/usr/bin/env python3
"""GNN4ID preprocessing: raw PCAPs → graph shards + manifest.

RAM is bounded by a byte budget (``--mem-budget-mb``) + streaming build; DISK is
bounded by subsampling (``--max-flows-per-class``), uint8 packet features, and
deleting each large CSV right after its shards are built. The manifest is written
incrementally so a crashed/OOM/disk-full run resumes (already-built pcaps skipped).

Usage:
    python baselines/gnn4id/preprocess.py \
        --raw-root            data/raw/14gb \
        --out                 baselines/gnn4id/outputs/graphs.manifest.json \
        --csv-dir             baselines/gnn4id/outputs/csv \
        --mem-budget-mb       2000 \
        --max-flows-per-class 30000   # workers auto-scale to CPU count by default
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

# Resolve imports whether run from repo root or baselines/gnn4id/
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from utils.functions import ShardWriter

_LOG = logging.getLogger("gnn4id.preprocess")


def _process_pcap(
    pcap_str: str, out_csv_str: str, label: str, max_pkts: int, n_meters: int
) -> tuple[str, str] | None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.feature_extractor import extract_pcap_to_csv
    from utils.additional_features import additional_features

    out_csv = Path(out_csv_str)
    if not out_csv.exists():
        extract_pcap_to_csv(pcap_str, out_csv_str, label=label, max_pkts=max_pkts, n_meters=n_meters)
    result = additional_features(out_csv_str)
    if result == "":
        return None
    return (out_csv_str, label)


def _can_admit(
    inflight_bytes: int, inflight_count: int, task_bytes: int,
    budget_bytes: int, max_workers: int,
) -> bool:
    """Size-aware admission test for the extraction scheduler.

    Peak RAM during extraction scales with the *total bytes of PCAPs processed
    concurrently*, so we cap concurrency by a byte budget rather than only a
    worker count. Rules:
      * never exceed ``max_workers`` concurrent tasks;
      * always allow at least one task (else a PCAP larger than the budget could
        never run);
      * otherwise admit only while in-flight bytes stay within ``budget_bytes``.
    """
    if inflight_count >= max_workers:
        return False
    if inflight_count == 0:
        return True
    return inflight_bytes + task_bytes <= budget_bytes


def _run_extraction_scheduled(
    tasks: list[tuple[str, str, str, int]],
    max_workers: int,
    budget_bytes: int,
    max_pkts: int,
    n_meters: int,
    on_done,
) -> int:
    """Extract PCAPs concurrently under a byte budget, invoking ``on_done(csv, label)``
    in the main process as each PCAP finishes (used to build shards + delete the CSV).

    Big PCAPs are started first (descending size) so the slow tail overlaps with the
    small ones, and admission is gated by :func:`_can_admit`. Returns #PCAPs done.
    """
    pending = sorted(tasks, key=lambda t: t[3], reverse=True)
    n_done = 0
    running: dict = {}  # future -> (name, size_bytes)
    inflight_bytes = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        while pending or running:
            # Admit as many pending tasks as the budget/worker cap allows.
            progressed = True
            while progressed and pending:
                progressed = False
                for idx in range(len(pending)):
                    pcap_str, out_csv_str, label, size = pending[idx]
                    if _can_admit(inflight_bytes, len(running), size, budget_bytes, max_workers):
                        fut = pool.submit(_process_pcap, pcap_str, out_csv_str, label, max_pkts, n_meters)
                        running[fut] = (Path(pcap_str).name, size)
                        inflight_bytes += size
                        pending.pop(idx)
                        progressed = True
                        break  # state changed → rescan from the top
            if not running:
                break  # nothing admitted and nothing running (defensive)
            _LOG.info(
                "  in-flight: %d task(s), %.0f MB / %.0f MB budget",
                len(running), inflight_bytes / 1e6, budget_bytes / 1e6,
            )
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                name, size = running.pop(fut)
                inflight_bytes -= size
                try:
                    result = fut.result()
                except Exception as exc:
                    _LOG.error("  %s failed: %s", name, exc)
                    continue
                if result is None:
                    _LOG.warning("  additional_features failed for %s, skipping", name)
                else:
                    csv_path, label = result
                    _LOG.info("  extracted: %s", name)
                    on_done(csv_path, label)
                    n_done += 1
    return n_done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw/14gb")
    ap.add_argument("--out", default="baselines/gnn4id/outputs/graphs.manifest.json",
                    help="Manifest JSON; graph shards go in a sibling <stem>_shards/ dir. "
                         "A legacy graphs.pt is still loadable by train.py.")
    ap.add_argument("--csv-dir", default="baselines/gnn4id/outputs/csv")
    ap.add_argument("--max-packets-per-flow", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0,
                    help="Max concurrent PCAP workers. 0 = auto (= CPU count). Peak "
                         "RAM is bounded by --mem-budget-mb regardless, so this is "
                         "just a concurrency ceiling — auto-scaling is safe.")
    ap.add_argument("--mem-budget-mb", type=int, default=2000,
                    help="Total size of PCAPs processed concurrently (MB). Caps "
                         "extraction peak RAM; a PCAP larger than this still runs alone.")
    ap.add_argument("--nfstream-meters", type=int, default=1,
                    help="nfstream metering processes PER pcap (default 1). 0=auto "
                         "(≈ all cores) — the setting that previously exploded RAM.")
    ap.add_argument("--shard-max-graphs", type=int, default=50000,
                    help="Flush a graph shard to disk every N graphs to bound build RAM.")
    ap.add_argument("--max-flows-per-class", type=int, default=30000,
                    help="Subsample at most N flows per class (uniform random, seeded). "
                         "0 = keep all (warning: float-free uint8 graphs are still large).")
    ap.add_argument("--keep-csv", action="store_true",
                    help="Keep intermediate CSVs. Default: delete each CSV right after "
                         "its shards are built, so disk never holds all CSVs at once.")
    ap.add_argument("--seed", type=int, default=42, help="Subsampling RNG seed.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    raw_root = Path(args.raw_root)
    csv_dir = Path(args.csv_dir)
    out_path = Path(args.out)
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover class folders → label mapping ──────────────────────────
    class_dirs = sorted(p for p in raw_root.iterdir() if p.is_dir())
    if not class_dirs:
        _LOG.error("No class subdirectories found under %s", raw_root)
        sys.exit(1)
    label_mapping: dict[str, int] = {d.name: i for i, d in enumerate(class_dirs)}
    _LOG.info("Classes (%d): %s", len(label_mapping), list(label_mapping))

    # ── 2. Build task list (with PCAP sizes for the byte-budget scheduler) ──
    tasks: list[tuple[str, str, str, int]] = []  # (pcap, out_csv, label, size_bytes)
    for cls_dir in class_dirs:
        label = cls_dir.name
        pcaps = sorted(cls_dir.glob("*.pcap"))
        _LOG.info("  [%s] %d pcap(s)", label, len(pcaps))
        for pcap in pcaps:
            out_csv = csv_dir / f"{label}__{pcap.stem}.csv"
            tasks.append((str(pcap), str(out_csv), label, pcap.stat().st_size))

    # ── 3-5. Extract → build shards → delete CSV, interleaved per pcap ─────
    # Interleaving bounds BOTH RAM (byte budget + streaming build) and DISK (each
    # large CSV is removed right after its compact uint8 shards are written).
    max_pkts = args.max_packets_per_flow
    budget_bytes = args.mem_budget_mb * 1_000_000
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    cap = args.max_flows_per_class if args.max_flows_per_class > 0 else None

    writer = ShardWriter(
        manifest_path=str(out_path),
        label_mapping=label_mapping,
        max_graphs_per_shard=args.shard_max_graphs,
        max_flows_per_class=cap,
        seed=args.seed,
    )

    # Resume: drop pcaps whose shards are already recorded in the manifest.
    todo = [t for t in tasks if not writer.already_built(t[1], t[2])]
    if len(todo) < len(tasks):
        _LOG.info("Resume: %d/%d pcap(s) already built, skipping", len(tasks) - len(todo), len(tasks))

    def _on_done(csv_path: str, label: str) -> None:
        # Build this pcap's shards first; only delete the CSV on success so a
        # failed build (e.g. disk full) leaves the CSV for a clean re-run.
        added = writer.add_csv(csv_path, label)
        _LOG.info("  built %s (+%d graphs)", Path(csv_path).name, added)
        if not args.keep_csv:
            try:
                os.remove(csv_path)
            except OSError as exc:
                _LOG.warning("  could not delete CSV %s: %s", csv_path, exc)

    _LOG.info(
        "Extract+build %d pcap(s): workers≤%d (RAM cap=--mem-budget-mb=%d MB), "
        "meters=%d/pcap, cap=%s flow/class, delete-csv=%s",
        len(todo), workers, args.mem_budget_mb, args.nfstream_meters,
        cap if cap is not None else "none", not args.keep_csv,
    )
    _run_extraction_scheduled(
        todo, max_workers=workers, budget_bytes=budget_bytes,
        max_pkts=max_pkts, n_meters=args.nfstream_meters, on_done=_on_done,
    )

    manifest = writer.finalize()
    _LOG.info("Built %d graph(s) across %d shard(s)", manifest["num_graphs"], len(manifest["shards"]))
    _LOG.info("Per-class counts: %s", manifest["per_class"])
    _LOG.info("Saved manifest → %s", out_path)


if __name__ == "__main__":
    main()
