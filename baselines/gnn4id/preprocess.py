#!/usr/bin/env python3
"""GNN4ID preprocessing: raw PCAPs → graph shards + manifest.

Extraction peak RAM is bounded by a byte budget (``--mem-budget-mb``), not just a
worker count, and graphs are streamed to disk shards instead of held in RAM — both
guard against the OOM that occurred when parallel nfstream workers each built a
full-CSV DataFrame at once.

Usage:
    python baselines/gnn4id/preprocess.py \
        --raw-root      data/raw/14gb \
        --out           baselines/gnn4id/outputs/graphs.manifest.json \
        --csv-dir       baselines/gnn4id/outputs/csv \
        --mem-budget-mb 2000        # workers auto-scale to CPU count by default
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

from utils.functions import write_graph_shards

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
) -> list[tuple[str, str]]:
    """Extract PCAPs concurrently under a byte budget; return [(csv, label), ...].

    Big PCAPs are started first (descending size) so the slow tail overlaps with
    the small ones, and admission is gated by :func:`_can_admit`.
    """
    pending = sorted(tasks, key=lambda t: t[3], reverse=True)
    csv_files: list[tuple[str, str]] = []
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
                    _LOG.info("  done: %s", name)
                    csv_files.append(result)
    return csv_files


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

    # ── 3. PCAP → CSV + additional_features (size-aware scheduler) ─────────
    max_pkts = args.max_packets_per_flow
    budget_bytes = args.mem_budget_mb * 1_000_000
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    _LOG.info(
        "Extracting %d pcap(s): workers≤%d (RAM cap = --mem-budget-mb), "
        "budget=%d MB, nfstream meters=%d/pcap",
        len(tasks), workers, args.mem_budget_mb, args.nfstream_meters,
    )
    csv_files = _run_extraction_scheduled(
        tasks, max_workers=workers, budget_bytes=budget_bytes,
        max_pkts=max_pkts, n_meters=args.nfstream_meters,
    )

    # ── 4-5. Stream graphs → on-disk shards + manifest (bounded build RAM) ──
    _LOG.info("Streaming PyG graphs from %d CSV file(s) → shards ...", len(csv_files))
    manifest = write_graph_shards(
        csv_files=csv_files,
        label_mapping=label_mapping,
        manifest_path=str(out_path),
        max_graphs_per_shard=args.shard_max_graphs,
    )
    _LOG.info("Built %d graph(s) across %d shard(s)", manifest["num_graphs"], len(manifest["shards"]))
    _LOG.info("Saved manifest → %s", out_path)


if __name__ == "__main__":
    main()
