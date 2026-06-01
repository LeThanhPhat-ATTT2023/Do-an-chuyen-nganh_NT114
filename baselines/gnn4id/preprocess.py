#!/usr/bin/env python3
"""GNN4ID preprocessing: raw PCAPs → outputs/graphs.pt.

Usage:
    python baselines/gnn4id/preprocess.py \
        --raw-root data/raw/14gb \
        --out      baselines/gnn4id/outputs/graphs.pt \
        --csv-dir  baselines/gnn4id/outputs/csv \
        --workers  4
"""
from __future__ import annotations
import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

# Resolve imports whether run from repo root or baselines/gnn4id/
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from utils.feature_extractor import extract_pcap_to_csv
from utils.additional_features import additional_features
from utils.functions import NIDSDataset

_LOG = logging.getLogger("gnn4id.preprocess")


def _process_pcap(pcap_str: str, out_csv_str: str, label: str, max_pkts: int) -> tuple[str, str] | None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.feature_extractor import extract_pcap_to_csv
    from utils.additional_features import additional_features

    out_csv = Path(out_csv_str)
    if not out_csv.exists():
        extract_pcap_to_csv(pcap_str, out_csv_str, label=label, max_pkts=max_pkts)
    result = additional_features(out_csv_str)
    if result == "":
        return None
    return (out_csv_str, label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw/14gb")
    ap.add_argument("--out", default="baselines/gnn4id/outputs/graphs.pt")
    ap.add_argument("--csv-dir", default="baselines/gnn4id/outputs/csv")
    ap.add_argument("--max-packets-per-flow", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel PCAP workers (default 1; set to #vCPU for speedup)")
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

    # ── 2. Build task list ─────────────────────────────────────────────────
    tasks: list[tuple[str, str, str]] = []  # (pcap, out_csv, label)
    for cls_dir in class_dirs:
        label = cls_dir.name
        pcaps = sorted(cls_dir.glob("*.pcap"))
        _LOG.info("  [%s] %d pcap(s)", label, len(pcaps))
        for pcap in pcaps:
            out_csv = csv_dir / f"{label}__{pcap.stem}.csv"
            tasks.append((str(pcap), str(out_csv), label))

    # ── 3. PCAP → CSV + additional_features (parallel) ────────────────────
    csv_files: list[tuple[str, str]] = []
    max_pkts = args.max_packets_per_flow

    if args.workers == 1:
        for pcap_str, out_csv_str, label in tasks:
            _LOG.info("  processing %s ...", Path(pcap_str).name)
            result = _process_pcap(pcap_str, out_csv_str, label, max_pkts)
            if result is None:
                _LOG.warning("  additional_features failed for %s, skipping", out_csv_str)
            else:
                csv_files.append(result)
    else:
        _LOG.info("Parallel extraction with %d workers ...", args.workers)
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for pcap_str, out_csv_str, label in tasks:
                f = pool.submit(_process_pcap, pcap_str, out_csv_str, label, max_pkts)
                futures[f] = Path(pcap_str).name
            for f in as_completed(futures):
                name = futures[f]
                try:
                    result = f.result()
                except Exception as exc:
                    _LOG.error("  %s failed: %s", name, exc)
                    continue
                if result is None:
                    _LOG.warning("  additional_features failed for %s, skipping", name)
                else:
                    _LOG.info("  done: %s", name)
                    csv_files.append(result)

    _LOG.info("Building PyG dataset from %d CSV files ...", len(csv_files))

    # ── 4. Build PyG graphs ────────────────────────────────────────────────
    dataset = NIDSDataset(csv_files=csv_files, label_mapping=label_mapping)
    _LOG.info("Built %d graphs", len(dataset))

    # ── 5. Save ────────────────────────────────────────────────────────────
    graphs = [dataset[i] for i in range(len(dataset))]
    torch.save({"graphs": graphs, "label_mapping": label_mapping}, str(out_path))
    _LOG.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
