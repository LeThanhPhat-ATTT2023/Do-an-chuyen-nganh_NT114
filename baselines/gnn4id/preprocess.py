#!/usr/bin/env python3
"""GNN4ID preprocessing: raw PCAPs → outputs/graphs.pt.

Usage:
    python baselines/gnn4id/preprocess.py \
        --raw-root data/raw/14gb \
        --out      baselines/gnn4id/outputs/graphs.pt \
        --csv-dir  baselines/gnn4id/outputs/csv
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import torch

# Resolve imports whether run from repo root or baselines/gnn4id/
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from utils.feature_extractor import extract_pcap_to_csv
from utils.additional_features import additional_features
from utils.functions import NIDSDataset

_LOG = logging.getLogger("gnn4id.preprocess")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw/14gb")
    ap.add_argument("--out", default="baselines/gnn4id/outputs/graphs.pt")
    ap.add_argument("--csv-dir", default="baselines/gnn4id/outputs/csv")
    ap.add_argument("--max-packets-per-flow", type=int, default=20)
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

    # ── 2. PCAP → CSV (nfstream) ───────────────────────────────────────────
    csv_files: list[tuple[str, str]] = []
    for cls_dir in class_dirs:
        label = cls_dir.name
        pcaps = sorted(cls_dir.glob("*.pcap"))
        _LOG.info("  [%s] %d pcap(s)", label, len(pcaps))
        for pcap in pcaps:
            out_csv = csv_dir / f"{label}__{pcap.stem}.csv"
            if not out_csv.exists():
                _LOG.info("    extracting %s ...", pcap.name)
                extract_pcap_to_csv(
                    str(pcap), str(out_csv), label=label,
                    max_pkts=args.max_packets_per_flow,
                )
            else:
                _LOG.debug("    skip (exists): %s", out_csv.name)
            # 3. Additional features (rolling-window, overwrites CSV)
            result = additional_features(str(out_csv))
            if result == "":
                _LOG.warning("    additional_features failed for %s, skipping", out_csv.name)
                continue
            csv_files.append((str(out_csv), label))

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
