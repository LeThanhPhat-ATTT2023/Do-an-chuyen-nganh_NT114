#!/usr/bin/env python3
"""Sweep alternative anchor definitions for EACS against the clean answer key.

For every suspect-class flow, compute per-(flow, matching-family):
  * n_ev    — number of evidence packets
  * frac_ev — fraction of the flow's packets carrying evidence
  * sum_w / max_w — aggregated evidence weight
then report anchor precision/recall for threshold grids on each statistic.
Goal: find a cheap aggregation that anchors the ~1k real attacks while
excluding the ~5.4k background flows the current (sum_w > 0) rule anchors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graphslm_ids.offline.training.noise_consensus import (  # noqa: E402
    _FAMILY_ORDER,
    class_to_family_from_csvs,
)

SUSPECT_CLASSES = ["CommandInjection", "XSS", "SqlInjection", "Uploading_Attack"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="outputs/v3_ob/graph.npz")
    ap.add_argument("--meta", default="outputs/v3_ob/graph.meta.json")
    ap.add_argument("--clean-labels", default="outputs/v3_ob/clean_eval_labels.npy")
    ap.add_argument("--mitre-dir", default="data/mitre")
    args = ap.parse_args()

    z = np.load(args.npz)
    meta = json.loads(Path(args.meta).read_text())
    label_mapping: dict[str, int] = meta["label_mapping"]
    flow_y = np.asarray(z["flow_y"], dtype=np.int64)
    clean = np.asarray(np.load(args.clean_labels), dtype=np.int64)
    benign_id = label_mapping["Benign"]
    num_flows = flow_y.shape[0]

    contains = np.asarray(z["contain_edge_index"], dtype=np.int64)
    pkt_to_flow = np.full(int(contains[1].max()) + 1, -1, dtype=np.int64)
    pkt_to_flow[contains[1]] = contains[0]
    pkts_per_flow = np.bincount(contains[0], minlength=num_flows).astype(np.float64)

    n_fam = len(_FAMILY_ORDER)
    n_ev = np.zeros((num_flows, n_fam))
    sum_w = np.zeros((num_flows, n_fam))
    max_w = np.zeros((num_flows, n_fam))
    for col, fam in enumerate(_FAMILY_ORDER):
        k = f"evidence_{fam}_edge_index"
        if k not in z.files:
            continue
        pkts = np.asarray(z[k], dtype=np.int64)[0]
        w = np.asarray(z[f"evidence_{fam}_edge_attr"], dtype=np.float64).reshape(len(pkts), -1)[:, 0]
        flows = pkt_to_flow[pkts]
        ok = flows >= 0
        flows, w = flows[ok], w[ok]
        # multiple evidence edges can share a packet; count unique evidence packets
        uniq_pairs = np.unique(np.stack([flows, pkts[ok]]), axis=1)
        n_ev[:, col] += np.bincount(uniq_pairs[0], minlength=num_flows)
        np.add.at(sum_w[:, col], flows, w)
        np.maximum.at(max_w[:, col], flows, w)

    class_to_family = class_to_family_from_csvs(args.mitre_dir, label_mapping, len(label_mapping)).numpy()
    sus_ids = sorted(label_mapping[c] for c in SUSPECT_CLASSES if c in label_mapping)
    in_cls = np.isin(flow_y, sus_ids)
    fam_of_flow = class_to_family[flow_y]
    rows = np.where(in_cls & (fam_of_flow >= 0))[0]
    famcol = fam_of_flow[rows]
    stats = {
        "n_ev": n_ev[rows, famcol],
        "frac_ev": n_ev[rows, famcol] / np.maximum(pkts_per_flow[rows], 1),
        "sum_w": sum_w[rows, famcol],
        "max_w": max_w[rows, famcol],
    }
    is_atk = clean[rows] == flow_y[rows]
    n_atk = int(is_atk.sum())
    print(f"web flows={len(rows)}  true attacks={n_atk}")
    for name, v in stats.items():
        print(f"\n--- anchor = {name} >= t ---")
        print(f"{'t':>8}{'anchors':>9}{'∩attack':>9}{'precision':>10}{'recall':>8}")
        qs = np.unique(np.quantile(v, [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]))
        grid = sorted(set(list(qs) + [1, 2, 3, 5, 8, 13, 21, 0.05, 0.1, 0.2, 0.5]))
        for t in grid:
            anc = v >= t
            na = int(anc.sum())
            if na == 0:
                continue
            inter = int((anc & is_atk).sum())
            print(f"{t:>8.3f}{na:>9}{inter:>9}{inter/na:>10.3f}{inter/n_atk:>8.3f}")
    # distribution summary attackers vs background
    print("\n--- distribution (median / p90) attack vs background ---")
    for name, v in stats.items():
        a, b = v[is_atk], v[~is_atk]
        print(f"{name:>8}: atk med={np.median(a):.3f} p90={np.quantile(a,0.9):.3f} | "
              f"bg med={np.median(b):.3f} p90={np.quantile(b,0.9):.3f}")


if __name__ == "__main__":
    main()
