"""Label-quality & separability audit for the NT114 IDS dataset.

The tabular ceiling test showed a strong model + payload content tops out at
macro-F1 ~0.47 (no class > 0.82, Benign only 0.48). That low ceiling usually
means LABELS or FLOW DEFINITION cap quality, not model capacity. This script
decides which, with three model-light analyses on the SAME flows/labels:

1. CONFUSION STRUCTURE  — where do a strong model's errors go?
   * errors clustered within a semantic group (web↔web, recon↔recon) => genuine
     content/behaviour overlap (fixable with richer features).
   * most classes bleeding into BENIGN => benign-background contamination
     (per-file labeling mislabels normal IoT chatter inside attack captures).

2. kNN LABEL PURITY     — model-free separability. For each flow, what fraction
   of its 10 nearest neighbours (in standardized rich+payload space) share its
   label? High purity => classes ARE separable (then the GNN, not the data, is
   the problem). Low purity => features can't tell classes apart / labels noisy.

3. PER-CLASS COMPOSITION — top dst ports, protocol mix, % zero-payload (pure
   control) flows, fan-out. Quantifies how much each attack class looks like
   infrastructure/benign background.

Reuses flow reconstruction + feature builders from tabular_baseline so flows and
labels are identical to the graph artifact. Read-only on interim data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graphslm_ids.offline.preprocessing.graph_csv_builder import build_graph_csv_tables  # noqa: E402
from tabular_baseline import (  # noqa: E402
    build_flow_features,
    build_payload_histograms,
    stratified_subsample,
)

# common infrastructure / benign-chatter destination ports (IoT background)
_INFRA_PORTS = {53, 67, 68, 123, 137, 138, 1900, 5353, 5355, 3702, 1883, 8883, 5683}


def composition_report(flow_nodes: pd.DataFrame, packet_nodes: pd.DataFrame,
                       label_map: dict[str, int]) -> dict:
    fn = flow_nodes.copy()
    fn["dst_port"] = pd.to_numeric(fn["dst_port"], errors="coerce").fillna(-1).astype(int)
    fn["total_payload_bytes"] = pd.to_numeric(fn["total_payload_bytes"], errors="coerce").fillna(0)
    out = {}
    print("\n=== PER-CLASS COMPOSITION ===")
    print(f"{'class':28s} {'flows':>9} {'%infra_port':>11} {'%zero_payld':>11} {'top_dst_ports'}")
    for cls in sorted(label_map):
        sub = fn[fn["label"].astype(str) == cls]
        n = len(sub)
        if n == 0:
            continue
        infra = float((sub["dst_port"].isin(_INFRA_PORTS)).mean())
        zero_pl = float((sub["total_payload_bytes"] == 0).mean())
        top_ports = Counter(sub["dst_port"].tolist()).most_common(4)
        top_str = ", ".join(f"{p}:{c*100//n}%" for p, c in top_ports)
        proto = Counter(sub["protocol"].astype(str).tolist()).most_common()
        out[cls] = {
            "flows": n, "pct_infra_port": round(infra, 4),
            "pct_zero_payload": round(zero_pl, 4),
            "top_dst_ports": top_ports, "protocol_mix": proto,
        }
        print(f"{cls:28s} {n:>9,} {infra*100:>10.1f}% {zero_pl*100:>10.1f}%  {top_str}")
    return out


def knn_purity(X: np.ndarray, y: np.ndarray, label_names: dict[int, str],
               sample: int, seed: int, k: int = 10) -> dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    if X.shape[0] > sample:
        idx = stratified_subsample(y, max(1, sample // len(np.unique(y))), seed)
        X, y = X[idx], y[idx]
    Xs = StandardScaler().fit_transform(np.nan_to_num(X.astype(np.float32)))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(Xs)
    _, ind = nn.kneighbors(Xs)
    neigh = ind[:, 1:]  # drop self
    same = (y[neigh] == y[:, None]).mean(axis=1)  # purity per sample
    out = {"overall_purity": round(float(same.mean()), 4), "per_class": {}}
    print(f"\n=== kNN LABEL PURITY (k={k}, n={X.shape[0]:,}) ===")
    print(f"  overall purity = {out['overall_purity']:.4f}   "
          f"(1.0 = perfectly separable, ~1/13={1/len(np.unique(y)):.3f} = random)")
    for c in sorted(np.unique(y)):
        p = float(same[y == c].mean())
        out["per_class"][label_names[c]] = round(p, 4)
    for cls, p in sorted(out["per_class"].items(), key=lambda kv: kv[1]):
        print(f"    {cls:28s} {p:.4f}")
    return out


def confusion_report(X: np.ndarray, y: np.ndarray, label_names: dict[int, str],
                     seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.class_weight import compute_sample_weight

    X = np.nan_to_num(X.astype(np.float32))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    sw = compute_sample_weight("balanced", ytr)
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.15,
                                         early_stopping=True, random_state=seed)
    clf.fit(Xtr, ytr, sample_weight=sw)
    pred = clf.predict(Xte)
    classes = sorted(label_names)
    cm = confusion_matrix(yte, pred, labels=classes)
    cm_row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # row-normalized (recall view)

    benign_id = next((i for i, n in label_names.items() if n.lower().startswith("benign")), None)
    print("\n=== CONFUSION (row-normalized; for each TRUE class, top predicted) ===")
    leak_to_benign = {}
    for i, c in enumerate(classes):
        row = cm_row[i]
        order = np.argsort(-row)[:3]
        tops = ", ".join(f"{label_names[classes[j]]}:{row[j]*100:.0f}%" for j in order)
        bleak = float(row[classes.index(benign_id)]) if benign_id is not None and benign_id != c else 0.0
        leak_to_benign[label_names[c]] = round(bleak, 4)
        print(f"  {label_names[c]:28s} -> {tops}")
    if benign_id is not None:
        avg_leak = np.mean([v for k, v in leak_to_benign.items()
                            if not k.lower().startswith("benign")])
        print(f"\n  avg fraction of NON-benign classes predicted as Benign = {avg_leak*100:.1f}%")
        print("  (high => benign-background contamination from per-file labeling)")
    return {"row_normalized": cm_row.round(4).tolist(),
            "classes": [label_names[c] for c in classes],
            "leak_to_benign": leak_to_benign}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-csv", default="data/interim/payload_dataset_14gb/metadata.csv")
    ap.add_argument("--payload-npy", default="data/interim/payload_dataset_14gb/payload_256.npy")
    ap.add_argument("--out-json", default="outputs/diagnostics/label_audit_results.json")
    ap.add_argument("--max-per-class", type=int, default=8000)
    ap.add_argument("--knn-sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/5] Loading metadata …", flush=True)
    md = pd.read_csv(args.metadata_csv, dtype={
        "pcap_file": "category", "label": "category", "src_ip": "category",
        "dst_ip": "category", "protocol": "category", "packet_index": np.int32,
        "src_port": np.int32, "dst_port": np.int32, "payload_len_raw": np.int32})

    print("[2/5] Reconstructing flows …", flush=True)
    tables = build_graph_csv_tables(md, flow_timeout_seconds=30.0, max_packets_per_flow=20)
    del md
    flow_nodes = tables.flow_nodes.reset_index(drop=True)
    packet_nodes = tables.packet_nodes.reset_index(drop=True)
    print(f"      flows={len(flow_nodes):,}", flush=True)

    label_values = sorted({str(v) for v in flow_nodes["label"].tolist()})
    label_map = {v: i for i, v in enumerate(label_values)}
    label_names = {i: v for v, i in label_map.items()}
    protocol_map = {v: i for i, v in enumerate(sorted({str(v) for v in flow_nodes["protocol"].tolist()}))}
    y_all = flow_nodes["label"].astype(str).map(label_map).to_numpy(np.int64)

    comp = composition_report(flow_nodes, packet_nodes, label_map)

    print("\n[3/5] Engineering features …", flush=True)
    _, rich_df, _ = build_flow_features(flow_nodes, packet_nodes, protocol_map)
    sel = stratified_subsample(y_all, args.max_per_class, args.seed)
    y = y_all[sel]
    flow_ids_sample = flow_nodes["flow_id"].to_numpy()[sel]
    rich_X = rich_df.to_numpy(np.float32)[sel]
    pay_X, _ = build_payload_histograms(flow_ids_sample, packet_nodes, Path(args.payload_npy))
    X = np.concatenate([rich_X, pay_X], axis=1)
    print(f"      audit matrix: {X.shape}", flush=True)

    print("\n[4/5] kNN purity …", flush=True)
    knn = knn_purity(X, y, label_names, args.knn_sample, args.seed)

    print("\n[5/5] Confusion structure …", flush=True)
    conf = confusion_report(X, y, label_names, args.seed)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "composition": comp, "knn_purity": knn, "confusion": conf,
        "label_names": label_names, "n_flows_total": int(y_all.shape[0]),
    }, indent=2))
    print(f"\n[OK] wrote {out}   (wall {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
