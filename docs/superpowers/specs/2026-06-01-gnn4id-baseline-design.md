# GNN4ID Baseline Adaptation — Design Spec

**Date:** 2026-06-01
**Purpose:** Adapt GNN4ID (Farrukh et al., 2025) as a comparable baseline for EG-HGT v3. Train on the same CIC-IoT-2023 13-class PCAP dataset, evaluate with random split, report macro F1 for apples-to-apples comparison.

---

## Scope

- Clone GNN4ID source into `baselines/gnn4id/` (minimal fork — only necessary changes).
- Convert Jupyter notebooks → Python scripts for headless server execution.
- Adapt for 13 classes (dynamic label mapping from folder names, not hardcoded).
- Output `results.json` with macro F1 + per-class F1 in same format as `v3_eval_both_splits.py`.
- **Out of scope:** temporal split, MITRE ATT&CK, knowledge graph, any changes to GNN4ID's core feature extraction or model architecture logic.

---

## Repository Layout

```
baselines/
└── gnn4id/
    ├── requirements.txt              # torch-geometric, scapy, dpkt, scikit-learn
    ├── preprocess.py                 # Stage 1: PCAP → PyG HeteroData graphs
    ├── train.py                      # Stage 2: train + eval, writes results.json
    ├── model.py                      # HeteroGNN / HeteroGNN_Edge, output_dim=13
    ├── utils/
    │   ├── feature_extractor.py      # GNN4ID's Feature_extractor_flow_packet_combined.py
    │   ├── functions.py              # GNN4ID's Functions.py
    │   └── additional_features.py   # GNN4ID's Additional_Features.py
    └── outputs/                      # gitignored
        ├── graphs.pt                 # serialised graph list
        ├── checkpoint.pt             # best model weights
        └── results.json              # final metrics
```

---

## Stage 1 — Preprocessing (`preprocess.py`)

**Input:** `--raw-root data/raw/14gb` (same root as EG-HGT v3 pipeline).

**Label mapping:** Dynamic — iterate `raw_root/*/` subdirectories; `label = folder_name`. No hardcoded class list. Supports any number of classes.

**Graph construction (GNN4ID's original logic, unchanged):**
- Per flow: extract 82 CICFlowMeter-style flow features → 1 flow node.
- Per flow: take up to 20 packets → extract 1500-dim byte-wise payload features → up to 20 packet nodes.
- Edges:
  - `contains`: flow → each of its packet nodes.
  - `link` (next_packet): packet_i → packet_{i+1}, attr = [delta_t].
- Each flow becomes one `torch_geometric.data.HeteroData` object.
- Flow node carries its integer label (0..12).

**Output:** `outputs/graphs.pt` — Python list of `HeteroData` objects, one per flow.

**CLI:**
```bash
python baselines/gnn4id/preprocess.py \
  --raw-root data/raw/14gb \
  --out     baselines/gnn4id/outputs/graphs.pt \
  --max-packets-per-flow 20   # GNN4ID default, kept fixed
```

---

## Stage 2 — Training & Evaluation (`train.py`)

**Split:** Random stratified 70 / 15 / 15 (train / val / test), `seed=42`. Independent of `splits.json` (incompatible task formulations: graph-level vs node-level).

**DataLoader:** PyG `DataLoader`, batch size 64 (graph-level batching).

**Model:** `HeteroGNN_Edge` (GATConv variant with edge attributes) — the stronger of the two GNN4ID variants.

**Loss:** `CrossEntropyLoss` with balanced class weights, cap 10.0 (same cap as EG-HGT v3 to ensure fair comparison under class imbalance).

**Optimizer:** Adam, lr=0.01, weight_decay=1e-5 (GNN4ID's original hyperparameters — not tuned).

**Training:** 50 epochs, early stopping patience=10 on val macro F1. Best checkpoint saved to `outputs/checkpoint.pt`.

**Evaluation (on held-out test set):**
- Macro F1, weighted F1, accuracy.
- Per-class F1 for all 13 classes.

**Output `results.json`:**
```json
{
  "split": "random",
  "macro_f1": 0.0,
  "weighted_f1": 0.0,
  "accuracy": 0.0,
  "per_class_f1": {
    "BruteForce": 0.0,
    "DDoS": 0.0,
    "...": 0.0
  },
  "label_mapping": {"BruteForce": 0, "...": 1}
}
```

---

## Model (`model.py`)

Ported directly from GNN4ID `Utility/Model.py`. Changes:

| Location | Original | Adapted |
|---|---|---|
| `HeteroGNN_Edge.__init__` classification head | `Linear(128, 8)` final layer | `Linear(128, 13)` |
| `HeteroGNN.__init__` classification head | `Linear(128, 8)` | `Linear(128, 13)` |
| Class count references | hardcoded `8` | passed as `num_classes` constructor arg |

Everything else (SAGEConv/GATConv layers, BatchNorm, LeakyReLU, global mean pooling, concatenation across node types) is **unchanged**.

---

## What Is and Isn't Changed from GNN4ID

| Component | Status |
|---|---|
| Feature extraction logic (82 flow + 1500 packet features) | Unchanged |
| Graph schema (flow + packet nodes, contains + link edges) | Unchanged |
| GNN architecture (HeteroConv, GAT, 2 layers, hidden=64) | Unchanged |
| Hyperparameters (lr, weight_decay, batch_size) | Unchanged |
| Number of output classes | **Changed: 8 → 13 (dynamic)** |
| Label mapping | **Changed: dynamic from folder names** |
| Delivery format | **Changed: notebooks → Python scripts** |
| Class weight cap | **Added: cap=10.0 for fair comparison** |

---

## Comparison Table (target)

| Model | Graph schema | Knowledge graph | Split | Macro F1 |
|---|---|---|---|---|
| GNN4ID (HeteroGNN_Edge) | flow + packet nodes | No | Random | TBD |
| EG-HGT v3 (ours) | flow + packet + host + technique + tactic | Yes (MSEE) | Random | TBD |
| EG-HGT v3 (ours) | — | — | Temporal | TBD |

---

## Non-Goals

- No temporal split for GNN4ID (random only, as agreed).
- No tuning of GNN4ID hyperparameters.
- No integration with EG-HGT v3's preprocessing pipeline.
- No MITRE ATT&CK or evidence edges added to GNN4ID.
