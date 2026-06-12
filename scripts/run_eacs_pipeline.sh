#!/bin/bash
# EACS full pipeline on a fresh GPU box (e.g. metis):
#   [1] rebuild the ordered-byte ORIGINAL-label graph (--no-attack-isolation)
#   [2] artifact sanity audit
#   [3] extract the eval-only clean answer key (LNL protocol grading key)
#   [4] 50-epoch EACS training (configs/eg_hgt_v6_ob_eacs.yaml)
#
# Run inside tmux:  tmux new -s eacs 'bash scripts/run_eacs_pipeline.sh'
# Requires: ~/venv with torch+cuda, data/raw -> pcap root (18 class dirs),
#           data/mitre -> MITRE CSVs + STIX json.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/venv/bin/activate
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

RAW_ROOT="${RAW_ROOT:-data/raw}"
OUT="${OUT:-outputs/v3_ob}"
mkdir -p "$OUT"

if [ ! -f "$OUT/graph.npz" ]; then
  echo "=== [1/4] graph rebuild (ORIGINAL labels, ordered-byte) $(date) ==="
  python -m graphslm_ids.offline.preprocessing.cli \
    --raw-root "$RAW_ROOT" \
    --out-npz "$OUT/graph.npz" \
    --out-meta "$OUT/graph.meta.json" \
    --pmi-table "$OUT/pmi_table.parquet" \
    --pmi-meta "$OUT/pmi.meta.json" \
    --splits-json "$OUT/splits.json" \
    --no-attack-isolation \
    --pmi-family-tau 0.6 \
    --max-per-class 50000 \
    --pmi-subsample-per-class 25000 \
    --temporal-train-frac 0.80 \
    --temporal-val-frac 0.10 2>&1 | tee "$OUT/rebuild.log"
else
  echo "=== [1/4] SKIP rebuild — $OUT/graph.npz exists $(date) ==="
fi

echo "=== [2/4] artifact audit $(date) ==="
python scripts/diagnostics/v3_artifact_audit.py "$OUT/graph.npz" 2>&1 | tee "$OUT/audit.log"

echo "=== [3/4] clean answer key (eval-only) $(date) ==="
python scripts/tools/extract_clean_eval_labels.py \
  --graph-meta "$OUT/graph.meta.json" \
  --raw-root "$RAW_ROOT" \
  --out-npy "$OUT/clean_eval_labels.npy" \
  --out-audit "$OUT/clean_eval_labels.audit.json"

echo "=== [4/4] EACS 50ep training $(date) ==="
python -m graphslm_ids.offline.training.train_hgt_flow_classifier \
  --config configs/eg_hgt_v6_ob_eacs.yaml --device cuda 2>&1 | tee outputs/eacs_full.log
echo "=== PIPELINE DONE $(date) ==="
