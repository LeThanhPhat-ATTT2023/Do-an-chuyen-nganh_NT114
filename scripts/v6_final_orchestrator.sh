#!/bin/bash
# Autonomous FINAL pipeline -- single-model >=0.89 attempt vs GNN4ID (0.8537).
#   wait for ordered-byte rebuild (tmux e3build) -> train focal + ordered-byte -> honest eval.
# ORIGINAL labels (graph rebuilt with --no-attack-isolation), 18 classes, 50ep. Unattended;
# survives disconnect (run inside tmux 'final').
cd ~/Do-an-chuyen-nganh_NT114 || exit 1
export PYTHONPATH=src
PY=/home/ubuntu/venv/bin/python

echo "=== [orch] waiting for ordered-byte rebuild (tmux e3build) ... $(date) ==="
while tmux has-session -t e3build 2>/dev/null; do sleep 60; done
if [ ! -f outputs/v3_ob/graph.npz ]; then
  echo "=== [orch] REBUILD FAILED (no outputs/v3_ob/graph.npz) ==="
  tail -n 60 outputs/v3_ob/rebuild.log
  exit 1
fi
echo "=== [orch] rebuild done; ordered-byte graph ready. $(date) ==="
ls -la outputs/v3_ob/graph.npz outputs/v3_ob/splits.json

echo "=================================================================="
echo "=== [orch] FINAL train: focal + ordered-byte (orig labels, 50ep) $(date) ==="
echo "=================================================================="
$PY -m graphslm_ids.offline.training.train_hgt_flow_classifier \
    --config configs/eg_hgt_v6_ob_focal.yaml --device cuda 2>&1 | tee outputs/v3_ob_focal_run.log
echo "=== [orch] FINAL train exit $?  $(date) ==="

echo "=== [orch] honest eval: raw macro-F1 + VAL-tuned tau sweep (no test-peek) ==="
$PY scripts/diagnostics/v4_confusion_analysis.py \
    --config configs/eg_hgt_v6_ob_focal.yaml \
    --checkpoint outputs/v3_ob_focal/hgt_flow_best.pt \
    --training-summary outputs/v3_ob_focal/training_summary.json \
    --out outputs/v3_ob_focal/confusion.json \
    --la-taus='-1.0,-0.5,-0.25,0.5' 2>&1 | tee -a outputs/v3_ob_focal_run.log
echo "=================================================================="
echo "=== [orch] FINAL DONE  $(date) ==="
echo "[orch] headline macro_f1: outputs/v3_ob_focal/training_summary.json  (target >=0.89, beat GNN4ID 0.8537)"
echo "[orch] per-class + tau:   outputs/v3_ob_focal/confusion.json / _la_sweep.json"
echo "=================================================================="
