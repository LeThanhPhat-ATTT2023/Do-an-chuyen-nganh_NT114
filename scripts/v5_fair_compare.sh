#!/bin/bash
# Fairness 2x2 — fill the KEY missing cell: HGT v5-recipe on the ORIGINAL-label graph,
# at GNN4ID's exact budget (50 epochs / patience 10). Single GPU, so this WAITS for the
# v5 (relabeled) run to finish, then trains at full speed. Unattended; tmux session 'fair'.
#
# After this, the 2x2 is:
#   GNN4ID / original  = 0.854 headline (50ep) ; 0.853 on our distribution (50ep) — already have
#   HGT    / original  = THIS run (50ep)                      <- fair model comparison vs GNN4ID
#   HGT    / relabeled = eg_hgt_v5 (~0.94)                    <- relabel ablation vs the row above
cd ~/Do-an-chuyen-nganh_NT114 || exit 1
export PYTHONPATH=src
PY=/home/ubuntu/venv/bin/python

echo "=== FAIR-COMPARE: waiting for tmux 'v5' to finish (single GPU) ... $(date) ==="
while tmux has-session -t v5 2>/dev/null; do sleep 60; done
echo "=== v5 finished — GPU free. Starting HGT/original (50ep, GNN4ID-matched). $(date) ==="

echo "=================================================================="
echo "=== HGT v5-recipe on ORIGINAL-label v3 graph  (50 epochs)  $(date) ==="
echo "=================================================================="
$PY -m graphslm_ids.offline.training.train_hgt_flow_classifier \
    --config configs/eg_hgt_v5_origlabels.yaml --device cuda
echo "=== HGT/original TRAIN exit $?  $(date) ==="

$PY scripts/diagnostics/v4_confusion_analysis.py \
    --config configs/eg_hgt_v5_origlabels.yaml \
    --checkpoint outputs/v3/hgt_v5_origlabels/hgt_flow_best.pt \
    --training-summary outputs/v3/hgt_v5_origlabels/training_summary.json \
    --out outputs/v3/hgt_v5_origlabels/confusion.json \
    --la-taus='-1.0,-0.5,-0.25,0.5'
echo "=== HGT/original EVAL exit $?  $(date) ==="

echo "=================================================================="
echo "=== FAIR-COMPARE DONE  $(date) ==="
echo "Read the 2x2:"
echo "  HGT/original  : outputs/v3/hgt_v5_origlabels/training_summary.json (macro_f1)"
echo "  HGT/relabeled : outputs/v3/hgt_v5/training_summary.json (~0.94)"
echo "  GNN4ID/orig   : baselines/gnn4id/outputs/outputs_v1/results.json (0.854 headline)"
echo "                  baselines/gnn4id/outputs/results_imbalanced_v3dist.json (0.853 our-dist)"
echo "=================================================================="
