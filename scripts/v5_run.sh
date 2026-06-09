#!/bin/bash
# EG-HGT v5 unattended runner: train (recipe re-calibration) -> eval (val-tuned tau sweep).
# Launched inside tmux session 'v5' on the L40S so it survives client disconnect.
cd ~/Do-an-chuyen-nganh_NT114 || exit 1
export PYTHONPATH=src
PY=/home/ubuntu/venv/bin/python

echo "=================================================================="
echo "=== EG-HGT v5  TRAIN START  $(date) ==="
echo "=================================================================="
$PY -m graphslm_ids.offline.training.train_hgt_flow_classifier \
    --config configs/eg_hgt_v5.yaml --device cuda
echo "=== V5 TRAIN EXIT CODE $?  $(date) ==="

echo "=================================================================="
echo "=== EG-HGT v5  EVAL (confusion + val-tuned logit-adjust)  $(date) ==="
echo "=================================================================="
$PY scripts/diagnostics/v4_confusion_analysis.py \
    --config configs/eg_hgt_v5.yaml \
    --checkpoint outputs/v3/hgt_v5/hgt_flow_best.pt \
    --training-summary outputs/v3/hgt_v5/training_summary.json \
    --out outputs/v3/hgt_v5/v5_confusion.json \
    --la-taus='-1.5,-1.0,-0.75,-0.5,-0.25,0.5'
echo "=== V5 EVAL EXIT CODE $?  $(date) ==="

echo "=================================================================="
echo "=== EG-HGT v5  ALL DONE  $(date) ==="
echo "Results: outputs/v3/hgt_v5/training_summary.json"
echo "         outputs/v3/hgt_v5/v5_confusion.json"
echo "         outputs/v3/hgt_v5/v5_confusion_la_sweep.json"
echo "=================================================================="
