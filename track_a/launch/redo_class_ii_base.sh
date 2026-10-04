#!/usr/bin/env bash
# Redo class-II cells whose summaries were archived after the 2026-05-04
# HF refresh added full as_paired_ii splits + the official 990-peptide
# class-II candidate pool. Runs in `base` env. Uses GPU passed as $1
# (default 3, which becomes free first).
set -e
GPU="${1:-3}"
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/redo_class_ii_base.log
exec > >(tee -a "$LOG") 2>&1
echo "[redo/base] started on GPU $GPU at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

# Order: as_trb_ii first (cheap, eval-only, checkpoint reused) then as_paired_ii (full redo)
for baseline in lstm mlp tcrbert; do
  for task in as_trb_ii as_paired_ii; do
    bash track_a/launch/resume_run.sh "$baseline" "$task" "$GPU" || \
      echo "[redo/base] $baseline/$task FAILED, continuing"
  done
done
echo "[redo/base] DONE at $(date -u +%FT%TZ)"
