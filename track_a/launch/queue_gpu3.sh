#!/usr/bin/env bash
# GPU 3: TCR-BERT × 4 main tasks. Frees up after ~6h, used for ESM-C 300M cells next.
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/gpu3.log
exec > >(tee -a "$LOG") 2>&1
echo "[gpu3] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

for task in as_trb_i as_paired_i as_trb_ii as_paired_ii; do
  echo "[gpu3] running tcrbert / $task at $(date -u +%FT%TZ)"
  bash track_a/launch/run_baseline.sh tcrbert "$task" 3 || { echo "[gpu3] FAILED tcrbert/$task"; break; }
done
echo "[gpu3] tcrbert done. queueing ESM-C 300M cells at $(date -u +%FT%TZ)"

for cell in esmc_300m-base esmc_300m-cpt; do
  for task in as_trb_i as_paired_i as_trb_ii as_paired_ii; do
    echo "[gpu3] running $cell / $task at $(date -u +%FT%TZ)"
    bash track_a/launch/run_baseline.sh "$cell" "$task" 3 || { echo "[gpu3] FAILED $cell/$task"; break 2; }
  done
done
echo "[gpu3] all cells complete at $(date -u +%FT%TZ)"
