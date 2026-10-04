#!/usr/bin/env bash
# Re-run TULIP-original on GPU at 2,000 test rows per condition (parity with
# every other baseline). The original CPU runs were capped at 200 rows because
# all 4 GPUs were saturated at the time; that cap left us with only 1–29 kept
# queries per condition after gold-in-pool filtering, which is statistically
# unreliable for the normalized-recall comparison.
#
# Sharing GPU $TULIP_GPU (default 0) with whatever else is on it.
# track_a_tulip env has torch 2.1.2+cu121, sm_89-compatible.
set -e
GPU="${TULIP_GPU:-0}"
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/tulip_redo_gpu.log
exec > >(tee -a "$LOG") 2>&1
echo "[tulip-redo] started on GPU $GPU at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate track_a_tulip
export PYTHONPATH=/home/ubuntu/quest
export CUDA_VISIBLE_DEVICES="$GPU"

for task in as_trb_i as_paired_i; do
  echo "[tulip-redo] running tulip-original / $task at $(date -u +%FT%TZ)"
  python -m track_a.tulip.score \
    --task "$task" \
    --baseline_name tulip-original \
    --device cuda \
    --max_test_rows 2000 \
    --results_root /home/ubuntu/quest/results || \
    { echo "[tulip-redo] FAILED $task"; break; }
done

# Class II auto-skipped by tulip/score.py (TULIP not pretrained for class II MHC)
for task in as_trb_ii as_paired_ii; do
  python -m track_a.tulip.score \
    --task "$task" \
    --baseline_name tulip-original \
    --device cuda \
    --results_root /home/ubuntu/quest/results || true
done

echo "[tulip-redo] DONE at $(date -u +%FT%TZ)"
