#!/usr/bin/env bash
# CPU queue for TULIP-original (zero-shot). Runs in track_a_tulip env (transformers 4.30).
# Class II not supported by upstream TULIP; we run TRB-I + Paired-I only.
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/cpu_tulip.log
exec > >(tee -a "$LOG") 2>&1
echo "[cpu/tulip] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate track_a_tulip
export PYTHONPATH=/home/ubuntu/quest

for task in as_trb_i as_paired_i; do
  echo "[cpu/tulip] running tulip-original / $task at $(date -u +%FT%TZ)"
  python -m track_a.tulip.score --task "$task" \
    --baseline_name tulip-original --device cpu \
    --max_test_rows 200 \
    --results_root /home/ubuntu/quest/results || { echo "[cpu/tulip] FAILED $task"; break; }
done
# Class II is skipped (TULIP-original was not pretrained on class II MHC); the
# wrapper auto-emits a 'skipped' record for class II if invoked.
for task in as_trb_ii as_paired_ii; do
  python -m track_a.tulip.score --task "$task" \
    --baseline_name tulip-original --device cpu \
    --results_root /home/ubuntu/quest/results || true
done
echo "[cpu/tulip] DONE at $(date -u +%FT%TZ)"
