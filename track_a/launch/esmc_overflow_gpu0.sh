#!/usr/bin/env bash
# Overflow queue: runs ESM-C 300M cpt cells on GPU 0 once esm2_8m queue completes.
# Idempotent — resume_run.sh's summary-file check makes overlapping with
# resume_gpu2/resume_gpu3 safe (they'll skip cells already done here).
# Pairs trb_i + trb_ii on this GPU; paired_i + paired_ii go to GPU 1 overflow.
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/esmc_overflow_gpu0.log
exec > >(tee -a "$LOG") 2>&1
echo "[overflow/gpu0] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

for task in as_trb_i as_trb_ii; do
  echo "[overflow/gpu0] running esmc_300m-cpt / $task"
  bash track_a/launch/resume_run.sh esmc_300m-cpt "$task" 0 2000 || \
    echo "[overflow/gpu0] FAILED esmc_300m-cpt/$task; continuing"
done
echo "[overflow/gpu0] DONE at $(date -u +%FT%TZ)"
