#!/usr/bin/env bash
# Overflow queue: runs ESM-C 300M cpt cells on GPU 1 once esm2_35m queue completes.
# Pairs paired_i + paired_ii on this GPU; trb_i + trb_ii go to GPU 0 overflow.
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/esmc_overflow_gpu1.log
exec > >(tee -a "$LOG") 2>&1
echo "[overflow/gpu1] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

for task in as_paired_i as_paired_ii; do
  echo "[overflow/gpu1] running esmc_300m-cpt / $task"
  bash track_a/launch/resume_run.sh esmc_300m-cpt "$task" 1 2000 || \
    echo "[overflow/gpu1] FAILED esmc_300m-cpt/$task; continuing"
done
echo "[overflow/gpu1] DONE at $(date -u +%FT%TZ)"
