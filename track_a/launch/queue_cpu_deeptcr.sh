#!/usr/bin/env bash
# CPU queue for DeepTCR. Runs in track_a_deeptcr env (TF 2.12 CPU).
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/cpu_deeptcr.log
exec > >(tee -a "$LOG") 2>&1
echo "[cpu/deeptcr] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate track_a_deeptcr
export PYTHONPATH=/home/ubuntu/quest

for task in as_trb_i as_paired_i as_trb_ii as_paired_ii; do
  echo "[cpu/deeptcr] running deeptcr / $task at $(date -u +%FT%TZ)"
  WORKING_DIR="/home/ubuntu/quest/track_a/deeptcr/.dt_state/${task}"
  mkdir -p "$WORKING_DIR"
  python -m track_a.deeptcr.run --task "$task" \
    --baseline_name deeptcr --epochs_min 10 \
    --max_test_rows 2000 --working_dir "$WORKING_DIR" \
    --results_root /home/ubuntu/quest/results || { echo "[cpu/deeptcr] FAILED $task"; break; }
done
echo "[cpu/deeptcr] DONE at $(date -u +%FT%TZ)"
