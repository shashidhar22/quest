#!/usr/bin/env bash
# Polls every 60s for tulip_redo_gpu.sh to exit; when it does, re-runs
# compute_normalized_recall.py and writes results to track_a/normalized_recall.csv.
# Self-terminates after one successful re-run.
set -uo pipefail
LOG=/home/ubuntu/quest/track_a/launch/auto_recompute_norm_recall.log
exec > >(tee -a "$LOG") 2>&1
echo "[auto-recompute] started at $(date -u +%FT%TZ) PID=$$"

is_alive() {
  pgrep -af "bash.*track_a/launch/tulip_redo_gpu.sh" 2>/dev/null \
    | grep -v grep | grep -v auto_recompute > /dev/null
}

while is_alive; do
  sleep 60
done

echo "[auto-recompute] tulip_redo_gpu finished at $(date -u +%FT%TZ); recomputing..."
PYTHONPATH=/home/ubuntu/quest /home/ubuntu/miniforge3/bin/python3 \
  /home/ubuntu/quest/track_a/launch/compute_normalized_recall.py
RC=$?
echo "[auto-recompute] compute_normalized_recall exited rc=$RC at $(date -u +%FT%TZ)"
