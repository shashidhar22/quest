#!/usr/bin/env bash
# Polls every 60s for resume_gpu0 / resume_gpu1 to exit; when each does, launches
# the ESM-C-cpt overflow queue on that GPU. Uses marker files so it won't
# double-launch even if interrupted/restarted. Self-terminates after both
# overflows are launched.
set -uo pipefail
LOG=/home/ubuntu/quest/track_a/launch/supervisor.log
MARKER_DIR=/home/ubuntu/quest/track_a/launch/.supervisor
mkdir -p "$MARKER_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "[supervisor] started at $(date -u +%FT%TZ) PID=$$"

# Returns 0 if `bash <script>` is currently running (matches by absolute path token)
is_alive() {
  local script="$1"
  pgrep -af "bash.*${script}" 2>/dev/null | grep -v grep | grep -v supervisor.sh > /dev/null
}

launch_overflow() {
  local script_basename="$1"
  local marker="$MARKER_DIR/${script_basename}.launched"
  if [[ -f "$marker" ]]; then
    return 0
  fi
  echo "[supervisor] launching $script_basename at $(date -u +%FT%TZ)"
  cd /home/ubuntu/quest
  nohup setsid bash "/home/ubuntu/quest/track_a/launch/${script_basename}" \
    </dev/null >/dev/null 2>&1 &
  local pid=$!
  disown $pid 2>/dev/null || true
  echo "[supervisor]   $script_basename PID=$pid"
  echo "$pid $(date -u +%FT%TZ)" > "$marker"
}

while true; do
  # GPU 0: when resume_gpu0.sh exits, launch esmc_overflow_gpu0.sh
  if ! is_alive "resume_gpu0.sh"; then
    launch_overflow "esmc_overflow_gpu0.sh"
  fi
  # GPU 1: when resume_gpu1.sh exits, launch esmc_overflow_gpu1.sh
  if ! is_alive "resume_gpu1.sh"; then
    launch_overflow "esmc_overflow_gpu1.sh"
  fi
  # Done when both overflows have been launched
  if [[ -f "$MARKER_DIR/esmc_overflow_gpu0.sh.launched" \
     && -f "$MARKER_DIR/esmc_overflow_gpu1.sh.launched" ]]; then
    echo "[supervisor] both overflows launched; supervisor exiting at $(date -u +%FT%TZ)"
    break
  fi
  sleep 60
done
