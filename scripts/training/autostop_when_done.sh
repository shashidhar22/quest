#!/usr/bin/env bash
# autostop_when_done.sh — arm this on the training box, in its own tmux session, BEFORE walking away:
#
#   tmux new-session -d -s autostop \
#     "MAX_HOURS=18 bash scripts/training/autostop_when_done.sh <driver_pid> '<publish command>'"
#
# Stops the instance (shutdown -h; instanceInitiatedShutdownBehavior=stop keeps the volume) when ANY
# of three guards fires. They are independent on purpose — they do not share a failure mode:
#
#   1. DONE     the driver PID is gone (kill -0 fails). A PID, never a name pattern. On 2026-10-03 a
#               `pgrep -f "[r]un_stage41_pipeline"` watcher matched the tmux SERVER, whose argv was the
#               original `tmux new-session … run_stage41_pipeline.py` command line, so it never fired
#               and a g6.24xlarge idled for 14.5 h (~USD 100). The watcher's own tmux session kept that
#               server alive — a self-sustaining deadlock.
#   2. IDLE     no training process AND every GPU at 0 % for IDLE_MIN consecutive minutes (default
#               20). Catches a driver that finishes training but hangs. The process pattern is
#               anchored to a cmdline that STARTS with python, so wrappers (tmux, bash -c) whose argv
#               merely contains the script name cannot match.
#   3. CEILING  wall clock past MAX_HOURS (default 24). Catches everything else, including bugs in
#               this script's own detection. Set it to ~1.5x the expected run: a false stop costs a
#               resume, no ceiling costs an unbounded bill.
#
# Before stopping it runs the publish command (if given) so results reach S3 even when the driver
# crashed. Arm-time self-test refuses to arm on a dead PID, a missing nvidia-smi, or an absurd
# MAX_HOURS. Every decision is logged with the guard that made it. DRY_RUN=1 logs instead of
# publishing/stopping (used by tests/test_autostop_when_done.sh).
set -u
PID="${1:?usage: autostop_when_done.sh <driver_pid> [publish_cmd]}"
PUBLISH_CMD="${2:-}"
MAX_HOURS="${MAX_HOURS:-24}"
IDLE_MIN="${IDLE_MIN:-20}"
POLL="${POLL:-60}"
LOG="${LOG:-$HOME/autostop.log}"
NVSMI="${NVSMI:-nvidia-smi}"
DRY_RUN="${DRY_RUN:-0}"
S3_LOG="${S3_LOG:-}"          # optional s3:// key to copy the log to before stopping

say(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }

# ---- arm-time self-test: refuse rather than arm something that cannot work ----------------------
kill -0 "$PID" 2>/dev/null            || { say "REFUSE: pid $PID is not alive"; exit 2; }
command -v "$NVSMI" >/dev/null 2>&1   || { say "REFUSE: $NVSMI not found"; exit 2; }
awk "BEGIN{exit !($MAX_HOURS > 0 && $MAX_HOURS < 240)}" \
                                      || { say "REFUSE: MAX_HOURS=$MAX_HOURS (want 0 < h < 240)"; exit 2; }
max_s=$(awk "BEGIN{printf \"%d\", $MAX_HOURS*3600}")
idle_s=$((IDLE_MIN*60))

training_alive(){ pgrep -f '^([^ ]*/)?python[0-9.]* .*(train_cpt|torchrun|run_stage|ray_train|esm_native)' >/dev/null 2>&1; }
gpu_busy(){ "$NVSMI" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
            | awk '{ if ($1+0 > 0) b=1 } END { exit !b }'; }

t0=$(date +%s); idle=0; fired=""
say "armed: pid=$PID max_hours=$MAX_HOURS idle_min=$IDLE_MIN poll=${POLL}s publish=${PUBLISH_CMD:-none}"
training_alive || say "WARN: no training process matches at arm time (preflight phase, or wrong box?)"

# ---- the three guards ---------------------------------------------------------------------------
while :; do
  if ! kill -0 "$PID" 2>/dev/null; then fired="DONE (pid $PID exited)"; break; fi
  if [ $(( $(date +%s) - t0 )) -ge "$max_s" ]; then fired="CEILING (MAX_HOURS=$MAX_HOURS reached; pid $PID still alive)"; break; fi
  if ! training_alive && ! gpu_busy; then idle=$((idle + POLL)); else idle=0; fi
  if [ "$idle" -ge "$idle_s" ]; then fired="IDLE (no training process and GPUs at 0% for ${IDLE_MIN} min; pid $PID still alive)"; break; fi
  sleep "$POLL"
done
say "guard fired: $fired  (elapsed $(( ($(date +%s) - t0) / 60 )) min)"

# ---- publish, then stop ---------------------------------------------------------------------------
if [ -n "$PUBLISH_CMD" ]; then
  if [ "$DRY_RUN" = 1 ]; then say "dry-run: would publish with: $PUBLISH_CMD"
  else say "publishing"; bash -c "$PUBLISH_CMD" >>"$LOG" 2>&1; say "publish rc=$?"; fi
fi
if [ -n "$S3_LOG" ] && [ "$DRY_RUN" != 1 ]; then aws s3 cp "$LOG" "$S3_LOG" --only-show-errors; fi
if [ "$DRY_RUN" = 1 ]; then say "dry-run: would run: sudo shutdown -h +1"; exit 0; fi
say "stopping instance in 1 min (shutdown -h; shutdown behavior=stop, volume survives)"
sudo shutdown -h +1
