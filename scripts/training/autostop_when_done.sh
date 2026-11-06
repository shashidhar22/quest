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
# crashed. That publish is BOUNDED at PUBLISH_TIMEOUT seconds (default 600) and so is the optional
# log copy, because an unbounded publish immediately before a shutdown is an unbounded bill: a
# wedged upload held the box open and defeated the whole watchdog (audit finding 3, 2026-11-06).
# Arm-time self-test refuses to arm on a dead PID, a missing nvidia-smi, or an absurd MAX_HOURS.
# Every decision is logged with the guard that made it. DRY_RUN=1 logs instead of
# publishing/stopping, and SHUTDOWN_CMD is overridable so the real publish-then-stop path is
# itself testable (both used by tests/test_autostop_when_done.sh).
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
PUBLISH_TIMEOUT="${PUBLISH_TIMEOUT:-600}"   # seconds; nothing may postpone the stop past this
LOG_COPY_TIMEOUT="${LOG_COPY_TIMEOUT:-120}" # the log copy is bounded separately and shorter
SHUTDOWN_CMD="${SHUTDOWN_CMD:-sudo shutdown -h +1}"   # overridable so the stop path is testable

say(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }

# run_bounded <seconds> <label> <command string> -- runs it, kills it if it overruns, returns.
#
# Audit finding 3, 2026-11-04 -> 2026-11-06: the publish step ran `bash -c "$PUBLISH_CMD"` with
# no bound, immediately before the shutdown. A stuck upload -- an `aws s3 sync` against a dead
# endpoint, a credential prompt, a retry loop -- therefore held the instance open indefinitely,
# which defeats the entire purpose of a cost watchdog. Publishing is worth waiting a bounded
# while for; it is never worth an unbounded bill. The job leads its own process group so a
# wedged grandchild (aws spawns its own) dies with it.
run_bounded(){
  local limit=$1 lbl=$2 cmd=$3 waited=0 pp rc=0
  set -m
  bash -c "$cmd" >>"$LOG" 2>&1 &
  pp=$!
  set +m
  while [ "$waited" -lt "$limit" ] && kill -0 "$pp" 2>/dev/null; do
    sleep 2; waited=$((waited+2))
  done
  if kill -0 "$pp" 2>/dev/null; then
    say "$lbl exceeded ${limit}s: terminating it and stopping anyway"
    kill -TERM -"$pp" 2>/dev/null || kill -TERM "$pp" 2>/dev/null || true
    sleep 2
    kill -KILL -"$pp" 2>/dev/null || kill -KILL "$pp" 2>/dev/null || true
    wait "$pp" 2>/dev/null || true
    say "$lbl rc=timeout"
    return 124
  fi
  wait "$pp" || rc=$?
  say "$lbl rc=$rc"
  return $rc
}

# ---- arm-time self-test: refuse rather than arm something that cannot work ----------------------
# Numeric BEFORE it reaches awk: `abc` inside an awk expression evaluates to 0, and anything
# else spliced there is awk program text (pre-launch review, 2026-11-06).
case "$MAX_HOURS" in (''|*[!0-9.]*|*.*.*) say "REFUSE: MAX_HOURS=$MAX_HOURS is not a number"; exit 2;; esac
kill -0 "$PID" 2>/dev/null            || { say "REFUSE: pid $PID is not alive"; exit 2; }
command -v "$NVSMI" >/dev/null 2>&1   || { say "REFUSE: $NVSMI not found"; exit 2; }
awk "BEGIN{exit !($MAX_HOURS > 0 && $MAX_HOURS < 240)}" \
                                      || { say "REFUSE: MAX_HOURS=$MAX_HOURS (want 0 < h < 240)"; exit 2; }
max_s=$(awk "BEGIN{printf \"%d\", $MAX_HOURS*3600}")
idle_s=$((IDLE_MIN*60))

# `track_a.esm_sft` is the judge. Without it the watchdog could not see judge-first's work at all
# and fell back on GPU utilisation alone, so 20 minutes of split loading, negative sampling or a
# slow model load on a lazily restored volume would have fired IDLE and stopped the box mid-run
# (pre-launch review, 2026-11-06).
training_alive(){ pgrep -f '^([^ ]*/)?python[0-9.]* .*(train_cpt|torchrun|run_stage|ray_train|esm_native|track_a[.]esm_sft)' >/dev/null 2>&1; }
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
  if [ "$DRY_RUN" = 1 ]; then say "dry-run: would publish with: $PUBLISH_CMD (bounded at ${PUBLISH_TIMEOUT}s)"
  else say "publishing (bounded at ${PUBLISH_TIMEOUT}s)"; run_bounded "$PUBLISH_TIMEOUT" publish "$PUBLISH_CMD" || true; fi
fi
if [ -n "$S3_LOG" ] && [ "$DRY_RUN" != 1 ]; then
  # Bounded for the same reason, and it is the LAST thing attempted, so losing it costs only
  # this log -- which is already on the volume the stop preserves.
  run_bounded "$LOG_COPY_TIMEOUT" "log copy" "aws s3 cp '$LOG' '$S3_LOG' --only-show-errors" || true
fi
if [ "$DRY_RUN" = 1 ]; then say "dry-run: would run: $SHUTDOWN_CMD"; exit 0; fi
say "stopping instance (shutdown behavior=stop, volume survives): $SHUTDOWN_CMD"
$SHUTDOWN_CMD
