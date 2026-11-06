#!/bin/bash
# The ONE command that starts the judge-first visit, on the box:
#
#   PUBLISH_DEST=s3://fh-pi-warren-h-eco/quest/tcrbench2/jf_out \
#     bash scripts/training/v04_box/launch_judge_first.sh
#
# It exists because the instruction it replaces was prose in a visit record: "launch the run,
# then arm autostop_when_done.sh on the pid in $O/judge_first.pid". Two manual steps in an order
# that matters, with no check that the second happened.
#
# WHAT THIS DOES, in order:
#   1. take the output root's lock, so two launches cannot interleave in one directory;
#   2. create a FRESH run directory for this invocation and generate a token for it;
#   3. start judge_first.sh there under a WAITABLE supervisor, with REQUIRE_WATCHDOG=1 so it
#      will not touch the GPU until it sees the handshake this script writes;
#   4. read the pid from the identity file THIS invocation's token appears in;
#   5. arm the watchdog on that pid, in its own session so it outlives the run it watches;
#   6. verify from the watchdog's own log that it armed on that exact pid;
#   7. only then write the handshake, releasing the run;
#   8. wait for the driver, record a status manifest, and publish this run's directory alone.
# On any failure before step 7 it terminates and reaps the driver IT started, and signals
# nothing else.
#
# WHY RUN IDENTITY, added 2026-11-06 after the second review of this file. The first version
# cleared `watchdog.armed` but not `judge_first.pid`, then read whatever pid file it found. With
# a live stale pid present it armed the watchdog on that pid and released the new driver -- the
# watchdog watching a process that was not the job, which is the 2026-10-03 idle-box failure
# this script exists to prevent. Its arm-failure branch then sent SIGTERM to that unverified
# pid, terminating an unrelated process; a test proved exactly that. A fresh directory per
# invocation makes a stale pid file unreachable by construction, and the token makes the
# identity positive evidence rather than an inference from directory freshness.
#
# WHY A WAITABLE SUPERVISOR. util-linux `setsid` calls setsid(2), which fails with EPERM when
# the caller already leads its process group -- which `set -m` guarantees for a background job.
# It then FORKS, and the parent returns EXIT_SUCCESS immediately unless --wait was given
# (sys-utils/setsid.c). So the first version's `$!` was not the driver on any box that has
# setsid, i.e. on the box, and its exit status was not the job's. This laptop has no setsid at
# all, which is why that went unnoticed here. Now: `setsid --wait` when it is supported, which
# waits and returns the child's status; otherwise no setsid for the driver, because a plain
# background job under `set -m` already leads its own process group and survives this wrapper
# (a non-interactive bash does not HUP its children). The watchdog still gets a plain `setsid`,
# where fork-and-return is exactly what is wanted: it must outlive everything here.
#
# The watchdog ceiling is deliberately ABOVE the launcher's: they stop different things. The
# launcher stops an overrunning job; the watchdog stops an idle or wedged box, including one a
# dead launcher left behind. A watchdog that fired first would mask the launcher's own refusal.
set -u -m
Q=${Q:-/home/ubuntu/quest}
OUT_ROOT=${O:-/home/ubuntu/jf_out}
MAX_HOURS=${MAX_HOURS:-8}
WATCHDOG_HOURS=${WATCHDOG_HOURS:-}
ARM_TIMEOUT=${ARM_TIMEOUT:-60}
IDENTITY_TIMEOUT=${IDENTITY_TIMEOUT:-60}
STOP_GRACE=${STOP_GRACE:-10}
AUTOSTOP=${AUTOSTOP:-$Q/scripts/training/autostop_when_done.sh}
JF=${JF:-$Q/scripts/training/v04_box/judge_first.sh}
PUBLISH_DEST=${PUBLISH_DEST:-}
PUBLISH_CMD=${PUBLISH_CMD:-}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
RUNDIR=$OUT_ROOT/runs/$RUN_ID
WLOG=${WLOG:-$RUNDIR/autostop.log}
LOCK=$OUT_ROOT/launch.lock
say(){ echo "[$(date -u '+%F %T')] [launch] $*"; }

[ -f "$JF" ] || { say "ABORT: no launcher at $JF"; exit 2; }
[ -f "$AUTOSTOP" ] || { say "ABORT: no watchdog at $AUTOSTOP"; exit 2; }
mkdir -p "$OUT_ROOT" || { say "ABORT: cannot create $OUT_ROOT"; exit 2; }

# ---- one active launch per output root ----------------------------------------------------
if [ -f "$LOCK" ]; then
  holder=$(cat "$LOCK" 2>/dev/null || true)
  case "$holder" in (''|*[!0-9]*) holder="" ;; esac
  if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
    say "ABORT: launch pid $holder is already active in $OUT_ROOT. Two drivers in one output"
    say "       root would interleave their artifacts and their watchdogs. Use a different O,"
    say "       or wait for that run to finish."
    exit 2
  fi
  say "clearing a stale lock left by pid ${holder:-unknown}"
fi
echo $$ > "$LOCK"
release_lock(){ [ "$(cat "$LOCK" 2>/dev/null || true)" = "$$" ] && rm -f "$LOCK"; return 0; }
trap 'release_lock' EXIT

mkdir -p "$RUNDIR"
ARMED=$RUNDIR/watchdog.armed
IDENT=$RUNDIR/run.identity
STATUS=$RUNDIR/launch_status.json
: > "$WLOG"
[ -n "$WATCHDOG_HOURS" ] || WATCHDOG_HOURS=$(awk "BEGIN{printf \"%.4f\", $MAX_HOURS*1.5}")
PID=""
SUP=""
ARMED_PID=""

# A status manifest, so a later reader can tell this run's outcome from the directory itself
# rather than from whatever files happen to be lying around. Finding 5 of the second review: a
# previous comparison.json survived a later refused run in a shared directory and would have
# been published beside it as if it belonged to it.
status(){
  local outcome=$1 rc=${2:-null}
  printf '{\n  "run_id": "%s",\n  "run_dir": "%s",\n  "outcome": "%s",\n  "driver_rc": %s,\n' \
    "$RUN_ID" "$RUNDIR" "$outcome" "$rc" > "$STATUS"
  printf '  "driver_pid": %s,\n  "watchdog_armed_pid": %s,\n  "supervisor": "%s",\n' \
    "${PID:-null}" "$([ -n "$ARMED_PID" ] && echo "\"$ARMED_PID\"" || echo null)" \
    "${DRIVER_SUP:-none}" >> "$STATUS"
  printf '  "max_hours": %s,\n  "watchdog_hours": %s,\n  "finished_utc": "%s"\n}\n' \
    "$MAX_HOURS" "$WATCHDOG_HOURS" "$(date -u '+%F %T')" >> "$STATUS"
}

# Terminate and reap the driver THIS invocation started, and nothing else. Only `$PID`, which
# was verified against this run's token, and `$SUP`, which is this shell's own child, are ever
# signalled.
stop_driver(){
  local w=0
  [ -n "$PID" ] && { kill -TERM "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true; }
  [ -n "$SUP" ] && kill -TERM "$SUP" 2>/dev/null
  while [ "$w" -lt "$STOP_GRACE" ]; do
    local live=0
    [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && live=1
    [ -n "$SUP" ] && kill -0 "$SUP" 2>/dev/null && live=1
    [ "$live" -eq 0 ] && break
    sleep 1; w=$((w+1))
  done
  [ -n "$PID" ] && { kill -KILL "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true; }
  [ -n "$SUP" ] && { kill -KILL "$SUP" 2>/dev/null || true; wait "$SUP" 2>/dev/null || true; }
  say "driver terminated and reaped after ${w}s"
}
die(){ say "ABORT: $*"; stop_driver; status "${DIE_OUTCOME:-launch_failed}"; exit 2; }

# ---- the supervisor ------------------------------------------------------------------------
SETSID=$(command -v setsid 2>/dev/null || true)
DRIVER_SUP=""
if [ -n "$SETSID" ] && "$SETSID" --wait true >/dev/null 2>&1; then
  DRIVER_SUP="$SETSID --wait"
fi
say "run_id=$RUN_ID dir=$RUNDIR"
say "driver supervisor: ${DRIVER_SUP:-none; plain background job in its own process group}"
[ -n "$PUBLISH_DEST" ] || [ -n "$PUBLISH_CMD" ] || \
  say "WARNING: no PUBLISH_DEST; results will be on the volume only"

say "starting the run (ceiling ${MAX_HOURS}h), held at preflight until the watchdog is verified"
REQUIRE_WATCHDOG=1 RUN_TOKEN=$RUN_ID Q=$Q O=$RUNDIR MAX_HOURS=$MAX_HOURS \
  $DRIVER_SUP bash "$JF" >>"$RUNDIR/launch.log" 2>&1 &
SUP=$!

# ---- identity: the pid must come from a file this invocation's token appears in -------------
w=0
while [ "$w" -lt "$IDENTITY_TIMEOUT" ]; do
  [ -s "$IDENT" ] && break
  kill -0 "$SUP" 2>/dev/null || break
  sleep 1; w=$((w+1))
done
if [ ! -s "$IDENT" ]; then
  say "the run published no identity in $IDENT. Its own output:"
  sed 's/^/       /' "$RUNDIR/launch.log" 2>/dev/null | tail -10
  DIE_OUTCOME=no_identity die "no verified driver identity, so there is no pid this launch can
  vouch for, and arming a watchdog on an unvouched pid is the defect this file was rewritten to
  remove. The driver must write \"\$RUN_TOKEN \$\$\" to \$O/run.identity as its first act;
  judge_first.sh does. A driver script that does not is not launchable through this wrapper."
fi
got=$(awk 'NR==1{print $1}' "$IDENT")
PID=$(awk 'NR==1{print $2}' "$IDENT")
[ "$got" = "$RUN_ID" ] || { PID=""; DIE_OUTCOME=identity_mismatch \
  die "identity token mismatch in $IDENT: found '$got', this launch is '$RUN_ID'"; }
case "$PID" in (''|*[!0-9]*) PID=""; DIE_OUTCOME=identity_mismatch \
  die "identity file records no usable pid: '$(cat "$IDENT")'" ;; esac
if ! kill -0 "$PID" 2>/dev/null; then
  say "the run exited before it could be watched; its own output:"
  sed 's/^/       /' "$RUNDIR/launch.log" 2>/dev/null | tail -10
  PID=""; DIE_OUTCOME=driver_exited_early die "nothing to watch"
fi
say "driver pid=$PID verified against this launch's token (supervisor pid $SUP)"

# ---- arm, and verify from the watchdog's own log --------------------------------------------
say "arming the watchdog on pid $PID (ceiling ${WATCHDOG_HOURS}h, log $WLOG)"
WATCH_PUBLISH=$PUBLISH_CMD
[ -n "$WATCH_PUBLISH" ] || [ -z "$PUBLISH_DEST" ] || \
  WATCH_PUBLISH="aws s3 sync '$RUNDIR' '$PUBLISH_DEST/$RUN_ID/' --only-show-errors"
MAX_HOURS=$WATCHDOG_HOURS LOG=$WLOG $SETSID bash "$AUTOSTOP" "$PID" "$WATCH_PUBLISH" \
  >>"$WLOG" 2>&1 &
w=0
while [ "$w" -lt "$ARM_TIMEOUT" ]; do
  grep -q "armed: pid=$PID " "$WLOG" 2>/dev/null && { ARMED_PID=$PID; break; }
  sleep 1; w=$((w+1))
done
if [ -z "$ARMED_PID" ]; then
  say "the watchdog did not arm on pid $PID within ${ARM_TIMEOUT}s. Watchdog log:"
  sed 's/^/       /' "$WLOG" 2>/dev/null | tail -5
  DIE_OUTCOME=watchdog_not_armed die "refusing to leave GPU work with no watchdog"
fi
say "watchdog armed and verified on pid $ARMED_PID"
: > "$ARMED"
say "handshake written: $ARMED — the run may now use the GPU"

# A stable path for the operator, rewritten (never read) so a stale copy cannot mislead.
echo "$PID" > "$OUT_ROOT/judge_first.pid"
rm -f "$OUT_ROOT/latest"
ln -s "$RUNDIR" "$OUT_ROOT/latest" 2>/dev/null || true
say "follow it with: tail -f $RUNDIR/run.log   (watchdog: tail -f $WLOG)"

wait "$SUP"
rc=$?
status "driver_exited_$rc" "$rc"
say "run finished rc=$rc; status in $STATUS"
if [ -n "$PUBLISH_DEST" ] && [ -z "$PUBLISH_CMD" ]; then
  say "publishing this run only: $PUBLISH_DEST/$RUN_ID/"
  aws s3 sync "$RUNDIR" "$PUBLISH_DEST/$RUN_ID/" --only-show-errors || \
    say "WARNING: publish failed; the watchdog will retry its own bounded copy"
fi
exit $rc
