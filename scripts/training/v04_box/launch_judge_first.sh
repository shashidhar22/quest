#!/bin/bash
# The ONE command that starts the judge-first visit, on the box:
#
#   bash scripts/training/v04_box/launch_judge_first.sh
#
# It exists because the previous instruction was prose in a visit record: "launch the run, then
# arm autostop_when_done.sh on the pid in $O/judge_first.pid". Two manual steps in an order that
# matters, with no check that the second happened. The 2026-11-06 audit put it plainly: the
# watchdog was a manual external instruction, not something the launch armed or verified. A visit
# whose watchdog was never armed is the USD 100 idle-box failure of 2026-10-03 waiting to recur.
#
# What this does, in order:
#   1. start judge_first.sh in its own session, with REQUIRE_WATCHDOG=1 so it will NOT begin any
#      GPU step until it sees the handshake file this script writes;
#   2. read the pid the run recorded for itself;
#   3. arm the watchdog on THAT pid, in its own session so it outlives the run it is watching;
#   4. verify from the watchdog's own log that it armed on that exact pid;
#   5. only then write the handshake, releasing the run.
# If arming cannot be verified, the run is terminated and nothing is spent.
#
# The watchdog ceiling is deliberately ABOVE the launcher's: they stop different things. The
# launcher stops an overrunning job; the watchdog stops an idle or wedged box, including one left
# by a launcher that died. A watchdog that fired first would mask the launcher's own refusal.
set -u -m
# `setsid` puts the run and the watchdog in sessions of their own, so neither dies with this
# wrapper's terminal. It is absent on macOS, where these tests run, so fall back to `set -m`:
# that still gives each job its own process GROUP, which is what signal isolation needs.
SETSID=$(command -v setsid 2>/dev/null || true)
Q=${Q:-/home/ubuntu/quest}
O=${O:-/home/ubuntu/jf_out}
MAX_HOURS=${MAX_HOURS:-8}
WATCHDOG_HOURS=${WATCHDOG_HOURS:-}
ARM_TIMEOUT=${ARM_TIMEOUT:-60}
AUTOSTOP=${AUTOSTOP:-$Q/scripts/training/autostop_when_done.sh}
JF=${JF:-$Q/scripts/training/v04_box/judge_first.sh}
PUBLISH_CMD=${PUBLISH_CMD:-}
WLOG=${WLOG:-$HOME/autostop.log}
mkdir -p "$O"
ARMED=$O/watchdog.armed
rm -f "$ARMED"
: > "$WLOG"
[ -n "$WATCHDOG_HOURS" ] || WATCHDOG_HOURS=$(awk "BEGIN{printf \"%.4f\", $MAX_HOURS*1.5}")
say(){ echo "[$(date -u '+%F %T')] [launch] $*"; }

[ -f "$JF" ] || { say "ABORT: no launcher at $JF"; exit 2; }
[ -f "$AUTOSTOP" ] || { say "ABORT: no watchdog at $AUTOSTOP"; exit 2; }
[ -n "$PUBLISH_CMD" ] || say "WARNING: no PUBLISH_CMD; results will be on the volume only"

say "starting the run (ceiling ${MAX_HOURS}h), held at preflight until the watchdog is verified"
REQUIRE_WATCHDOG=1 Q=$Q O=$O MAX_HOURS=$MAX_HOURS $SETSID bash "$JF" >>"$O/launch.log" 2>&1 &
RUN=$!

# The run writes its own pid; that file, not $! , is what the watchdog must watch, because the
# two can differ (a wrapper shell) and the watchdog keying on the wrong pid is guard #1 failing.
for _ in $(seq 1 30); do [ -s "$O/judge_first.pid" ] && break; sleep 1; done
PID=$(cat "$O/judge_first.pid" 2>/dev/null || echo "")
case "$PID" in (''|*[!0-9]*)
  say "ABORT: the run never recorded a usable pid. Its own output:"
  sed 's/^/       /' "$O/launch.log" 2>/dev/null | tail -10
  kill -TERM "$RUN" 2>/dev/null; exit 2;; esac
kill -0 "$PID" 2>/dev/null || { say "ABORT: the run exited before it could be watched; see $O/launch.log"; exit 2; }
say "run pid=$PID (wrapper $RUN)"

say "arming the watchdog on pid $PID (ceiling ${WATCHDOG_HOURS}h, log $WLOG)"
MAX_HOURS=$WATCHDOG_HOURS LOG=$WLOG $SETSID bash "$AUTOSTOP" "$PID" "$PUBLISH_CMD" >>"$WLOG" 2>&1 &

for _ in $(seq 1 "$ARM_TIMEOUT"); do grep -q "armed: pid=$PID " "$WLOG" 2>/dev/null && break; sleep 1; done
if ! grep -q "armed: pid=$PID " "$WLOG" 2>/dev/null; then
  say "ABORT: the watchdog did not arm on pid $PID within ${ARM_TIMEOUT}s. Terminating the run"
  say "       rather than leaving GPU work with no watchdog. Watchdog log:"
  sed 's/^/       /' "$WLOG" 2>/dev/null | tail -5
  kill -TERM "$PID" 2>/dev/null
  exit 2
fi
say "watchdog armed and verified on pid $PID"
: > "$ARMED"
say "handshake written: $ARMED — the run may now use the GPU"
say "follow it with: tail -f $O/run.log   (watchdog: tail -f $WLOG)"
wait "$RUN"
rc=$?
say "run finished rc=$rc"
exit $rc
