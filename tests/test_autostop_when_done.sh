#!/usr/bin/env bash
# Falsifiable tests for scripts/training/autostop_when_done.sh, runnable on a laptop (no GPU):
# nvidia-smi is mocked, DRY_RUN=1 so nothing is published or stopped. Each case states what a
# failure looks like. Exit non-zero on the first failure.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
W="$HERE/scripts/training/autostop_when_done.sh"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
MOCK="$T/nvidia-smi"; printf '#!/bin/sh\nprintf "%s\\n%s\\n" "${GPU_UTIL:-0}" "${GPU_UTIL:-0}"\n' > "$MOCK"; chmod +x "$MOCK"
pass=0; fail=0
check(){ if grep -q "$2" "$1"; then echo "PASS: $3"; pass=$((pass+1)); else echo "FAIL: $3 — log lacks '$2'"; sed 's/^/    /' "$1"; fail=$((fail+1)); fi; }

echo "== T1 DONE: driver exits -> fires DONE, would stop (failure: no DONE line, or IDLE/CEILING fires first)"
sleep 8 & pid=$!
LOG="$T/t1.log" NVSMI="$MOCK" DRY_RUN=1 POLL=2 MAX_HOURS=1 IDLE_MIN=60 bash "$W" "$pid" "echo publish-ran" >/dev/null 2>&1
check "$T/t1.log" "guard fired: DONE" "T1 DONE fires"
check "$T/t1.log" "would run: sudo shutdown" "T1 reaches the stop step"
check "$T/t1.log" "would publish with: echo publish-ran" "T1 publish step runs before stop"

echo "== T2 CEILING: driver alive past MAX_HOURS -> fires CEILING (failure: DONE/IDLE instead, or never fires)"
sleep 120 & pid=$!
GPU_UTIL=50 LOG="$T/t2.log" NVSMI="$MOCK" DRY_RUN=1 POLL=1 MAX_HOURS=0.0015 IDLE_MIN=60 bash "$W" "$pid" >/dev/null 2>&1
kill "$pid" 2>/dev/null
check "$T/t2.log" "guard fired: CEILING" "T2 CEILING fires while pid alive and GPUs busy"

echo "== T3 IDLE: driver alive, GPUs 0%, no training process -> fires IDLE (failure: never fires within IDLE_MIN)"
sleep 300 & pid=$!
GPU_UTIL=0 LOG="$T/t3.log" NVSMI="$MOCK" DRY_RUN=1 POLL=10 MAX_HOURS=1 IDLE_MIN=1 bash "$W" "$pid" >/dev/null 2>&1
kill "$pid" 2>/dev/null
check "$T/t3.log" "guard fired: IDLE" "T3 IDLE fires after IDLE_MIN of idle"

echo "== T4 IDLE must NOT fire while GPUs busy (failure: IDLE line present)"
sleep 20 & pid=$!
GPU_UTIL=30 LOG="$T/t4.log" NVSMI="$MOCK" DRY_RUN=1 POLL=2 MAX_HOURS=1 IDLE_MIN=1 bash "$W" "$pid" >/dev/null 2>&1
if grep -q "guard fired: IDLE" "$T/t4.log"; then echo "FAIL: T4 IDLE fired while GPUs busy"; fail=$((fail+1)); else echo "PASS: T4 busy GPUs hold IDLE off; ended by $(grep -o 'guard fired: [A-Z]*' "$T/t4.log")"; pass=$((pass+1)); fi

echo "== T5 REFUSE: dead pid -> exit 2 (failure: arms anyway)"
LOG="$T/t5.log" NVSMI="$MOCK" DRY_RUN=1 bash "$W" 999999 >/dev/null 2>&1; rc=$?
if [ "$rc" = 2 ] && grep -q "REFUSE: pid" "$T/t5.log"; then echo "PASS: T5 refuses dead pid (rc=2)"; pass=$((pass+1)); else echo "FAIL: T5 rc=$rc"; fail=$((fail+1)); fi

echo "== T6 pattern: a tmux/bash wrapper whose argv CONTAINS the script name must not count as training"
# Simulate the 2026-10-03 shape: a bash process whose cmdline contains 'python … run_stage41_pipeline.py'
#
# A negative pgrep match only means something if pgrep can see the process list at all. The
# round-two review ran this where pgrep printed "Cannot get process list", and the case counted
# that as a pass: a check that cannot fail proves nothing. So the positive control comes first,
# and an unusable pgrep is reported INCONCLUSIVE rather than green.
bash -c 'exec -a "tmux new-session -d -s stage41 cd /x && python scripts/training/run_stage41_pipeline.py" sleep 15' & wpid=$!
# the positive control: a cmdline that SHOULD match the pattern, so a silent pgrep is detectable
bash -c 'exec -a "python scripts/training/train_cpt_stage2_single.py --fake" sleep 15' & cpid=$!
sleep 1
PAT='^([^ ]*/)?python[0-9.]* .*(train_cpt|torchrun|run_stage|ray_train|esm_native)'
ctl_err=$(pgrep -f "$PAT" 2>&1 >/dev/null)
if ! pgrep -f "$PAT" >/dev/null 2>&1; then
  echo "INCONCLUSIVE: T6 pgrep cannot see the positive control (${ctl_err:-no match}); the"
  echo "              negative case below would pass for the wrong reason, so it is not counted."
  inconclusive=$((${inconclusive:-0}+1))
else
  # pgrep works here, so a non-match for the wrapper is real evidence
  kill "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
  sleep 1
  if pgrep -f "$PAT" >/dev/null 2>&1; then echo "FAIL: T6 wrapper matched the training pattern"; fail=$((fail+1));
  else echo "PASS: T6 wrapper argv does not match the anchored pattern"; pass=$((pass+1)); fi
fi
kill "$wpid" "$cpid" 2>/dev/null

# ---- audit finding 3, 2026-11-06: the publish step must not be able to postpone the stop -----
# These run the REAL publish path (DRY_RUN=0) with the stop command injected, because DRY_RUN=1
# only logs what it would publish and therefore cannot exercise the bound at all.
echo "== T7 a hung publisher cannot postpone the stop (failure: the stop never happens, or waits out the publisher)"
sleep 2 & pid=$!
t0=$(date +%s)
LOG="$T/t7.log" NVSMI="$MOCK" DRY_RUN=0 POLL=1 MAX_HOURS=1 IDLE_MIN=60 PUBLISH_TIMEOUT=3 \
  SHUTDOWN_CMD="echo WOULD-SHUTDOWN" bash "$W" "$pid" "sleep 300" >/dev/null 2>&1
el=$(( $(date +%s) - t0 ))
check "$T/t7.log" "publish exceeded" "T7 the publisher is bounded and says so"
check "$T/t7.log" "WOULD-SHUTDOWN" "T7 the stop still happens"
if [ "$el" -lt 30 ]; then echo "PASS: T7 stopped in ${el}s, not after the 300s publisher"; pass=$((pass+1));
else echo "FAIL: T7 took ${el}s — the publisher postponed the stop"; fail=$((fail+1)); fi
if pgrep -f "^sleep 300$" >/dev/null 2>&1; then echo "FAIL: T7 left the hung publisher running"; fail=$((fail+1));
else echo "PASS: T7 the hung publisher was terminated"; pass=$((pass+1)); fi

echo "== T8 a publisher that finishes normally is NOT killed (failure: the bound kills everything, so T7 proves nothing)"
sleep 2 & pid=$!
LOG="$T/t8.log" NVSMI="$MOCK" DRY_RUN=0 POLL=1 MAX_HOURS=1 IDLE_MIN=60 PUBLISH_TIMEOUT=30 \
  SHUTDOWN_CMD="echo WOULD-SHUTDOWN" bash "$W" "$pid" "echo published-fine" >/dev/null 2>&1
check "$T/t8.log" "published-fine" "T8 a normal publish runs to completion"
check "$T/t8.log" "publish rc=0" "T8 its exit status is recorded"
if grep -q "publish exceeded" "$T/t8.log"; then echo "FAIL: T8 a normal publish was reported as hung"; fail=$((fail+1));
else echo "PASS: T8 a normal publish is not bounded out"; pass=$((pass+1)); fi

echo "== T9 the log copy to S3 is bounded too (failure: a stuck aws cp holds the box open)"
printf '#!/bin/sh\nsleep 300\n' > "$T/aws"; chmod +x "$T/aws"
sleep 2 & pid=$!
t0=$(date +%s)
PATH="$T:$PATH" LOG="$T/t9.log" NVSMI="$MOCK" DRY_RUN=0 POLL=1 MAX_HOURS=1 IDLE_MIN=60 \
  PUBLISH_TIMEOUT=3 LOG_COPY_TIMEOUT=3 S3_LOG="s3://example/nope.log" \
  SHUTDOWN_CMD="echo WOULD-SHUTDOWN" \
  bash "$W" "$pid" >/dev/null 2>&1
el=$(( $(date +%s) - t0 ))
check "$T/t9.log" "WOULD-SHUTDOWN" "T9 the stop happens despite a stuck log copy"
if [ "$el" -lt 30 ]; then echo "PASS: T9 stopped in ${el}s"; pass=$((pass+1));
else echo "FAIL: T9 took ${el}s — the log copy postponed the stop"; fail=$((fail+1)); fi

echo; echo "passed=$pass failed=$fail inconclusive=${inconclusive:-0}"
[ "${inconclusive:-0}" = 0 ] || echo "NOTE: an inconclusive case is not a pass. See the T6 note."
[ "$fail" = 0 ]
