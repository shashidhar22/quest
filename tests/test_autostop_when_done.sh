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
bash -c 'exec -a "tmux new-session -d -s stage41 cd /x && python scripts/training/run_stage41_pipeline.py" sleep 15' & wpid=$!
sleep 1
if pgrep -f '^([^ ]*/)?python[0-9.]* .*(train_cpt|torchrun|run_stage|ray_train|esm_native)' >/dev/null; then echo "FAIL: T6 wrapper matched the training pattern"; fail=$((fail+1)); else echo "PASS: T6 wrapper argv does not match the anchored pattern"; pass=$((pass+1)); fi
kill "$wpid" 2>/dev/null

echo; echo "passed=$pass failed=$fail"; [ "$fail" = 0 ]
