#!/bin/bash
# The judge-checkpoint diagnostic (step 12, gate 4): exploratory, evaluation only, no verdict.
#
# Start it ONLY through the wrapper, which arms the watchdog on this run's own pid:
#
#   JF=$Q/scripts/training/pilot12_box/run_diagnostic.sh \
#   Q=… O=… PY=… MANIFEST_A=… MANIFEST_A_SHA256=… MANIFEST_B=… MANIFEST_B_SHA256=… \
#   CKPT_BASE=… CKPT_D=… ADAPTER_D=… MAX_HOURS=2 \
#     bash scripts/training/v04_box/launch_judge_first.sh
#
#   preflight  CPU  files exist, both manifest sha256s match, nvidia-smi present (unless ALLOW_CPU)
#   A          GPU  15-query manifest: base, arm D, then diagnostic_analyze -> A_analysis.json
#   B          GPU  100-per-condition manifest: the same -> B_analysis.json
#
# The launch contract is judge_first.sh's, and so are its reasons: "$RUN_TOKEN $$" in
# $O/run.identity first, so the wrapper can vouch for the pid it arms the watchdog on; under
# REQUIRE_WATCHDOG=1 no GPU step before the wrapper's watchdog.armed handshake; MAX_HOURS checked
# before every step and enforced DURING it by killing the child's process group; status.json on
# every exit path, so the outcome is read from the run directory rather than inferred from which
# files exist. A step whose output already exists is skipped (resume) and never overwritten.
#
# Each python step is also wrapped in coreutils `timeout` where it exists: the scorer's
# --max-seconds is checked only between batches and cannot stop a hung kernel. The kill comes
# KILL_SLACK seconds after --max-seconds, because that clock also covers model load and writing.
set -u -m
START=$(date +%s)
if [ -n "${O:-}" ]; then
  mkdir -p "$O" || exit 2
  [ -n "${RUN_TOKEN:-}" ] && echo "$RUN_TOKEN $$" > "$O/run.identity"
fi
STAGE=env
COMPLETED=""
clean(){ printf '%s' "$1" | tr -d '"\\' | tr '\n' ' '; }      # a value safe inside a JSON string
write_status(){
  [ -n "${O:-}" ] || return 0
  {
    printf '{\n  "run_id": "%s",\n  "outcome": "%s",\n  "stage": "%s",\n  "completed": "%s",\n' \
      "$(clean "${RUN_TOKEN:-unnamed}")" "$1" "$STAGE" "${COMPLETED# }"
    printf '  "detail": "%s",\n  "elapsed_min": %s,\n  "finished_utc": "%s"\n}\n' \
      "$(clean "${2:-}")" "$(( ($(date +%s) - START) / 60 ))" "$(date -u '+%F %T')"
  } 2>/dev/null > "$O/status.json"
}
say(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$O/run.log"; }

missing=""
for v in Q O PY MANIFEST_A MANIFEST_A_SHA256 MANIFEST_B MANIFEST_B_SHA256 CKPT_BASE CKPT_D ADAPTER_D MAX_HOURS; do
  [ -n "${!v:-}" ] || missing="$missing $v"
done
if [ -n "$missing" ]; then
  echo "REFUSE: required environment not set:$missing" >&2
  write_status env_refused "not set:$missing"; exit 2
fi
# A quote, backslash or newline in a path would break status.json or a command line.
for v in Q O PY MANIFEST_A MANIFEST_B CKPT_BASE CKPT_D ADAPTER_D; do
  case "${!v}" in (*$'\n'*|*'"'*|*\\*)
    echo "REFUSE: $v holds a newline, a quote or a backslash" >&2
    write_status env_refused "$v holds a newline, quote or backslash"; exit 2;; esac
done
ALLOW_CPU=${ALLOW_CPU:-}
MAX_SECONDS_A=${MAX_SECONDS_A:-600}
MAX_SECONDS_B=${MAX_SECONDS_B:-3000}
KILL_SLACK=${KILL_SLACK:-120}
# POLL and SHUTDOWN_GRACE are read by the watchdog too, which adds POLL in integer shell
# arithmetic: a fraction there breaks it after it has armed. So this driver's own wait interval
# has its own name (STEP_POLL, fractions allowed), and the shared ones must be bounded integers.
STEP_POLL=${STEP_POLL:-1}
[[ $STEP_POLL =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "REFUSE: STEP_POLL=$STEP_POLL is not a number" >&2; write_status env_refused "a wait knob is not a number"; exit 2; }
for v in POLL SHUTDOWN_GRACE; do
  [ -n "${!v+x}" ] || continue
  { [[ ${!v} =~ ^[0-9]+$ ]] && [ "${!v}" -le 300 ]; } || {
    echo "REFUSE: $v=${!v} is not an integer from 0 to 300 (the watchdog reads it too)" >&2; write_status env_refused "a wait knob is not a number"; exit 2; }
done
[ "${POLL:-1}" != 0 ] || { echo "REFUSE: POLL=0 is not a wait interval" >&2; write_status env_refused "a wait knob is not a number"; exit 2; }
SHUTDOWN_GRACE=${SHUTDOWN_GRACE:-15}
# Numeric BEFORE awk sees it: inside an awk expression `abc` evaluates to 0 (judge_first.sh).
for v in MAX_HOURS MAX_SECONDS_A MAX_SECONDS_B KILL_SLACK; do
  [[ ${!v} =~ ^[0-9]+(\.[0-9]+)?$ ]] || {
    echo "REFUSE: $v=${!v} is not a number" >&2; write_status env_refused "$v not a number"; exit 2; }
done
# The outcome is read from status.json, so a launch that cannot write it scores nothing: a stale
# file left in place would report an earlier launch's outcome as this one's.
write_status running "started"
grep -q '"outcome": "running"' "$O/status.json" 2>/dev/null || {
  echo "REFUSE: cannot write $O/status.json; refusing to run with no record of the outcome" >&2; exit 2; }
# 2 h is this visit's frozen ceiling (plans/step12-diagnostic-and-pilot.md). The wrapper's own
# default is 8, so a launch that forgets MAX_HOURS must refuse here rather than run under it.
awk "BEGIN{exit !($MAX_HOURS >= 0 && $MAX_HOURS <= 2)}" || {
  echo "REFUSE: MAX_HOURS=$MAX_HOURS is above this visit's frozen ceiling (want 0 <= hours <= 2)" >&2
  write_status env_refused "MAX_HOURS out of range"; exit 2; }
MAX_S=$(awk "BEGIN{printf \"%d\", $MAX_HOURS*3600}")
TIMEOUT=$(command -v timeout 2>/dev/null || true)
export PYTHONPATH=$Q TOKENIZERS_PARALLELISM=false
cd "$Q" || { write_status env_refused "cannot cd to Q"; exit 2; }

JOBS=""
signal_tree(){ kill -"$2" -"$1" 2>/dev/null || kill -"$2" "$1" 2>/dev/null || true; }
cleanup(){
  local p w=0
  [ -n "${JOBS// /}" ] || return 0
  for p in $JOBS; do signal_tree "$p" TERM; done
  while [ "$w" -lt "$SHUTDOWN_GRACE" ]; do
    local live=0; for p in $JOBS; do kill -0 "$p" 2>/dev/null && live=1; done
    [ "$live" -eq 0 ] && break
    sleep 1; w=$((w+1))
  done
  for p in $JOBS; do signal_tree "$p" KILL; wait "$p" 2>/dev/null || true; done
  JOBS=""
}
trap 'say "signal received: cleaning up"; cleanup; write_status terminated "signal received"; exit 130' INT TERM
die(){ say "ABORT: $*"; cleanup; write_status failed "$*"; exit 1; }
used_s(){ echo $(( $(date +%s) - START )); }
expired(){ [ "$(used_s)" -ge "$MAX_S" ]; }
check_ceiling(){ ! expired || die "MAX_HOURS=$MAX_HOURS ceiling reached before $1; stopping rather than overrunning"; }

# run_step <label> <log> <timeout seconds> <command...> -- tracked, own process group, deadline
# enforced while it runs (judge_first.sh run_gpu, same reasons, same bash-3.2 polling).
run_step(){
  local lbl=$1 log=$2 secs=$3 rc=0 p
  shift 3
  if [ -n "$TIMEOUT" ]; then
    ( export CUDA_VISIBLE_DEVICES=0; exec "$TIMEOUT" -k 30 "$secs" "$@" > "$log" 2>&1 ) &
  else
    ( export CUDA_VISIBLE_DEVICES=0; exec "$@" > "$log" 2>&1 ) &
  fi
  p=$!
  JOBS="$JOBS $p"
  while kill -0 "$p" 2>/dev/null; do
    if expired; then
      say "$lbl: MAX_HOURS=$MAX_HOURS ceiling reached WHILE running; terminating it"
      return 124     # still in JOBS: the caller's die -> cleanup kills its process group
    fi
    sleep "$STEP_POLL"
  done
  wait "$p" || rc=$?
  JOBS=""
  say "$lbl rc=$rc"
  return $rc
}

# step_or_die <label> <log> <timeout seconds> <command...>: run_step, with a stop at the ceiling
# told apart from a failure in status.json.
step_or_die(){
  local rc=0
  run_step "$@" || rc=$?
  [ "$rc" -ne 124 ] || die "$1: stopped at the ceiling while it was running"
  [ "$rc" -eq 0 ] || die "$1 failed (rc=$rc; its log is $2)"
}

sha(){ if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | awk '{print $1}'; }

# ---------- preflight ----------
STAGE=preflight
say "DIAGNOSTIC start pid=$$ max_hours=$MAX_HOURS timeout=${TIMEOUT:-none}"
bad=""
for f in "$MANIFEST_A" "$MANIFEST_B" "$CKPT_BASE" "$CKPT_D" \
         "$Q/track_a/esm_sft/diagnostic_eval.py" "$Q/track_a/esm_sft/diagnostic_analyze.py"; do
  [ -f "$f" ] || bad="$bad; missing file $f"
done
[ -d "$ADAPTER_D" ] || bad="$bad; missing adapter directory $ADAPTER_D"
[ -f "$MANIFEST_A" ] && [ "$(sha "$MANIFEST_A")" != "$MANIFEST_A_SHA256" ] && bad="$bad; MANIFEST_A sha256 $(sha "$MANIFEST_A") is not $MANIFEST_A_SHA256"
[ -f "$MANIFEST_B" ] && [ "$(sha "$MANIFEST_B")" != "$MANIFEST_B_SHA256" ] && bad="$bad; MANIFEST_B sha256 $(sha "$MANIFEST_B") is not $MANIFEST_B_SHA256"
[ -n "$ALLOW_CPU" ] || command -v nvidia-smi >/dev/null 2>&1 || bad="$bad; nvidia-smi not found and ALLOW_CPU is unset"
if [ -n "$bad" ]; then
  say "ABORT: preflight failed, nothing was started${bad}"
  write_status preflight_failed "${bad#; }"; exit 2
fi
if [ "${REQUIRE_WATCHDOG:-0}" = 1 ]; then
  waited=0
  while [ ! -f "$O/watchdog.armed" ] && [ "$waited" -lt "${ARM_WAIT:-120}" ]; do sleep 1; waited=$((waited+1)); done
  [ -f "$O/watchdog.armed" ] || { say "ABORT: no watchdog handshake at $O/watchdog.armed after ${waited}s;
  refusing to start GPU work unwatched. Launch through scripts/training/v04_box/launch_judge_first.sh."
  write_status no_watchdog "handshake absent"; exit 2; }
  say "watchdog handshake present after ${waited}s"
elif [ -z "$ALLOW_CPU" ] || command -v nvidia-smi >/dev/null 2>&1; then
  # REQUIRE_WATCHDOG=1 is set by the wrapper. Without it nothing is watching this box, and only
  # an explicit off-box dry run may go on: ALLOW_CPU=1 on a machine with no GPU. ALLOW_CPU does
  # not force the CPU, so on a GPU box it would be a way to score unwatched.
  say "ABORT: not started through scripts/training/v04_box/launch_judge_first.sh (REQUIRE_WATCHDOG is
  not 1), so no watchdog is armed; refusing to start GPU work."
  write_status no_watchdog "not launched through the wrapper"; exit 2
fi
COMPLETED="$COMPLETED preflight"
CK_BASE_SHA=$(sha "$CKPT_BASE")
CK_D_SHA=$(sha "$CKPT_D")
# names <file> <string>...: the file holds every string, quoted, as a JSON value. An output is
# reused only when it names the manifest and checkpoint this run would have scored: a file that
# merely exists (stale, empty, another checkpoint's) is not a result of this run.
names(){ local f=$1 s; shift; for s in "$@"; do grep -qF "\"$s\"" "$f" || return 1; done; }
CPU_FLAG=$([ -n "$ALLOW_CPU" ] && echo "--allow-cpu" || echo "")

# stage <A|B> <manifest> <max seconds> <manifest sha256>
stage(){
  local s=$1 man=$2 maxs=$3 mansha=$4 arm adapter ckpt cksha out secs
  STAGE=$s
  secs=$(awk "BEGIN{printf \"%d\", $maxs + $KILL_SLACK}")
  for arm in base armD; do
    out=$O/${s}_$arm.json
    if [ "$arm" = base ]; then adapter=none; ckpt=$CKPT_BASE; cksha=$CK_BASE_SHA
    else adapter=$ADAPTER_D; ckpt=$CKPT_D; cksha=$CK_D_SHA; fi
    if [ -f "$out" ]; then
      names "$out" judge-diagnostic-results-v1 "$mansha" "$cksha" || die "$out exists and is not
      a scoring of this manifest with this checkpoint. It is not reused or overwritten; move it aside."
      say "$s $arm: $out is this manifest and checkpoint; reused, not overwritten"; continue
    fi
    check_ceiling "$s $arm"
    step_or_die "$s $arm" "$O/${s}_$arm.log" "$secs" \
      "$PY" -m track_a.esm_sft.diagnostic_eval --manifest "$man" --ckpt "$ckpt" --adapter "$adapter" \
        --task as_paired_i --tcr_variant cdr3 --mhc_variant full --max_len 1024 --batch-size 100 \
        --precision native --max-seconds "$maxs" --out "$out" $CPU_FLAG
    [ -f "$out" ] || die "stage $s: the $arm scoring reported success but wrote no $out"
  done
  out=$O/${s}_analysis.json
  if [ -f "$out" ]; then
    names "$out" "$mansha" "$CK_BASE_SHA" "$CK_D_SHA" || die "$out exists and is not the analysis
    of this manifest and these two checkpoints. It is not reused or overwritten; move it aside."
    say "$s analysis: $out is this manifest and these checkpoints; reused, not overwritten"
  else
    check_ceiling "$s analysis"
    step_or_die "$s analysis" "$O/${s}_analysis.log" 600 \
      "$PY" -m track_a.esm_sft.diagnostic_analyze --manifest "$man" \
        --results "base=$O/${s}_base.json" "armD=$O/${s}_armD.json" --out "$out"
    [ -f "$out" ] || die "stage $s: the analysis reported success but wrote no $out"
    cat "$O/${s}_analysis.log" >> "$O/run.log"
  fi
  COMPLETED="$COMPLETED $s"
}

stage A "$MANIFEST_A" "$MAX_SECONDS_A" "$MANIFEST_A_SHA256"
stage B "$MANIFEST_B" "$MAX_SECONDS_B" "$MANIFEST_B_SHA256"
STAGE=done
write_status complete "stages A and B scored and analysed"
say "DIAGNOSTIC complete in $(( $(used_s) / 60 )) min: $O/A_analysis.json $O/B_analysis.json"
say "  Exploratory and native precision only; the analysis assigns no pass/fail label."
