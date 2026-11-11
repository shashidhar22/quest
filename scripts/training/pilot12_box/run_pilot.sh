#!/bin/bash
# STEP-12 PILOT: two CPT arms, then three SFT cells, on one GPU, in one box visit.
#
#   0  CPU    experiment_lock.py verify: the code, both corpus markers and the splits are the
#             locked ones; MAX_HOURS within the lock's visit ceiling
#   1  GPU    CPT arm P (repaired_proportional_10M), argv from the lock; then check-cpt
#   2  GPU    CPT arm E (task_enriched_cap1000), likewise; then check-cpt and check-pair
#             (one initial LoRA state, masked targets within tolerance of each other and of
#             the pilot budget). Stages 1-2 are also bounded by CPT_MAX_HOURS.
#   3  GPU    SFT train + validation eval for B (no adapter), P, E; check-eval after each
#   4  CPU    pilot_compare.py on the three cells -> $O/pilot.json
#
# Started ONLY through scripts/training/v04_box/launch_judge_first.sh with JF=<this file>,
# whose contract this keeps exactly as judge_first.sh does: "$RUN_TOKEN $$" in $O/run.identity
# first; under REQUIRE_WATCHDOG=1 no GPU work before $O/watchdog.armed; a MAX_HOURS deadline
# checked before every GPU step and enforced on a running one through its process group;
# status.json written on every exit path.
#
# WHY EVERY ARGUMENT COMES FROM THE LOCK. The plan freezes every open value in
# experiment-lock.json; a driver that also carried hyperparameters would be a second place for
# them to drift, and the trainer's legacy defaults (max_length 128, bucket boundaries up to 128,
# early stopping on, wandb on) would fill any flag left out. cpt-args and sft-args print the
# whole argv, and check-cpt / check-eval hold what each run recorded to the same lock.
#
# THE LOCK IS HELD, NOT JUST CHECKED. Stage 0 verifies it and copies the verified bytes into the
# run directory; every later read is of that copy, and before each GPU step the lock file, the
# copy and the artifacts are verified again. A run directory started under one lock is never
# continued under another, and pilot.json is reused only for the exact eval files it was made from.
#
# RESUME. A stage whose check passes on existing output is reused and logged as such. Output
# that exists and fails its check stops the run: it is never overwritten and never trained on,
# because a half-written arm is indistinguishable from a finished one without that check.
#
# Required: Q O PY EXPERIMENT_LOCK CORPUS_P CORPUS_E SPLITS_DIR CANONICAL_EVAL MAX_HOURS.
# Optional: CPT_MAX_HOURS (default and ceiling: the lock's cpt_stage_max_hours), ALLOW_CPU=1
# (off-box dry runs only). The trainer's leakage-audit check always runs, against the lock's
# own benchmark hashes ($O/benchmark_split_sha256.json, written by cpt-args).
# EXPERIMENT_LOCK, not LOCK: the wrapper assigns LOCK=$OUT_ROOT/launch.lock in its own shell,
# which overwrites an exported LOCK, so this driver would have been handed the launch lock.
set -u -m
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
if [ -n "${O:-}" ]; then
  mkdir -p "$O" || { echo "ABORT: cannot create O=$O" >&2; exit 2; }
  [ -n "${RUN_TOKEN:-}" ] && echo "$RUN_TOKEN $$" > "$O/run.identity"
fi
START=$(date +%s)
STAGES=""
CURRENT=""
LOCK_SHA=""
PILOT_PRODUCED=0

say(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$O/run.log"; }
used_s(){ echo $(( $(date +%s) - START )); }
clean(){ printf '%s' "$1" | tr -d '"\\' | tr '\n' ' '; }      # a value safe inside a JSON string
sha(){ if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | awk '{print $1}'; }
mark(){ STAGES="$STAGES${STAGES:+, }\"$1\": \"$2\""; }
write_status(){
  local outcome=$1 detail=${2:-}
  {
    printf '{\n  "run_id": "%s",\n  "outcome": "%s",\n  "detail": "%s",\n' \
      "$(clean "${RUN_TOKEN:-unnamed}")" "$outcome" "$(clean "$detail")"
    printf '  "lock": "%s",\n  "lock_sha256": %s,\n  "stages": {%s},\n' \
      "$(clean "${EXPERIMENT_LOCK:-}")" "$([ -n "$LOCK_SHA" ] && echo "\"$LOCK_SHA\"" || echo null)" "$STAGES"
    printf '  "pilot": %s,\n  "elapsed_min": %s,\n  "finished_utc": "%s"\n}\n' \
      "$([ "$PILOT_PRODUCED" = 1 ] && echo "\"$(clean "$O")/pilot.json\"" || echo null)" \
      "$(( $(used_s) / 60 ))" "$(date -u '+%F %T')"
  } 2>/dev/null > "$O/status.json"
}

missing=""
for v in Q O PY EXPERIMENT_LOCK CORPUS_P CORPUS_E SPLITS_DIR CANONICAL_EVAL MAX_HOURS; do
  [ -n "${!v:-}" ] || missing="$missing $v"
done
if [ -n "$missing" ]; then
  echo "ABORT: required environment unset:$missing" >&2
  [ -n "${O:-}" ] && write_status refused "required environment unset:$missing"
  exit 2
fi
# Every value below ends up as one line of an argv file or inside status.json. A newline would
# add trainer flags the lock never set; a quote or backslash would break the status file.
for v in Q O PY EXPERIMENT_LOCK CORPUS_P CORPUS_E SPLITS_DIR CANONICAL_EVAL MAX_HOURS; do
  case "${!v}" in (*$'\n'*|*'"'*|*\\*)
    echo "ABORT: $v holds a newline, a quote or a backslash" >&2
    write_status refused "$v holds a newline, quote or backslash"; exit 2;; esac
done
# POLL and SHUTDOWN_GRACE are read by the watchdog too, which adds POLL in integer shell
# arithmetic: a fraction there breaks it after it has armed. So this driver's own wait interval
# has its own name (STEP_POLL, fractions allowed), and the shared ones must be bounded integers.
STEP_POLL=${STEP_POLL:-1}
[[ $STEP_POLL =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "ABORT: STEP_POLL=$STEP_POLL is not a number" >&2; write_status refused "a wait knob is not a bounded number"; exit 2; }
for v in POLL SHUTDOWN_GRACE; do
  [ -n "${!v+x}" ] || continue
  { [[ ${!v} =~ ^[0-9]+$ ]] && [ "${!v}" -le 300 ]; } || {
    echo "ABORT: $v=${!v} is not an integer from 0 to 300 (the watchdog reads it too)" >&2; write_status refused "a wait knob is not a bounded number"; exit 2; }
done
[ "${POLL:-1}" != 0 ] || { echo "ABORT: POLL=0 is not a wait interval" >&2; write_status refused "a wait knob is not a bounded number"; exit 2; }
SHUTDOWN_GRACE=${SHUTDOWN_GRACE:-15}
# The outcome is read from status.json, so a launch that cannot write it does no GPU work: a
# stale file left in place would report an earlier launch's outcome as this one's.
write_status running "started"
grep -q '"outcome": "running"' "$O/status.json" 2>/dev/null || {
  echo "ABORT: cannot write $O/status.json; refusing to run with no record of the outcome" >&2; exit 2; }
ALLOW_CPU=${ALLOW_CPU:-}
LP=$Q/scripts/training/pilot12_box/experiment_lock.py
TRAINER=$Q/scripts/training/train_cpt_stage2_single.py
CMP=$Q/scripts/training/pilot12_box/pilot_compare.py
CPU_JUDGE=$([ -n "$ALLOW_CPU" ] && echo "--allow_cpu" || echo "")
# The judge resolves splits and pool through QUEST_SPLITS_DIR; the lock's digests are of
# SPLITS_DIR, so the two must name one directory or check-eval refuses every cell.
export PYTHONPATH=$Q QUEST_SPLITS_DIR=$SPLITS_DIR \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd "$Q" || { echo "ABORT: cannot cd to Q=$Q" >&2; write_status refused "no Q"; exit 2; }

JOBS=""
register(){ JOBS="$JOBS $1"; }
unregister(){ local keep="" t; for t in $JOBS; do [ "$t" = "$1" ] || keep="$keep $t"; done; JOBS=$keep; }
signal_tree(){ kill -"$2" -"$1" 2>/dev/null || kill -"$2" "$1" 2>/dev/null || true; }
cleanup(){
  local t waited=0 still
  [ -n "${JOBS// /}" ] || return 0
  say "cleanup: terminating$JOBS"
  for t in $JOBS; do signal_tree "$t" TERM; done
  while [ "$waited" -lt "$SHUTDOWN_GRACE" ]; do
    still=0
    for t in $JOBS; do kill -0 "$t" 2>/dev/null && still=1; done
    [ "$still" -eq 0 ] && break
    sleep 1; waited=$((waited+1))
  done
  for t in $JOBS; do signal_tree "$t" KILL; wait "$t" 2>/dev/null || true; done
  JOBS=""
}
trap 'say "signal received: cleaning up before exit"; cleanup; [ -n "$CURRENT" ] && mark "$CURRENT" failed; write_status terminated "signal received"; exit 130' INT TERM
die(){
  say "ABORT: $*"
  cleanup
  [ -n "$CURRENT" ] && mark "$CURRENT" failed
  write_status failed "$*"
  say "PILOT FAILED"
  exit 1
}
# lockpy <subcommand> [args]: the lock module, its output in run.log, its status returned.
lockpy(){
  "$PY" "$LP" "$1" --lock "$LOCKFILE" "${@:2}" 2>&1 | tee -a "$O/run.log"
  return "${PIPESTATUS[0]}"
}
# lockval <dotted.key>: one value from the lock; a list prints space-separated.
lockval(){
  "$PY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    d = d[k]
print(" ".join(map(str, d)) if isinstance(d, list) else d)' "$LOCKFILE" "$1"
}
num_ok(){ [[ $1 =~ ^[0-9]+(\.[0-9]+)?$ ]]; }
# read_argv <file>: the lines of an argv file into ARGV (bash 3.2 has no mapfile).
read_argv(){ ARGV=(); local l; while IFS= read -r l; do ARGV+=("$l"); done < "$1"; }

# ---------- 0: the lock ----------
LOCKFILE=$EXPERIMENT_LOCK
PINNED=$O/experiment-lock.verified.json
say "PILOT start pid=$$ lock=$EXPERIMENT_LOCK max_hours=$MAX_HOURS"
num_ok "$MAX_HOURS" || die "MAX_HOURS=$MAX_HOURS is not a number"
CURRENT=verify
verify_lock(){
  lockpy verify --corpus "P=$CORPUS_P" --corpus "E=$CORPUS_E" --splits-dir "$SPLITS_DIR" --repo "$Q"
}
verify_lock > "$O/verify.out" || die "the experiment lock does not match the artifacts; nothing was started"
LOCK_SHA=$(sed -n 's/^lock sha256: //p' "$O/verify.out")
[ -n "$LOCK_SHA" ] || die "verify printed no lock digest"
if [ -f "$PINNED" ]; then
  cmp -s "$PINNED" "$EXPERIMENT_LOCK" || die "this run directory was started under another lock
  ($PINNED). A changed or regenerated lock needs a fresh run directory."
else
  cp "$EXPERIMENT_LOCK" "$PINNED" || die "cannot keep a copy of the verified lock in $O"
fi
[ "$(sha "$PINNED")" = "$LOCK_SHA" ] || die "the copy of the verified lock is not the lock that was verified"
LOCKFILE=$PINNED
# hold_lock <what>: before a GPU step, the lock file is still the verified one and the code,
# corpora and splits still match it. Verified once and then trusted is not a lock.
# finals_digest: both arms' adapter directories, by the digest the judge records for them.
finals_digest(){ local a; for a in P E; do echo "$a $("$PY" "$LP" adapter-digest --final "$O/cpt_$a/final")"; done; }
hold_lock(){
  [ "$(sha "$EXPERIMENT_LOCK")" = "$LOCK_SHA" ] && [ "$(sha "$PINNED")" = "$LOCK_SHA" ] \
    || die "the experiment lock changed after it was verified; stopping before $1"
  # Once the pair is checked, an SFT cell is trained on, and evaluated with, those adapters.
  [ ! -f "$O/cpt_finals.sha256" ] || [ "$(finals_digest)" = "$(cat "$O/cpt_finals.sha256")" ] \
    || die "a CPT adapter is not the one the pair check passed on; stopping before $1"
  verify_lock > /dev/null || die "the code, corpora or splits no longer match the lock; stopping before $1"
}
VISIT_MAX=$(lockval spending.visit_max_hours) || die "the lock has no spending.visit_max_hours"
CPT_LOCK_MAX=$(lockval spending.cpt_stage_max_hours) || die "the lock has no spending.cpt_stage_max_hours"
awk "BEGIN{exit !($MAX_HOURS <= $VISIT_MAX)}" || die "MAX_HOURS=$MAX_HOURS exceeds the lock's
  visit_max_hours=$VISIT_MAX"
CPT_MAX_HOURS=${CPT_MAX_HOURS:-$CPT_LOCK_MAX}
num_ok "$CPT_MAX_HOURS" || die "CPT_MAX_HOURS=$CPT_MAX_HOURS is not a number"
awk "BEGIN{exit !($CPT_MAX_HOURS <= $CPT_LOCK_MAX)}" || die "CPT_MAX_HOURS=$CPT_MAX_HOURS exceeds
  the lock's cpt_stage_max_hours=$CPT_LOCK_MAX"
VISIT_END=$(( START + $(awk "BEGIN{printf \"%d\", $MAX_HOURS*3600}") ))
mark verify passed
CURRENT=""
say "0: lock verified, sha256 $LOCK_SHA; ceilings: visit ${MAX_HOURS}h, CPT ${CPT_MAX_HOURS}h"

# The watchdog handshake, before any GPU work: the deadline here stops an overrunning job, the
# watchdog stops an idle or wedged box, and neither substitutes for the other.
if [ "${REQUIRE_WATCHDOG:-0}" = 1 ]; then
  waited=0
  while [ ! -f "$O/watchdog.armed" ] && [ "$waited" -lt "${ARM_WAIT:-120}" ]; do sleep 1; waited=$((waited+1)); done
  [ -f "$O/watchdog.armed" ] || { say "ABORT: no watchdog handshake at $O/watchdog.armed after
  ${waited}s; refusing to start GPU work unwatched. Launch through launch_judge_first.sh."
    write_status no_watchdog "handshake absent"; exit 2; }
  say "0: watchdog handshake present after ${waited}s"
elif [ -z "$ALLOW_CPU" ] || command -v nvidia-smi >/dev/null 2>&1; then
  # REQUIRE_WATCHDOG=1 is set by the wrapper. Without it nothing is watching this box, and only
  # an explicit off-box dry run may go on: ALLOW_CPU=1 on a machine with no GPU. ALLOW_CPU does
  # not force the CPU, so on a GPU box it would be a way to train unwatched.
  say "ABORT: not started through scripts/training/v04_box/launch_judge_first.sh (REQUIRE_WATCHDOG is
  not 1), so no watchdog is armed; refusing to start GPU work."
  write_status no_watchdog "not launched through the wrapper"; exit 2
fi

# check_deadline <what> <epoch>: nothing new starts past a ceiling.
check_deadline(){
  say "  elapsed $(( $(used_s) / 60 )) min; next: $1"
  [ "$(date +%s)" -lt "$2" ] || die "ceiling reached before $1; stopping rather than overrunning"
}
# run_gpu <label> <log> <deadline epoch> <command...>: waited on with an active deadline, so a
# step that starts in time cannot run past the ceiling; on expiry its process group is
# terminated and the step returns 124.
run_gpu(){
  local lbl=$1 log=$2 dl=$3 rc=0 p waited
  shift 3
  ( export CUDA_VISIBLE_DEVICES=0; exec "$@" >> "$log" 2>&1 ) &
  p=$!
  register "$p"
  while kill -0 "$p" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$dl" ]; then
      say "$lbl: ceiling reached WHILE this step was running; terminating it"
      signal_tree "$p" TERM
      waited=0
      while [ "$waited" -lt "$SHUTDOWN_GRACE" ] && kill -0 "$p" 2>/dev/null; do sleep 1; waited=$((waited+1)); done
      signal_tree "$p" KILL
      wait "$p" 2>/dev/null || true
      unregister "$p"
      say "$lbl rc=124 (terminated at the ceiling)"
      return 124
    fi
    sleep "$STEP_POLL"
  done
  wait "$p" || rc=$?
  unregister "$p"
  say "$lbl rc=$rc"
  return $rc
}
# gpu_or_die <label> <log> <deadline> <command...>: run_gpu, with the two ways it can end badly
# told apart in status.json.
gpu_or_die(){
  local rc=0
  run_gpu "$@" || rc=$?
  [ "$rc" -ne 124 ] || die "$1: stopped at the ceiling while it was running"
  [ "$rc" -eq 0 ] || die "$1 failed (rc=$rc; log $2)"
}
nonempty(){ [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

# ---------- 1-2: CPT ----------
# cpt_arm <arm> <version dir>
cpt_arm(){
  local arm=$1 corpus=$2 dir=$O/cpt_$1
  CURRENT=cpt_$arm
  if [ -f "$dir/final/trainer_state.json" ]; then
    lockpy check-cpt --arm "$arm" --final "$dir/final" || die "cpt_$arm: $dir/final exists and
    fails check-cpt. It is not reused, overwritten or trained on; move it aside to re-run."
    mark "cpt_$arm" reused; CURRENT=""; say "cpt_$arm: reused, check-cpt passes"; return 0
  fi
  nonempty "$dir" && die "cpt_$arm: partial output in $dir with no final/ state. It is not
  overwritten or resumed; move it aside to re-run the arm."
  lockpy cpt-args --arm "$arm" --dataset-path "$corpus/dataset" --output-dir "$dir" \
    --canonical-eval-path "$CANONICAL_EVAL" --benchmark-sha256 "$O/benchmark_split_sha256.json" \
    > "$O/cpt_$arm.argv" || die "cpt_$arm: the lock refused to derive its arguments"
  read_argv "$O/cpt_$arm.argv"
  hold_lock "CPT $arm"
  check_deadline "CPT $arm" "$CPT_END"
  gpu_or_die "cpt $arm" "$O/cpt_$arm.log" "$CPT_END" "$PY" "$TRAINER" "${ARGV[@]}"
  [ -f "$dir/final/trainer_state.json" ] || die "cpt_$arm: the trainer exited 0 with no final state"
  lockpy check-cpt --arm "$arm" --final "$dir/final" || die "cpt_$arm: the run does not match the lock"
  mark "cpt_$arm" ran; CURRENT=""
}
CPT_END=$(( $(date +%s) + $(awk "BEGIN{printf \"%d\", $CPT_MAX_HOURS*3600}") ))
[ "$CPT_END" -le "$VISIT_END" ] || CPT_END=$VISIT_END
cpt_arm P "$CORPUS_P"
cpt_arm E "$CORPUS_E"
CURRENT=pair
lockpy check-pair --final-p "$O/cpt_P/final" --final-e "$O/cpt_E/final" \
  || die "the two CPT arms are not a pair (initial LoRA state or masked targets); no SFT is run"
# Written once, by the first launch whose pair check passes; hold_lock compares against it
# before every later GPU step, in this launch and in any resumed one.
[ -f "$O/cpt_finals.sha256" ] || finals_digest > "$O/cpt_finals.sha256"
mark pair passed; CURRENT=""

# ---------- 3: SFT cells ----------
# sft_cell <cell> <adapter>
sft_cell(){
  local cell=$1 adapter=$2 dir=$O/sft_$1 final=()
  [ "$adapter" = none ] || final=(--final "$adapter")     # binds the cell to its arm's adapter
  CURRENT=sft_$cell
  if [ -f "$dir/eval.json" ]; then
    lockpy check-eval --eval "$dir/eval.json" --arm "$cell" ${final[@]+"${final[@]}"} || die "sft_$cell: $dir/eval.json
    exists and fails check-eval. It is not reused or overwritten; move it aside to re-run."
    mark "sft_$cell" reused; CURRENT=""; say "sft_$cell: reused, check-eval passes"; return 0
  fi
  # `judge train` writes best.pt after any improving epoch, so only train.complete, written here
  # after a zero exit, lets an interrupted evaluation skip retraining.
  if [ -f "$dir/train.complete" ] && [ -f "$dir/best.pt" ]; then
    say "sft_$cell: training completed in an earlier launch; evaluating only"
  else
    nonempty "$dir" && die "sft_$cell: partial training output in $dir with no train.complete.
    It is not trained on or overwritten; move it aside to re-run the cell."
    lockpy sft-args --stage train --adapter "$adapter" --out-dir "$dir" > "$O/sft_$cell.train.argv" \
      || die "sft_$cell: the lock refused to derive its training arguments"
    read_argv "$O/sft_$cell.train.argv"
    hold_lock "SFT $cell training"
    check_deadline "SFT $cell training" "$VISIT_END"
    gpu_or_die "sft $cell train" "$O/sft_$cell.train.log" "$VISIT_END" \
      "$PY" -m track_a.esm_sft.judge "${ARGV[@]}" $CPU_JUDGE
    [ -f "$dir/best.pt" ] || die "sft_$cell: training exited 0 with no $dir/best.pt"
    date -u '+%F %T' > "$dir/train.complete"
  fi
  lockpy sft-args --stage eval --adapter "$adapter" --ckpt "$dir/best.pt" --out "$dir/eval.json" \
    > "$O/sft_$cell.eval.argv" || die "sft_$cell: the lock refused to derive its eval arguments"
  read_argv "$O/sft_$cell.eval.argv"
  hold_lock "SFT $cell evaluation"
  check_deadline "SFT $cell evaluation" "$VISIT_END"
  gpu_or_die "sft $cell eval" "$O/sft_$cell.eval.log" "$VISIT_END" \
    "$PY" -m track_a.esm_sft.judge "${ARGV[@]}" $CPU_JUDGE
  [ -f "$dir/eval.json" ] || die "sft_$cell: evaluation exited 0 with no eval.json"
  lockpy check-eval --eval "$dir/eval.json" --arm "$cell" ${final[@]+"${final[@]}"} \
    || die "sft_$cell: the evaluation does not match the lock"
  mark "sft_$cell" ran; CURRENT=""
}
sft_cell B none
sft_cell P "$O/cpt_P/final"
sft_cell E "$O/cpt_E/final"

# ---------- 4: the read-out ----------
CURRENT=compare
# pilot.json is the read-out OF three eval files. Their digests are written beside it, and it
# is reused only for exactly those files: a cell re-run since would otherwise be reported
# under numbers it did not produce.
cells_digest(){ local c; for c in B P E; do echo "$c $(sha "$O/sft_$c/eval.json")"; done; }
if [ -f "$O/pilot.json" ]; then
  [ "$(cells_digest)" = "$(cat "$O/pilot.cells.sha256" 2>/dev/null)" ] || die "$O/pilot.json was
  not made from the eval files now in this run directory. It is not reused or overwritten; move it
  aside to compare the current cells."
  say "4: $O/pilot.json exists and was made from these eval files; reused"
  mark compare reused
else
  TASK=$(lockval sft.task) || die "the lock has no sft.task"
  CONTRASTS="$(lockval primary_contrast) $(lockval secondary_contrasts)" || die "the lock has no contrasts"
  NBOOT=$(lockval evaluation.comparator.n_boot) || die "the lock has no comparator n_boot"
  CSEED=$(lockval evaluation.comparator.seed) || die "the lock has no comparator seed"
  "$PY" "$CMP" --cells "B=$O/sft_B/eval.json" "P=$O/sft_P/eval.json" "E=$O/sft_E/eval.json" \
    --val-split "$SPLITS_DIR/${TASK}_val.parquet" --contrasts $CONTRASTS \
    --n-boot "$NBOOT" --seed "$CSEED" --out "$O/pilot.json" 2>&1 | tee -a "$O/run.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] && [ -f "$O/pilot.json" ] || die "pilot_compare produced no read-out"
  cells_digest > "$O/pilot.cells.sha256"
  mark compare ran
fi
CURRENT=""
PILOT_PRODUCED=1
write_status complete "validation triage on one CPT seed and one SFT seed"
say "PILOT complete in $(( $(used_s) / 60 )) min. Read-out: $O/pilot.json"
say "  Triage only: validation selected each checkpoint, so the numbers are in-sample."
