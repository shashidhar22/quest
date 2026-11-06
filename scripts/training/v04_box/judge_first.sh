#!/bin/bash
# JUDGE-FIRST: the smallest run that answers the first missing comparison.
#
# Decision 2 of 2026-11-04, approved by Yuexin. V04 as specified bought six CPT re-baselines and
# four judge cells before any correct comparison had ever been produced. This runs the cheap half
# only, in the order that lets the cheap part refuse the expensive part:
#
#   P0  CPU     preflight: paths, checkpoints, config, adapter
#   P1  GPU 0   probe both arms — the scorer's output must vary with the peptide
#   P2  GPU 0   re-score the EXISTING base SFT checkpoint under the length-matched draw (eval only)
#   G1  CPU     locked-read readability gate on the NEW-draw ranks
#   P3  GPU 0   train + evaluate ONE paired legacy arm-D SFT cell, only on a G1 pass
#   C1  CPU     the paired comparison on the locked primary read
#
# WHAT THIS DOES NOT DO, by instruction: no CPT training, no arm A or arm D re-baseline, no
# beta-chain cells, no 100M scaling. It evaluates an EXISTING adapter, so it needs no decision
# about the CPT loss objective. A result here applies to that legacy recipe: a negative does not
# reject all CPT objectives, and a positive does not validate the newly adopted one. Controls
# under the new objective must be judge-scored on their own.
#
# One GPU is enough, so this needs NO RESIZE: the box is already g6.2xlarge. That is most of why
# it is cheap.
#
# CEILING. MAX_HOURS is enforced here as well as by the watchdog, because the watchdog's job is to
# stop an idle box and this one's job is to stop an overrunning job. On expiry the run terminates
# its children through their process groups and exits nonzero, leaving no success artifact.
#
# Lifecycle, failure propagation and cleanup follow run_v04.sh, including why: trainers are started
# from this shell and `exec`ed so the tracked PID is the real process, `set -m` gives each job its
# own process group so cleanup reaches torchrun's descendants, and every child status is collected
# rather than swallowed by a following `say`.
set -u -m
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
Q=${Q:-/home/ubuntu/quest}; O=${O:-/home/ubuntu/jf_out}
PY=${PY:-python}
DRAW=${DISTRACTOR_DRAW:-length_matched}
RHO_FLOOR=${RHO_FLOOR:-0.8}
PREREG=${PREREG_DELTA:-0.05}
MAXQ=${MAXQ:-5000}
MINP=${MINP:-100}
UNITS=${UNITS:-$Q/tests/fixtures/s42_as_paired_i_units.csv}
PROV=$Q/scripts/training/v04_box/provenance.py
PRE=$Q/scripts/training/v04_box/preflight.py
# What must import on the box. Overridable so the off-box tests can exercise everything else.
PREFLIGHT_REQUIRE=${PREFLIGHT_REQUIRE-torch,peft,transformers}
# ALLOW_CPU=1 drops the CUDA requirement. Unset on a GPU visit, always: an evaluation that
# silently fell back to the CPU is a result nobody chose to buy (audit finding 5, 2026-11-06).
ALLOW_CPU=${ALLOW_CPU:-}
S42=$Q/scripts/training/stage42_box/s42_check.py
CMP=$Q/scripts/training/stage42_box/paired_compare.py
POOL=${POOL_FILE:-$Q/benchmark_v2/splits/as_class_i_candidate_pool.tsv}
# What the JUDGE optimises. It is not a CPT objective, and the two must never share a field:
# both manifests used to say `objective=legacy_row_share_cpt`, which the base arm never had and
# arm D never trained under either (its trainer_state.json records the fixed-batch path).
SFT_OBJECTIVE=${SFT_OBJECTIVE:-bce-pair-logits/v1}
CONDITIONS=${CONDITIONS:-"test_level4 test_novel_pep test_iid"}
ADAPTER_D=${ADAPTER_D:-$Q/checkpoints/cpt_stage41_pointD_run05/final}
BASE_SRC=${BASE_CKPT:-/home/ubuntu/s42_out/base_paired/best.pt}
MAX_HOURS=${MAX_HOURS:-8}
SHUTDOWN_GRACE=${SHUTDOWN_GRACE:-15}
POLL=${POLL:-1}                  # how often a running child is checked against the deadline;
                                 # also the worst-case signal latency, since bash defers a trap
                                 # behind a foreground sleep. One second per phase is free.
SMOKE=${SMOKE:-}
export PYTHONPATH=$Q PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p $O; cd $Q
say(){ echo "[$(date -u '+%F %T')] $*" | tee -a $O/run.log; }
echo $$ > $O/judge_first.pid     # the watchdog waits on THIS pid, never a name pattern
START=$(date +%s)
# In SECONDS, via awk, so a fractional MAX_HOURS works and the deadline is testable in one run.
# Integer shell arithmetic on hours could only ever express a ceiling no test could reach.
# An unparseable or absurd ceiling is REFUSED rather than defaulted: a cost ceiling that quietly
# becomes "no ceiling" on a typo is the failure this whole guard exists to prevent. MAX_HOURS=0
# stays legal and means a ceiling already exceeded, so nothing is spent.
awk "BEGIN{exit !($MAX_HOURS >= 0 && $MAX_HOURS < 240)}" 2>/dev/null || {
  echo "ABORT: MAX_HOURS=$MAX_HOURS is not a usable ceiling (want 0 <= hours < 240)" >&2; exit 2; }
MAX_S=$(awk "BEGIN{printf \"%d\", $MAX_HOURS*3600}")

JOBS=""
FAILED=0
register(){ JOBS="$JOBS $1=$2"; }
unregister(){ local keep="" t; for t in $JOBS; do [ "${t%%=*}" = "$1" ] || keep="$keep $t"; done; JOBS=$keep; }
signal_tree(){ kill -"$2" -"$1" 2>/dev/null || kill -"$2" "$1" 2>/dev/null || true; }
cleanup(){
  local t p waited=0 still
  [ -n "${JOBS// /}" ] || return 0
  say "cleanup: terminating$JOBS"
  for t in $JOBS; do signal_tree "${t%%=*}" TERM; done
  while [ "$waited" -lt "$SHUTDOWN_GRACE" ]; do
    still=0
    for t in $JOBS; do p=${t%%=*}; kill -0 "$p" 2>/dev/null && still=1; done
    [ "$still" -eq 0 ] && break
    sleep 1; waited=$((waited+1))
  done
  for t in $JOBS; do signal_tree "${t%%=*}" KILL; done
  for t in $JOBS; do wait "${t%%=*}" 2>/dev/null || true; done
  say "cleanup: children gone after ${waited}s"
  JOBS=""
}
trap 'say "signal received: cleaning up before exit"; cleanup; say "JUDGE-FIRST TERMINATED"; exit 130' INT TERM
die(){
  say "ABORT: $*"
  FAILED=1
  cleanup
  say "JUDGE-FIRST FAILED"
  exit 1
}
used_s(){ echo $(( $(date +%s) - START )); }
expired(){ [ "$(used_s)" -ge "$MAX_S" ]; }
# Checked before every GPU step, so nothing new starts past the ceiling.
check_ceiling(){
  say "  elapsed $(( $(used_s) / 60 )) min of $(( MAX_S / 60 )) min ceiling"
  ! expired || die "MAX_HOURS=$MAX_HOURS ceiling reached before $1; stopping rather than overrunning"
}
# run_gpu <label> <logfile> <command...>
#
# Tracked, so a signal is not deferred behind it, and waited on with an ACTIVE DEADLINE rather
# than a bare `wait`. Audit finding 3 of 2026-11-06: checking the ceiling only between phases
# meant a phase starting at hour 7:59 could run for days, and the visit record's claim that
# expiry terminates children from inside the launcher was untrue. On expiry the child is
# terminated through its process group, the step returns 124, and `die` stops the run.
#
# The cost of the poll loop is bounded signal latency: a trap now fires up to POLL seconds late
# instead of immediately, because bash defers traps behind a foreground `sleep`. `wait -n` would
# avoid that, but it needs bash 4.3 and the laptop that runs these tests has 3.2, so the deadline
# would then be the untested path. A bounded, uniform latency is the better trade.
run_gpu(){
  local lbl=$1 log=$2 rc=0 p
  shift 2
  ( export CUDA_VISIBLE_DEVICES=0; exec "$@" > "$log" 2>&1 ) &
  p=$!
  register "$p" "${lbl// /_}"
  while kill -0 "$p" 2>/dev/null; do
    if expired; then
      say "$lbl: MAX_HOURS=$MAX_HOURS ceiling reached WHILE this step was running; terminating it"
      signal_tree "$p" TERM
      local waited=0
      while [ "$waited" -lt "$SHUTDOWN_GRACE" ] && kill -0 "$p" 2>/dev/null; do
        sleep 1; waited=$((waited+1))
      done
      signal_tree "$p" KILL
      wait "$p" 2>/dev/null || true
      unregister "$p"
      say "$lbl rc=124 (terminated at the ceiling after ${waited}s grace)"
      return 124
    fi
    sleep "$POLL"
  done
  wait "$p" || rc=$?
  unregister "$p"
  say "$lbl rc=$rc"
  return $rc
}

# write_manifest <cell dir> <adapter> <label> -- derived from the cell's own evaluation.
write_manifest(){
  $PY $PROV write --out "$1/manifest.json" --from-eval "$1/eval.json" \
    --sft-objective "$SFT_OBJECTIVE" --cotenants none 2>&1 | tee -a $O/run.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "$3: its evaluation cannot be certified, so no manifest
  was written. A result whose inputs cannot be established is not a result."
}
# reuse_or_die <cell dir> <adapter> <label> -- an existing evaluation may be reused only if it
# is the same result: its checkpoint is present, its own provenance is present and unaltered,
# and every artifact it recorded still hashes the same. The audit planted a stale eval with no
# checkpoint and a manifest naming another pool; the launcher overwrote the manifest, skipped
# training and produced a comparison. Each of the four checks below refuses that scenario alone.
reuse_or_die(){
  local cell=$1 adapter=$2 lbl=$3
  say "P3: $cell already has an eval; validating before reusing it"
  [ -f "$cell/best.pt" ] || die "$lbl: an evaluation exists at $cell/eval.json but the
  checkpoint that would have produced it does not. Nothing may be certified from it."
  [ -f "$cell/manifest.json" ] || die "$lbl: an evaluation exists with no manifest beside it, so
  its provenance would have to be invented now. Delete $cell to re-run it honestly."
  $PY $PROV verify-eval --eval "$cell/eval.json" --ckpt "$cell/best.pt" --adapter "$adapter" \
    2>&1 | tee -a $O/run.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "$lbl: the artifacts behind the existing evaluation have
  changed, so reusing it would compare numbers produced from different inputs."
  $PY $PROV write --out "$cell/.manifest.derived.json" --from-eval "$cell/eval.json" \
    --sft-objective "$SFT_OBJECTIVE" --cotenants none >/dev/null \
    || die "$lbl: the existing evaluation cannot be certified."
  $PY $PROV check --a "$cell/manifest.json" --b "$cell/.manifest.derived.json" --kind strict \
    2>&1 | tee -a $O/run.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "$lbl: the stored manifest does not describe the
  evaluation beside it. The manifest is NOT being overwritten; resolve which one is wrong."
  say "P3: reuse validated — checkpoint present, provenance unaltered, inputs unchanged"
}

say "JUDGE-FIRST start pid=$$ draw=$DRAW rho_floor=$RHO_FLOOR max_hours=$MAX_HOURS adapter=$ADAPTER_D"

# ---------- P0: preflight ----------
say "P0: preflight"
bad=0
# Hash the pool the SPLIT LOADER resolves, not a path named here: the launcher used to hash
# $POOL while the evaluator read load_candidate_pool(), so a manifest could record the digest of
# a file nothing opened (audit finding 1, 2026-11-06).
POOL_ID=$($PY $PROV fingerprint --pool 2>/dev/null || echo MISSING)
[ "$POOL_ID" = MISSING ] && POOL_ID=POOL_FILE_MISSING
[ "$POOL_ID" = POOL_FILE_MISSING ] && { say "  preflight: candidate pool not found at $POOL"; bad=1; }
[ -d "$ADAPTER_D" ] || { say "  preflight: arm D adapter not found at $ADAPTER_D"; bad=1; }
for p in "$UNITS" "$PROV" "$S42" "$CMP" "$PRE"; do
  [ -f "$p" ] || { say "  preflight: missing file $p"; bad=1; }
done
[ -f "$O/base_paired/best.pt" ] || [ -f "$BASE_SRC" ] || {
  say "  preflight: no base SFT checkpoint to re-score (looked for $BASE_SRC)"; bad=1; }
[ "$bad" -eq 0 ] || { say "ABORT: preflight failed; nothing was started"; exit 2; }

# The real preflight: imports, CUDA, the pool, EVERY split file including train and val, the
# adapter's contents and the base checkpoint's loadability. Until 2026-11-06 this phase tested
# that paths existed, so a missing train split surfaced after the base evaluation was paid for
# and a driverless box evaluated on the CPU without a word.
CUDA_FLAG=$([ -n "$ALLOW_CPU" ] && echo "" || echo "--require-cuda")
# Kept consistent: if this run is allowed on the CPU, the judge must be allowed too, or the
# preflight would pass and every judge call would then refuse.
CPU_JUDGE=$([ -n "$ALLOW_CPU" ] && echo "--allow_cpu" || echo "")
$PY $PRE --task as_paired_i --conditions train val $CONDITIONS \
  --adapter "$ADAPTER_D" --ckpt "$([ -f $O/base_paired/best.pt ] && echo $O/base_paired/best.pt || echo $BASE_SRC)" \
  --require "$PREFLIGHT_REQUIRE" $CUDA_FLAG --out $O/preflight.json 2>&1 | tail -40 | tee -a $O/run.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "ABORT: the environment preflight failed; nothing was
  started. Its report is in $O/preflight.json."; exit 2; }
say "P0 environment preflight ok; report in $O/preflight.json"
# The watchdog handshake. REQUIRE_WATCHDOG=1 (set by launch_judge_first.sh) holds the run here
# until that script has verified, from the watchdog's own log, that it armed on THIS pid. The
# ceiling in this launcher stops an overrunning job; the watchdog stops an idle or wedged box,
# including one this launcher leaves behind by dying. Neither substitutes for the other, so a run
# that needs both must not start GPU work having only one (audit finding 3, 2026-11-06).
if [ "${REQUIRE_WATCHDOG:-0}" = 1 ]; then
  ARMED=$O/watchdog.armed
  waited=0
  while [ ! -f "$ARMED" ] && [ "$waited" -lt "${ARM_WAIT:-120}" ]; do sleep 1; waited=$((waited+1)); done
  [ -f "$ARMED" ] || { say "ABORT: no watchdog handshake at $ARMED after ${waited}s; refusing to
  start GPU work unwatched. Launch through scripts/training/v04_box/launch_judge_first.sh."; exit 2; }
  say "P0 watchdog handshake present after ${waited}s: $ARMED"
fi
say "P0 ok  pool_id=$POOL_ID"

# ---------- P1: the probe ----------
# On 2026-10-07 every candidate tied and a cell burned 18 GPU-hours before the post-hoc gate
# noticed. The probe costs about a minute and would have caught it from the start.
check_ceiling "the probe"
# The base probe loads the ACTUAL base checkpoint. Audit finding 2: a probe of a freshly
# built model vouches for no checkpoint, and the base checkpoint is the one artifact here that
# nothing else in this run validates before it is read for the paid evaluation.
BASE=$O/base_paired; mkdir -p $BASE
[ -f $BASE/best.pt ] || cp "$BASE_SRC" $BASE/best.pt 2>/dev/null
[ -f $BASE/best.pt ] || die "the base SFT checkpoint vanished between preflight and use"
for arm in none "$ADAPTER_D"; do
  tag=$([ "$arm" = none ] && echo base || echo armD)
  ckpt_flag=$([ "$arm" = none ] && echo "--ckpt $BASE/best.pt" || echo "")
  run_gpu "probe $tag" "$O/probe_$tag.log" \
    $PY -m track_a.esm_sft.judge probe --task as_paired_i --adapter "$arm" $ckpt_flag \
      --tcr_variant cdr3 --mhc_variant full $CPU_JUDGE \
    || die "the probe failed for $tag: the scorer's output does not depend on its input, so nothing below is worth running"
  grep -h '^\[probe\]' "$O/probe_$tag.log" | tail -1 | sed 's/^/    /' | tee -a $O/run.log
done

# ---------- P2: re-score the existing base checkpoint ----------
# Eval only, no retraining. The 10-08 number was scored against the uniform draw and is not
# comparable to anything produced under the length-matched one.
check_ceiling "the base re-score"
run_gpu "P2 base re-score" "$BASE/eval.log" \
  $PY -m track_a.esm_sft.judge eval --task as_paired_i --adapter none \
    --tcr_variant cdr3 --mhc_variant full --ckpt $BASE/best.pt --out $BASE/eval.json \
    --conditions $CONDITIONS --max_queries $MAXQ --min_peptides $MINP \
    --distractor_draw $DRAW $SMOKE $CPU_JUDGE \
  || die "the base re-score failed; the gate has nothing to read"
# The manifest is written AFTER the evaluation and DERIVED FROM IT. Writing it first, as this
# did until 2026-11-06, meant the manifest described an intention; derived from the eval it
# describes the pool, splits, checkpoint and adapter the evaluation actually read.
write_manifest $BASE none "the base re-score"

# ---------- G1: the gate ----------
say "G1: locked-read readability gate on the NEW-draw ranks"
$PY $S42 gate --eval $BASE/eval.json --units $UNITS --prereg-delta $PREREG \
  --rho-floor $RHO_FLOOR --primary-condition test_novel_pep --primary-unit peptide \
  --primary-metric norm_rank 2>&1 | tee -a $O/run.log
G1=${PIPESTATUS[0]}
say "G1 exit=$G1"
if [ "$G1" -ge 2 ]; then
  die "the gate reported a TECHNICAL error (exit $G1): missing or invalid evidence, not a refusal.
  The arm-D cell is not being trained. Fix the evidence and re-run."
fi
if [ "$G1" -ne 0 ]; then
  say "G1 REFUSED (exit 1): the locked primary read cannot resolve the pre-registered effect under"
  say "  the $DRAW draw, so no base-versus-CPT number from this task would be interpretable."
  say "  The arm-D cell is NOT being trained. This is the pre-agreed refusal, and it is the"
  say "  cheap half of the run declining to buy the expensive half."
  say "JUDGE-FIRST complete: refused before the paid step. Base re-score is in $BASE/eval.json."
  exit 0
fi

# ---------- P3: the one paired arm-D cell ----------
check_ceiling "the arm D cell"
CELL=$O/armD_paired; mkdir -p $CELL
# Train-then-evaluate, the pattern from stage42_box/wave1.sh. A resumed visit does not pay
# twice -- but reuse is VALIDATED first, and the manifest is written from the evaluation rather
# than over it.
if [ -f $CELL/eval.json ]; then
  reuse_or_die $CELL "$ADAPTER_D" "the arm D cell"
else
  run_gpu "P3 armD train" "$CELL/train.log" \
    $PY -m track_a.esm_sft.judge train --task as_paired_i --adapter "$ADAPTER_D" \
      --tcr_variant cdr3 --mhc_variant full --out_dir $CELL $SMOKE $CPU_JUDGE \
    || die "the arm D SFT cell failed to train"
  [ -f $CELL/best.pt ] || die "training reported success but produced no $CELL/best.pt"
  check_ceiling "the arm D eval"
  run_gpu "P3 armD eval" "$CELL/eval.log" \
    $PY -m track_a.esm_sft.judge eval --task as_paired_i --adapter "$ADAPTER_D" \
      --tcr_variant cdr3 --mhc_variant full --ckpt $CELL/best.pt --out $CELL/eval.json \
      --conditions $CONDITIONS --max_queries $MAXQ --min_peptides $MINP \
      --distractor_draw $DRAW $SMOKE $CPU_JUDGE \
    || die "the arm D evaluation failed"
  write_manifest $CELL "$ADAPTER_D" "the arm D cell"
fi

# ---------- C1: the comparison ----------
# `cells` waives co-tenancy and nothing else, and prints the waiver it used. Here both cells run
# alone on one GPU, so there is nothing to waive and the check should report no tolerated
# difference at all.
$PY $PROV check --a $BASE/manifest.json --b $CELL/manifest.json --kind cells 2>&1 | tee -a $O/run.log
[ "${PIPESTATUS[0]}" -eq 0 ] || die "provenance refused the base-versus-armD comparison"
say "C1: paired comparison on the locked primary read"
$PY $CMP --a $BASE/eval.json --b $CELL/eval.json --units $UNITS --out $O/comparison.json \
  --primary-condition test_novel_pep --primary-unit peptide --primary-metric norm_rank \
  2>&1 | tee -a $O/run.log
CMP_RC=${PIPESTATUS[0]}
say "C1 exit=$CMP_RC"
[ "$CMP_RC" -eq 0 ] && [ -f $O/comparison.json ] \
  || die "the comparison could not be produced; this run has no usable result"

say "JUDGE-FIRST complete in $(( ($(date +%s) - START) / 60 )) min. Result: $O/comparison.json"
say "  Scope: this compares the base SFT checkpoint against the LEGACY Stage-4.1 arm-D adapter."
say "  It says nothing about the newly adopted CPT objective, which has no judge-scored arm yet."
