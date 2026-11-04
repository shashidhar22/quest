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
S42=$Q/scripts/training/stage42_box/s42_check.py
CMP=$Q/scripts/training/stage42_box/paired_compare.py
POOL=${POOL_FILE:-$Q/benchmark_v2/splits/as_class_i_candidate_pool.tsv}
ADAPTER_D=${ADAPTER_D:-$Q/checkpoints/cpt_stage41_pointD_run05/final}
BASE_SRC=${BASE_CKPT:-/home/ubuntu/s42_out/base_paired/best.pt}
MAX_HOURS=${MAX_HOURS:-8}
SHUTDOWN_GRACE=${SHUTDOWN_GRACE:-15}
SMOKE=${SMOKE:-}
export PYTHONPATH=$Q PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p $O; cd $Q
say(){ echo "[$(date -u '+%F %T')] $*" | tee -a $O/run.log; }
echo $$ > $O/judge_first.pid     # the watchdog waits on THIS pid, never a name pattern
START=$(date +%s)

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
# Checked before every GPU step, so the ceiling binds between steps rather than only at the end.
check_ceiling(){
  local used=$(( ($(date +%s) - START) / 60 ))
  local cap=$(( MAX_HOURS * 60 ))
  say "  elapsed ${used} min of ${cap} min ceiling"
  [ "$used" -lt "$cap" ] || die "MAX_HOURS=$MAX_HOURS ceiling reached before $1; stopping rather than overrunning"
}
# run_gpu <label> <logfile> <command...> -- tracked, so a signal is not deferred behind it
run_gpu(){
  local lbl=$1 log=$2 rc=0 p
  shift 2
  ( export CUDA_VISIBLE_DEVICES=0; exec "$@" > "$log" 2>&1 ) &
  p=$!
  register "$p" "${lbl// /_}"
  wait "$p" || rc=$?
  unregister "$p"
  say "$lbl rc=$rc"
  return $rc
}

say "JUDGE-FIRST start pid=$$ draw=$DRAW rho_floor=$RHO_FLOOR max_hours=$MAX_HOURS adapter=$ADAPTER_D"

# ---------- P0: preflight ----------
say "P0: preflight"
bad=0
POOL_ID=$([ -f "$POOL" ] && $PY $PROV fingerprint --path "$POOL" || echo POOL_FILE_MISSING)
[ "$POOL_ID" = POOL_FILE_MISSING ] && { say "  preflight: candidate pool not found at $POOL"; bad=1; }
[ -d "$ADAPTER_D" ] || { say "  preflight: arm D adapter not found at $ADAPTER_D"; bad=1; }
for p in "$UNITS" "$PROV" "$S42" "$CMP"; do
  [ -f "$p" ] || { say "  preflight: missing file $p"; bad=1; }
done
[ -f "$O/base_paired/best.pt" ] || [ -f "$BASE_SRC" ] || {
  say "  preflight: no base SFT checkpoint to re-score (looked for $BASE_SRC)"; bad=1; }
[ "$bad" -eq 0 ] || { say "ABORT: preflight failed; nothing was started"; exit 2; }
say "P0 ok  pool_id=$POOL_ID"

# ---------- P1: the probe ----------
# On 2026-10-07 every candidate tied and a cell burned 18 GPU-hours before the post-hoc gate
# noticed. The probe costs about a minute and would have caught it from the start.
check_ceiling "the probe"
for arm in none "$ADAPTER_D"; do
  tag=$([ "$arm" = none ] && echo base || echo armD)
  run_gpu "probe $tag" "$O/probe_$tag.log" \
    $PY -m track_a.esm_sft.judge probe --task as_paired_i --adapter "$arm" \
      --tcr_variant cdr3 --mhc_variant full \
    || die "the probe failed for $tag: the scorer's output does not depend on its input, so nothing below is worth running"
  grep -h '^\[probe\]' "$O/probe_$tag.log" | tail -1 | sed 's/^/    /' | tee -a $O/run.log
done

# ---------- P2: re-score the existing base checkpoint ----------
# Eval only, no retraining. The 10-08 number was scored against the uniform draw and is not
# comparable to anything produced under the length-matched one.
check_ceiling "the base re-score"
BASE=$O/base_paired; mkdir -p $BASE
[ -f $BASE/best.pt ] || cp "$BASE_SRC" $BASE/best.pt 2>/dev/null
[ -f $BASE/best.pt ] || die "the base SFT checkpoint vanished between preflight and use"
$PY $PROV write --out $BASE/manifest.json --objective legacy_row_share_cpt --token-budget n/a \
  --bucket-boundaries n/a --steps n/a --early-stop n/a --draw $DRAW --pool-size 100 \
  --pool-id $POOL_ID --dataset as_paired_i --seed 42 --cotenants none >/dev/null \
  || die "could not write the base manifest"
run_gpu "P2 base re-score" "$BASE/eval.log" \
  $PY -m track_a.esm_sft.judge eval --task as_paired_i --adapter none \
    --tcr_variant cdr3 --mhc_variant full --ckpt $BASE/best.pt --out $BASE/eval.json \
    --conditions test_level4 test_novel_pep test_iid --max_queries $MAXQ --min_peptides $MINP \
    --distractor_draw $DRAW $SMOKE \
  || die "the base re-score failed; the gate has nothing to read"

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
$PY $PROV write --out $CELL/manifest.json --objective legacy_row_share_cpt --token-budget n/a \
  --bucket-boundaries n/a --steps n/a --early-stop n/a --draw $DRAW --pool-size 100 \
  --pool-id $POOL_ID --dataset as_paired_i --seed 42 --cotenants none >/dev/null \
  || die "could not write the arm D manifest"
# Train-then-evaluate, the pattern from stage42_box/wave1.sh. The cell is skipped if an eval
# already exists, so a resumed visit does not pay twice.
if [ -f $CELL/eval.json ]; then
  say "P3: armD_paired already has an eval, skipping"
else
  run_gpu "P3 armD train" "$CELL/train.log" \
    $PY -m track_a.esm_sft.judge train --task as_paired_i --adapter "$ADAPTER_D" \
      --tcr_variant cdr3 --mhc_variant full --out_dir $CELL $SMOKE \
    || die "the arm D SFT cell failed to train"
  [ -f $CELL/best.pt ] || die "training reported success but produced no $CELL/best.pt"
  check_ceiling "the arm D eval"
  run_gpu "P3 armD eval" "$CELL/eval.log" \
    $PY -m track_a.esm_sft.judge eval --task as_paired_i --adapter "$ADAPTER_D" \
      --tcr_variant cdr3 --mhc_variant full --ckpt $CELL/best.pt --out $CELL/eval.json \
      --conditions test_level4 test_novel_pep test_iid --max_queries $MAXQ --min_peptides $MINP \
      --distractor_draw $DRAW $SMOKE \
    || die "the arm D evaluation failed"
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
