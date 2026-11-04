#!/bin/bash
# V04 on the one volume. Encodes the decision chain so no step can be skipped by hand.
#
#   P1  GPU 0,1,2  arm A re-baseline, one seed each, token-share, 7800 steps, NO early stop
#       GPU 3      cell 1 base judge re-score under the length-matched draw (eval only)
#   G1  CPU        locked-read readability gate on cell 1's NEW-draw per-peptide ranks
#   P2  GPU 0,1,2  arm D re-baseline, one seed each, same objective
#       GPU 3      judge cells 2-4  ONLY IF G1 PASSED
#
# Why the chain and not a checklist. Every readability number computed so far is power under the
# UNIFORM draw, and this visit runs the LENGTH-MATCHED draw, so the decision to spend the
# comparison cells rests on a pool that will not be used. G1 recomputes it against the real new
# ranks in seconds, because the predicate imports only math and random. Cells 2-4 therefore
# launch on a measurement, not an intention. box-jobs/README.md 3.3-3.4.
#
# Seeds of one arm share co-tenancy with each other by construction: all three run in the same
# phase on GPUs 0-2. A CROSS-arm band comparison does not, since GPU 3's occupant differs between
# P1 and P2; that asymmetry is recorded in each manifest rather than planned away, and the
# comparison kind `cells` waives co-tenancy and nothing else, printing the waiver it used.
#
# PROCESS OWNERSHIP (repaired 2026-11-04 after the external Stage-4 audit). Trainers are started
# from THIS shell with `&`, and `$!` is captured here. The previous version called train_arm through
# command substitution, so the PID it echoed belonged to the substitution subshell: every `wait`
# failed with "not a child of this shell", the driver reached "V04 complete" within seconds, and
# `autostop_when_done.sh` -- which waits on the driver PID written below -- would have stopped the
# instance on top of six live trainings. The driver outliving its children is what makes that
# watchdog safe, so no child may ever be started from a subshell again.
#
# Failure handling is explicit, not `set -e`: a failing trainer or evaluator used to be followed by
# a successful `say`, which swallowed its status and produced a visit that exited 0 with no result.
# Every child status is now collected, logged by label, and propagated. The driver still waits for
# all live children before exiting on an abort, so a failure never orphans a training job.
#
# Overridable so dryrun.sh can exercise the whole chain on a laptop with stubs.
set -u
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
Q=${Q:-/home/ubuntu/quest}; O=${O:-/home/ubuntu/v04_out}
PY=${PY:-python}; TORCHRUN=${TORCHRUN:-torchrun}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)}
LW=${LOSS_WEIGHTING:-token}          # objective adopted 2026-11-03; default `row` is rejected
DRAW=${DISTRACTOR_DRAW:-length_matched}
SEEDS=${SEEDS:-1 2 3}
STEPS=${STEPS:-7800}
RHO_FLOOR=${RHO_FLOOR:-0.8}
PREREG=${PREREG_DELTA:-0.05}
UNITS=${UNITS:-$Q/tests/fixtures/s42_as_paired_i_units.csv}
PROV=$Q/scripts/training/v04_box/provenance.py
S42=$Q/scripts/training/stage42_box/s42_check.py
POOL=${POOL_FILE:-$Q/benchmark_v2/splits/as_class_i_candidate_pool.tsv}
# Checkpoints go here. Separable so a test can never write into, or clean up, the real tree:
# dryrun.sh used to point $Q at the live checkout and then remove `checkpoints/v04_arm*_seed*`,
# which after a real visit is exactly the six trained adapters the visit was bought to produce.
CKPT_ROOT=${CKPT_ROOT:-$Q/checkpoints}
export PYTHONPATH=$Q PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p $O $CKPT_ROOT; cd $Q
say(){ echo "[$(date -u '+%F %T')] $*" | tee -a $O/run.log; }
echo $$ > $O/run_v04.pid       # the watchdog waits on THIS pid, never a name pattern

JOBS=""        # space-separated "<pid>=<label>" tokens for children of THIS shell
FAILED=0       # sticky: any child or cell failure makes the visit fail

kill_jobs(){ local t; for t in $JOBS; do kill "${t%%=*}" 2>/dev/null; done; }
trap 'say "signal received: terminating children"; kill_jobs; exit 130' INT TERM

# wait_all <phase>  -- waits for every child started since the last call. Never returns while a
# training job is alive, because the watchdog reads driver liveness as visit liveness.
wait_all(){
  local phase=$1 t p l rc=0 st
  for t in $JOBS; do
    p=${t%%=*}; l=${t#*=}
    if wait "$p"; then
      say "  $phase: $l ok"
    else
      st=$?
      say "  $phase: FAIL $l (pid $p) exited $st"
      rc=1
    fi
  done
  JOBS=""
  [ $rc -eq 0 ] || FAILED=1
  return $rc
}

# die <message>  -- abort, but never leave a training job orphaned.
die(){
  say "ABORT: $*"
  FAILED=1
  if [ -n "$JOBS" ]; then
    say "waiting for live children before exiting, so none is orphaned"
    wait_all abort || true
  fi
  say "V04 FAILED"
  exit 1
}

POOL_ID=$([ -f "$POOL" ] && $PY $PROV fingerprint --path "$POOL" || echo POOL_FILE_MISSING)
say "V04 start pid=$$ ngpu=$NGPU objective=$LW draw=$DRAW seeds='$SEEDS' steps=$STEPS rho_floor=$RHO_FLOOR pool_id=$POOL_ID"
[ "$LW" = row ] && { say "ABORT: --loss_weighting row is rejected for production (findings 9.5)"; exit 2; }
# A missing pool file used to become the literal string POOL_FILE_MISSING in every manifest, so it
# matched itself and every provenance check passed on an absent pool. The draw is built from this
# file and random Recall@10 is k/pool, so an absent pool is not a runnable visit.
[ "$POOL_ID" = POOL_FILE_MISSING ] && { say "ABORT: candidate pool not found at $POOL"; exit 2; }
# The old fallback put concurrent jobs on GPU 0 and reused master_port 29800 for all of them. V04 is
# specified for g6.12xlarge; refuse rather than silently run a different experiment.
[ "$NGPU" -ge 4 ] || { say "ABORT: V04 needs 4 GPUs (g6.12xlarge); got $NGPU"; exit 2; }

TRAIN_COMMON="--tcr_variant cdr3 --max_length 1024 --batch_size 16 \
  --token_budget 16384 --bucket_boundaries 64 128 256 512 1024 --loss_weighting $LW \
  --base_model esmc_300m --canonical_eval_path $Q/data/cpt_canonical_eval_v2 --eos_strategy raw \
  --num_epochs 1 --lr 2e-4 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --eval_steps 500 --save_steps 500 --logging_steps 50 --no_compile \
  --attn_implementation sdpa --early_stop_n_patience 0 --report_to wandb"
# NOTE on `--attn_implementation sdpa`: it is INERT for ESM-C. ESMC.from_pretrained sets
# _use_flash_attn=True itself and never consults this arg (findings 9.2), so every run here is on
# varlen flash attention and the log line saying "sdpa" is misleading. Kept because Stage 4.1 and
# 4.3 passed it, so removing it would change the recorded command without changing the numerics.
# It matters for provenance: flash-attn's backward uses atomics, which is ONE candidate source of
# the 1.31 % run-to-run difference the band measures. Training-time MLM masking is also unseeded
# (quest/training/collators.py uses bare torch.rand; only EVAL masking takes an explicit
# Generator), so the 1.31 % is not yet apportioned between the two. See the audit triage.
#
# NOTE on STEPS: it is recorded in every manifest but the trainer has no --max_steps; it derives
# the optimizer and scheduler length from dataset size and --num_epochs. 7800 is what the current
# 7,987,764 rows happen to yield at this effective batch size. Nothing here enforces that, so a
# data or batching change makes the manifest's step count a claim rather than a fact.

# train_arm <label> <dataset> <mhc_variant> <seed> <gpu> <cotenants>
# Runs the trainer in the FOREGROUND of its own subshell, so its exit status is the subshell's and
# the caller's `$!` names a child of the driver.
train_arm(){
  local lbl=$1 ds=$2 mhc=$3 seed=$4 gpu=$5 cot=$6
  local out=$CKPT_ROOT/v04_${lbl}_seed${seed}      # derived: own line, or set -u trips
  mkdir -p $out
  $PY $PROV write --out $out/manifest.json --objective $LW --token-budget 16384 \
    --bucket-boundaries "64,128,256,512,1024" --steps $STEPS --early-stop off \
    --draw n/a --pool-size n/a --pool-id n/a --dataset $ds --seed $seed \
    --cotenants "$cot" >/dev/null || return 3
  CUDA_VISIBLE_DEVICES=$gpu $TORCHRUN --nproc_per_node=1 --master_port=$((29800+gpu)) \
    scripts/training/train_cpt_stage2_single.py --dataset_path $Q/data/cpt_datasets/$ds \
    --mhc_variant $mhc --gradient_accumulation_steps 64 --seed $seed $TRAIN_COMMON \
    --output_dir $out --wandb_run_name v04-${lbl}-s${seed} > $out/training.log 2>&1
}

# start_arm <label> <dataset> <mhc_variant> <seed> <gpu> <cotenants>
start_arm(){
  train_arm "$1" "$2" "$3" "$4" "$5" "$6" &
  JOBS="$JOBS $!=${1}_seed${4}"
}

# run_eval <dir> <adapter> <gpu> <task> <label>
run_eval(){
  local d=$1 adapter=$2 gpu=$3 task=$4 lbl=$5 rc=0
  CUDA_VISIBLE_DEVICES=$gpu $PY -m track_a.esm_sft.judge eval --task $task \
    --adapter $adapter --tcr_variant cdr3 --mhc_variant full --ckpt $d/best.pt \
    --distractor_draw $DRAW --min_peptides 100 --out $d/eval.json > $d/eval.log 2>&1 || rc=$?
  say "judge $lbl rc=$rc task=$task"
  return $rc
}

# judge_cell <name> <adapter> <gpu> <task> <cotenants>
judge_cell(){
  local name=$1 adapter=$2 gpu=$3 task=$4 cot=$5
  local d=$O/$name                                      # derived: own line
  mkdir -p $d
  # Nothing in this launcher trains a comparison checkpoint: cells 2-4 were only ever given
  # `--ckpt $O/<cell>/best.pt`, which no step creates. Evaluating a missing checkpoint either
  # crashes late or scores an untrained head and calls it a comparison, so refuse here instead.
  # Wiring train-then-evaluate for these cells (as stage42_box/wave1.sh does) is unfinished work
  # and costs GPU time that the visit estimate does not include.
  [ -f $d/best.pt ] || {
    say "REFUSED cell $name: no trained comparison checkpoint at $d/best.pt"
    return 4
  }
  $PY $PROV write --out $d/manifest.json --objective $LW --token-budget 16384 \
    --bucket-boundaries "64,128,256,512,1024" --steps n/a --early-stop n/a \
    --draw $DRAW --pool-size 100 --pool-id $POOL_ID --dataset $task --seed 42 \
    --cotenants "$cot" >/dev/null || return 3
  run_eval $d $adapter $gpu $task $name
}

# ---------- P1 ----------
say "P1: arm A x seeds on GPUs 0-2, cell 1 base re-score on GPU 3"
i=0
for s in $SEEDS; do
  start_arm armA run01_cdr3_pocket_proportional_10M pocket $s $i "armA seeds $SEEDS"
  i=$((i+1))
done
G3=3
# Cell 1 re-scores the EXISTING base checkpoint under the new draw. eval only, no retraining:
# the 10-08 number was scored against the uniform draw and is not comparable to anything new.
BASE=$O/base_paired; mkdir -p $BASE
[ -f $BASE/best.pt ] || cp ${BASE_CKPT:-/home/ubuntu/s42_out/base_paired/best.pt} $BASE/best.pt 2>/dev/null
[ -f $BASE/best.pt ] || die "no base SFT checkpoint to re-score (looked for ${BASE_CKPT:-/home/ubuntu/s42_out/base_paired/best.pt})"
$PY $PROV write --out $BASE/manifest.json --objective $LW --token-budget 16384 \
  --bucket-boundaries "64,128,256,512,1024" --steps n/a --early-stop n/a --draw $DRAW \
  --pool-size 100 --pool-id $POOL_ID --dataset as_paired_i --seed 42 --cotenants "armA" >/dev/null \
  || die "could not write the base re-score manifest"
run_eval $BASE none $G3 as_paired_i "cell 1 base re-score" \
  || die "the base re-score failed; the G1 gate has nothing to read"
wait_all P1 || die "an arm A re-baseline failed; not buying arm D on top of an incomplete P1"
say "P1 done"

# ---------- G1: the gate ----------
say "G1: locked-read readability gate on the NEW-draw ranks"
$PY $S42 gate --eval $BASE/eval.json --units $UNITS --prereg-delta $PREREG \
  --rho-floor $RHO_FLOOR --primary-condition test_novel_pep --primary-unit peptide \
  --primary-metric norm_rank 2>&1 | tee -a $O/run.log
G1=${PIPESTATUS[0]}
say "G1 exit=$G1"
# KNOWN GAP (audit finding 7): s42_check.py returns 0 when `--units` was passed but the eval
# carries no per-query ranks, and it uses one exit code for an affordability refusal and for a
# software error. So a nonzero G1 is read here as the pre-agreed refusal, and a zero G1 does not
# yet prove ranks existed. Making the gate fail closed is the next fix in this file's programme.

# ---------- P2 ----------
say "P2: arm D x seeds on GPUs 0-2"
i=0
for s in $SEEDS; do
  start_arm armD run05_cdr3_fullmhc_proportional_10M full $s $i "armD seeds $SEEDS"
  i=$((i+1))
done
if [ "$G1" -eq 0 ]; then
  say "G1 PASSED: running comparison cells 2-4 on GPU $G3"
  CELLS_RC=0
  judge_cell armD_paired ${ARMD_ADAPTER:-$Q/checkpoints/cpt_stage41_pointD_run05/final} $G3 \
    as_paired_i armD || CELLS_RC=1
  judge_cell base_trb none $G3 as_trb_i armD || CELLS_RC=1
  judge_cell armD_trb ${ARMD_ADAPTER:-$Q/checkpoints/cpt_stage41_pointD_run05/final} $G3 \
    as_trb_i armD || CELLS_RC=1
  if [ $CELLS_RC -eq 0 ]; then
    # `cells` waives co-tenancy and nothing else, and prints the waiver it used: GPU 3's neighbour
    # differs between P1 and P2 by construction, every other key must match.
    $PY $PROV check --a $BASE/manifest.json --b $O/armD_paired/manifest.json --kind cells \
      2>&1 | tee -a $O/run.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { say "provenance refused the base-vs-armD comparison"; FAILED=1; }
  else
    say "a comparison cell failed or refused; the comparison is not certified"
    FAILED=1
  fi
else
  say "G1 FAILED: comparison cells 2-4 SKIPPED. The locked primary read cannot resolve the"
  say "  pre-registered effect under the $DRAW draw, so no base-vs-CPT number from this task"
  say "  would be interpretable. This is the pre-agreed refusal, not a run failure."
fi
wait_all P2 || say "an arm D re-baseline failed"
say "P2 done"

if [ $FAILED -ne 0 ]; then
  say "V04 FAILED: see the per-job statuses above. No comparison result from this visit is usable."
  exit 1
fi
say "V04 complete. Band: seeds $SEEDS per arm; read every verdict against the measured band,"
say "  and report measured cross-arm rho against the assumed $RHO_FLOOR and the breakeven in G1."
# Still missing before a verdict can be read off this visit: the launcher runs no paired
# comparison. `judge compare` is never invoked, and its estimator is query-level Recall@10 with a
# row bootstrap, not the locked peptide-clustered normalized-rank read the gate enforces.
