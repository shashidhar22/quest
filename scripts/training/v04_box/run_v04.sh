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
# P1 and P2; that asymmetry is recorded in each manifest rather than planned away.
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
export PYTHONPATH=$Q PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p $O; cd $Q
say(){ echo "[$(date -u '+%F %T')] $*" | tee -a $O/run.log; }
echo $$ > $O/run_v04.pid       # the watchdog waits on THIS pid, never a name pattern

POOL_ID=$([ -f "$POOL" ] && $PY $PROV fingerprint --path "$POOL" || echo POOL_FILE_MISSING)
say "V04 start pid=$$ ngpu=$NGPU objective=$LW draw=$DRAW seeds='$SEEDS' steps=$STEPS rho_floor=$RHO_FLOOR pool_id=$POOL_ID"
[ "$LW" = row ] && { say "ABORT: --loss_weighting row is rejected for production (findings 9.5)"; exit 2; }

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
# It matters for provenance: flash-attn's backward uses atomics, which is the irreducible source
# of the 1.31 % run-to-run difference the band measures.

# train_arm <label> <dataset> <mhc_variant> <seed> <gpu> <cotenants>
train_arm(){
  local lbl=$1 ds=$2 mhc=$3 seed=$4 gpu=$5 cot=$6
  local out=$Q/checkpoints/v04_${lbl}_seed${seed}      # derived: own line, or set -u trips
  mkdir -p $out
  $PY $PROV write --out $out/manifest.json --objective $LW --token-budget 16384 \
    --bucket-boundaries "64,128,256,512,1024" --steps $STEPS --early-stop off \
    --draw n/a --pool-size n/a --pool-id n/a --dataset $ds --seed $seed \
    --cotenants "$cot" >/dev/null
  CUDA_VISIBLE_DEVICES=$gpu $TORCHRUN --nproc_per_node=1 --master_port=$((29800+gpu)) \
    scripts/training/train_cpt_stage2_single.py --dataset_path $Q/data/cpt_datasets/$ds \
    --mhc_variant $mhc --gradient_accumulation_steps 64 --seed $seed $TRAIN_COMMON \
    --output_dir $out --wandb_run_name v04-${lbl}-s${seed} > $out/training.log 2>&1 &
  echo $!
}

# judge_cell <name> <adapter> <gpu>
judge_cell(){
  local name=$1 adapter=$2 gpu=$3
  local d=$O/$name                                      # derived: own line
  mkdir -p $d
  $PY $PROV write --out $d/manifest.json --objective $LW --token-budget 16384 \
    --bucket-boundaries "64,128,256,512,1024" --steps n/a --early-stop n/a \
    --draw $DRAW --pool-size 100 --pool-id $POOL_ID --dataset as_paired_i --seed 42 \
    --cotenants "armD" >/dev/null
  CUDA_VISIBLE_DEVICES=$gpu $PY -m track_a.esm_sft.judge eval --task as_paired_i \
    --adapter $adapter --tcr_variant cdr3 --mhc_variant full --ckpt $d/best.pt \
    --distractor_draw $DRAW --min_peptides 100 --out $d/eval.json > $d/eval.log 2>&1
  say "judge cell $name rc=$?"
}

# ---------- P1 ----------
say "P1: arm A x seeds on GPUs 0-2, cell 1 base re-score on GPU 3"
PIDS=""; i=0
for s in $SEEDS; do
  g=$i; [ "$NGPU" -lt 3 ] && g=0
  PIDS="$PIDS $(train_arm armA run01_cdr3_pocket_proportional_10M pocket $s $g "armA seeds $SEEDS")"
  i=$((i+1))
done
G3=3; [ "$NGPU" -lt 4 ] && G3=0
# Cell 1 re-scores the EXISTING base checkpoint under the new draw. eval only, no retraining:
# the 10-08 number was scored against the uniform draw and is not comparable to anything new.
BASE=$O/base_paired; mkdir -p $BASE
[ -f $BASE/best.pt ] || cp ${BASE_CKPT:-/home/ubuntu/s42_out/base_paired/best.pt} $BASE/best.pt 2>/dev/null
$PY $PROV write --out $BASE/manifest.json --objective $LW --token-budget 16384 \
  --bucket-boundaries "64,128,256,512,1024" --steps n/a --early-stop n/a --draw $DRAW \
  --pool-size 100 --pool-id $POOL_ID --dataset as_paired_i --seed 42 --cotenants "armA" >/dev/null
CUDA_VISIBLE_DEVICES=$G3 $PY -m track_a.esm_sft.judge eval --task as_paired_i --adapter none \
  --tcr_variant cdr3 --mhc_variant full --ckpt $BASE/best.pt --distractor_draw $DRAW \
  --min_peptides 100 --out $BASE/eval.json > $BASE/eval.log 2>&1
say "cell 1 base re-score rc=$? draw=$DRAW"
for p in $PIDS; do wait $p; done
say "P1 done"

# ---------- G1: the gate ----------
say "G1: locked-read readability gate on the NEW-draw ranks"
$PY $S42 gate --eval $BASE/eval.json --units $UNITS --prereg-delta $PREREG \
  --rho-floor $RHO_FLOOR --primary-condition test_novel_pep --primary-unit peptide \
  --primary-metric norm_rank 2>&1 | tee -a $O/run.log
G1=${PIPESTATUS[0]}
say "G1 exit=$G1"

# ---------- P2 ----------
say "P2: arm D x seeds on GPUs 0-2"
PIDS=""; i=0
for s in $SEEDS; do
  g=$i; [ "$NGPU" -lt 3 ] && g=0
  PIDS="$PIDS $(train_arm armD run05_cdr3_fullmhc_proportional_10M full $s $g "armD seeds $SEEDS")"
  i=$((i+1))
done
if [ "$G1" -eq 0 ]; then
  say "G1 PASSED: running comparison cells 2-4 on GPU $G3"
  judge_cell armD_paired ${ARMD_ADAPTER:-$Q/checkpoints/cpt_stage41_pointD_run05/final} $G3
  judge_cell base_trb none $G3
  judge_cell armD_trb ${ARMD_ADAPTER:-$Q/checkpoints/cpt_stage41_pointD_run05/final} $G3
  $PY $PROV check --a $BASE/manifest.json --b $O/armD_paired/manifest.json --kind strict \
    2>&1 | tee -a $O/run.log
else
  say "G1 FAILED: comparison cells 2-4 SKIPPED. The locked primary read cannot resolve the"
  say "  pre-registered effect under the $DRAW draw, so no base-vs-CPT number from this task"
  say "  would be interpretable. This is the pre-agreed refusal, not a run failure."
fi
for p in $PIDS; do wait $p; done
say "P2 done"
say "V04 complete. Band: seeds $SEEDS per arm; read every verdict against the measured band,"
say "  and report measured cross-arm rho against the assumed $RHO_FLOOR and the breakeven in G1."
