#!/bin/bash
# Stage 4.2 wave 1 — the downstream judge that gates stages 4.4-4.7.
#
# Four cells: {base ESM-C, Stage 4.1 arm D} x {as_paired_i, as_trb_i}, each trained then scored on
# test_level4 and test_novel_pep (test_novel_tcr is excluded: 44-65 % of its peptides also appear in
# train). Both arms are serialized cdr3 x full MHC — the way arm D was continued-pretrained — so the
# comparison isolates the adapter and not the input format (plans/stage4-2.md, change 1).
#
# Cells run in DECISION ORDER, and results publish after every cell, because box time is the scarce
# resource: cells 1-2 alone answer "does CPT pay for itself downstream" on the paired split, which
# is the gate. Cells 3-4 add the beta-chain split. A ceiling that stops the box mid-way therefore
# costs the least important half.
#
# GATE: after cell 1, base Recall@10 on test_iid must beat the analytic random rate (10 of a
# 100-peptide pool) with its CI clear of it. If it does not, the harness is broken and no later
# number means anything, so the run stops there rather than spending the box on noise.
set -u
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
Q=${Q:-/home/ubuntu/quest}; O=${O:-/home/ubuntu/s42_out}; PY=${PY:-python}
CHECK=${CHECK:-/home/ubuntu/s42_check.py}
ADAPTER_D=${ADAPTER_D:-$Q/checkpoints/cpt_stage41_pointD_run05/final}
DST=${DST:-s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/stage42_wave1}
MAXQ=${MAXQ:-5000}; MINQ=${MINQ:-500}; SMOKE=${SMOKE:-}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)}
export PYTHONPATH=$Q TOKENIZERS_PARALLELISM=false
mkdir -p $O; cd $Q
say(){ echo "[$(date -u '+%F %T')] $*" >> $O/wave1.log; }
publish(){ [ -n "${NO_PUBLISH:-}" ] || aws s3 sync $O $DST --only-show-errors; }

say "wave 1 start: ngpu=$NGPU adapter_D=$ADAPTER_D max_queries=$MAXQ"
[ -d "$ADAPTER_D" ] || { say "ABORT: arm D adapter not found at $ADAPTER_D"; publish; exit 2; }

# cell <name> <task> <adapter> <gpu> ; trains, evaluates, publishes. Non-zero if either step fails.
cell(){
  local name=$1 task=$2 adapter=$3 gpu=${4:-0}   # separate statements: in one `local`, $name is unset
  local C=$O/$name
  local t rc
  if [ -f $C/eval.json ]; then say "$name: already done, skipping"; return 0; fi
  mkdir -p $C
  t=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu $PY -m track_a.esm_sft.judge train --task $task --adapter "$adapter" \
      --tcr_variant cdr3 --mhc_variant full $SMOKE --out_dir $C > $C/train.log 2>&1
  rc=$?; say "$name train rc=$rc secs=$(( $(date +%s)-t )) gpu=$gpu"
  [ $rc -eq 0 ] || { publish; return $rc; }
  t=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu $PY -m track_a.esm_sft.judge eval --task $task --adapter "$adapter" \
      --tcr_variant cdr3 --mhc_variant full $SMOKE --ckpt $C/best.pt --out $C/eval.json \
      --conditions test_level4 test_novel_pep test_iid --max_queries $MAXQ --min_queries $MINQ \
      > $C/eval.log 2>&1
  rc=$?; say "$name eval rc=$rc secs=$(( $(date +%s)-t ))"
  publish
  return $rc
}

compare(){   # compare <base_cell> <cpt_cell> <label>
  local a=$O/$1/eval.json
  local b=$O/$2/eval.json
  [ -f "$a" ] && [ -f "$b" ] || { say "$3: skipped (missing eval.json)"; return 0; }
  $PY -m track_a.esm_sft.judge compare --a "$a" --b "$b" > $O/compare_$3.txt 2>&1
  say "$3 compare rc=$?"; sed 's/^/    /' $O/compare_$3.txt >> $O/wave1.log
  publish
}

# --- cell 1: base on the paired split. Also carries the harness gate. ---
cell base_paired as_paired_i none 0 || { say "ABORT: base/as_paired_i failed"; exit 3; }
if $PY $CHECK gate --eval $O/base_paired/eval.json >> $O/wave1.log 2>&1; then
  say "GATE PASS: base recall@10 on test_iid is above the random rate, CI clear"
else
  say "GATE FAIL: base is at or below the random rate on test_iid — the harness is broken."
  say "Not spending the box on cells 2-4. Fix the judge before any wave-1 number is reported."
  publish; exit 4
fi

# --- cells 2-4. Cells 1+2 are THE DECISION; 3-4 add the beta-chain split. On a 4-GPU box the
# three remaining cells run side by side (they are independent); on 1 GPU, one after another. ---
if [ "$NGPU" -ge 4 ]; then
  say "cells 2-4 in parallel on GPUs 1,2,3"
  cell armD_paired as_paired_i "$ADAPTER_D" 1 & P2=$!
  cell base_trb    as_trb_i    none         2 & P3=$!
  cell armD_trb    as_trb_i    "$ADAPTER_D" 3 & P4=$!
  wait $P2 || say "WARN: armD/as_paired_i failed"
  compare base_paired armD_paired paired
  wait $P3 || say "WARN: base/as_trb_i failed"
  wait $P4 || say "WARN: armD/as_trb_i failed"
  compare base_trb armD_trb trb
else
  cell armD_paired as_paired_i "$ADAPTER_D" 0 || say "WARN: armD/as_paired_i failed"
  compare base_paired armD_paired paired
  cell base_trb    as_trb_i    none         0 || say "WARN: base/as_trb_i failed"
  cell armD_trb    as_trb_i    "$ADAPTER_D" 0 || say "WARN: armD/as_trb_i failed"
  compare base_trb armD_trb trb
fi

$PY $CHECK report --dir $O >> $O/wave1.log 2>&1
say "wave 1 done"
publish

# Hand the box to phase 3 (the v2-sampler equivalence re-test) if the kit shipped one.
PHASE3=${PHASE3:-/home/ubuntu/phase3_v2.sh}
if [ -f "$PHASE3" ] && [ -z "${NO_PHASE3:-}" ]; then say "handing over to $PHASE3"; bash "$PHASE3"; fi
