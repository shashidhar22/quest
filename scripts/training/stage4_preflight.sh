#!/bin/bash
# TCRBench v3 Stage 4 pre-flight checks.
# Exits non-zero if any of the 8 prerequisites is missing.
set -u

FAIL=0
log() { echo "[preflight] $*"; }
check() {
  local label="$1"; shift
  if "$@"; then
    log "OK    $label"
  else
    log "FAIL  $label"
    FAIL=1
  fi
}

# 1. Source data dir present
check "source dedup data exists" \
  test -d /home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched

# 2. Required existing CPT datasets
for ds in run01_cdr3_pocket_proportional_10M \
          run05_cdr3_fullmhc_proportional_10M \
          run06c_cdr3_pocket_balanced_100M \
          run08_cdr3_pocket_proportional_100M \
          run09_full_fullmhc_proportional_10M; do
  check "CPT dataset $ds" \
    test -d "/home/ubuntu/quest/data/cpt_datasets/$ds"
done

# 3. Stage 2 Run 03 checkpoint (reused for Point E)
check "Stage 2 Run 03 checkpoint" \
  test -d /home/ubuntu/quest/checkpoints/cpt_stage2_run03_full/final
check "Stage 2 Run 03 adapter file" \
  test -f /home/ubuntu/quest/checkpoints/cpt_stage2_run03_full/final/adapter_model.safetensors

# 4. Canonical eval v2
check "canonical eval v2 exists" \
  test -d /home/ubuntu/quest/data/cpt_canonical_eval_v2

# 5. cpt_training_lib importable with region_restricted_eval + early-stop callback
LIB_DIR=/home/ubuntu/quest/scripts/training
if PYTHONPATH="$LIB_DIR" python3 -c "
from cpt_training_lib import region_restricted_eval, setup_lora_model, CdrPplPlateauCallback, CPTTrainer, CPTConfig
" 2>/dev/null; then
  log "OK    cpt_training_lib imports clean"
else
  log "FAIL  cpt_training_lib import failed"
  FAIL=1
fi

# 6. Disk space (need ~50 GB free)
FREE_GB=$(df -BG /home/ubuntu/quest/ | tail -1 | awk '{gsub("G","",$4); print $4}')
if [ "${FREE_GB:-0}" -ge 50 ]; then
  log "OK    disk free ${FREE_GB} GB (need >= 50)"
else
  log "FAIL  disk free ${FREE_GB} GB (need >= 50)"
  FAIL=1
fi

# 7. WandB authenticated
if wandb login --verify 2>&1 | grep -qi "Currently logged in as"; then
  log "OK    wandb authenticated"
else
  log "FAIL  wandb not authenticated (run: wandb login)"
  FAIL=1
fi

# 8. ESM-C 300M loadable via cpt_training_lib (same loader the trainer uses)
if PYTHONPATH="$LIB_DIR" python3 -c "
from cpt_training_lib import load_base_model
m = load_base_model('esmc_300m')
n = sum(p.numel() for p in m.parameters())
assert n > 1e8, n
print(f'ESM-C 300M params: {n:,}')
" 2>/dev/null; then
  log "OK    ESM-C 300M loadable (via cpt_training_lib.load_base_model)"
else
  log "FAIL  ESM-C 300M not loadable via cpt_training_lib"
  FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
  log "PRE-FLIGHT FAILED"
  exit 1
fi
log "PRE-FLIGHT OK"
exit 0
