#!/usr/bin/env bash
# run_contract_build.sh — one contract build and its verification on a box (corpus repair step 10).
#
# Builds an example-sampled contract version against the real source, then runs the verifier
# WITHOUT --allow-background-subset, so the build only passes if the single-receptor rows are the
# ones the contract was sized on. CPU only: run it on a memory-sized instance, not a GPU one.
#
# Everything it needs is named in the environment and checked before any work starts; nothing has
# a default that could silently select a different input. It never writes into an existing run
# directory. Outcome: $RUN_DIR/status.json (written last) and the build's own _build_artifacts.
#
#   QUEST=/home/ubuntu/quest-cr  PY=/path/to/python  RUN_DIR=/home/ubuntu/cpt_repair/<run>
#   CONTRACT=… CONTRACT_SHA256=…  EXCLUSION_LIST=… EXCLUSION_SHA256=…  HLA_REF=… HLA_SHA256=…
#   VERSION=task_enriched_cap1000 SCALE=7571616 THREADS=30 MEMORY_GB=200
#   SOURCE_ROOT=… CLUSTERS_DIR=… SPLIT_DIR=…   bash run_contract_build.sh
#
# DISTRIBUTION (default contract) is `proportional` for a row-sampled contract version, which is
# then built with the md5 partition sampler and checked against the contract's partition targets.
set -euo pipefail

for v in QUEST PY RUN_DIR CONTRACT CONTRACT_SHA256 EXCLUSION_LIST EXCLUSION_SHA256 HLA_REF HLA_SHA256 \
         VERSION SCALE THREADS MEMORY_GB SOURCE_ROOT CLUSTERS_DIR SPLIT_DIR; do
  [ -n "${!v:-}" ] || { echo "REFUSE: $v is not set"; exit 2; }
done
say(){ echo "[$(date -u '+%F %T')] $*"; }
check_sha(){ local got; got=$(sha256sum "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] || { echo "REFUSE: $1 has sha256 $got, expected $2"; exit 2; }; }

# ---- preflight: refuse before spending anything --------------------------------------------------
[ ! -e "$RUN_DIR" ] || { echo "REFUSE: $RUN_DIR exists; a run never reuses a directory"; exit 2; }
check_sha "$CONTRACT" "$CONTRACT_SHA256"
check_sha "$EXCLUSION_LIST" "$EXCLUSION_SHA256"
check_sha "$HLA_REF" "$HLA_SHA256"
for d in "$SOURCE_ROOT/subset_key=trb" "$SOURCE_ROOT/subset_key=tra" "$SOURCE_ROOT/subset_key=tra_trb_peptide_mhc_one" \
         "$CLUSTERS_DIR" "$SPLIT_DIR" "$QUEST/scripts/data_processing"; do
  [ -d "$d" ] || { echo "REFUSE: $d is missing"; exit 2; }
done
"$PY" -c "import duckdb, datasets, pyarrow; print('duckdb', duckdb.__version__, 'datasets', datasets.__version__)" \
  || { echo "REFUSE: $PY cannot import duckdb, datasets and pyarrow"; exit 2; }
mkdir -p "$RUN_DIR"
free_gb=$(df -BG --output=avail "$RUN_DIR" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 150 ] || { echo "REFUSE: only ${free_gb} GB free under $RUN_DIR; DuckDB spills while sorting"; exit 2; }
say "preflight ok: $(git -C "$QUEST" rev-parse HEAD) free=${free_gb}GB threads=$THREADS memory=${MEMORY_GB}GB"

DISTRIBUTION=${DISTRIBUTION:-contract}
case "$DISTRIBUTION" in
  contract) BUILD_ONLY=() ;;
  proportional) BUILD_ONLY=(--partition-sampler md5) ;;
  *) echo "REFUSE: DISTRIBUTION=$DISTRIBUTION"; exit 2 ;;
esac
COMMON=(--tcr-variant cdr3 --mhc-variant full --distribution "$DISTRIBUTION" --scale "$SCALE"
        --eligibility-policy reference-aware-v1 --hla-reference "$HLA_REF"
        --split-routing receptor-hash-v2 --benchmark-exclusion heldout-peptides-v2
        --exclusion-list "$EXCLUSION_LIST" --sampling-contract "$CONTRACT" --contract-version "$VERSION"
        --source-root "$SOURCE_ROOT" --clusters-dir "$CLUSTERS_DIR" --split-dir "$SPLIT_DIR" --seed 42
        --output-dir "$RUN_DIR/dataset" --run-id "$(basename "$RUN_DIR")")
cd "$QUEST"

t0=$(date +%s); builder_rc=0
"$PY" scripts/data_processing/cpt_dataset_builder.py "${COMMON[@]}" "${BUILD_ONLY[@]}" --threads "$THREADS" \
  --memory-gb "$MEMORY_GB" --duckdb-tmp "$RUN_DIR/duckdb_tmp" > "$RUN_DIR/build.log" 2>&1 || builder_rc=$?
t1=$(date +%s); say "builder rc=$builder_rc in $(( (t1 - t0) / 60 )) min"

verifier_rc=not_run
if [ "$builder_rc" = 0 ]; then
  verifier_rc=0
  "$PY" scripts/data_processing/cpt_dataset_verifier.py "${COMMON[@]}" --poll-timeout-sec 60 \
    > "$RUN_DIR/verify.log" 2>&1 || verifier_rc=$?
  say "verifier rc=$verifier_rc in $(( ($(date +%s) - t1) / 60 )) min"
fi
rm -rf "$RUN_DIR/duckdb_tmp"
printf '{"builder_rc": "%s", "verifier_rc": "%s", "build_minutes": %d, "quest_head": "%s", "version": "%s"}\n' \
  "$builder_rc" "$verifier_rc" $(( (t1 - t0) / 60 )) "$(git -C "$QUEST" rev-parse HEAD)" "$VERSION" > "$RUN_DIR/status.json"
say "status: $(cat "$RUN_DIR/status.json")"
[ "$builder_rc" = 0 ] && [ "$verifier_rc" = 0 ]
