#!/usr/bin/env bash
# audit_and_publish.sh — audit one built corpus from its own files, then publish it (step 11).
#
# Assembles a version directory beside the build (the dataset is MOVED into it, not copied, so
# there is one copy on the volume), runs cpt_built_audit.py, then publish_corpus.py, which
# refuses unless the preflight passes and the destination prefix is empty. PUBLISH=0 stops after
# the audit and a dry-run of the publication.
#
#   QUEST=… PY=… RUN_DIR=<run dir holding dataset/> VERSION_DIR=<new dir> CONTRACT=… CONTRACT_SHA256=…
#   SPLITS_DIR=<benchmark splits> HLA_REF=… DEST=s3://…/<version>/ PUBLISH=0|1  bash audit_and_publish.sh
set -euo pipefail
for v in QUEST PY RUN_DIR VERSION_DIR CONTRACT CONTRACT_SHA256 SPLITS_DIR HLA_REF DEST PUBLISH; do
  [ -n "${!v:-}" ] || { echo "REFUSE: $v is not set"; exit 2; }
done
say(){ echo "[$(date -u '+%F %T')] $*"; }
[ "$(sha256sum "$CONTRACT" | cut -d' ' -f1)" = "$CONTRACT_SHA256" ] || { echo "REFUSE: contract sha256 differs"; exit 2; }
[ -d "$RUN_DIR/dataset/_build_artifacts" ] || { echo "REFUSE: $RUN_DIR/dataset is not a built dataset"; exit 2; }
[ ! -e "$VERSION_DIR" ] || { echo "REFUSE: $VERSION_DIR exists"; exit 2; }
grep -q '^{"builder_rc": "0", "verifier_rc": "0"' "$RUN_DIR/status.json" || { echo "REFUSE: $RUN_DIR did not build and verify"; exit 2; }
mkdir -p "$VERSION_DIR"
mv "$RUN_DIR/dataset" "$VERSION_DIR/dataset"
cp "$CONTRACT" "$VERSION_DIR/sampling-contract.json"
cd "$QUEST"
say "audit of $(basename "$VERSION_DIR")"
"$PY" scripts/data_processing/cpt_built_audit.py --dataset "$VERSION_DIR/dataset" --splits-dir "$SPLITS_DIR" \
  --task as_paired_i --heldout-parts val test_iid test_novel_pep test_level4 --hla-reference "$HLA_REF" \
  --out-dir "$VERSION_DIR" > "$VERSION_DIR.audit.log" 2>&1 || { say "AUDIT FAILED"; tail -5 "$VERSION_DIR.audit.log"; exit 1; }
say "audit PASS: $(grep -o '"distinct_built_peptides": [0-9]*' "$VERSION_DIR.audit.log")"
if [ "$PUBLISH" = 1 ]; then
  "$PY" scripts/data_processing/contract_build_box/publish_corpus.py --version-dir "$VERSION_DIR" --dest "$DEST"
else
  "$PY" scripts/data_processing/contract_build_box/publish_corpus.py --version-dir "$VERSION_DIR" --dest "$DEST" --dry-run
fi
