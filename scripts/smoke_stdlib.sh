#!/bin/bash
# Fresh-clone smoke: do the dependency-free parts of this repo actually run anywhere?
#
# Why this exists. Every verdict that cost this programme a visit came from code that could not be
# run away from the training box, and the fix was to put the decision logic in the repo with no
# heavy imports. That property was ASSERTED, not checked: the full suite has 36 failures and 6
# collection errors from missing local dependencies, so "the tests pass" is not a claim anyone can
# verify on another machine. Worse, tests/fixtures/*.csv was swallowed by a `*.csv` ignore rule, so
# the Stage 4.2 cluster-CI tests depended on a file that was never committed and passed only on the
# author's laptop. This target would have caught that on its first run.
#
# It clones THIS repo at HEAD into a temp dir, so only committed content is visible, then runs the
# subset that needs nothing beyond the standard library plus pytest. A missing fixture, a
# gitignored file or a new heavy import fails it.
#
# Usage: bash scripts/smoke_stdlib.sh [python]
set -euo pipefail
PY=${1:-python3}
REPO=$(cd "$(dirname "$0")/.." && pwd)
BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Tests that must run on a bare interpreter. Add a test here only if it imports no torch/peft/
# transformers/tidytcells, directly or transitively.
STDLIB_TESTS=(
  tests/test_s42_gate.py
  tests/test_s42_readability.py
  tests/test_token_budget_sampler.py
)

echo "[smoke] cloning $BRANCH at HEAD into a fresh tree (committed content only)"
git clone --quiet --no-hardlinks --depth 1 --branch "$BRANCH" "file://$REPO" "$TMP/quest"
cd "$TMP/quest"

echo "[smoke] confirming the fixtures the stdlib tests need are actually in the clone"
missing=0
while IFS= read -r f; do
  [ -e "$f" ] || { echo "[smoke] MISSING FROM CLONE: $f"; missing=1; }
done < <(grep -rhoE '"[^"]*fixtures/[A-Za-z0-9_.-]+"' "${STDLIB_TESTS[@]}" 2>/dev/null \
         | tr -d '"' | sed 's#^#tests/#' | sort -u)
for f in tests/fixtures/s42_as_paired_i_units.csv \
         tests/fixtures/s42_as_paired_i_true_peptides.csv \
         tests/fixtures/s42_eval_base_paired_ranks_20261008.json \
         tests/fixtures/s42_eval_base_paired_20261008.json \
         tests/fixtures/s42_eval_bf16_tied_20261007.json; do
  [ -e "$f" ] || { echo "[smoke] MISSING FROM CLONE: $f"; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "[smoke] FAIL: fixtures are not committed — the tests only pass where they were written"; exit 1; }

echo "[smoke] asserting the predicate imports nothing heavy"
$PY - <<'PYCHK'
import ast, pathlib, sys
HEAVY = {"torch", "peft", "transformers", "tidytcells", "datasets", "pandas", "numpy", "pyarrow"}
src = pathlib.Path("scripts/training/stage42_box/readability.py")
mods = set()
for n in ast.walk(ast.parse(src.read_text())):
    if isinstance(n, ast.Import):
        mods |= {a.name.split(".")[0] for a in n.names}
    elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
        mods.add(n.module.split(".")[0])
bad = mods & HEAVY
print(f"[smoke]   readability.py imports: {sorted(mods)}")
sys.exit(1 if bad else 0)
PYCHK

echo "[smoke] running the stdlib test subset"
$PY -m pytest -q "${STDLIB_TESTS[@]}"
echo "[smoke] PASS — the decision logic runs on a fresh clone with no training dependencies"
