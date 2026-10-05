# CPT Stage 4.7 — data scaling (10M → 100M)

**Status:** planned, last · **Depends on:** a positive 4.2 judge, the 4.4/4.5 recipe fixed, and
4.3's resume · **Knowledge:** `~/Documents/ACS/quest-cpt/plan.md` step 6

## Goal
Run the scaling curve the original Stage 4 was meant to be — at the final recipe, judged
downstream, once and not repeatedly.

## Evidence
`foundation_100M` was used by Gen C and nowhere since; `foundation_500M` has never been trained
on. Every Gen D stage, 4.1 included, trained on a 10M `cpt_datasets/runXX` subset. The planned
1M/10M/100M curve never ran.

## Approach
1. **Build a 100M set at the winning serialization and sampling** — unless one already fits.
   Two 100M sets exist in S3: `run08_cdr3_pocket_proportional_100M` and
   `run06c_cdr3_pocket_balanced_100M`. **If 4.4 settles on cdr3 × pocket balanced, run06c makes
   this step a pure training cost with no build at all**; a cdr3 × full recipe still needs one.
   Budget ≈ 9 h CPU at run11's measured rate (10M took 56 min), on a cheap CPU-only box — not a
   GPU box.
2. Train one epoch at a fixed step budget, same config as the 10M winner, no early stopping.
3. **500M only if 10M → 100M moves the judge.** Otherwise stop: the curve has answered.

**Why last.** Scale is the most expensive thing on the list and the only one that must be
repeated if the recipe changes afterwards. It is worth buying only when the recipe is settled and
a downstream metric can score it.

## Cost
- Build: ≈ 9 h CPU, a few dollars on an r6i box. **Disk:** run11's 10M is 3.2 GB, so 100M is
  ≈ 32 GB plus heavy DuckDB spill — the Stage 4.1 box had 401 GB free, which is comfortable, but
  check before starting.
- Run: ≈ 10× a 10M run → **≈ USD 60–80** proportional after 4.3 (≈ USD 380 at today's
  throughput, which is the clearest reason 4.3 comes first). Several hundred if balanced at
  matched rows, because of the token multiple.
- Needs 4.3's mid-run resume before spot is safe on a run this long.

## Verification
- **Preflight:** measured max tokens ≤ 1024; row count within ±1 % of 100M; effective batch
  exactly 1024.
- **Decision:** Recall@10 on `test_level4` improves from the 10M winner to 100M, outside the CI.
  Perplexity improving alone does not count — it is expected from more data and says nothing
  about specificity.
- Stages 7 (ESM-C 600M) and 8 (production build) stay deferred until this lands, since both are
  "@ winner" steps.
