# CPT Stage 4.2 — downstream SFT judge

**Status:** planned, do first · **Depends on:** Stage 4.1 adapters (available:
`s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/cpt_stage41_point{A_run01,D_run05,F_run11}/final/`)
· **Knowledge:** `~/Documents/ACS/quest-cpt/plan.md` step 1

## Goal
Build the metric every later CPT stage is judged by: Recall@10 with a bootstrap CI on held-out
specificity splits. Perplexity stops being the decision metric here.

## Why first
- No ESM-C class-I SFT result exists. In W&B the `esmc_300m-base-as_paired_i` cell crashed on its
  only attempt, and no ESM-C **CPT** cell on a class-I task ever finished (the two that finished
  are class-II eval cells).
- Stages 4.4–4.7 each train a new adapter. Without a judge they can only be ranked by perplexity,
  which disagreed with the downstream winner the one time both were measured.

## Approach
Reuse `track_a/esm_sft/` (frozen base + LoRA, AttentionPooling + InfoNCE head). Four changes:

1. **Serialize as the adapter was trained.** Verified feasible: the benchmark splits carry the
   full ~250 AA MHC chain in `mhc_one`/`mhc_two` *and* the pocket forms
   (`mhc_one_pocket`, `_contact`, `_pocket_contact`) — `build_benchmark_splits.py:66-72`,
   `docs/wiki/benchmark_v2/BENCHMARK_CLUSTERING.md:36`. So: full-MHC arms (D, F) score on
   `mhc_one`; arm A on `mhc_one_pocket`; the Gen C adapter keeps `mhc_one_pocket_contact`.
   Join segments with `<eos>` as `build_input_text_sql()` does, **not** Track A's `XX`.
   Testing a cdr3 × full adapter on pocket-contact text measures format transfer, not the adapter.
2. **Bounded eval, with a floor on n.** Track A's 2,000-row cap left `as_trb_i test_level4` at
   **70 queries**, where a bootstrap CI cannot separate anything. Cap at **5,000 queries per
   condition** instead (covers `as_paired_i test_level4` = 4,872 entirely; samples `as_trb_i`'s
   83,178 with seed 42). **Abort a cell's claim if n < 500** after filtering rather than reporting
   it — record it as underpowered.
3. Bootstrap CI over queries (1,000 resamples, seed 42) on Recall@10 and MRR.
4. **Fix the ESM-C crash first** — read the crashed run's log in W&B before touching code.

## Cells, in two waves (the gate is wave 1)
- **Wave 1, 4 cells:** base ESM-C and arm D, on `as_paired_i` and `as_trb_i`. This alone answers
  "does CPT pay for itself downstream".
- **Wave 2, 6 cells, only if wave 1 is positive:** arms A and F plus the Gen C adapter, same
  tasks — these rank serializations, which is only worth money once CPT has cleared the bar.

Conditions scored: `test_level4` and `test_novel_pep`. `test_novel_tcr` is excluded — 44–65 % of
its peptides also appear in train (`track_a/peptide_overlap_audit.csv`).

## Cost
Track A's ESM-C cells ran 0.9–3.3 GPU-h. Wave 1 ≈ 4–13 GPU-h ≈ **USD 4–11** on one g6.xlarge
(≈ USD 0.80/h); wave 2 another ≈ USD 6–16. Engineering ≈ 1 day.

## Verification
- **Smoke, before any new number:** base ESM-C on `as_trb_i` reproduces Track A's base Recall@10
  within its CI. If it does not, the harness changed and nothing downstream is comparable.
- **Power:** every reported cell has n ≥ 500 queries.
- **Decision:** arm D's CI excludes base on `test_level4`. If it does not, stages 4.4–4.7 have no
  downstream justification — stop and rethink rather than spending on them.
