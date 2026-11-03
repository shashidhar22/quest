# CPT Stage 4.2 — downstream SFT judge

**Status (2026-10-07):** first GPU cell ran and the harness gate **FAILED as designed** — base
Recall@10 = 0.0 on every query (all 100 candidates tied: the pool + head ran in bf16 and candidates
differing by a 9-residue peptide fell within one bf16 ulp). Fixed: pool + head now fp32
(`tests/test_judge_scorer_fp32.py`). Also found: the benchmark splits are **exploded across
segment orderings** — as_paired_i test_level4 is 4,872 rows for **202 molecules**, test_iid 6,696
for 201, train 553,440 for 15,145 — so the judge now de-duplicates on `row_id` (one ordering per
molecule), which also makes the paired key unique and cuts a cell from 18 GPU-h to under one.
**Deviations from the text below, forced by that:** the 2,000→5,000 query cap is moot (≈ 200
queries exist on the decision conditions); the n ≥ 500 floor is unattainable there and is now a
reported flag at 150, not an abort; the paired class-I CI will be wide (n ≈ 200) and that is a
property of the benchmark, to be stated with the result. New pre-gate (≈ 1 min, before any cell):
`judge probe` — scores must vary with the peptide and be finite, for both arms. Wave 1 is
otherwise unchanged and still unrun. · **Depends on:** Stage 4.1 adapters (available:
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

Conditions scored: `test_level4` (the decision condition), `test_novel_pep` (the breadth check)
and `test_iid` (reported, never gated). `test_novel_tcr` is excluded — 44–65 % of its peptides
also appear in train (`track_a/peptide_overlap_audit.csv`).

## What the splits actually contain — measured 2026-10-15

The condition names describe something the files do not hold. Measured from the splits against
train (15,145 molecules / 1,455 peptides / 448 peptide clusters):

| condition | shares a peptide with train | shares a peptide cluster | trb_cdr3 overlap |
|---|---|---|---|
| `test_level4` | none | none | 0.0 % |
| `test_iid` | none | none | 4.0 % |
| `test_novel_pep` | none | none | 9.4 % |

So `test_iid` is **not** in-distribution, and `test_level4` is **not** "both seen" — it shares
nothing with training at all. Every condition is peptide-disjoint; they differ only in receptor
overlap. The earlier reading, that recall tracked peptide familiarity, is **retracted** — this
measurement refutes it (findings §9.6).

What recall actually tracked is the true peptide's **length** against a candidate pool that is
76 % 9-mers: inside `test_level4`, 9-mers scored 0.553 and 14-mers 0.013. That is a property of
the evaluation rather than the splits, and it is removed by the length-matched distractor draw
added 2026-11-03 (`judge._draw_distractors`). Matching is partial for long peptides, because the
pool holds too few siblings, so every condition reports `frac_distractors_same_length`.

## Cost
Track A's ESM-C cells ran 0.9–3.3 GPU-h. Wave 1 ≈ 4–13 GPU-h ≈ **USD 4–11** on one g6.xlarge
(≈ USD 0.80/h); wave 2 another ≈ USD 6–16. Engineering ≈ 1 day.

## Verification
- **Harness check, before any new number:** the base arm's peptide-clustered Recall@10 on
  `test_level4` — the condition that has signal — clears the analytic random rate of 0.10 on its
  CI lower bound. Three outcomes, PASS / UNDERPOWERED / BROKEN, and only BROKEN stops the box
  (`scripts/training/stage42_box/s42_check.py`). *Changed 2026-10-15:* this check keyed on
  `test_iid` until then, whose 201 molecules over 53 peptides cannot separate 0.144 from 0.100; on
  2026-10-08 it reported "the harness is broken" about a working harness and skipped cells 2–4 for
  nothing. *Changed 2026-10-05:* the earlier check, "reproduce Track A's base recall", is void
  because Track A's inputs carried no MHC and its arms had unequal capacity (findings finding 3).
- **Power:** the unit of independence is the **distinct true peptide**, not the query. In
  `test_level4` 87 % of peptides are all-hit or all-miss, so a per-query bootstrap ran 3–4× too
  narrow and manufactured a "below random" reading of `test_novel_pep`. CIs cluster-bootstrap over
  peptides; the floor is `--min_peptides`, default 100. `as_paired_i test_iid` has 53 and is
  reported, never gated.
- **Decision:** arm D's interval excludes base's point estimate on `test_level4`. If it does not,
  that means `as_paired_i` **cannot answer** the question — its interval is 0.46 wide — and the
  next move is `as_trb_i`, which carries 9,734 molecules on `test_level4`. An inconclusive wave 1
  does not condemn CPT. Only a CPT arm whose interval sits *below* base's point estimate is
  evidence against it.
