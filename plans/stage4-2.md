# CPT Stage 4.2 — downstream SFT judge

**Status:** planned · **Depends on:** Stage 4.1 adapters (3 arms) · **Doc:** `QUEST_CHECKPOINTS_AND_DATA.md` §9.1 step 1

## Goal
Build the metric every later CPT stage is judged by: Recall@10 with a bootstrap CI on held-out
specificity splits. Perplexity is not the decision metric from here on.

## Why first
- No ESM-C class-I SFT result exists. In W&B the `esmc_300m-base-as_paired_i` cell crashed on its
  only attempt; no ESM-C CPT cell on a class-I task ever finished.
- Stages 4.4–4.7 each train a new adapter. Without a judge they can only be ranked by perplexity,
  which disagreed with the downstream winner the one time both were measured.

## Approach
Reuse `track_a/esm_sft/` (frozen base + LoRA, AttentionPooling + InfoNCE head). Changes:
1. **Serialize as the adapter was trained**: `<eos>`-joined segments, full MHC chain for full-MHC
   arms, built with `build_input_text_sql()` logic. Track A's `XX` + pocket-contact format is kept
   only for the Gen C adapter it was made for.
2. Lift the 2,000-row eval cap for `test_level4`.
3. Bootstrap CI over queries (1,000 resamples) on Recall@10 and MRR.
4. Fix the ESM-C crash first; read the crashed run's log in W&B before changing code.

## Cells (10)
Encoders: ESM-C base, Gen C adapter, Stage 4.1 arms 1/2/3. Tasks: `as_paired_i`, `as_trb_i`.
Conditions scored: `test_level4`, `test_novel_pep`. `test_novel_tcr` excluded (contaminated).

## Cost
Track A ESM-C cells ran 0.9–3.3 GPU-h. 10 cells ≈ 10–30 GPU-h on one L4 (g6.xlarge,
≈ USD 0.80/h on-demand) → **USD 10–25**. Engineering ≈ 1 day.

## Verification
- Smoke: base ESM-C on `as_trb_i` reproduces Track A's base recall within its CI.
- Decision: at least one CPT arm's CI excludes base on `test_level4`. If none does, 4.4–4.7 have no
  downstream justification — stop and rethink before spending on them.
