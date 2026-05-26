# CPT Stage 3 → Stage 4+ Planning Report

**Status.** Stage 3 complete. Full MHC (Run 05) won the MHC-representation
sweep decisively (−63 % pocket_ppl vs pocket-only baseline). But the Run 05
training itself was severely sequence-truncated; the win is a **lower bound**
on what full-MHC training can do. This report:

1. Quantifies the truncation and its effect on Stage 3's headline.
2. Lays out the decisions that need to be made before Stage 4 launches.
3. Maps the remaining stages (4 → 8) with revised assumptions.

---

## 1. What the truncation actually did

Stage 3 Run 05 ran with `max_length=128`. The original plan justified this
with "mean tokens ~16.5, most rows fit easily in 64; 99th percentile is
unimportant since MHC rows are rare." That justification was wrong: MHC
rows are NOT unimportant — they are the entire signal for the
representation-axis question, and the rare-row 99th percentile is the
mode for the rows that matter.

Token-length distribution for the canonical eval at `(tcr=cdr3, mhc=full)`,
by `mhc_class`:

| mhc_class           | n_rows | median | p95 | p99 | max | % > 128 | % > 256 | % > 384 |
|---------------------|-------:|-------:|----:|----:|----:|--------:|--------:|--------:|
| I                   |     23 |   193  | 203 | 275 | 275 |  100 %  |   4.3 % |   0.0 % |
| II_complete         |  2,371 |   535  | 562 | 570 | 578 |  100 %  | **100 %** | **100 %** |
| II_partial_DP_beta  |    165 |   276  | 303 | 310 | 310 |  100 %  | 100 %   |   0.0 % |
| II_partial_DQ_beta  |    206 |   279  | 308 | 312 | 313 |  100 %  |  99.0 % |   0.0 % |
| II_partial_DR_beta  |  1,609 |   285  | 298 | 316 | 338 |  100 %  |  99.9 % |   0.0 % |
| none                | 15,334 |    20  |  32 | 265 | 308 |    1 %  |   1.0 % |   0.0 % |

The class II_complete rows are huge because the input concatenates **both**
MHC-II α and β chains (e.g. DR-α ~265 AA + DR-β ~260 AA + linkers), and
Stage 3's `max_length=128` truncated all of them at the end of the α
chain. Class I rows fit in 256. Class II partials need 384.

In Stage 3:
- Run 05 had **0** pocket masked tokens for `mhc_class=I` (full alpha chain
  was tail-truncated past the pocket — confirmed in the per-class table).
- For class II_complete, Run 05 saw only the first 128 tokens of a typical
  ~535-token sequence (~24 %), yet pocket_ppl on those tokens was 1.538
  vs 4.574 baseline (−66 %). The win was real even with severe truncation.

**Implication.** Run 05's headline is a strict lower bound. With
non-truncating `max_length`, the win is mechanically at least as big and
likely bigger.

## 2. Decisions needed before Stage 4 launches

These are the calls I'd like input on. My recommended default is in **bold**
for each.

### Decision A — re-run Stage 3 Run 05 with non-truncating max_length?

- **Option A1 (recommended): roll the fix into Stage 4.** Don't retrain
  Run 05. Stage 4 trains 1M and 100M with the corrected `max_length` and
  retrains a 10M anchor anyway (so Run 05 becomes a "truncated 10M"
  baseline we can compare against the corrected 10M, giving a free
  quantification of how much truncation cost). Saves ~3 hr / $11.
- Option A2: spend 3 hr / $11 retraining Run 05 cleanly so the Stage 3
  decision is published with a confident margin. The decision (full > pocket)
  doesn't change either way — only the magnitude does — so the value is
  mostly cosmetic. Skip unless a downstream stakeholder needs a clean
  10M number now.

### Decision B — Stage 4 `max_length`

- **Option B1 (recommended): max_length = 384, with aggressive length bucketing.**
  - Fits 100 % of class I, partial DP/DQ/DR, and all non-MHC rows.
  - Class II_complete still truncates (still gets ~70 % of its β chain;
    the full β tail is mostly framework not pocket, so the cost is small).
  - Length buckets: `[16, 32, 64, 128, 256, 384]`. The 256/384 buckets
    contain ~5 % of rows (the MHC-bearing ones); only those batches
    pay the long-sequence cost.
- Option B2: max_length = 512. Fits all class II_partial cleanly and
  captures ~95 % of class II_complete content. Cost ~30 % higher per
  long-bucket batch.
- Option B3: max_length = 640. Fits 100 % including class II_complete.
  Cost ~70 % higher per long-bucket batch.
- Option B4: max_length = 256. Fits class I + non-MHC; class II
  partials are 5-15 % truncated. Probably the wrong call — class II is
  ~95 % of the MHC signal in the corpus.

Cost analysis on g6.12xlarge (4× L4 24 GB) with length bucketing: the
long-bucket batches are the slow ones, but they contain ~5 % of rows.
Going from 128 → 384 increases total wall time by an estimated 1.7×
(not 9× — most rows still run at len ≤ 64). Going to 640 is 2.3×.
**Recommendation: 384.**

### Decision C — class II_complete handling

For class II_complete rows, the input includes both MHC chains:
`[CDR3] <eos> peptide <eos> mhc_one <eos> mhc_two`. Both chains are needed
for the binding groove (α-1 + β-1 form the cleft). Two sub-options:

- **Option C1 (recommended): include both chains, accept ~30 % truncation
  on the β tail.** The truncated β-tail is mostly framework (the binding
  pocket is in β-1 which sits at positions 1-90), so the pocket-relevant
  context is preserved.
- Option C2: order chains by allele-priority so the predicted-pocket
  chain comes first. Lower risk of truncating away pocket positions,
  but complicates the `_parse_order_key` logic.
- Option C3: bake an "essential MHC" extraction into the dataset
  builder — e.g. α-1 + β-1 helix regions only (~180 AA total),
  matching class I's α-1 + α-2 length. Cleaner, but needs a fresh
  dataset build (cost: ~$5, 1 hr).

C1 is the cheapest defensible call. C3 is the "right" answer in
principle but is a separate stage of work.

### Decision D — Stage 4 scaling-curve points

Stage 3 settled the representation axis. Stage 4 is the data-scaling axis,
holding `(tcr=cdr3, mhc=full, max_length=384)` fixed. Original plan
called for 1M / 10M / 100M.

- **Option D1 (recommended): Run 07 = 1M, Run 09 = 10M (FRESH with
  corrected max_length), Run 08 = 100M.** Three points on the curve, all
  trained with the corrected settings. Cost: ~$5 + ~$13 + ~$50 = $68,
  wall ~17 hr.
- Option D2: 1M / 100M only; reuse Stage 3 Run 05 as the
  truncated-10M data point with a footnote. Cost: ~$5 + $50 = $55,
  wall ~12 hr. Saves $13 and 5 hr at the cost of a non-comparable mid
  point.

The 100M run is the long pole (~10 hr) regardless of choice. Both
options put us inside the original Stage 4 budget envelope ($60-80).

### Decision E — gradient checkpointing / ZeRO

At `max_length=384` + bf16 + LoRA, ESM-C 300M fits comfortably on L4
24 GB without checkpointing (we used `batch=128 × max_len=128` ≈ 16k
tokens/GPU; `batch=64 × max_len=384` ≈ 25k tokens/GPU is comparable).

- **Option E1 (recommended): no gradient checkpointing.** Halve
  `batch/GPU` from 128 → 64 to keep memory headroom; double
  `grad_accum` from 2 → 4 to keep effective batch ≈ 1024.
- Option E2: enable checkpointing, keep batch=128. Slower step time
  (~1.5×) but more aggressive batch.

E1 is simpler and matches Stage 2 Run 03 (which used larger sequences
than Run 05).

## 3. What's wrong with `mhc_one_complete` for class II

(Side note for Stage 4 dataset builder.) Class II_complete encodes a
heterodimer (DR-α + DR-β) as two separate `mhc_*` columns. The Stage 3
input projection serializes both, which is the right thing for an MLM
trying to learn the joint groove context, but it means class II_complete
sequences are roughly 2× the size of class I or class II_partial.

Stage 4 could consider concatenating α+β with a chain-boundary token
(rather than `<eos>`-separated segments) so the model treats them as
one continuous antigen-presenting unit. This is a Stage 4 data-pipeline
decision; defer until D-decision is settled.

## 4. Sanity-check fix already applied (and its scope)

The Stage 3 preflight caught a real bug in `identify_segment_regions`:
MHC pocket residues are NOT a contiguous substring of pocket_contact or
the full chain — they're a positional subsequence interleaved with
other residues by chain position. The fix was a 13-line greedy
subsequence-match helper, scoped to the MHC overlays only (TCR
`cdr3-in-tra_full` overlay is genuinely a contiguous substring and is
unchanged). This affects evaluation/region-attribution; it does not
affect training itself.

This bug was latent in `cpt_training_lib.py` for the entire Stage 2
duration. Stage 2 was not affected because all three Stage 2 runs used
`mhc_variant=pocket` — bare pocket segments are uniformly labeled as
pocket, no overlay needed. Any Stage 0/1 ESM-2 run that used
`mhc_variant=pocket_contact` or `full` would have produced unreliable
region-restricted metrics. Worth scanning the Stage 1 retroactive
checkpoints for which mhc_variant they used.

## 5. Compute budget through Stage 8 (revised)

| stage | scope | wall (est.) | $ (est.) | cumulative |
|---|---|---:|---:|---:|
| 0  | shakedown (done) | — | — | $5 |
| 1  | ESM-2 150M baselines (done) | — | — | $40 |
| 2  | TCR-rep sweep (done) | — | — | $135 |
| 3  | MHC-rep sweep (done) | 3.4 hr | $15 | $150 |
| 4  | proportional scaling 1M/10M/100M @ full+384 | 17 hr | $68 | $218 |
| 5  | balanced vs proportional @ chosen scale | 5 hr | $20 | $238 |
| 6  | masking strategy (IC-weighted) @ chosen scale | 6 hr | $24 | $262 |
| 7  | ESM-C 600M large-model validation @ winner | 14 hr | $100 | $362 |
| 8  | production T4 build @ winner | 24 hr | $200 | $562 |
| ret. | position-IC-weighted retroactive eval (7 ckpts × 1 pass) | 2 hr | $5 | $567 |

Original plan envelope: $540-650. Revised total: ~$567. Still in
envelope. Buffer for re-runs and Stage 5/6 iteration: ~$80.

## 6. Open questions

These don't block Stage 4 launch but should be answered before Stage 7:

- **Q1**: For class II_complete, is α-1+β-1 (~180 AA) sufficient context
  to predict pocket residues, or does the model need the framework tail?
  Test by an ablation in Stage 5 or 6.
- **Q2**: Run 05's `framework_ppl=1.557` is very low — is the model
  memorizing allele identity (and predicting framework from that)
  rather than learning per-position chemistry? Could check by holding
  out unseen alleles in the canonical eval.
- **Q3**: 15 % random masking under-emphasizes pocket positions
  (~19/250 ≈ 8 % of an MHC chain is pocket). An IC-weighted masking
  scheme would up-weight pocket positions during training. Stage 6
  is the right place to test; cheap (~$24) and may be the largest
  outstanding lever.
- **Q4**: Stage 8 production wants downstream-task performance, not
  region-restricted MLM. Need to choose: (a) frozen-encoder linear probe
  for binding affinity, (b) end-to-end fine-tune on a labelled
  TCR-pMHC binder/non-binder dataset, (c) zero-shot retrieval. Defer
  until Stage 5/6 settles.
- **Q5**: Do we revisit Stage 2's TCR-rep decision (CDR3 won)? With
  full-MHC context, does `cdr123` or `full` TCR start to add value?
  Probably not, but the Stage 2 result was conditioned on
  `mhc=pocket`. One cheap sanity run after Stage 4 picks the data
  scale would re-validate.

## 7. Recommended Stage 4 launch spec

Ready to plan / execute when these decisions are confirmed:

- (A1) Don't retrain Stage 3 Run 05; quantify truncation cost from
  the Stage 4 10M anchor instead.
- (B1) max_length = 384, length buckets = [16, 32, 64, 128, 256, 384].
- (C1) Keep both chains for class II_complete; accept partial β-tail
  truncation at len 384.
- (D1) Three runs: Run 07 (1M), Run 09 (10M, full+384), Run 08 (100M).
- (E1) batch=64, grad_accum=4, no gradient checkpointing.
- WandB project: `tcrbench-v3-cpt-stage4`.
- Early-stop metric: `pocket overall _ perplexity` (same as Stage 3,
  validated working).
- Estimated wall: 17 hr (1M ≈ 1 hr, 10M ≈ 6 hr, 100M ≈ 10 hr).
- Estimated cost: ~$68.

Stage 4 winning data scale gates the rest of the plan (Stage 5/6
ablations only train at the chosen scale).
