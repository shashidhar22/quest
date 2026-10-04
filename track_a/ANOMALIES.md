# Track A — Methodology Anomalies

This file logs every deviation from a baseline's canonical published usage that arose during Track A. Each entry documents what was done, why, and what the impact is on results — for transparent reporting in the paper's methods section.

## TCR-BERT — Empty-TRA shim for TRB-only tasks

**Date**: 2026-05-04
**Affects**: `as_trb_i`, `as_trb_ii` cells of TCR-BERT
**Deviation**: The upstream TCR-BERT-with-peptide setup expects paired (TRA, TRB) chains. AS-TRB-I and AS-TRB-II have no TRA data — `tra_cdr3` is NaN for all rows. Track A's wrapper feeds an empty-string token sequence into the TRA encoder branch.
**Why**: Filtering TRB-only rows would drop the entire dataset. We cannot launch a TRB-only baseline against TCR-BERT-paired without some shim.
**Impact**: The TRA encoder produces a fixed embedding for the empty input that the head learns to ignore. Effectively the model becomes a single-chain TRB model with extra inert capacity. Predictions still depend on (TRB, peptide, MHC). Results should be interpreted as "TCR-BERT with TRA branch zeroed out," not as canonical paired-TCR-BERT.
**Reference**: `verification/tcrbert_verification.md` § Single-chain shim.

## DeepTCR — UNK V/D/J placeholders

**Date**: 2026-05-04
**Affects**: All DeepTCR cells (TRB-I, Paired-I, TRB-II, Paired-II)
**Deviation**: DeepTCR canonically uses V/D/J gene one-hot embeddings. The benchmark_v2/splits parquet schema does NOT include `trav_gene/trbv_gene/...` columns (dropped at the enriched stage). Track A feeds `'UNK'` placeholders for all V/D/J inputs.
**Why**: Extracting V/D/J via ANARCI from `tra_full/trb_full` would add ~30 min of preprocessing per task; for a baseline with otherwise modest expected performance, the V/D/J signal is unlikely to be the deciding factor. We chose speed over fidelity.
**Impact**: DeepTCR's V/D/J channel is effectively disabled. Predictions rely on CDR3 sequence + MHC only. Reported numbers should be labeled "DeepTCR (CDR3 + MHC, V/D/J = UNK)" in the paper.
**Escalation path**: If smoke-test or full-scale results suggest V/D/J would meaningfully change the picture, run ANARCI on the train/test sets and re-train. This adds ~30 min preprocessing + ~30 min retrain per task.
**Reference**: `verification/deeptcr_verification.md` § Inputs.

## ESM SFT — "Decoder SFT" = scorer head, not generative decoder

**Date**: 2026-05-04
**Affects**: All 24 cells of the ESM 2×3 matrix
**Deviation**: The user prompt's phrase "decoder SFT" is interpreted as fine-tuning a small AttentionPooling + MLP scoring head on top of the (frozen-or-LoRA-adapted) encoder, NOT training a generative transformer decoder. The InfoNCE loss is contrastive retrieval, matching how Recall@K is measured.
**Why**: Training a generative decoder on benchmark_v2 splits is out of scope — it would require a separate seq2seq stack (similar to `quest/models/seq2seq_model.py`) and would likely take 50+ GPU-hours per cell. Retrieval-head SFT is the standard methodology for ESM-based retrieval and produces directly comparable numbers across the 6 model variants.
**Impact**: Reported numbers reflect ESM2/ESM-C encoders + small SFT heads, NOT end-to-end generative training. Paper should label results "ESM2-X, retrieval head SFT" or similar.
**Reference**: `verification/esm_sft_verification.md` § Architecture and Methodology caveats.

## TULIP — `<MIS>` placeholder for TRA on TRB-only tasks

**Date**: 2026-05-04
**Affects**: TULIP-original and TULIP-retrained on `as_trb_i`, `as_trb_ii`
**Deviation**: TRB-only tasks feed the `<MIS>` token in place of the TRA chain.
**Why**: This is the canonical TULIP convention from the upstream paper — TULIP was trained with `<MIS>` placeholders to handle missing chains. NOT a Track A invention; in-distribution for the model.
**Impact**: None — this is the documented and pre-trained-on usage of TULIP for missing chains.
**Reference**: `verification/tulip_verification.md` § Mode.
**Note**: Logged here for completeness even though it isn't strictly an anomaly.

## Class II training data

**Date**: 2026-05-04
**Affects**: All retrained baselines on `as_trb_ii`, `as_paired_ii`
**Deviation**: Class II has no train/val splits (eval-only by design — see PHASE0_CLASSII_INVESTIGATION.md). For "retrained" cells, baselines train on the matching class-I `_train.parquet` and evaluate on `as_*_ii_eval.parquet`.
**Why**: This is the benchmark's design. Class II measures cross-class generalization.
**Impact**: Class II numbers should be interpreted as "trained on class I, zero-shot evaluated on class II" — a transfer-learning result, not an in-distribution result.
**Reference**: `PHASE0_CLASSII_INVESTIGATION.md`.

## Class-II candidate pool synthesis

**Date**: 2026-05-04
**Affects**: All retrieval evaluations on `as_*_ii` (every retrained baseline)
**Deviation**: No published class-II candidate pool exists. Track A synthesizes one as `union(class-I pool, peptides in as_*_ii_eval.parquet)`.
**Why**: The published `as_class_i_candidate_pool.tsv` covers only class-I peptides (3,563 unique). Only ~19% of class-II eval peptides overlap with the class-I pool — if we evaluated class II against the class-I-only pool, ~80% of test rows would have no reachable true peptide and would be excluded from metrics. The synthesized pool (~4,359 peptides for as_trb_ii) keeps every class-II test peptide reachable while preserving the class-I distractors.
**Impact**: Class-II Recall@K is on a slightly larger candidate pool than class I. Random baselines differ accordingly: random recall@10 = 10/4359 ≈ 0.0023 for class II vs 10/3563 ≈ 0.0028 for class I. Reportable.
**Reference**: `track_a/lstm_mlp/convert_data.py:load_candidates`. Same logic should be reused by the other baselines' evaluate.py.

## Eval row cap = 2000 per condition

**Date**: 2026-05-04
**Affects**: All baselines × all conditions × all tasks
**Deviation**: We cap evaluation at the first 2,000 rows of each test partition. Some conditions (e.g. `as_paired_i_test_novel_tcr` = 715,200 rows; `as_trb_i_test_novel_allele` = 200,106) would take days of GPU time at full scale × 3,563-candidate retrieval scoring × 6 ESM cells × 4 tasks.
**Why**: Time budget. Joint-encoder retrieval scoring is O(test_rows × candidates) forwards through the model.
**Impact**: Reported metrics are over the first 2,000 rows of each condition (deterministic — no shuffle, so cross-baseline comparable). The score-sidecar parquet captures all 2000 × 3563 scored pairs per condition. If reviewers want full-scale numbers for the smaller conditions (test_iid is 1,044–6,696 rows), we can re-run those without subsampling — no architectural change required.
**Reference**: `track_a/launch/run_baseline.sh` passes `--max_test_rows 2000` to evaluate.py. Subsampling is row-order based (no random seed), so results are deterministic and reproducible.

## Class-II Paired-II upgraded to full splits (2026-05-04 mid-day refresh)

**Date**: 2026-05-04 17:30 UTC
**Affects**: `as_paired_ii` task across all baselines + the official class-II candidate pool
**Change**: `sravisha/tcrbench` HF dataset was updated with full `as_paired_ii_{train,val,test_iid,test_novel_tcr,test_novel_pep,test_novel_allele,test_level4,test_mixed}.parquet` files plus `as_class_ii_candidate_pool.tsv` (990 peptides). `as_trb_ii` and `as_tra_ii` remain eval-only.
**Impact**:
- `as_paired_ii` cells now train on ~20K class-II positives (their own train file) instead of class-I companion data, and report 5 conditions (test_iid 360, test_novel_tcr 58K, test_novel_pep 15K, test_novel_allele 3.7K, test_level4 840).
- All class-II cells now use the official 990-peptide candidate pool instead of the prior synthesized union (3,563 + class-II eval peptides ≈ 4,300). Random-baseline metrics differ accordingly: random recall@10 = 10/990 ≈ 0.0101 (vs 0.0023 with the synthesized pool).
- 9 stale class-II `_summary.json` files archived to `results_archive/old_class_ii_<UTC>/`. Cells redo'd with new data + new pool.
- Expected total cells: 132 → 176 (paired_ii went from 1 → 5 conditions × 11 baselines = +44).
**Note on TULIP-original**: still skipped on class II (model architecture limitation). Skipped placeholders carry through.
**Reference**: `track_a/shared/load_splits.py:CLASS_II_FULL_TASKS`; `track_a/lstm_mlp/convert_data.py:load_train_val/load_candidates`.

## test_novel_tcr peptides are NOT novel — major caveat for the paper

**Date**: 2026-05-04 (audit at 17:55 UTC)
**Affects**: All baselines × `test_novel_tcr` condition × all 4 main tasks
**Finding**: An audit (`track_a/peptide_overlap_audit.csv`) of train↔test peptide overlap shows that `test_novel_tcr` rows have **44–65% of their peptides also in train** for class I (AS-TRB-I 64.7%, AS-Paired-I 52.6%, AS-Paired-II 12.1%). Only the TCR axis is held novel; the peptide axis is not.

This explains why all baselines score dramatically higher on `test_novel_tcr` than on `test_iid`:
- LSTM TRB-I: recall@10 = 0.023 (iid) vs 0.542 (novel_tcr)
- MLP TRB-I:  0.046 vs 0.680
- DeepTCR:    0.0   vs 0.833
- TULIP-original: 0.24 vs 1.000

The model isn't generalizing to novel TCRs — it's largely retrieving peptides it memorized during training. The "novel TCR" framing of the condition name is misleading without this qualifier.

**Why this happens**: per `docs/BENCHMARK_SPLITS.md` §4 step 8, the `iid` condition is *stratified by peptide cluster* into train/val/test_iid. test_iid peptides are cluster-disjoint from train. By contrast, `test_novel_tcr` rows are constructed from rows whose **TCR cluster** is novel — there's no constraint on the peptide axis, so train peptides can (and do) recur there. The same logic explains why test_novel_allele on Paired-I has 42.9% peptide overlap and Paired-II has 50%.

**Per-condition novelty** (audit summary, AS-TRB-I as the headline):

| Condition | Pep ∩ train (%) | TCR ∩ train (%) | Effective novelty |
|---|---|---|---|
| test_iid | 0% | 5.7% | peptide-novel (despite "iid" naming) |
| test_novel_tcr | **64.7%** | 0% | TCR-novel only — peptides are train-set |
| test_novel_pep | 0% | 6.6% | peptide-novel |
| test_novel_allele | 7.9% | 1.8% | allele-novel; peptides mostly novel |
| test_level4 | 0% | 0% | all-novel — strictest condition |
| test_mixed | 2.9% | 2.3% | exactly 2 axes novel |

**Action items for the paper / FINAL_REPORT**:
1. Always co-report `test_novel_tcr` numbers WITH the peptide-overlap percentage so reviewers see the leakage.
2. The headline "novel-axis" number for true generalization should be `test_level4` or `test_novel_pep`, not `test_novel_tcr`.
3. `test_novel_allele` on Paired-* is similarly contaminated — flag it.
4. Consider reporting `test_iid` as "novel-peptide-only" (it effectively is) for clarity.

**Reference**: `track_a/peptide_overlap_audit.csv`; the audit script lives at `/tmp/overlap_audit.py`.

Caveat on the audit itself: the script used `trb_cdr3` as the TCR-overlap key. AS-TRA-* rows have NaN `trb_cdr3` → the audit's `tcr_overlap_pct=100%` for those tasks is a script artifact. Re-run with `tra_cdr3` for AS-TRA-* if those numbers matter. Class II eval-only tasks (`as_trb_ii`, `as_tra_ii`) were correctly skipped (no train file).

## TULIP-original "zero-shot" numbers are largely peptide memorization from pretrain

**Date**: 2026-05-04 18:05 UTC
**Affects**: All TULIP-original results across the entire matrix
**Finding**: TULIP's published model weights were pretrained on `data/SeenPeptides/tulip2_pretune.csv` (948,661 rows, **332,014 unique peptides**) plus smaller fine-tune sets — most likely NetMHCpan-style synthetic data given the corpus size. We audited overlap between TULIP's pretrain peptide vocabulary and the AS test sets.

**Headline:** the AS Class I candidate pool has **43.4% overlap (1,546 / 3,563)** with TULIP's pretrain peptides; Class II pool has **27.4% (271 / 990)**.

### Per-condition pretrain-leak (TRB-I, the headline column)

| Condition | unique test peps | in TULIP-train | % | TULIP-original recall@10 |
|---|---|---|---|---|
| test_iid | 39 | 31 | **79.5%** | 0.240 |
| test_novel_tcr | 561 | 206 | 36.7% | **1.000** |
| test_novel_pep | 297 | 162 | **54.5%** | **0.820** |
| test_novel_allele | 1,583 | 654 | 41.3% | **0.750** |
| test_level4 | 335 | 150 | **44.8%** | **1.000** |
| test_mixed | 1,257 | 742 | 59.0% | (n/a) |

### The smoking gun

For **AS-Paired-I**:

| Condition | TULIP-train pep overlap | TULIP-original recall@10 |
|---|---|---|
| test_iid | 81.1% | 1.000 |
| test_novel_tcr | 39.5% | 0.760 |
| test_novel_pep | 51.8% | 0.480 |
| test_novel_allele | 61.2% | 0.240 |
| **test_level4** | **11.1%** | **0.000** |

The Paired-I `test_level4` cell — where TULIP-pretrain overlap is **only 11.1%** — is the *single* condition where TULIP-original scored 0. Every condition where TULIP scored high has ≥40% pretrain overlap.

### What this means for the paper

1. TULIP-original's apparent "zero-shot generalization" is **mostly retrieval of peptides it saw during pretraining**, not novel-peptide generalization. The model has a vocabulary of 332K pretrain peptides; ~half of our "novel" test peptides are in that vocabulary.
2. The Paired-I `test_level4` result (recall@10 = 0 on the one condition with low pretrain leakage) is the **only honest zero-shot generalization number** TULIP produces on AS.
3. We should either:
   - (a) Always report TULIP numbers alongside the per-condition pretrain-overlap %, OR
   - (b) Compute a "filtered" recall@10 that restricts evaluation to test rows whose peptide is **NOT** in TULIP-pretrain, OR
   - (c) Both.
4. The same audit should run for any other "zero-shot" baseline that uses an external pretrain corpus (e.g. PanPep on Track B).
5. TULIP's strength on AS-Class-II `test_novel_pep` (84.2% pretrain overlap) and `test_iid` (85.7%) should be similarly caveated.

### Files

- `track_a/tulip_pretrain_leakage_audit.txt` — full per-condition table for all AS tasks
- `/tmp/tulip_pretrain_audit.py` — the audit script (uncommitted)
- TULIP pretrain data: `track_a/tulip/repo/data/{Seen,Unseen}Peptides.zip` (extracted under `/tmp/tulip_data/`)

## (Future entries appended below)
