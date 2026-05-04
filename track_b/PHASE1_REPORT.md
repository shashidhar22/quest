# Track B Phase 1 — Verification + Smoke-Test Report

**Status**: All 4 baselines have a working env, a verification document reviewed before code, and a passing 100-row smoke test on AS-TRB-I.

**Date**: 2026-04-30

## Top-line summary

| Baseline    | Env built | Verification doc | Smoke test | AUC (test_iid 50) | Accuracy | Wall-clock |
|-------------|-----------|------------------|------------|-------------------|----------|-----------|
| ERGO-II     | ✓ track_b_ergo_ii (Py 3.7, torch 1.7.1)         | ✓ ergo_ii_verification.md     | ✓ | 0.523 | 0.647 | ~10s |
| NetTCR-2.2  | ✓ track_b_nettcr_2_2 (Py 3.10, TF 2.11)         | ✓ nettcr_2_2_verification.md  | ✓ | 0.561 | 0.833 | ~10s |
| PanPep      | ✓ track_b_panpep (Py 3.9, torch 1.10.2)         | ✓ panpep_verification.md      | ✓ | 0.505 | 0.646 | ~15s |
| TEIM-Seq    | ✓ track_b_teim (Py 3.8, torch 1.10.1)           | ✓ teim_verification.md        | ✓ | 0.499 | 0.833 | ~5s  |

Smoke tests use 100 train rows + 50 val + 50 test_iid (each + 5× negatives). 2 training epochs. CPU-only on a g6.12xlarge (GPU 2/3 idle as required, GPU 0/1 untouched throughout).

## Per-baseline status and Phase-2 estimates

### ERGO-II (LSTM, vdj-mode)
- **Mode used**: ERGO-II.vdj LSTM with `--use_vj --use_mhc`. TRB-only on AS-TRB-I.
- **Smoke runtime**: 2-epoch train on 492 rows = ~7 s; eval on 150 rows = ~3 s. Total ~10 s.
- **Phase-2 wall-clock estimate**: 60K positives × 6 (with negatives) = 360K rows. 2-3 hours per epoch on CPU; 25 epochs ≈ 2 days. **Phase 2 GPU question still open**: torch 1.7 doesn't support sm_89 (L4 GPU). Either (a) accept 2-day CPU wall-clock, (b) modernize the source to PL ≥ 2.0, (c) pin a different GPU.
- **V/J extraction**: smoke test used `'UNK'` placeholder. Phase 2 needs ANARCI or tidytcells germline matching on `trb_full` for the proper .vdj signal.
- **Open question**: VDJdb overlap with bundled vdjdb-pretrained model for the "original" arm.

### NetTCR-2.2 (pretrained mode)
- **Mode used**: pan for smoke (single training round); pretrained for Phase 2 (pan + per-peptide fine-tune).
- **Smoke runtime**: 2-epoch train on 468 rows = ~5 s; eval on 288 rows = ~3 s. Total ~10 s.
- **Phase-2 wall-clock estimate**: pretrained mode trains a pan model on ~50K rows then fine-tunes ~800 peptide-specific models. At 30 s pan epoch + 200 epochs + ~50 s × 800 fine-tune = ~12 hours on CPU. With CUDA libs installed (~2GB extra disk), ~30 min on L4.
- **Critical limitation**: model doesn't use MHC. `test_novel_allele` predictions can't differ from `test_iid` on the allele axis.
- **Architecture quirk**: TRB-only → empty A1/A2/A3 → constant CNN output on those branches. Trains fine in smoke; needs monitoring for Phase 2 stability.
- **Drop rate**: 17.7% of AS-TRB-I rows have B1/B2 null and are dropped.

### PanPep (zero-shot only on TCRBench)
- **Mode used**: zero-shot. **Major finding**: TCRBench's iid stratification deliberately disjoints peptide clusters between train and test files (verified empirically: 0/1044 peptide-cluster overlap on as_trb_i_test_iid). PanPep's majority/few-shot modes need exact peptide-string matches in the support set; with zero overlap, those modes are unreachable. **All TCRBench conditions → zero-shot for PanPep.**
- **Smoke runtime**: zero-shot inference on 229 rows = ~12 s.
- **Phase-2 wall-clock estimate**: zero-shot is just a forward pass through the meta-learner. ~1000 rows/min on CPU; full Phase 2 at ~360K rows = ~6 hours.
- **No retraining**: PanPep ships only inference code (no training script). The "retrained" arm is skipped per resolved decision.
- **Train-set overlap caveat**: PanPep's published meta-learner was trained on VDJdb/McPAS-TCR; TCRBench includes VDJdb. AUC numbers may be inflated for VDJdb-sourced rows. **Phase 2 should report dropped fraction filtered to non-VDJdb rows as an ablation**.
- **2025 reusability report (PMC12458544)**: documents that PanPep's majority-mode advantage depends on negative-sampling strategy (collapses with hard negatives). Not relevant to our zero-shot-only deployment, but document.

### TEIM-Seq (binding head only)
- **Mode used**: TEIM-Seq sequence-level binding. TEIM-Res (residue-level) skipped per the brief — TCRBench has no structural data.
- **Smoke runtime**: 2-epoch train on 504 rows = ~3 s; eval on 102 rows = ~2 s. Total ~5 s.
- **Phase-2 wall-clock estimate**: ANARCI alignment ~200 rows/sec ≈ 5 min for 60K positives one-time. Training: 25 epochs × ~300 batches = ~10 min on CPU; ~2 min on L4.
- **Hard length caps**: peptide ∈ [8, 12] AA, CDR3 ≤ 20 AA. **Drops ~30% of AS-TRB-I positives** (TCRBench has many 13–17 AA peptides). This is the largest data loss of any baseline.
- **VDJdb/McPAS overlap**: bundled `teim_seq.ckpt` was trained on VDJdb + McPAS + NCOV. Same caveat as PanPep for the "original" arm.

## Cross-baseline observations

### MHC handling
Three of four baselines (NetTCR-2.2, PanPep, TEIM-Seq) **do not use MHC** as input. Only ERGO-II takes MHC (as a categorical embedding). For the `test_novel_allele` condition, only ERGO-II will produce allele-aware predictions; the other three will show the same metrics as `test_iid` modulo allele-correlated noise. This is a critical caveat for the paper's allele-generalization story.

### V/J handling
Only ERGO-II uses V/J as features. The other three use only CDR3β + peptide. ERGO-II's V/J extraction is a Phase-2 TODO (currently UNK placeholders). NetTCR-2.2 uses CDR1/2 explicitly (we drop 17.7% null rows on AS-TRB-I).

### GPU compatibility
- **ERGO-II, PanPep, TEIM**: pinned to torch ≤ 1.10 because of `pytorch_lightning` API divergence; doesn't support L4 (sm_89). All smoke tests run on CPU.
- **NetTCR-2.2**: TF 2.11 supports newer GPUs but needs CUDA 11 system libs not installed on the host. CPU fallback works.
- **Phase 2 decision**: either (a) CPU all-the-things for ~2-day wall-clock total, (b) modernize each baseline's PL/torch deps, or (c) install CUDA 11 system libs (~2 GB) for NetTCR-2.2 and accept CPU for the other three.

### Train/test peptide-cluster disjointness (PanPep finding generalizes)
TCRBench's iid stratification within the iid pool is BY peptide cluster (BENCHMARK_SPLITS.md §4 step 8). This means:
- **Train file ↔ test_iid file**: 0 peptide-cluster overlap.
- All 5 conditions (iid, novel_tcr, novel_pep, novel_allele, level4) are effectively peptide-novel from a baseline's perspective if the baseline only sees `*_train.parquet` peptides.
- This is correct behavior for measuring generalization, but PanPep majority/few-shot modes (which need exact peptide overlap) are unreachable. Other baselines learn cluster-level peptide patterns from CDR3-peptide co-occurrence — not affected.

### Negative-sampling consistency
All four baselines use **the same negatives** via `track_b/shared/negative_sampling.sample_negatives()`: 5×, peptide-only swap, seed=43, positive-set filter. This ensures cross-baseline AUC is apples-to-apples. Each baseline has its own native negative-sampling that we override:
- ERGO-II: native joint (peptide, MHC) swap.
- NetTCR-2.2: native peptide-swap with Levenshtein-≥3 filter.
- PanPep: caller-provided.
- TEIM: native per-peptide stratified shuffle, 5×.

Phase 2 may report each baseline's native-negatives AUC as an ablation if reviewers ask.

## Open questions for Phase 2

1. **GPU strategy** (cross-cutting): CPU-only (slow but works) vs. modernize three of four baselines' deps to support sm_89? Lean: stick with CPU for Phase 1; revisit if Phase 2 wall-clock is unacceptable.
2. **VDJdb overlap (PanPep + TEIM original arms)**: report AUC + caveat for Phase 2 main results; add an ablation filtering VDJdb-sourced rows.
3. **TEIM length filter (~30% data loss)**: accept the drop or relax the upstream length assertion (small source patch)? Lean: accept for Phase 2 fidelity.
4. **NetTCR-2.2 alpha-zero-padding stability** at full scale: monitor pan-training loss curve; if unstable, surgically prune α branches.
5. **ERGO-II V/J extraction**: ANARCI (slow, ~30 min on 60K rows) vs. tidytcells germline-match (faster but less accurate). Lean: ANARCI in a separate Python-3.10 env, write results to `track_b/shared/cache/vj_calls.parquet`, share across baselines if any others adopt V/J.

## File inventory

```
track_b/
├── verification/
│   ├── ergo_ii_verification.md      # 270 lines, addresses all 8 brief sub-points
│   ├── nettcr_2_2_verification.md   # ditto
│   ├── panpep_verification.md       # ditto + revised mode mapping after data finding
│   └── teim_verification.md         # ditto + length cap + ANARCI dependency
├── shared/
│   ├── load_splits.py               # parquet → pandas DataFrame
│   ├── negative_sampling.py         # 5× peptide-swap, seed=43
│   ├── allele_utils.py              # HLA name vs. protein-seq fallback
│   ├── extract_vj.py                # Phase-1 stub returning UNK
│   └── cache/                       # gitignored
├── ergo_ii/   {env.yml, repo/, convert_data.py, train.py, evaluate.py, results/, README.md}
├── nettcr_2_2/ {same structure}
├── panpep/    {same structure}
├── teim/      {same structure}
└── PHASE1_REPORT.md  (this file)
```

## Verification gates honored

For each baseline, the workflow was:
1. Set up conda env + clone repo.
2. Read paper + repo.
3. Write verification doc.
4. **STOP** — print "VERIFICATION DOCUMENT WRITTEN: review before proceeding to code" and wait.
5. Resume on user approval — write conversion + training + evaluation code.
6. Run 100-row smoke test.
7. Mark complete.

All 4 verification documents were written before any conversion/training code, and user review was solicited at each gate.

## Pre-conditions met (per the brief)

- [x] GPUs 0 and 1 untouched (Track A2 CPT). Confirmed via `nvidia-smi` at start.
- [x] All training and evaluation on CPU; ready to retarget GPU 2 or 3 for Phase 2 if drivers allow.
- [x] `benchmark_v2/splits/` parquet files used directly; never modified.
- [x] All baselines run sequentially; no parallel env conflicts.

Phase 1 complete. Ready for Phase 2 scoping.
