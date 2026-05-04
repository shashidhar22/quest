# PanPep Verification Document

**Status**: DRAFT — pending human review before any conversion / training code is written.

**Env**: `/home/ubuntu/miniforge3/envs/track_b_panpep` built from `track_b/panpep/env.yml` (Python 3.9, torch 1.10.2+cu102). Imports verified. Same L4-incompat issue as ERGO-II (torch 1.10 supports up to sm_75); smoke test runs on CPU. Source code uses hardcoded `.cuda()` calls — we'll patch to fall back to CPU. Repo cloned to `track_b/panpep/repo/`.

## Source material

- **Paper**: Gao Y, Gao Y, Liu Q, et al. *Pan-Peptide Meta Learning for T-Cell Receptor-Antigen Binding Recognition.* Nature Machine Intelligence 5:236–249, 2023.
- **2025 reusability report**: PMC12458544 — independent reproduction with critical caveats (see §8).
- **Repo**: https://github.com/bm2-lab/PanPep (cloned to `track_b/panpep/repo/`).
- **Local files inspected**: `repo/README.md`, `repo/PanPep.py`, `repo/Data/Example_{majority,few-shot,zero-shot}.csv`, `repo/Requirements/*.py`, `repo/Requirements/*.pkl`.

## 1. Paper Methods summary

PanPep is a meta-learning framework for TCR-peptide binding prediction with three modes determined by the number of known binders for the test peptide:
- **Majority** (>100 known binders): MAML fine-tuning of the meta-learner with many gradient steps (default 1000) on a per-peptide support set.
- **Few-shot** (5–100 binders): MAML fine-tuning with fewer steps (default 3).
- **Zero-shot** (no binders): Neural Turing Machine (NTM) — the meta-learner predicts directly using its memory module without any per-peptide fine-tuning.

The architecture: amino acids are encoded with **Atchley factors** (5-dim biophysical features per AA, hard-coded in `dic_Atchley_factors.pkl`); positional encoding (sinusoidal) is added; peptide embedding is concatenated with each TCR embedding; transformer-based feature extractor + MLP produces a binding score. The meta-learner was trained on VDJdb / McPAS-TCR with peptide-level episodes.

**Critical**: MHC is NOT used as input. Only peptide + CDR3β.

The 2025 reusability report (PMC12458544) found that PanPep's majority-mode advantage depends critically on the negative-sampling strategy: with "hard negatives" (sequence-similar peptides, like NetTCR's Levenshtein-≥3 filter inverted), majority-mode performance collapses. With "background draw" negatives (random repertoire), majority-mode looks great. Document this in the integration write-up.

## 2. Repo canonical workflow

**The repo ships ONLY inference code — there is no training script.** The released `model.pt` is the published meta-learner. Users provide a CSV with peptide / CDR3 / Label columns; PanPep runs in one of three modes determined by `--learning_setting`.

```bash
python PanPep.py --learning_setting majority --update_step_test 1000 \
    --input ./Data/Example_majority.csv --output ./Output/Example_majority_output.csv
```

Sanity check (uses bundled example):
```bash
cd repo
python PanPep.py --learning_setting zero-shot \
    --input Data/Example_zero-shot.csv --output Output/sanity_zero.csv
```

## 3. Data format requirements

CSV with header. Columns differ by mode:

### Majority / Few-shot
```csv
Peptide,CDR3,Label
ATDALMTGY,CAISESQGNTEAFF,1
ATDALMTGY,CAISEDRALVSYTF,1
ATDALMTGY,CAVQPGQGMQPQHF,Unknown
```

- `Label = 1`: positive support pair (used for fine-tuning)
- `Label = 0`: negative support pair (used for fine-tuning)
- `Label = Unknown`: query pair (gets a score in the output)

The CSV must be sorted such that all rows for a single peptide are contiguous (per the README's note: *"you should sort the peptides in the input csv, before predicting their binding probabilities"*).

### Zero-shot
```csv
Peptide,CDR3
FVYVFTTHL,CASSFNYNEQFF
KLAKPLPYT,CASDPDSLIHNTGELFF
```

No labels; everything is a query.

### Output (all modes)
```csv
Peptide,CDR3,Score
```

`Score` is the predicted binding probability ∈ [0, 1].

### Sequence-length caps (`PanPep.py:aamapping`)
- Peptide: 15 (longer truncated with a printed warning)
- CDR3: 25
- Atchley encoding: 5 features per AA → max-len matrices padded with zeros.

## 4. Negative sampling strategy

PanPep's training-time negative sampling is documented in the paper (not exposed in the repo); the inference workflow expects the **caller** to provide explicit (TCR, peptide, label) pairs. For `Label=0` rows, the caller is responsible for ensuring those are real non-binders.

For our integration: we use `track_b/shared/negative_sampling.sample_negatives()` (5×, seed=43, peptide-only swap) as the **support-set negatives** for majority/few-shot modes. Per the 2025 reusability report, this is closer to the "background draw" strategy that flatters PanPep majority-mode performance.

## 5. Train/test split assumptions

PanPep trains the meta-learner on episodes (one peptide per episode), so by design the model handles unseen peptides via the zero-shot / few-shot modes. **There is no concept of "training data overlap" with our TCRBench data because PanPep itself wasn't retrained.** We use the published meta-learner as-is for the "original" Lu et al.-style evaluation arm.

For the "retrained" arm: per the brief, we should retrain on TCRBench. **This is non-trivial because PanPep does not ship training code** — only the meta-learner checkpoint and inference path. Options:

| Option | Effort | Faithfulness |
|--------|--------|--------------|
| (a) Skip retrained arm; report only "original" PanPep | Low | Matches Lu et al.'s approach |
| (b) Heavy fine-tune via majority mode (`update_step_test=10000+`) on TCRBench data | Medium | Approximates retraining but not the same as MAML retrain from scratch |
| (c) Reimplement MAML training from the paper Methods | High | True retrained variant |

**Recommendation**: (a) for Phase 1 + Phase 2 main results, (b) as an ablation if reviewers ask, (c) only if specifically demanded. The 2025 reusability report (which reviewer ejRT cited) also evaluated PanPep without retraining.

### Mapping TCRBench conditions → PanPep modes (REVISED after data inspection)

The brief's mode mapping assumed that the `iid` condition has peptide overlap between train and test. **Empirical inspection shows this is false** for TCRBench:
- Train file ↔ test_iid file peptide-string overlap: 0 / 1044 (test_iid sample)
- Train file ↔ test_iid file peptide-cluster overlap: **0 / 1044**

This is because the splits doc (BENCHMARK_SPLITS.md §4 step 8) stratifies the iid POOL into train/val/test_iid **by peptide cluster** with 85/7.5/7.5 ratios — i.e., the train and test_iid peptide-cluster sets are deliberately disjoint within the shared iid pool. iid means "the row's TCR cluster and allele are in train; the row's peptide cluster is in train pool but NOT in train file specifically". That's a distinct semantics from PanPep's "majority = lots of binders for this peptide in train".

Revised mapping:

| TCRBench condition | PanPep mode | Why |
|--------------------|-------------|-----|
| `iid` | **zero-shot** | peptide strings & clusters disjoint from train |
| `novel_tcr` | **zero-shot** | same |
| `novel_pep` | **zero-shot** | same — strictly novel by design |
| `novel_allele` | **zero-shot** | same |
| `level4` | **zero-shot** | strictly novel |

**All conditions → zero-shot.** Majority and few-shot modes are unreachable on TCRBench because no test peptide string is in the train file. Document this clearly: PanPep on TCRBench is effectively a zero-shot benchmark, period.

This is consistent with PanPep's published meta-learning premise (peptides are tasks; the model should generalize across tasks). It also matches reviewer ejRT's specific ask for zero-shot PanPep numbers on held-out epitopes.

## 6. Configuration choices

Defaults from the repo:
- `update_lr = 0.01`
- `update_step_test = 3` (few-shot) or `1000` (majority)
- `C = 3` (number of bases)
- `R = 3` (peptide index matrix vector length)
- `L = 75` (peptide embedding length)

For Phase-1 smoke test: use `update_step_test = 10` (faster than 1000) to validate the pipeline. For Phase-2: revert to 1000 per the paper.

## 7. Cross-baseline format reconciliation

| Question | Answer |
|----------|--------|
| Needs V/J? | **No.** Only peptide + CDR3β. |
| Needs TRA on TRB-only? | **No.** TRB CDR3 only. |
| MHC representation? | **Not used.** PanPep doesn't take MHC. |
| CDR3 columns? | CDR3β only (`trb_cdr3`). cdr1/cdr2 unused. |
| Negative sampling? | We provide negatives in the support set (`Label=0` rows) via `track_b/shared/negative_sampling.sample_negatives()`. The query set is purely the test rows we want to score (with `Label=Unknown`). |
| Internal splits / clustering? | None. PanPep trains on episodes (one peptide per task); the published meta-learner has already seen many TCR-peptide pairs from VDJdb/McPAS, including ones that may overlap our TCRBench data. Document this as a known limitation. |

## 8. Known divergences from TCRBench

1. **No retraining**: as discussed in §5 — there is no training script. We use the published meta-learner. This means our "retrained" arm for PanPep is either skipped or approximated via heavy fine-tuning. Document the choice clearly.
2. **Train-set overlap**: the published meta-learner was trained on VDJdb / McPAS-TCR (Feb–Jul 2020 versions). TCRBench AS-Class-I includes VDJdb among its source databases (`source_db_set` includes 'vdjdb'). **There is data overlap between PanPep's training and our TCRBench test sets.** This is a fundamental issue: the "iid" condition for PanPep is not truly held-out, and AUC numbers may be inflated. Phase 2 needs to filter or report this.
3. **MHC ignored**: same as NetTCR-2.2; PanPep can't differentiate alleles.
4. **The 2025 reusability caveat**: per PMC12458544, PanPep's majority-mode advantage is sensitive to the negative-sampling strategy. Hard negatives → performance collapses. Background draw → looks great. Document the negative-sampling strategy used in our integration.
5. **GPU compatibility**: torch 1.10.2 doesn't support sm_89 (L4). Smoke test runs on CPU. Source code uses hardcoded `.cuda()` calls (`PanPep.py:170`); we'll patch to allow CPU fallback.

## 9. Open questions

1. **Retrained arm**: skip (option a) or approximate (option b)? Recommend (a). Confirm.
2. **VDJdb overlap with TCRBench**: how do we want to handle this? Options:
   - (i) Filter TCRBench test rows whose source_db_primary is `vdjdb` (loses ~30%? — depends).
   - (ii) Just report AUC and note the overlap caveat.
   - (iii) Restrict to peptides not in PanPep's training set (would need to extract that list — non-trivial).
3. **Mode-switching is moot on TCRBench**: per §5 (revised), all conditions resolve to zero-shot because the iid stratification deliberately disjoints peptide clusters between train and test files. Phase 2 reports only zero-shot AUC.

---

**STOP gate**: review before conversion / training code is written.
