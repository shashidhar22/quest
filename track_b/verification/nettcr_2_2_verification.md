# NetTCR-2.2 Verification Document

**Status**: DRAFT — pending human review before any conversion / training code is written.

**Env**: `/home/ubuntu/miniforge3/envs/track_b_nettcr_2_2` built from `track_b/nettcr_2_2/env.yml` (Python 3.10, tensorflow 2.11). Imports verified. **GPU note**: TF 2.11.0 needs CUDA 11 system libs (`libcudart.so.11.0`, `libcudnn.so.8`, etc.) which are not installed — TF falls back to CPU. Phase 1 smoke test runs on CPU; Phase 2 needs either CUDA 11 system install or `pip install nvidia-cudnn-cu11 nvidia-cublas-cu11` (≈2 GB disk). Repo cloned to `track_b/nettcr_2_2/repo/`.

## Source material

- **Paper**: Jensen MF, Nielsen M. *NetTCR-2.2 — Improved TCR specificity predictions by combining pan- and peptide-specific training strategies, loss-scaling and integration of sequence similarity.* eLife 12:RP93934, 2024. https://elifesciences.org/articles/93934
- **Repo**: https://github.com/mnielLab/NetTCR-2.2 (cloned to `track_b/nettcr_2_2/repo/`).
- **Local files inspected**: `repo/README.md`, `repo/environment.yml`, `repo/src/train_nettcr_2_2_pretrained.py`, `repo/src/nettcr_archs.py`, `repo/data/examples/train_example.csv`, `repo/data/small_example.csv`.

## 1. Paper Methods summary

NetTCR-2.2 is a CNN-based binary classifier for paired αβ TCR-peptide binding. Inputs: peptide (max 12 AA) + 6 CDR sequences (CDR1/2/3 of α and β). Each input is BLOSUM50-encoded then divided by 5 for normalization, padded to fixed length (a1=7, a2=8, a3=22, b1=6, b2=7, b3=23). Each input feeds 5 parallel Conv1D filters with kernel sizes {1,3,5,7,9}, each Conv1D has 16 filters with ReLU activation, followed by GlobalMaxPooling1D. The 35 pooled vectors (7 inputs × 5 kernels) are concatenated, dropout (default 0.6), Dense(64) → Dense(1) sigmoid (*Embedding*; `nettcr_archs.py:CNN_CDR123_global_max_two_step_pre_training`).

Training: BCE loss, Adam (lr=1e-3 default), batch 64, early stopping on `val_auc_01` (AUC restricted to FPR≤0.1) with patience 100, max 200 epochs. Sample-weighted to compensate for peptide imbalance — weights are `log2(N_total / count(pep)) / log2(unique_peps)` then re-normalized so loss is comparable (`train_nettcr_2_2_pretrained.py:83-87`). Negative sampling: peptide-swap within partition, with the constraint that swapped peptides must have Levenshtein distance ≥ 3 from the original peptide; 1:5 positive:negative ratio (paper *Data partitioning and generation of swapped negatives*).

**Critical finding**: **MHC is not used as input** in NetTCR-2.2. Paper Results section: *"Due to the limited number of peptides (and HLAs), HLA is not included in the model."* The training data CSV has an `allele` column but it's metadata only — never fed to the model.

Evaluation: nested cross-validation (5-fold outer, 4-fold inner), reports peptide-weighted and unweighted AUC + AUC0.1.

## 2. Repo canonical workflow

### Pretrained mode (the variant we use)
```bash
python src/train_nettcr_2_2_pretrained.py \
    --train_data <train.csv> --val_data <val.csv> \
    --outdir <outdir> --model_name <name>
```

This script does TWO rounds:
1. **Pan training**: trains all peptides jointly with sample-weighted loss; saves `<outdir>/checkpoint/<name>.h5`.
2. **Per-peptide fine-tune**: for each peptide in `pep_list`, loads the pan checkpoint, freezes the `first_*` layers, unfreezes the `second_*` layers, trains only on rows of that peptide, saves `<outdir>/<peptide>/checkpoint/<name>.h5`.

So the final output is one `.h5` per peptide (plus a `.tflite`). At inference time, predict.py loads the per-peptide model.

### Inference
```bash
python src/predict.py --test_data <csv> --outdir <outdir> --model_name <name> --model_type pretrained
```

### Sanity check (use bundled pretrained models)
```bash
cd repo
python src/predict.py --test_data data/small_example.csv \
    --outdir models/nettcr_2_2_peptide --model_name "t.1.v.2" --model_type peptide
```

## 3. Data format requirements

CSV with header. Required columns (assertions in `train_nettcr_2_2_pretrained.py:77-80`):
- `peptide` — amino acid string (max 12 AA)
- `A1`, `A2`, `A3` — TRA CDR1/2/3 amino acid strings
- `B1`, `B2`, `B3` — TRB CDR1/2/3 amino acid strings
- `binder` — 1 (positive) or 0 (negative)

Optional columns seen in `train_example.csv`: `index, allele, origin, partition, original_peptide, original_index, 10x, reference`. These are metadata only.

Single-row example matching our AS-TRB-I structure (TRB-only; A1/A2/A3 zero-padded → empty string):
```
peptide,A1,A2,A3,B1,B2,B3,binder
AVFDRKSDAK,,,,MNHEY,SVGAGI,ASSLGGLAADTQYF,1
```

**Sequence length max** (model-fixed): a1=7, a2=8, a3=22, b1=6, b2=7, b3=23, pep=12. Sequences longer than max are silently truncated by `enc_list_bl_max_len`.

## 4. Negative sampling strategy

NetTCR-2.2's native negative sampling (paper *Data partitioning and generation of swapped negatives*):
1. For each (positive) peptide, swap with TCRs that bind a peptide having Levenshtein distance ≥ 3 from it.
2. 1:5 positive:negative ratio.
3. Within partition (so negatives don't leak across folds).

**Comparison to our convention**: identical except for the Levenshtein-distance filter. Ours samples any peptide from the train pool; theirs filters out near-duplicate peptides. Our negatives may therefore be *easier* (some negatives may have peptides Levenshtein-1 from the original, which the model could call positive if the encoder isn't sharp enough). For Phase 1 smoke test we use **our convention** for cross-baseline consistency. For Phase 2 we may add the Levenshtein-distance filter as an option.

**Decision for Phase 1**: use `track_b/shared/negative_sampling.sample_negatives()` (5×, seed=43, positive-set filter only). Document the divergence in §8.

## 5. Train/test split assumptions

NetTCR-2.2 uses random partitioning (5 outer folds × 4 inner) for cross-validation, with redundancy reduction (Hobohm1 algorithm + kernel similarity, threshold 0.95 for CDR3) — they remove near-duplicate TCRs but DO NOT cluster epitopes for split-disjointness.

**TCRBench AS-TRB-I** uses cluster-disjoint splits. Our integration **bypasses NetTCR's CV** entirely:
- Pass `as_trb_i_train.parquet` → `train.csv`.
- Pass `as_trb_i_val.parquet` → `val.csv` (used by NetTCR for early stopping on `val_auc_01`).
- Eval on each `as_trb_i_test_<condition>.parquet` separately, reporting AUC + AUC0.1 per condition.

Reviewer note: NetTCR's redundancy reduction is internal to their dataset; our TCRBench data is pre-clustered to a stricter standard. We don't re-run their Hobohm1 step. Document this.

## 6. Configuration choices

We use **pretrained mode** per the brief (`train_nettcr_2_2_pretrained.py`). Hyperparameters use the script defaults: dropout 0.6, lr 1e-3, batch 64, patience 100, max 200 epochs. Smoke-test override: max 5 epochs, patience 5.

Paper note: pretrained mode is the recommended general-purpose mode (combines pan + peptide-specific signal). For peptides with only a few binders in the training set, pan would be more appropriate; for those with hundreds, peptide is the gold-standard. Pretrained is the compromise.

## 7. Cross-baseline format reconciliation

| Question | Answer |
|----------|--------|
| Needs V/J? Format? | **No**. NetTCR-2.2 takes only peptide + 6 CDRs as input. V/J are not used (CDRs are inferred from V gene + CDR3 by tooling outside NetTCR; we provide them directly from `*_cdr1/2/3` columns). |
| Needs TRA on TRB-only? | **Yes** (architecturally). The CNN has fixed input shapes for A1/A2/A3. **Missing TRA is handled by zero-padding** (empty string → BLOSUM50 encoding of all-zeros after right-padding). The CNN branches for A1/A2/A3 will produce constant/near-zero activations on these branches, and the dense layer learns to weight them low. This matches the Lu et al. 2025 precedent for evaluating NetTCR-2.2 on β-only datasets. |
| MHC representation? | **Not used.** The model never sees MHC. We can ignore `mhc_one_allele` entirely for this baseline. (Allele could be added as a metadata column for ranking analysis but doesn't affect predictions.) |
| CDR3 columns? | **All 6 CDRs needed** (A1, A2, A3, B1, B2, B3). Our parquet has `tra_cdr1/2/3` (100% null in TRB-I — zero-pad), `trb_cdr1/2/3` (B1/B2 17.7% null, B3 0% null). For B1/B2 nulls: drop those rows for now (simpler than imputing CDR1/2 from V gene without V calls). 17.7% drop rate is acceptable for AS-TRB-I (60K → ~50K rows). |
| Negative sampling? | We use `track_b/shared/negative_sampling.sample_negatives()`. Output rows get `binder=0`; positives get `binder=1`. NetTCR-2.2's native sample-weight scheme (log-scaled by peptide frequency) is computed inside `train_nettcr_2_2_pretrained.py:83-87` from the data we pass in — no override needed. |
| Internal splits / clustering? | NetTCR's redundancy-reduction (Hobohm1, 0.95 threshold) is applied to their training data prior to release; their training script does NOT re-cluster. We pass our pre-clustered TCRBench train + val directly. |

## 8. Known divergences from TCRBench

1. **Negative-sampling Levenshtein filter**: NetTCR-2.2 enforces ≥3 Levenshtein distance between original and swapped peptide; we don't. May make our negatives easier. Phase 2 should add an option to apply this filter. (§4)
2. **TRA zero-padding**: We feed empty strings for A1/A2/A3 in AS-TRB-I. The model's first-layer Conv1D on these branches produces constant output — effectively a learned bias. The pan-training step on a TRB-only dataset will likely *not* learn to ignore the α branches gracefully if no rows have α data; the gradient through those branches is just constant noise. **Phase 1 smoke test will show whether this trains stably; if not, we may need to surgically zero-out the α branches in the architecture before training.**
3. **B1/B2 17.7% null**: drop those rows. AS-TRB-I → ~50K rows after filter.
4. **No MHC signal**: The model can't learn allele-specific binding. For our `test_novel_allele` condition, NetTCR-2.2 effectively gives the same prediction regardless of allele — a known limitation that's reviewer-relevant.
5. **Fixed sequence-length caps**: peptides > 12 AAs are truncated. Our AS-TRB-I has some 14–17 AA peptides (e.g., `FAFRWVLGIAY`, `TFEYVSQPFLMDLE` from a quick spot-check of test_iid). Truncation may degrade those rows. Phase 2 may want to filter peptides to len≤12 and report the dropped fraction.

## 9. Open questions

1. **Should we re-train per outer fold like the paper, or just train once on TCRBench's train→val→test split?** Recommend single train/val/test (matches our other baselines and the brief). Cross-validation is only useful if we want to also report variance bars on AUC.
2. **Should we apply Levenshtein-≥3 filter to our negatives for NetTCR-2.2 specifically?** Phase 1: no (cross-baseline consistency). Phase 2: optional ablation. Confirm.
3. **Long peptides**: filter `len(peptide) > 12` rows or accept truncation? Phase 1 smoke test will show how many are truncated.
4. **TRA zero-padding stability**: real concern. If smoke test shows training instability or very-low AUC, may need to either (a) add a small synthetic α placeholder string (`AAAAAA` etc.) or (b) modify the architecture to skip α branches when β-only mode is active. Decide after seeing smoke-test loss curve.

---

**STOP gate**: review before conversion / training code is written.
