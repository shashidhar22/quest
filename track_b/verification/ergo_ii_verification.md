# ERGO-II Verification Document

**Status**: DRAFT — pending human review before any conversion / training code is written.

**Env**: `/home/ubuntu/miniforge3/envs/track_b_ergo_ii` built from `track_b/ergo_ii/env.yml`. Imports verified: torch 1.7.1, pytorch-lightning 0.7.6, Python 3.7. Repo cloned to `track_b/ergo_ii/repo/`. AE pre-trained weights symlinked at `repo/TCR_Autoencoder/`. Author Predict.py with bundled VDJdb model fails on L4 (CUDA capability mismatch — see §8.5); smoke test will use LSTM mode trained from scratch on CPU.

## Source material

- **Paper**: Springer I, Tickotsky N, Louzoun Y. *Contribution of T Cell Receptor Alpha and Beta CDR3, MHC Typing, V and J Genes to Peptide Binding Prediction.* Frontiers in Immunology 12:664514, 2021. https://www.frontiersin.org/articles/10.3389/fimmu.2021.664514/full
- **Predecessor**: Springer I, Besser H, Tickotsky-Moskovitz N, Dvorkin S, Louzoun Y. *Prediction of Specific TCR-Peptide Binding From Large Dictionaries of TCR-Peptide Pairs.* Frontiers in Immunology 11:1803, 2020.
- **Repo**: https://github.com/IdoSpringer/ERGO-II (cloned to `track_b/ergo_ii/repo/`).
- **Local files inspected**: `repo/readme.md`, `repo/requirements.txt`, `repo/example.csv`, `repo/Trainer.py`, `repo/Loader.py`, `repo/Sampler.py`, `repo/Predict.py`.

## 1. Paper Methods summary

ERGO-II is a feature-flexible binary classifier for TCR-peptide binding. Inputs: TCRβ CDR3 (always), peptide (always), optionally TCRα CDR3, V/J genes (α and/or β), MHC, and T-cell type (CD4/CD8). TCRs are encoded with either an autoencoder (AE) or LSTM; peptides are always LSTM-encoded; categorical features (V/J/MHC/T-type) use 50-dim learned embeddings (*Categorical Features Encoding*). Two MLP heads — one for samples with TCRα present, one without — produce a sigmoid score (*Configurations*). Training uses Adam (lr 1e-4, L2 1e-5), batch size 128, weighted BCE loss with positives weighted 5× negatives. Early stopping triggers after 3 consecutive epochs of declining validation AUC (*Configurations and Hyperparameters Tuning*). Negatives are constructed by random pairing — TCR features (CDR3α, CDR3β, V/J/T-type) from sample i with peptide and MHC from sample j — at a 5:1 ratio (*Data Sampling*). Evaluation reports AUC across SPB (Single Peptide Binding), TPP-I (random 80/20), TPP-II (held-out TCRβ), and TPP-III (held-out TCRβ + held-out peptide) test scenarios (*Accuracy Tests*, Table 1).

## 2. Repo canonical workflow

### Training
The repo ships only inference (Predict.py) plus the training scaffold (Trainer.py with `if __name__ == '__main__': pass`). The author's training script is invoked via `Trainer.ergo_ii_experiment()`:

```bash
python Trainer.py <iter> <gpu> <dataset> <tcr_encoding_model> --use_alpha --use_vj --use_mhc --use_t_type
```

Where `dataset ∈ {mcpas_human, vdjdb, vdjdb_no10x}`, `tcr_encoding_model ∈ {LSTM, AE}`. Flags `--use_alpha --use_vj --use_mhc --use_t_type` are individually toggleable.

But the training data loader hard-codes `Samples/{dataset}_train_samples.pickle` — a pickled list of dicts produced by `Sampler.sample_data()`. So our integration must:
1. Convert TCRBench AS-TRB-I to a pickled list of `dict` rows (not CSV).
2. Either rename our dataset key to one of the existing keys (e.g., `mcpas_human`) so paths resolve, or patch `Trainer.py` to accept a custom dataset name. The path-rewriting approach is cleaner and avoids touching their code.

### Inference
```bash
python Predict.py <dataset> <input.csv>
```
Reads CSV with columns `TRA, TRB, TRAV, TRAJ, TRBV, TRBJ, T-Cell-Type, Peptide, MHC` (see `repo/example.csv`), prints predictions. The trained model is loaded from `Models/version_<flags>/checkpoints/`.

### Sanity check (uses pre-trained models)
```bash
cd repo
python Predict.py vdjdb example.csv  # uses bundled VDJdb-trained model
```
This is the smallest working end-to-end test — use it once the env is built to confirm the env actually runs the model.

## 3. Data format requirements

### Inference format (CSV with header)

Columns and exact format from `repo/example.csv`:

| Column | Format | Example | Required? |
|--------|--------|---------|-----------|
| `TRA` | CDR3 amino acid string OR empty | `CAVSAASGGSYIPTF` | optional (empty allowed) |
| `TRB` | CDR3 amino acid string | `CASSFSGNTGELFF` | **required** |
| `TRAV` | IMGT gene name w/o allele | `TRAV3` | optional |
| `TRAJ` | IMGT gene name w/o allele | `TRAJ6` | optional |
| `TRBV` | IMGT gene name w/o allele | `TRBV12-3` | optional |
| `TRBJ` | IMGT gene name w/o allele | `TRBJ2-2` | optional |
| `T-Cell-Type` | `CD4` / `CD8` / empty | `CD8` | optional |
| `Peptide` | amino acid string | `RAKFKQLL` | **required** |
| `MHC` | HLA name, 2-digit form usual | `HLA-B*08` | optional |

A single-row example matching our AS-TRB-I rows (TRB-only, TRBV/J extracted from `trb_full`):
```
TRA,TRB,TRAV,TRAJ,TRBV,TRBJ,T-Cell-Type,Peptide,MHC
,CASSLGGLAADTQYF,,,TRBV6-5,TRBJ2-3,CD8,AVFDRKSDAK,HLA-A*11:01
```

**Important format notes** discovered from reading `Loader.py`:
- V/J/MHC are categorical — the model embeds them as **strings**. Whatever string we use must be consistent between train and test, and at inference time only strings that appeared in training have learned embeddings (`mhctox` lookup at `Loader.py:153`); unseen strings get `UNK` (index 0).
- This means we have flexibility on MHC format: we can use `HLA-A*11:01` (4-digit), `HLA-A*11` (2-digit), or even the raw protein sequence — as long as we're consistent. The example.csv uses 2-digit, but the paper says `HLA-A*02:01 and HLA-A*02 have different encodings` so **both forms work; the model just learns separate embeddings for each.**
- Author's pretrained models were trained on McPAS / VDJdb with the 4-digit form (`HLA-A*02:01`), so the **original-config evaluation** (using their pretrained model) requires the 4-digit form. We use 4-digit consistently.
- Empty / NaN values for optional columns become `'UNK'` and are mapped to index 0 (padding).

### Training format (pickled list of dicts)

Each dict (see `Sampler.read_data` for the schema):
```python
{
    'tcra': str | 'UNK',     # CDR3 alpha AA seq
    'tcrb': str,             # CDR3 beta AA seq (required)
    'va': str | 'UNK',       # e.g., 'TRAV3'
    'ja': str | 'UNK',
    'vb': str | 'UNK',
    'jb': str | 'UNK',
    't_cell_type': str,      # 'CD4' / 'CD8' / nan
    'peptide': str,          # required
    'protein': str,          # epitope source protein name; only used for diabetes filter — pass any string
    'mhc': str | 'UNK',
    'sign': int,             # 1 = positive, 0 = negative
}
```

Validation rules from `Sampler.invalid()`: TCRβ and peptide must be non-NaN and contain only the 20 standard AAs. The implementation drops 'X', 'B', etc.

## 4. Negative sampling strategy

ERGO-II's native strategy (`Sampler.negative_examples`):
1. For each negative slot, pick two random positive samples i and j.
2. Take **TCR-side** features (`tcra, tcrb, va, vb, ja, jb, t_cell_type`) from sample i.
3. Take **peptide-side** features (`peptide, protein, mhc`) from sample j.
4. Reject if the resulting (tcra, tcrb, ..., mhc) tuple already exists in the positive set OR was previously generated.
5. Generate `5 × |positives|` negatives.

**Comparison to our convention** (from Track B brief, mirrored from `BENCHMARK_SPLITS.md` §12):
- Our convention: hold (tcr, mhc) constant, swap only the peptide; 5×; filter against positive set; seed=43.
- ERGO-II: swaps (peptide, mhc) jointly, decouples TCR from MHC.

**Decision**: use **our convention** (peptide-only swap, mhc held constant). Reason: this matches the rest of Track B (NetTCR, PanPep, TEIM all get the same negatives) and produces apples-to-apples comparisons. The downside is that ERGO-II "sees" a different negative distribution than its native training; this is a known divergence and is documented here. If reviewer questions arise, we can also report ERGO-II with its native negatives as an ablation.

## 5. Train/test split assumptions

ERGO-II's native split is a random 80/20 of the positive set (`Sampler.train_test_split` line 79–92), with negatives generated separately from train and test pools. Their TPP-II / TPP-III protocols re-do this split with TCRβ-cluster and peptide-cluster constraints respectively (paper §Accuracy Tests).

**TCRBench AS-TRB-I provides** strict cluster-disjoint partitions: train/val/test_iid + 4 novelty-condition test files. Our integration **uses TCRBench's splits directly** (no re-shuffling), and reports per-condition AUC. Mapping:

| ERGO-II protocol | Closest TCRBench condition |
|---|---|
| TPP-I (random) | `test_iid` |
| TPP-II (new TCRβ) | `test_novel_tcr` |
| TPP-III (new TCRβ + new peptide) | `test_level4` (also requires new allele, slightly stricter) |
| — | `test_novel_pep` (held-out peptide; ERGO-II didn't have this exact protocol) |
| — | `test_novel_allele` (held-out HLA; ERGO-II didn't have this exact protocol) |

Reviewer note: TCRBench's `test_novel_pep` is a stricter peptide-novelty bar than ERGO-II's TPP-II/III because we use Levenshtein-clustered peptide groups, not raw string disjointness. Phrase results carefully.

## 6. Configuration choices

We use **ERGO-II.vdj** per Lu et al. 2025 (top-performer config in CDR3β+others category):
- `tcr_encoding_model = LSTM` (LSTM gave better AUC than AE in the paper for VDJdb)
- `--use_vj` (V and J gene embeddings)
- `--use_mhc`
- `--use_t_type` (CD4/CD8)
- `--use_alpha = False` (TRB-only on AS-TRB-I; turn on for AS-Paired-I)

Hyperparameters use the paper defaults (Adam lr 1e-4, L2 1e-5, batch 128, dropout 0.1, lstm_dim 500, encoding_dim 100, aa_embedding_dim 10, cat_embedding_dim 50).

## 7. Cross-baseline format reconciliation (per the plan)

| Question | Answer |
|----------|--------|
| Needs V/J? Format? | **Yes**, format: bare IMGT gene names (`TRBV12-3`, `TRBJ2-2`) without `*01` allele suffix. Our parquet has no V/J columns → **extract from `trb_full` via tidytcells** (per resolved decision in plan). Cache to `track_b/shared/cache/vj_calls.parquet`. |
| Needs TRA on TRB-only? | **No** — `TRA` field can be empty / NaN; ERGO-II's "MLP I" path handles it. Just leave the TRA column blank in the CSV and pass `'UNK'` for `tcra/va/ja` in the training-pickle dicts. |
| MHC representation? | HLA name string. We use 4-digit form `HLA-A*11:01` (matches author's pretrained model conventions). Uses `track_b/shared/allele_utils.resolve_allele()` to recover HLA names from protein-sequence-fallback rows. Drop rows that resolve to `None`. |
| CDR3 columns? | CDR3β only (`trb_cdr3`). cdr1/cdr2 are not used by ERGO-II at all. Our 17.7% null rate on `trb_cdr1/2` is therefore irrelevant for this baseline. |
| Negative-sampling integration? | We construct negatives ourselves via `track_b/shared/negative_sampling.sample_negatives()` (peptide-only swap, 5×, seed=43, positive-filtered). Output rows get `sign: 0` in the training-pickle dict. **We do NOT call ERGO-II's native `negative_examples()`** — we override it. |
| Internal splits / clustering? | ERGO-II's `Sampler.train_test_split` is random 80/20. **We do NOT call it** — we feed TCRBench-curated train and per-condition test pickles directly. The `Trainer.train_dataloader()` reads `Samples/{dataset}_train_samples.pickle`; we write our train pickle there with the dataset name set to e.g. `tcrbench_as_trb_i`. |

## 8. Known divergences from TCRBench

1. **Negative sampling**: their joint (peptide, MHC) swap → our peptide-only swap. Documented in §4.
2. **No `protein` field in our data**: ERGO-II's data has an `Antigen.protein` field (used only for the diabetes weight-up dataset, not standard training). We pass empty / `'UNK'`.
3. **No CD4/CD8 annotation in TCRBench**: AS-TRB-I rows don't have T-cell-type. We pass `nan` / `'UNK'`. Paper says T-cell-type is one of the optional features; setting it to UNK for all rows means the embedding becomes a constant (the UNK index 0), so the model effectively trains without the T-cell-type signal. Acceptable — matches one of the paper's documented configurations.
4. **MHC class II support**: ERGO-II's MHC encoding is a flat embedding lookup; it doesn't distinguish I from II (just embeds whatever string). Phase 1 is Class I only (`mhc_one_allele`).
5. **GPU compatibility (CONFIRMED)**: pytorch 1.7.1 / pytorch-lightning 0.7.6 do not support L4 (sm_89). Tested: `RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED` with `UserWarning: NVIDIA L4 with CUDA capability sm_89 is not compatible with the current PyTorch installation. The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75.` Smoke test must run on **CPU**. For Phase 2 we need to either (a) keep CPU training (~10× slower, but feasible at 60K positives × 6 batch ≈ a few hours), (b) modernize the ERGO-II source to pytorch-lightning ≥ 2.0, or (c) run on a Turing/Volta-or-older GPU we don't have access to. **Open question for Phase 2 scoping**.
6. **V/J extraction quality**: tidytcells V/J inference from CDR3β alone is ~95% accurate. About 5% of rows may get wrong V/J calls or `None`. Phase 2 should QC the extraction-failure rate and consider using a J-gene heuristic.

## 9. Open questions

1. **Original-config evaluation** (Lu et al.'s "original" arm): the bundled VDJdb-pretrained model in `repo/Models/version_1veajht/` will only score TCRs/peptides that were in its training vocabulary — others get `UNK` embeddings. AUC may be deflated for novel categorical values. Do we (a) score everything anyway, (b) restrict to the intersection, or (c) report both? Recommend (c) but want confirmation before Phase 2.
2. **Peptide length filter**: ERGO-II uses an LSTM with no explicit length cap; in practice, the longest peptide in McPAS/VDJdb is ~25 AAs. Our AS-Class-I has peptides up to ~14–17 AAs typical, but a few outliers reach 20+. Should we cap or filter? Recommend keeping all and letting the LSTM handle it (matches paper's permissive treatment).
3. **GPU strategy** (see §8.5): is CPU-only training acceptable for Phase 2, or do we modernize the source?

---

**STOP gate**: This document needs review before any conversion / training code is written. The smoke test pipeline (Step 1e in the plan) will be implemented after approval.
