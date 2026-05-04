# TEIM Verification Document

**Status**: DRAFT — pending human review before any conversion / training code is written.

**Env**: `/home/ubuntu/miniforge3/envs/track_b_teim` built from `track_b/teim/env.yml` (Python 3.8, torch 1.10.1, pytorch-lightning 1.6.4, ANARCI 2024.05.21 from bioconda). Imports verified; bundled `inference_seq.py` produces predictions on the example input. **Three install gotchas**: (a) pip 24.1+ rejects pl 1.6.4 metadata → pin `pip<24.1`; (b) torch 1.10 + setuptools≥59 hits `distutils.version` stub → pin `setuptools<59`; (c) ANARCI must be installed via `conda install -c bioconda anarci` because TEIM-Seq requires it for CDR3 numbering at inference time. Same L4 sm_89 incompat as ERGO-II / PanPep — smoke test on CPU. Source already CPU-aware. Repo cloned to `track_b/teim/repo/`.

## Source material

- **Paper**: Peng X, Lei Y, Feng P, Jia L, Ma J, Zhao D, Zeng J. *Characterizing the interaction conformation between T-cell receptors and epitopes with deep learning.* Nature Machine Intelligence 5:395–407, 2023. https://www.nature.com/articles/s42256-023-00634-4
- **Repo**: https://github.com/pengxingang/TEIM (cloned to `track_b/teim/repo/`).
- **Local files inspected**: `repo/README.md`, `repo/scripts/inference_seq.py`, `repo/scripts/models.py`, `repo/train_teim/scripts/train_seqlevel.py`, `repo/train_teim/utils/dataset.py`, `repo/train_teim/configs/seqlevel_all.yml`, `repo/data/binding_data/{vdj_all,mcpas_seen,mcpas_unseen,ncov_seen,ncov_unseen}.tsv`, `repo/inputs/inputs_bd.csv`.

## 1. Methods summary

TEIM has two heads sharing a sequence encoder:
- **TEIM-Seq** (sequence-level binding): predicts P(binding) ∈ [0,1] for a (CDR3β, epitope) pair.
- **TEIM-Res** (residue-level): predicts pairwise distance + contact map between CDR3β residues and epitope residues. Requires structural training data.

The brief restricts us to **TEIM-Seq** since TCRBench has no structural data. From `train_teim/scripts/train_seqlevel.py`:
- Architecture: CNN encoder for CDR3 + autoencoder-pretrained encoder for epitope (with a frozen `epi_ae.ckpt`); these are combined via a learned interaction module (`inter_type: mul` for multiplicative). Final dense produces one logit per pair.
- Loss: BCE (`F.binary_cross_entropy(y_hat, y)`) — no class weighting beyond what comes from data balance.
- Optimizer: AdamW, lr 2e-4, weight_decay 0; StepLR scheduler decays gamma=0.5 every 200 steps.
- Batch size: 512, max 25 epochs.
- Per-epitope AUC and per-epitope AUPR are tracked; mean and overall metrics are logged.

Inputs: CDR3β amino acid string + epitope amino acid string. **MHC is NOT used.** V/J columns exist in the upstream TSVs but are not read by `SeqLevelDataset` (it loads only `cdr3, epitope, label`).

## 2. Repo canonical workflow

### Inference (TEIM-Seq, sequence-level binding)
```bash
# Put TCR-epitope pairs in inputs/inputs_bd.csv with columns cdr3,epitope.
python scripts/inference_seq.py
# Output at outputs/sequence_level_binding.csv with column 'binding'.
```

The pretrained checkpoint is `ckpt/teim_seq.ckpt`, loaded via `BindPredictor`.

### Training (TEIM-Seq from scratch)
```bash
cd repo/train_teim
python scripts/train_seqlevel.py --config configs/seqlevel_all.yml
```

Default config trains on the union of `vdj_all, ncov_unseen, mcpas_unseen, ncov_seen, mcpas_seen` TSVs (~13K positive pairs combined) with `negative: shuffle` (peptide-swap), 80/20 train-val split, batch 512, 25 epochs.

### Sanity check
```bash
cd repo
python scripts/inference_seq.py
# Reads inputs/inputs_bd.csv (3 example rows shipped) → outputs/sequence_level_binding.csv
```

## 3. Data format requirements

### Inference CSV (input)
```csv
cdr3,epitope
CASSLGGLAADTQYF,AVFDRKSDAK
```
Two columns. No labels needed at inference time.

### Training TSV (`data/binding_data/*.tsv`)
```tsv
cdr3	epitope	v_gene	j_gene	label
CASSYQTGAAYGYTF	NLVPMVATV	TRBV6-5*01	TRBJ1-2*01	1
```
Five columns; `SeqLevelDataset.load_data` only reads `[cdr3, epitope, label]`. v_gene/j_gene are ignored. Label is 0 or 1.

There's also an **optional** `positive_epi_dist.tsv` keyed by epitope, used only for per-epitope AUC reporting (assigns each epitope an integer ID). Missing the file is graceful: `epi_id` becomes `-1` and per-epitope metrics are skipped.

### Single-row example for our integration:
```tsv
cdr3	epitope	label
CASSLGGLAADTQYF	AVFDRKSDAK	1
```

## 4. Negative sampling strategy

TEIM's native sampling (`SeqLevelDataset.make_shuffled_nega`):
1. Pivot positive pairs into a binary matrix (epi × cdr3).
2. Cells where `label != 1` are candidate negatives.
3. For each epitope, sample `5 × n_positives_for_this_epitope` negatives from its candidate cells.
4. Output is the union of positives + sampled negatives.

This is **per-peptide stratified peptide-swap, ratio 5:1** — almost identical to our `track_b/shared/negative_sampling.sample_negatives()`. The key differences:
- TEIM's sampling deduplicates against the positive set implicitly via the pivot (negatives can't be positives).
- TEIM samples negatives from any (cdr3 ∈ training set, epi) pair where the (cdr3, epi) is NOT a positive — i.e., uses any TCR for this epi, not just TCRs paired with other epis. Slight conceptual difference from ours.

For Phase 1: use **our** `sample_negatives()` for cross-baseline consistency (same negatives across ERGO-II / NetTCR / TEIM). Phase 2 may run TEIM with its native shuffling as an ablation.

## 5. Train/test split assumptions

TEIM offers three split modes (paper §Methods, repo configs):
- `cv-shuffle`: random 5-fold cross-validation (default for `seqlevel_cv_shuffle.yml`).
- `cv-new_epitope`: held-out epitope-clusters per fold (CD-HIT 50% identity cluster, file `data/cluster/seqlevel_epi_cluster_0.5.pkl`).
- `train-val`: simple 80/20 random split (used by `seqlevel_all.yml`).

For our integration: bypass TEIM's CV. Pass `as_trb_i_train.parquet` as the training file and use TCRBench-curated train/val/test splits.

## 6. Configuration choices

We use **TEIM-Seq, binding head only**, per the brief. Hyperparameters use `seqlevel_all.yml` defaults (lr 2e-4, batch 512, 25 epochs, dim_hid 256, layers_inter 2, inter_type 'mul'). Smoke test override: 2 epochs, batch 32.

The `epi_ae.ckpt` autoencoder for the epitope encoder must be present at `ckpt/epi_ae.ckpt` — it's bundled in the repo.

## 7. Cross-baseline format reconciliation

| Question | Answer |
|----------|--------|
| Needs V/J? | **No.** TSVs have V/J columns but `SeqLevelDataset` ignores them. CDR3 + epitope only. |
| Needs TRA on TRB-only? | **No.** TEIM-Seq is CDR3β + epitope only. TRA is unused. |
| MHC representation? | **Not used.** TEIM-Seq doesn't take MHC as input. Same as NetTCR-2.2 / PanPep. |
| CDR3 columns? | CDR3β only (`trb_cdr3`). cdr1/cdr2 unused. |
| Negative sampling? | We use `track_b/shared/negative_sampling.sample_negatives()` and write the labeled rows directly into the training TSV. We disable TEIM's `negative: shuffle` config (set `negative: original` so it uses our pre-labeled rows as-is). |
| Internal splits / clustering? | TEIM's training script uses one of the provided splits. We bypass via `negative: original` and our own train/val files. |

## 8. Known divergences from TCRBench

1. **Negative-sampling alignment**: TEIM's `shuffle` is per-peptide stratified, ours is per-row independent peptide-swap. Slight distributional difference; documented.
2. **MHC ignored**: same caveat as NetTCR-2.2 / PanPep — `test_novel_allele` condition can't be predicted differently than `test_iid` on the MHC axis.
3. **Pretrained model overlap**: the bundled `teim_seq.ckpt` was trained on VDJdb + McPAS + NCOV. TCRBench AS-Class-I includes VDJdb among its sources → train/test contamination for the "original" arm. Document.
3a. **Hard length caps** (CONFIRMED via source inspection):
    - `data_process.py:168` — `assert max_len_cdr3 <= 20, 'The cdr3 length must <= 20'`
    - `data_process.py:142` — `assert (np.max(len_seqs) <= 12) and (np.min(len_seqs)>=8), ValueError('Lengths of epitopes must be within [8, 12]')`
    Our TCRBench data has CDR3s up to ~25 AA and peptides 8–17 AA. Rows outside `[8,12]` for peptide or `>20` for CDR3 will raise an assertion. **`convert_data.py` must filter these rows before passing to TEIM.** Document the dropped fraction in the smoke-test result.
3b. **ANARCI dependency**: TEIM-Seq pads/aligns CDR3 via ANARCI numbering. ANARCI calls hmmer + germline DBs at runtime — single-row inference takes ~1–3 seconds (slow!). For Phase 2 batch inference on 60K+ rows, this is a non-trivial cost: the script writes a temp FASTA, runs ANARCI as a subprocess, parses the output CSV. Throughput ~200 rows/sec with the full ANARCI invocation; 60K positives ≈ 5 min one-time alignment cost. Accept for now.
4. **GPU compatibility**: torch 1.10.1 doesn't support sm_89 (L4). Smoke test on CPU. Source uses `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` (`inference_seq.py:17`) — already CPU-aware. ✓
5. **Pip metadata issue**: pytorch-lightning 1.6.4 has invalid metadata that recent pip versions reject. Env pinned to `pip<24.1`.

## 9. Open questions

1. **Original-arm overlap with VDJdb**: how do we want to report TEIM's "original" (pretrained) numbers given likely train-test contamination on VDJdb-sourced rows? Same options as PanPep §9. Recommend: (ii) report AUC + caveat for Phase 1; (i) Phase 2 ablation filtering VDJdb-sourced rows.
2. **AUPR vs AUC reporting**: TEIM reports both. Our other baselines report AUC-only. Standardize on AUC + AUC0.1 (matches NetTCR-2.2)? Or add AUPR for TEIM?
3. **Epitope autoencoder cap**: `len_epi: 12` in the seqlevel config implies peptides longer than 12 are truncated by the autoencoder. Same as NetTCR-2.2.

---

**STOP gate**: review before conversion / training code is written.
