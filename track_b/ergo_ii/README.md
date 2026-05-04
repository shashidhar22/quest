# ERGO-II on TCRBench AS Class I

End-to-end pipeline: AS-TRB-I parquet → ERGO-II training format → train → evaluate.

## Setup (one-time)

```bash
conda env create -f env.yml          # creates track_b_ergo_ii (Python 3.7)
git clone --depth 1 https://github.com/IdoSpringer/ERGO-II.git repo
ln -s Models/AE repo/TCR_Autoencoder  # symlink for the bundled AE weights
```

## Run smoke test (100 rows, ~30s on CPU)

```bash
conda activate track_b_ergo_ii
cd /home/ubuntu/quest/track_b/ergo_ii

# 1. Convert
python convert_data.py --task as_trb_i --split train     --n-rows 100 --dataset-name smoke --with-negatives
python convert_data.py --task as_trb_i --split test_iid  --n-rows 50  --dataset-name smoke --with-negatives
# ERGO-II's val_dataloader hard-codes <dataset>_test_samples.pickle:
cp repo/Samples/smoke_test_iid_samples.pickle repo/Samples/smoke_test_samples.pickle

# 2. Train (CPU; sm_89 incompat — see verification doc §8.5)
CUDA_VISIBLE_DEVICES="" python train.py --dataset-name smoke --epochs 2 --device cpu

# 3. Evaluate
CUDA_VISIBLE_DEVICES="" python evaluate.py --dataset-name smoke --split test_iid \
    --train-dataset-name smoke --device cpu
```

## Configuration

ERGO-II.vdj LSTM mode (per Lu et al. 2025 top-performer config):
- `tcr_encoding_model = LSTM`
- `--use_vj` (bare IMGT names: `TRBV6-5`)
- `--use_mhc` (HLA name: `HLA-A*11:01`)
- `--use_alpha = False` (TRB-only on AS-TRB-I; toggle on for AS-Paired-I)
- `--use_t_type = False` (TCRBench has no CD4/CD8 annotation)

V/J: Phase-1 stub returns `'UNK'`. Phase 2 will plug in ANARCI / tidytcells germline matching (see `track_b/shared/extract_vj.py`).

## Phase 1 smoke-test result (this commit)

```
AUC: 0.5232
Accuracy: 0.647
n_pos: 25, n_neg: 125
```

Above-majority-class accuracy and slightly-above-chance AUC after 2 epochs on 100 training rows confirm the pipeline is functional. Phase 2 trains on the full 60K AS-TRB-I positives + 300K negatives.

## Files

- `env.yml` — conda env spec
- `repo/` — git clone (gitignored)
- `convert_data.py` — parquet → ERGO-II pickle (training) + CSV (inference)
- `train.py` — wraps repo/Trainer.py with our hparams
- `evaluate.py` — loads checkpoint, scores eval pickle, writes metrics JSON
- `results/` — checkpoints, metrics JSON, predict CSVs

See `track_b/verification/ergo_ii_verification.md` for the full integration audit.
