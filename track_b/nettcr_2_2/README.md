# NetTCR-2.2 on TCRBench AS Class I

## Setup

```bash
conda env create -f env.yml          # creates track_b_nettcr_2_2 (Python 3.10, TF 2.11)
git clone --depth 1 https://github.com/mnielLab/NetTCR-2.2.git repo
```

## Run smoke test (100 train rows / 50 val / 50 test, ~10s on CPU)

```bash
conda activate track_b_nettcr_2_2
cd /home/ubuntu/quest/track_b/nettcr_2_2

python convert_data.py --task as_trb_i --split train     --n-rows 100 --out smoke_train.csv  --with-negatives
python convert_data.py --task as_trb_i --split val       --n-rows 50  --out smoke_val.csv    --with-negatives
python convert_data.py --task as_trb_i --split test_iid  --n-rows 50  --out smoke_test_iid.csv --with-negatives

CUDA_VISIBLE_DEVICES="" python train.py --train smoke_train.csv --val smoke_val.csv \
    --outdir smoke --model-name smoke --epochs 2 --patience 2 --batch-size 32

CUDA_VISIBLE_DEVICES="" python evaluate.py --test smoke_test_iid.csv \
    --model-path smoke/checkpoint/smoke.h5 --out smoke_test_iid_metrics.json
```

## Why we wrote our own train.py instead of calling repo/src/train_nettcr_2_2_*.py

The upstream training scripts call `pd.read_csv(...)` without
`keep_default_na=False`. That converts our empty α-chain CDR cells (TRB-only
data zero-pads A1/A2/A3 with empty strings) to NaN, which crashes
`keras_utils.enc_list_bl_max_len` because `len(NaN)` raises `TypeError`.

Our train.py imports `nettcr_archs.CNN_CDR123_global_max` directly and
re-implements the training loop with `keep_default_na=False` on CSV reads.
The architecture is identical to upstream; only data loading differs.

## Configuration

- Pan-mode for smoke test (single training round); pretrained-mode planned for Phase 2.
- TRB-only data: A1/A2/A3 zero-padded as empty strings → BLOSUM50 padding rows (-5).
- B1/B2-null rows dropped (~17.7% of AS-TRB-I).
- MHC ignored — NetTCR-2.2 doesn't use MHC as input (paper Results).
- Negatives: `track_b/shared/negative_sampling.sample_negatives()` (5×, seed=43, no Levenshtein-distance filter).

## Phase 1 smoke-test result

```
AUC: 0.561
AUC0.1: 0.486
Accuracy: 0.833
n_pos: 48, n_neg: 240
```

After 2 epochs on 78 positives / 390 negatives. Slightly-above-chance AUC + correct accuracy at 5:1 imbalance confirm the pipeline works.

See `track_b/verification/nettcr_2_2_verification.md` for the full integration audit (and the Lu et al. 2025 alpha-zero-padding precedent).
