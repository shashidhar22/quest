# TEIM-Seq on TCRBench AS Class I

## Setup

```bash
conda env create -f env.yml          # creates track_b_teim
git clone --depth 1 https://github.com/pengxingang/TEIM.git repo

# After clone, ANARCI must also be installed (TEIM-Seq calls it for CDR3 numbering):
conda activate track_b_teim
conda install -c bioconda anarci -y

# Two source patches (already applied to the cloned repo):
# - repo/train_teim/utils/misc.py: add map_location='cpu' to torch.load
```

## Run smoke test (~5 s on CPU)

```bash
conda activate track_b_teim
cd /home/ubuntu/quest/track_b/teim

# 1. Convert (filters peptide len ∉ [8,12] and CDR3 len > 20).
python convert_data.py --task as_trb_i --split train     --n-rows 100 --out smoke_train.tsv     --with-negatives --copy-to-repo
python convert_data.py --task as_trb_i --split val       --n-rows 50  --out smoke_val.tsv       --with-negatives --copy-to-repo
python convert_data.py --task as_trb_i --split test_iid  --n-rows 50  --out smoke_test_iid.tsv  --with-negatives --copy-to-repo

# 2. Train TEIM-Seq from scratch on CPU.
CUDA_VISIBLE_DEVICES="" python train.py --train smoke_train --val smoke_val \
    --epochs 2 --device cpu --batch-size 64 --name smoke

# 3. Evaluate.
CUDA_VISIBLE_DEVICES="" python evaluate.py --checkpoint smoke/final.ckpt \
    --test smoke_test_iid.tsv --out smoke_test_iid_metrics.json --device cpu
```

## Why we wrote our own train.py

The upstream `train_teim/scripts/train_seqlevel.py`:
- Hard-codes `gpus=1` (would crash on CPU); we need an explicit cpu/cuda switch.
- Reads from a relative `../data/binding_data/` and tries to merge with
  `positive_epi_dist.tsv` (TEIM's training peptides). Our peptides aren't in
  that file, so the merge raises `KeyError`. Our train.py points at our
  `results/` dir which has no `positive_epi_dist.tsv` — `SeqLevelDataset`
  catches the `FileNotFoundError` and falls back to `epi_id = -1`.

## Configuration

- TEIM-Seq, sequence-level binding head only (skip TEIM-Res — needs structural data).
- Hyperparameters from `seqlevel_all.yml`: lr 2e-4, batch 512 (smoke 64), 25 epochs (smoke 2),
  dim_hid 256, layers_inter 2, inter_type 'mul'.
- Length caps: peptide ∈ [8, 12] AA, CDR3 ≤ 20 AA. **Rows outside these bounds are dropped**
  (~30% of TCRBench positives — many have peptides 13–17 AA).
- Negatives: `track_b/shared/negative_sampling.sample_negatives()` (5×, seed=43, peptide-only swap).
- ANARCI for CDR3 numbering at inference time (~200 rows/sec).
- MHC, V/J, TRA: not used.

## Phase 1 smoke-test result

```
AUC: 0.499
Accuracy: 0.833
n_pos: 17, n_neg: 85
```

Chance AUC + correct accuracy at 5:1 imbalance after 2 epochs on 84 train positives.
Pipeline runs end-to-end. Phase 2 will train 25 epochs on the full AS-TRB-I train set.

See `track_b/verification/teim_verification.md` for the full integration audit.
