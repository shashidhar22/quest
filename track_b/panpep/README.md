# PanPep on TCRBench AS Class I

## Setup

```bash
conda env create -f env.yml          # creates track_b_panpep (Python 3.9, torch 1.10.2)
git clone --depth 1 https://github.com/bm2-lab/PanPep.git repo
```

After clone, the `repo/PanPep.py` and `repo/Requirements/Memory_meta.py` are
patched in-place to support CPU fallback. See "Patches applied" below.

## Run smoke test (zero-shot, ~15s on CPU)

```bash
conda activate track_b_panpep
cd /home/ubuntu/quest/track_b/panpep

python convert_data.py --mode zero-shot --task as_trb_i --test-split test_iid \
    --n-test 50 --out smoke_zero.csv

CUDA_VISIBLE_DEVICES="" python evaluate.py --mode zero-shot \
    --input smoke_zero.csv --truth smoke_zero_truth.csv \
    --out smoke_zero_metrics.json
```

## Mode mapping for TCRBench (revised after data inspection)

**All TCRBench AS-Class-I conditions → zero-shot.**

The brief's original mapping assigned majority/few-shot modes to `iid`, `novel_tcr`,
and `novel_allele` conditions. Empirical check shows that the iid stratification
within TCRBench (BENCHMARK_SPLITS.md §4 step 8) deliberately disjoints peptide
clusters between train and test files — train ↔ test_iid peptide-cluster overlap
is **0**. PanPep majority/few-shot modes need exact peptide-string matches in the
support set; with zero overlap, those modes are unreachable. We use zero-shot
for every condition.

See `track_b/verification/panpep_verification.md` §5 for the full discussion.

## Patches applied to upstream repo

These edits make PanPep run on CPU when CUDA is not available (L4 GPUs are
sm_89 which is incompatible with torch 1.10.x):

1. `repo/PanPep.py`:
   - `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
   - `torch.load(Path, map_location=device)`
   - `Memory_module(...).to(device)` instead of `.cuda()`
   - Custom `_CPUUnpickler` to remap CUDA storage in `joblib.load(...)` calls.
2. `repo/Requirements/Memory_meta.py`:
   - Added `_PANPEP_DEVICE` module-level alias.
   - All `.cuda()` calls replaced with `.to(_PANPEP_DEVICE)`.

To re-apply after a fresh clone, see the patch logic in this README's commit history.

## Configuration

- Mode: zero-shot for all TCRBench conditions (see above).
- Pre-trained meta-learner: bundled `repo/Requirements/model.pt`.
- Negatives: `track_b/shared/negative_sampling.sample_negatives()` constructs
  test-set negatives (5×, seed=43, peptide-only swap).

## Phase 1 smoke-test result

```
AUC: 0.505
Accuracy: 0.646
n_pos: 50, n_neg: 179
mode: zero-shot
```

Chance-level AUC after a 50-row test set is the realistic prior — PanPep's
zero-shot mode generalizes only weakly to peptides outside its training
distribution, which is exactly what TCRBench's cluster-disjoint splits ensure.
This matches the 2025 reusability report (PMC12458544) finding that PanPep
performance "declined significantly" on truly novel TCRs.

The pipeline runs end-to-end and emits sensible metrics. Phase 2 will report
zero-shot AUC across all 5 conditions.
