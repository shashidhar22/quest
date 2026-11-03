# Stage 4.2 wave 1 — V03 evidence, pulled from S3 2026-10-15

Source: `s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/stage42_wave1/`, everything but the
62 MB `best.pt` / `last.pt`. Visit: `quest-cpt/box-jobs/visits/v03-2026-10-07.md`.
Reading: `quest-cpt/findings.md` §9.6.

## What ran

Cell 1 only (base × `as_paired_i`), 2026-10-08 00:33 → 04:08 UTC. **Cells 2–4 never ran**, so no
arm-D number exists. Both probes passed, so the fp32-head fix holds on hardware.

## Result — the judge works; the gate was keyed to the wrong condition

The intervals this run printed are **3–4x too narrow**: it bootstrapped over queries, and queries
sharing a true peptide are not independent. Corrected (peptide-clustered, `cluster_bootstrap_ci`):

| condition | molecules | peptides | recall@10 | CI as printed | **corrected CI** | random |
|---|---:|---:|---:|---|---|---|
| test_level4 | 202 | 108 | **0.337** | [0.267, 0.406] | **[0.172, 0.633]** | 0.10 |
| test_iid | 201 | 53 | 0.144 | [0.095, 0.194] | [0.027, 0.353] | 0.10 |
| test_novel_pep | 1,589 | 463 | 0.081 | [0.068, 0.094] | [0.043, 0.147] | 0.10 |

Only `test_level4` clears random, so the harness is sound and the gate belongs there.
`test_novel_pep` is **not** below random once the interval is right.

Training (`base_paired/train_summary.json`): val AUC **0.702** epoch 1, 0.465 epoch 2; train loss
0.4530 → 0.4496 against H(1/6) = 0.4506. Epoch 1 correctly kept.

The gate read `test_iid`, could not separate 0.144 from 0.100, and logged "the harness is broken".
It is not: `test_level4` clears random by a wide margin in the same file. Fixed in
`scripts/training/stage42_box/s42_check.py` (gates on `test_level4`, three outcomes); both eval
files here are the fixtures in `tests/test_s42_gate.py`.

A peptide-familiarity explanation for the gradient was proposed and then **refuted** by measuring
the splits: no condition shares a peptide or peptide cluster with training, so `test_iid` is not
in-distribution and `test_level4` is not "both seen" — it shares no TCR with train either. Recall
tracks the true peptide's **length** against a pool that is 76 % 9-mers; inside `test_level4`,
9-mers score 0.553 and 14-mers 0.013. See `quest-cpt/findings.md` §9.6.

## Files

| Path | What |
|---|---|
| `wave1.log` | both attempts: 10-07 (tied scorer) and 10-08 (this one), with the gate lines |
| `chain.log` | `run_all.sh` driver: Stage 4.3 then wave 1, per launch |
| `probe_base.log`, `probe_armD.log` | pre-gate probes, both PASS |
| `base_paired/` | **the result**: `train.log`, `train_summary.json`, `eval.json`, `eval.log` |
| `base_paired_bf16_20261007/` | the superseded bf16 run: recall 0.0, MRR 0.01 = every candidate tied. Kept as the BROKEN fixture |

`eval.json` carries per-query ranks under `conditions.<cond>.ranks`, keyed by `row_id`, which is
what a paired comparison against arm D will use. The row-to-true-peptide map needed to cluster a CI
over peptides is `tests/fixtures/s42_as_paired_i_true_peptides.csv` (newer eval runs write it into
`eval.json` as `true_peptides`).
