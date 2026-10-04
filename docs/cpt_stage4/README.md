# CPT Stage 4 — results recovered from S3

**Provenance:** Stage 4 executed on the training box 2026-05-28 → 2026-05-29 and was synced to
S3 on 2026-05-29 17:58 UTC — about 1h15m after the driver log records the run finishing at
16:41:55 UTC, consistent with a manual post-run sync rather than streaming uploads. It was never committed to this repo — HEAD `897a62b` (2026-05-26)
contains only `docs/cpt_stage3/PLANNING.md`, which describes Stage 4 as *planned*. These files
were pulled back down from
`s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/` on 2026-09-08 to close that gap.

## Contents

| File | What it is |
|---|---|
| `SUMMARY.md` | Generated Stage 4 report (`build_stage4_summary.py`), the authoritative result |
| `stage4_combined_DRIVER.log` | Orchestrator log — contains the exact `torchrun` command per point |
| `point*_done.txt` | Per-point `_done` marker: STATUS, wall time, headline perplexities |
| `point*_canonical_v2_results.json` | Full region-restricted eval output (19,708 rows, 9 strata) |

Point E was a re-evaluation of a reused Stage 2 checkpoint, so it has a `_done` marker but no
`canonical_v2_results.json`.

## Result

Point **D** (`tcr=cdr3`, `mhc=full`) wins on the primary metric `pocket_ppl` at **2.001**,
against a baseline (Point A, `cdr3` × `pocket`) of 4.732 — a 57.7% reduction. Points B, C and E
all regress badly (+200% to +227%), so enriching the TCR side does not help and coupled
enrichment does not compound.

## Weights

Adapter weights are **not** in git (`checkpoints/` is gitignored at `.gitignore:59`). The
`final/` adapters for points A–D plus `checkpoint_best/` for the winner were synced to the
local `checkpoints/` tree on 2026-09-08. To re-fetch:

```bash
export AWS_PROFILE=DataUserAccess-953915750371 AWS_REGION=us-west-2
aws s3 sync s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/cpt_stage4_pointD_run05/final/ \
  checkpoints/cpt_stage4_pointD_run05/final/
```

Each adapter is a 56.3 MiB PEFT LoRA over `esmc_300m` (r=32, α=64, dropout 0.05). Note that
`base_model_name_or_path` is null in `adapter_config.json` — the base must be supplied by hand.

## Canonical eval set — RECOVERED (corrected 2026-09-17)

> An earlier version of this file said the eval set "was **not** synced to S3" and that re-running
> the evaluation "requires recovering that directory from the training instance or rebuilding it."
> That was true when written on 2026-09-08. It was recovered from the `neurips_two` snapshot and
> synced on **2026-09-11**, so the blocker is gone.

`cpt_canonical_eval_v2` (19,708 rows, 9 strata) is in S3:

```bash
aws s3 sync s3://fh-pi-warren-h-eco/quest/tcrbench2/cpt_canonical_eval_v2/ \
            data/cpt_canonical_eval_v2/      # 15 objects, 13.4 MB
```

Verified 2026-09-17 against `manifest.json`: `v2_row_count = 19708`, and the per-stratum counts
match the `point*_canonical_v2_results.json` files in this directory exactly (I: 23,
II_complete: 2371, II_partial_DP_beta: 165, II_partial_DQ_beta: 206, II_partial_DR_beta: 1609).

## ⚠️ Read before using the numbers in `SUMMARY.md`

The Stage 4 headline ("Point D wins on `pocket_ppl`, −57.7%") does **not** survive scrutiny.
Each point is evaluated on its **own input serialization**, so the prediction task changes between
points and the perplexities are not on a common scale:

- On the regions where serialization is identical across points there is **no effect**.
  `cdr3_ppl`: A = 3.968 vs D = 4.002 (both `tcr=cdr3`, ~45k tokens each) — D is slightly *worse*.
  `peptide_ppl` (identical in all five points): 15.35 / 15.86 / 17.61 / 15.35 / 16.90 — flat.
- `overall_ppl` is tabulated as if comparable but scores wildly different token sets:
  A = 78,099, D = 324,701, C = 1,112,467 masked tokens. D's 1.746 is 72% MHC framework at 1.283.
- `pocket_ppl` token counts *are* comparable (23.5k–26.6k), but 71–75% come from `II_complete`
  alone; `mhc_class=I` contributes 114–134 tokens (~0.5%, 23 rows).
- D's `pocket_ppl` tracks **HLA allele diversity, not biochemistry** (DP 1.301 < DQ 1.466 <
  II_complete 1.945 < DR 2.282 << class I 11.448). With `framework_ppl` = 1.283 and no allele
  hold-out in the eval, memorization is the leading explanation. This is Stage 3 `PLANNING.md`
  Q2, never answered.
- `mhc_class=none` rows carry 835–888 pocket-labelled tokens in **every** point, contradicting
  SUMMARY §3's "MHC-bearing rows only, by construction" — the region attributor is mislabelling.
- Budgets differ: early-stop `global_step` A = 5200, B = 3500, C = 5000, D = 7000.

The real signal is C vs D (both `mhc=full`, 14.836 vs 2.001): pocket tokens are 2.4% of C's
masked tokens vs 8.1% of D's vs 30% of A's. Under uniform 15% masking, adding full TCR chains
dilutes the MHC signal — which points at IC-weighted masking (Stage 3 Q3) as the lever, not at
D's representation being better.

**Next step**: re-score all five checkpoints on ONE fixed serialization (inference only, no
retraining), and build an allele-held-out split to test the memorization hypothesis. Both are
now unblocked by the recovered eval set above.

## Plan drift worth knowing

Stage 3 `PLANNING.md` specified Stage 4 as a **data-scaling** sweep (1M/10M/100M at `cdr3+full`,
`max_length=384`). What executed was a **representation** sweep at fixed 10M with `max_length`
varying per point (64 / 256 / 1024 / 1024). The scaling curve meant to gate Stages 5–8 was never
run, and the 384 recommendation was never used.
