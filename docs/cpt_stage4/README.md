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

## Caveat carried forward

The canonical eval set itself (`data/cpt_canonical_eval_v2/`, 19,708 rows) was **not** synced to
S3 — only its result JSONs. Re-running or extending the canonical evaluation requires recovering
that directory from the training instance or rebuilding it.
