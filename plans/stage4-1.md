# CPT Stage 4.1 — Stage 4 re-run at a fixed, non-truncating `max_length`

**Date:** 2026-10-02 · **Status: COMPLETE 2026-10-03** — all three arms PASS at 7,800 steps.
Results and the post-run idle incident: `docs/cpt_stage4.1/README.md` §10–11. Headline:
`pocket_ppl` A 3.368 / D **1.862** / F 2.090; arm D reproduces Stage 4 Point D.
**Box:** `i-0030c1a9c845b3714` (`neurips_two_s41`), g6.24xlarge, 4× L4, us-west-2a — chosen only
because g6.12xlarge had no capacity; it was oversized (4 % CPU, 10 of 23 GB per GPU).

## Goal

Re-run three arms of the Stage 4 representation sweep with **one fixed `max_length`
across every arm**, so that the serialization is the only thing that varies.

Stage 4 set `max_length` per point (64 / 256 / 1024 / 1024), always tracking the
serialization under test. Because `train_cpt_stage2_single.py` passes its `max_length`
straight into `region_restricted_eval` (`cpt_training_lib.py:1626-1638`), each point was
also *scored* on its own truncated view of the same 19,708-row eval set. Representation
effect and context-window effect were therefore inseparable. See
`AWSconnection/QUEST_CHECKPOINTS_AND_DATA.md` §8 finding 2.

## The fixed window: `max_length = 1024`

Measured over **every row of every split** with `scripts/training/measure_maxlen.py`
(2026-10-02), tokenizer `EsmSequenceTokenizer`, `truncation=False`:

| Dataset | Serialization | train max | test max | val max | p50 | p99 |
|---|---|---:|---:|---:|---:|---:|
| `run01_cdr3_pocket_proportional_10M` | cdr3 × pocket | **162** | 114 | 114 | 17 | 22 |
| `run05_cdr3_fullmhc_proportional_10M` | cdr3 × full | **577** | 578 | 578 | 17 | 22–29 |
| `run11_cdr123_fullmhc_proportional_10M` | cdr123 × full | **603** | 607 | **608** | 27–32 | 35–56 |

Global max across all three is **608 tokens** (run11, validation split), measured after the
build completed. run05's 578 independently matches the canonical eval set's longest
cdr3 × full row (578, `cpt_stage3/PLANNING.md:23-33`) — a useful cross-check. **0.0% of
run11 rows exceed 1024 in any split**, so the chosen window truncates nothing. The measurement is exact rather than sampled: token count can
exceed character count only by the 2 framing special tokens, so scanning the top 20,000
rows by character length and proving no unexamined row could beat the maximum bounds it
rigorously. The script prints `coverage=OK`/`INCOMPLETE`, so a shortfall is visible.

**1024 chosen over the smallest safe value (640) for three reasons:**
1. Headroom for run11 — cdr123 adds CDR1+CDR2 per chain (~30 tokens) over cdr3.
2. It makes the cdr3 × full arm **bit-identical to original Point D**, which already ran
   at 1024 and therefore was never truncating. That arm becomes a free **reproducibility
   control**: it should land near `pocket_ppl` 2.001.
3. Memory config at 1024 on 4× L4 is already proven (Stage 4 Points C and D).

Cost of 1024 over 640 is near zero: length bucketing is on and the collator pads
dynamically, so compute tracks actual sequence lengths, not the window.

## Arms

All three share `max_length=1024`, buckets `[64,128,256,512,1024]`, and **effective batch
1024**. Everything else is byte-identical to Stage 4's `COMMON_TRAIN_ARGS`.

| Arm | TCR × MHC | Dataset | Stage 4 counterpart | bs × accum × 4 | grad ckpt |
|---|---|---|---|---|---|
| `pointA_run01` | cdr3 × pocket | run01 | Point A — ran at **64, truncating** (max 162) | 128 × 2 = 1024 | no |
| `pointD_run05` | cdr3 × full | run05 | Point D — ran at 1024; **control** | 16 × 16 = 1024 | yes |
| `pointF_run11` | cdr123 × full | run11 (**new**) | none — Point B was cdr123 × *pocket_contact* | 16 × 16 = 1024 | yes |

Two deliberate departures from Stage 4, both of which *remove* confounds rather than add them:

- **Effective batch is pinned at 1024.** Stage 4 did not hold this constant — Point A ran
  1536 (192×2×4) while B/C/D ran 1024. 1024 is the value 3 of the 4 original points used.
  `batch_size × grad_accum` still differs per arm to suit real sequence lengths, which
  changes throughput only, not the optimization.
- **Gradient checkpointing stays a per-arm memory knob** (off for run01, on for the two
  full-MHC arms, exactly as Stage 4 did). It is numerically equivalent — activations are
  recomputed in the backward pass — so it cannot bias the comparison.

## Steps

1. ~~Recover the box + env~~ **done.** Snapshot `snap-0aa5bccb08a3012e3` registered as
   `ami-0b855eb5871217bce`; g6.12xlarge had no capacity in any AZ, so this is g6.24xlarge.
   Training env is miniforge3 **base** (torch 2.11.0+cu130, transformers 5.7.0, peft 0.19.1,
   esm 3.2.1.post1). The documented ESM-C/transformers pin conflict is gone in esm 3.2.1.
2. ~~Measure `max_length`~~ **done** (table above).
3. **Build `run11_cdr123_fullmhc_proportional_10M`** — `scripts/data_processing/build_run11.py`,
   adapted from `build_run10.py`. In flight; run10's identical build took 110 min.
4. **Re-measure run11** and confirm max ≤ 1024.
5. **Launch** `scripts/training/run_stage41_pipeline.py` — 3 arms sequentially in tmux,
   resumable via per-arm `_done` markers.
6. **Emit docs** — `docs/cpt_stage4.1/README.md` + `SUMMARY.md`, via an adapted
   `build_stage4_summary.py`.
7. **Sync** `checkpoints/cpt_stage41_*` to
   `s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/` (never into an existing prefix).

## Verification — defined before the run

The preflight in `run_stage41_pipeline.py` **refuses to launch** unless all of these hold,
so the central promise is enforced in code rather than assumed:

- every dataset directory and the canonical eval set exist;
- each arm's `batch_size × grad_accum × n_gpus` equals 1024 exactly;
- **every dataset's measured max token length is ≤ 1024** — a dataset that would be
  truncated aborts the stage;
- the length scan itself is complete (it reports `INCOMPLETE` and aborts if the top-N
  window did not cover all candidate rows).

After the run:
- `pointD_run05` reproduces Stage 4 Point D's `pocket_ppl` ≈ 2.001 within noise. **If it
  does not, the whole re-run is suspect** and that is the first thing to chase.
- `cdr3_ppl` for `pointA_run01` and `pointD_run05` should sit near each other (both are
  tcr=cdr3; Stage 4 gave 3.968 vs 4.002), since that region's serialization is identical.
- Step budgets get recorded per arm. Early stopping is kept as-is (Stage 4 parity), so
  budgets may again diverge; `save_steps 2000` means a matched-budget comparison can be
  made afterwards from intermediate checkpoints **without retraining**.

## Open risks

- **Capacity.** g6.12xlarge is unavailable in every AZ; this box is g6.24xlarge. If it is
  lost mid-run, the per-arm `_done` markers make the driver resumable, but a replacement
  box is not guaranteed.
- **Cost unverified.** The AWS Pricing API is denied to our SSO role, so no confirmed
  hourly rate. g6.24xlarge is the larger box (96 vCPU / 364 GB vs the original's 48 / 181)
  at a correspondingly higher rate — **check the console**. Estimated GPU time is
  ~15 hr across the three arms (2 + 6 + 7), plus ~2 hr for the run11 build.
- **Disk.** The DuckDB build spills heavily. Free space on `/home/ubuntu`: 406 GB at launch
  (18:44), 275 GB at +40 min, 237 GB at +70 min, with 170 GB in `data/duckdb_tmp_cpt`.
  Trending fine but worth watching; the box has no swap.
- **run11 max length is unmeasured** until the build completes. If it exceeds 1024 the
  preflight will abort and the window must be raised (implausible: cdr123 adds ~30 tokens
  to run05's 578).
- **Row count will not be exactly 10M.** run10 produced 9,998,527; the builder drops
  near-empty rows. `build_run11.py` accepts ±100,000, matching run10's gate.

## Also fixed during prep — the `mhc_class=none` pocket tokens

Diagnosed and fixed 2026-10-02; full write-up in `docs/cpt_stage4.1/README.md` §9. Short
version: 157 of the 19,708 canonical-eval rows carry a class-II **alpha** chain in the
`mhc_one` column, and `derive_mhc_class`'s `mhc_one`-only branch collapsed every
non-class-I chain to `none`. The pocket **labels were always correct** — only the stratum
label was wrong — so `pocket_ppl` overall and the early-stop metric were never affected,
contrary to how this was previously filed. Fixed in `cpt_dataset_verifier.py`, the mirrored
`cpt_dataset_builder.py` SQL, and `region_restricted_eval` (which now buckets on a
re-derived label, conservatively: stored-`none` rows carrying an allele only).

**Consequence for this plan:** Stage 4.1 runs with the repair in place, so its per-`mhc_class`
breakdown is correct from the first arm. Validation is a re-score of the existing Stage 4
Point D adapter, which must return `pocket.overall` **unchanged at 2.001** while the `none`
stratum empties into `II_partial_alpha` (~862 tokens, ppl ≈ 2.317). Any movement in
`pocket.overall` means the patch is doing more than relabelling.

## Provenance note

The Stage 4 driver (`run_stage4_pipeline.py`, `run_overnight_stage4.sh`,
`stage4_preflight.sh`), the summary generator (`build_stage4_summary.py`) and
`build_run10.py` were all absent from the repo and recorded as lost. They were recovered
from the `neurips_two` root snapshot on 2026-10-02 and are now committed locally, along with
the Stage 0–3 summaries, driver logs and `canonical_v2_results.json` files under
`docs/cpt_recovered/`, and the `track_a/` SFT harness. Stage 4.1's driver and summary
generator are adaptations of those recovered originals rather than rewrites.
