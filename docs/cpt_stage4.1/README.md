# CPT Stage 4.1 — the Stage 4 sweep at a fixed, non-truncating `max_length`

**Status (2026-10-04): COMPLETE.** All three arms PASS at 7,800 steps each. Results in §10;
generated table in `SUMMARY.md`; per-arm `_done`, `canonical_v2_results.json` and
`early_stop_history.json` in `results/`; driver and publish logs in `logs/`. Full arms
(adapters + step 2000/4000/6000 checkpoints) are in
`s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/cpt_stage41_*/`, verified byte-exact
against the box. The box sat idle 14.5 h after finishing; see §11.

Companion to `docs/cpt_stage4/README.md`, which should be read first — it documents why
Stage 4's headline does not survive scrutiny. Stage 4.1 is the controlled re-run.

---

## 1. Why this stage exists

Stage 4 set `max_length` per point: 64 for Point A, 256 for B, 1024 for C and D. The
window therefore tracked the serialization being tested, and the two effects cannot be
separated in that design.

It is worse than a training-side confound. `train_cpt_stage2_single.py` passes its
`max_length` straight into the evaluator:

```python
# scripts/training/cpt_training_lib.py:1626-1638
region_restricted_eval(unwrapped, self.canonical_ds,
                       cfg.tcr_variant, cfg.mhc_variant, ...,
                       max_length=cfg.max_length, ...)
```

So each point was also *scored* on its own truncated view of the same 19,708-row canonical
eval set. Total masked tokens ranged from 78,099 (A) to 1,112,467 (C) — not comparable
denominators. Stage 4 additionally did not hold the effective batch constant (Point A ran
1536, B/C/D ran 1024) and early stopping produced unequal step budgets
(5200 / 3500 / 5000 / 7000 on equal data).

Stage 4.1 fixes the window and the effective batch, leaving the serialization as the only
intended difference.

## 2. The fixed window: `max_length = 1024`

Measured over **every row of every split** by `scripts/training/measure_maxlen.py`
(tokenizer `EsmSequenceTokenizer`, `truncation=False`, 2026-10-02):

| Dataset | Serialization | train max | test max | val max | p50 | p99 |
|---|---|---:|---:|---:|---:|---:|
| `run01_cdr3_pocket_proportional_10M` | cdr3 × pocket | **162** | 114 | 114 | 17 | 22 |
| `run05_cdr3_fullmhc_proportional_10M` | cdr3 × full | **577** | 578 | 578 | 17 | 22–29 |
| `run11_cdr123_fullmhc_proportional_10M` | cdr123 × full | **603** | 607 | **608** | 27–32 | 35–56 |

Global max across all three is **608 tokens** (run11 validation split), measured 2026-10-02
after the build completed. run05's 578 independently matches the canonical eval set's longest
cdr3 × full row (578, `cpt_stage3/PLANNING.md:23-33`).

Rows exceeding each candidate window, from run11 (the longest dataset):

| Window | 64 | 128 | 256 | 384 | 512 | **1024** |
|---|---:|---:|---:|---:|---:|---:|
| % run11 rows truncated | 0.33–0.51% | 0.31–0.48% | 0.31–0.48% | 0.12–0.24% | 0.12–0.24% | **0.0%** |

Two things follow. **1024 truncates nothing**, so the premise of this stage holds. And Stage 4's
*planned* window of 384 would still have truncated 0.12–0.24% of cdr123 × full rows — even the
original plan's own recommendation was short.

The measurement is exact, not sampled. A token spans at least one character except the two
framing special tokens, so `tokens ≤ chars + 2`; scanning the top 20,000 rows by character
length and proving no unexamined row could exceed the maximum bounds it rigorously. The
script prints `coverage=OK` or `INCOMPLETE`, so a shortfall is visible rather than silent.
(The first version of that script asserted `tokens ≤ chars` and was caught by run01's train
split reporting 162 tokens against 160 characters — the two framing tokens. The bound was
corrected; the 162 was already exact.)

**Why 1024 rather than the smallest safe value, 640:**

1. Headroom for run11 — cdr123 adds CDR1+CDR2 per chain (~30 tokens) over cdr3.
2. It makes the cdr3 × full arm **bit-identical to original Point D**, which already ran at
   1024 and so was never truncating. That arm is therefore a free **reproducibility
   control** and should land near `pocket_ppl` 2.001.
3. The memory configuration at 1024 on 4× L4 is already proven (Stage 4 Points C and D).

The cost of 1024 over 640 is near zero: length bucketing is on and the collator pads
dynamically, so compute tracks real sequence lengths rather than the window. This is also
why the cdr3 × pocket arm stays cheap at a 1024 window despite its 64-token original.

## 3. Arms

All three: `max_length=1024`, buckets `[64,128,256,512,1024]`, effective batch **1024**.
Everything else is byte-identical to Stage 4's `COMMON_TRAIN_ARGS` (`esmc_300m`, LoRA
r=32/α=64/dropout 0.05, lr 2e-4, 1 epoch, MLM p=0.15, seed 42, `eos_strategy raw`, sdpa,
`--no_compile`, early stop on `pocket overall _ perplexity` patience 3 / min_rel 0.01).

| Arm | TCR × MHC | Dataset | Stage 4 counterpart | bs × accum × 4 GPUs | grad ckpt |
|---|---|---|---|---|---|
| `pointA_run01` | cdr3 × pocket | run01 | Point A — ran at **64, truncating** (max 162) | 128 × 2 = 1024 | no |
| `pointD_run05` | cdr3 × full | run05 | Point D — ran at 1024 already; **control** | 16 × 16 = 1024 | yes |
| `pointF_run11` | cdr123 × full | run11 (**new build**) | none — Point B was cdr123 × *pocket_contact* | 16 × 16 = 1024 | yes |

`pointF` is lettered F, not B, precisely because it is not Point B's serialization.

**Two deliberate departures from Stage 4**, both removing confounds rather than adding them:

- **Effective batch pinned at 1024** — the value 3 of the 4 original points used.
  `batch_size × grad_accum` still varies per arm to suit real sequence lengths, which
  changes throughput only, not the optimization.
- **Gradient checkpointing was passed per arm** (off for run01, on for the full-MHC arms, as
  in Stage 4). *Correction 2026-10-03:* the flag is a no-op in `cpt_training_lib.py` (a config
  field with no consumer), so no arm checkpointed, in Stage 4 or here. It cannot bias the
  comparison either way.

**Kept from Stage 4 even though it is imperfect:** early stopping, so step budgets may again
diverge. `save_steps 2000` means a matched-budget comparison can be made afterwards from
intermediate checkpoints *without retraining*. Report the per-arm `global_step` alongside
any comparison.

## 4. The new dataset

`run11_cdr123_fullmhc_proportional_10M` did not exist. The serialization is baked into
`input_text` at build time by `build_input_text_sql(tcr_variant, mhc_variant)`
(`cpt_dataset_builder.py:184`); the trainer's `--tcr_variant` / `--mhc_variant` flags only
drive the canonical-eval projection, so a cdr123 × full arm *requires* a new build.

Built by `scripts/data_processing/build_run11.py`, adapted from `build_run10.py` (the
ad-hoc cdr123 × pocket_contact precedent) with three changes: `--mhc-variant full`; its own
phase marker so Stage 4's `_phase2_done` is never clobbered; and a widened `input_text`
length gate, because run10's gate assumed the much shorter pocket_contact serialization.
Builder args otherwise match: `--distribution proportional --scale 10000000 --seed 42`,
with DuckDB sized to 60% of detected RAM and a cgroup hard cap.

## 5. How to run

```bash
# on the box, from a tmux session
cd /home/ubuntu/quest
PYTHONPATH=$PWD /home/ubuntu/miniforge3/bin/python \
  scripts/training/run_stage41_pipeline.py --nproc_per_node 4
```

The driver is resumable: each arm writes a `_done` marker and any arm already `STATUS=PASS`
is skipped. `--only <name>` runs a single arm. Afterwards:

```bash
PYTHONPATH=$PWD /home/ubuntu/miniforge3/bin/python scripts/training/build_stage41_summary.py
```

## 6. Verification, defined before the run

The driver's `preflight()` **refuses to launch** unless all of the following hold, so the
central promise of this stage is enforced in code rather than assumed:

- every dataset directory and the canonical eval set exist;
- each arm's `batch_size × grad_accum × n_gpus` equals 1024 exactly;
- **every dataset's measured max token length is ≤ 1024** — a dataset that would be
  truncated aborts the stage;
- the length scan is complete (it aborts with `INCOMPLETE` if the top-N window did not
  cover all candidate rows).

Post-run checks:

- `pointD_run05` should reproduce Stage 4 Point D's `pocket_ppl` ≈ 2.001 within noise.
  **If it does not, treat the whole re-run as suspect** and chase that first.
- `cdr3_ppl` should be close between `pointA_run01` and `pointD_run05` (Stage 4 gave 3.968
  vs 4.002), since that region's serialization is identical in both.
- Record `global_step` per arm; unequal budgets weaken any cross-arm claim.

## 7. Provenance and environment

Run on `i-0030c1a9c845b3714` (`neurips_two_s41`), **g6.24xlarge**, 4× L4 (23 GB each),
96 vCPU, 364 GB RAM, us-west-2a.

The original Stage 4 box (`neurips_two`, `i-0db51debae46eb1c0`, g6.12xlarge) still cannot
start — `InsufficientInstanceCapacity` in us-west-2a, unchanged since 2026-09-10, and
g6.12xlarge was refused in 2b as well. Its root snapshot `snap-0aa5bccb08a3012e3` was
registered as AMI `ami-0b855eb5871217bce` so it could launch on any type, and g6.24xlarge
had capacity. Same 4× L4 GPUs; double the vCPU and RAM.

**Training environment: miniforge3 `base`** (not one of the named conda envs, which are all
Track A/B baselines) — torch 2.11.0+cu130, transformers 5.7.0, peft 0.19.1,
**esm 3.2.1.post1**, datasets 4.8.5, duckdb 1.5.2. The driver calls bare `python`, which
only resolves with conda activated, which is why Stage 4 ran inside tmux.

> **The documented ESM-C version conflict no longer applies.**
> `docs/CPT_HF_CHECKPOINTS.md:88-90` says ESM-C needs a separate venv because the `esm`
> package pins `transformers < 4.48.2` against quest's `>= 4.57.1`. Verified 2026-10-02:
> `esm 3.2.1.post1` imports `ESMC` cleanly alongside `transformers 5.7.0`, with all 4 GPUs
> visible. No sidecar env is needed for CPT training. (A sidecar `track_a_tulip` env at
> transformers 4.30 does exist, for an unrelated TULIP API breakage.)

The HF cache on the box already holds `EvolutionaryScale/esmc-300m-2024-12` and the Gen C
adapter `sravisha/esmc300m-foundation100m-c3-lora-r32`, so no downloads are required.

## 8. Known risks

- **Capacity.** g6.12xlarge is unavailable in every AZ. If this box is lost mid-run the
  `_done` markers make the driver resumable, but a replacement is not guaranteed.
- **Cost unverified.** The AWS Pricing API is denied to our SSO role, so there is no
  confirmed hourly rate recorded here — check the console. Estimated ~15 GPU-hours across
  the three arms, plus ~2 hr for the run11 build.
- **Disk.** The DuckDB build spills heavily — 406 GB free at launch, 260 GB at ~55 min,
  with 147 GB in `data/duckdb_tmp_cpt`.
- **run11's max token length is unmeasured** until its build completes. The preflight will
  abort if it exceeds 1024 (implausible — cdr123 adds ~30 tokens to run05's 578).
- **The `mhc_class=none` pocket tokens are diagnosed and fixed** — and the earlier framing
  of this as a "region-attribution bug" was wrong. See §9.

---

## 9. The `mhc_class=none` pocket tokens — diagnosed and fixed (2026-10-02)

Every Stage 4 point reports pocket-labelled tokens for `mhc_class=none` (Point D: 157 rows,
862 masked tokens), which contradicts `cpt_stage4/SUMMARY.md` §3's claim that `pocket` covers
"MHC-bearing rows only, by construction". This was previously recorded as a live
*region-attribution* bug, in the same family as the `identify_segment_regions` subsequence
bug fixed in Stage 3. **That diagnosis was wrong.** The region labeller is correct; the
stratum metadata is not.

### Root cause

Exactly **157 of the 19,708** canonical-eval rows are labelled `mhc_class='none'` while
carrying a fully populated MHC — `mhc_one` (255 aa), `mhc_one_pocket` (37 aa) and
`mhc_one_allele`. Their alleles are **145 `HLA-DQA1*`, 7 `HLA-DRA*`, 5 `HLA-DPA1*`**: all
class-II **alpha** chains, all sitting in the `mhc_one` column. `hla_class()` classifies
every one of them as `II_alpha`.

That column placement is not an error — it is the imgthla standardizer's documented
convention: for class II the **alpha** chain goes in `mhc_one` and the beta chain in
`mhc_two`. The defect is in the class derivation, whose two single-chain branches are
asymmetric (`cpt_dataset_verifier.py::derive_mhc_class`, and the mirrored SQL at
`cpt_dataset_builder.py:675-695`):

```python
if has2:                                      # mhc_two only
    if c2 == 'II_beta':        return _ii_beta_subclass(...)
    if c2 in ('II_alpha', 'II_unknown_chain'): return 'II_partial_alpha'   # handled
if has1:                                      # mhc_one only
    return 'I' if c1 == 'I' else 'none'       # <-- any class-II chain falls to 'none'
```

A class-II chain found in the `mhc_one` position had nowhere to go. (A class-II *beta* chain
there would have fallen through identically.)

### Why this is far less serious than "pocket_ppl is mislabelled"

Those 862 tokens **are** genuine MHC pocket residues and **were** correctly labelled
`pocket`. `region_restricted_eval`'s `bucket_cols` are purely reporting dimensions — its
docstring is explicit that every row is projected, tokenized and masked identically and
losses are merely *accumulated* per `(region, bucket_col, bucket_value)`. So:

- `pocket_ppl` **overall is correct and unaffected**, as is every other `overall` aggregate;
- the early-stopping metric (`pocket overall _ perplexity`) is **unaffected**;
- only the **per-`mhc_class` breakdown** is wrong — 157 rows were filed under `none`
  instead of a class-II-alpha stratum.

Arithmetic check: 157 rows × 37 pocket residues × 15% masking ≈ 871 expected masked tokens;
Point D's `none` stratum reports 862 across exactly 157 rows. The account closes.

### The fix, in three tiers

**Tier 1 — the defect itself (done).** `derive_mhc_class`
(`cpt_dataset_verifier.py`) now applies the same class-II handling in both single-chain
branches, so a class-II alpha or beta chain in `mhc_one` is labelled `II_partial_alpha` /
`II_partial_D{R,P,Q}_beta` rather than `none`. Eight unit cases pass, including regressions
confirming class I, the `mhc_two` branch and the paired case are unchanged. The mirrored
correction is applied to the builder's `mhc_class_sql`
(`cpt_dataset_builder.py`, both single-chain branches now symmetric), so future eval-set and
dataset builds are right at source. Note that `run11` is being built by the pre-fix builder
already resident in memory, so it will carry the old labels in its own `mhc_class` column —
harmless, because training never reads that column and the *eval* set's labels are what drive
the breakdown.

**Tier 2 — make Stage 4.1 report correctly (done).** The existing result JSONs store only
per-stratum *aggregates*, so the 157 rows cannot be moved after the fact — a corrected
breakdown has to be bucketed correctly at eval time. `region_restricted_eval` therefore
re-derives the label where `meta_cols` is assembled, and logs how many rows it moved. The
repair is deliberately conservative: it only ever reclassifies a row whose **stored** label
is `none` and which carries a non-empty allele, so it can never silently move an
already-classified row, and it cannot touch any aggregate.

**Tier 3 — not done, deliberately.** Regenerating `cpt_canonical_eval_v2`'s `mhc_class`
column would be cleaner long-term, but it mutates a shared S3 artifact, and this project has
already been burned by same-name/different-content prefixes. Tier 2 achieves the same
reporting outcome without forking the eval set.

### Validation

Re-scoring the Stage 4 Point D adapter on the unchanged eval set with the patch in place had
to show **both** aggregate neutrality and a repaired breakdown. A failure would have been any
movement in `pocket.overall`, meaning the patch did more than relabel.

**Both held — ran 2026-10-02, `EXIT=0`**, result in
`checkpoints/stage41_validate_pointD_rescore.json`.

Aggregate neutrality — every overall region reproduces the stored Stage 4 Point D value:

| Region | Stored Stage 4 | Re-scored with repair |
|---|---:|---:|
| overall | 1.7460 | 1.746 |
| cdr3 | 4.0016 | 4.002 |
| **pocket (primary)** | **2.0007** | **2.001** |
| framework | 1.2828 | 1.283 |

Breakdown repaired — the log reports
`repaired mhc_class on 157 rows stored as 'none' but carrying an MHC allele ->
{'II_partial_alpha': 157}`, and the `none` stratum is **absent** from the output:

| `mhc_class` | ppl | masked | rows |
|---|---:|---:|---:|
| II_complete | 1.945 | 19,662 | 2,370 |
| II_partial_DR_beta | 2.282 | 4,574 | 1,526 |
| **II_partial_alpha** (was `none`) | **2.317** | **862** | **157** |
| II_partial_DQ_beta | 1.466 | 574 | 191 |
| II_partial_DP_beta | 1.301 | 490 | 161 |
| I | 11.448 | 125 | 23 |

The 157 rows moved intact — ppl 2.317 and 862 masked tokens are exactly what the `none`
stratum previously held — and every other stratum's ppl, token count and row count is
unchanged. The repair is provably relabel-only.

**Secondary result worth keeping:** this re-score also reproduces Stage 4 Point D exactly,
which independently validates the recovered environment, the adapter-loading path, the eval
set and the masking determinism. That is step 1 of
`AWSconnection/QUEST_CHECKPOINTS_AND_DATA.md` §9, discharged.

### Consequence for Stage 4 and Stage 4.1

Stage 4's **headline `pocket_ppl` numbers need no correction.** Their per-`mhc_class` tables
do: the `none` column should not exist as a pocket category, and the Stage 4 README's
"the labeller is wrong" note is superseded by this section. Stage 4.1 runs with Tier 2 in
place, so its breakdown is correct from the first arm.

## 10. Results (2026-10-03)

Every arm ran the full epoch, **7,800 optimizer steps**, with no early stop, so the step
budgets are equal. The confound noted in §3 did not materialise.

| Region | A cdr3 × pocket | D cdr3 × full | F cdr123 × full | Stage 4 Point D |
|---|---:|---:|---:|---:|
| `pocket_ppl` (primary) | 3.368 | **1.862** | 2.090 | 2.001 |
| `overall_ppl` | 4.405 | 1.726 | 1.966 | 1.746 |
| `cdr3_ppl` | 3.849 | 4.024 | 5.786 | 4.002 |
| `framework_ppl` | — | 1.275 | 1.376 | 1.283 |
| `peptide_ppl` | 14.908 | 15.214 | 16.649 | — |
| wall (h, 4× L4) | 1.58 | 5.98 | 6.05 | 5.67 |

Per-`mhc_class` `pocket_ppl`:

| mhc_class | A | D | F |
|---|---:|---:|---:|
| I | 17.690 | 7.555 | 8.538 |
| II_complete | 3.083 | 1.800 | 2.006 |
| II_partial_DP_beta | 1.608 | 1.350 | 1.316 |
| II_partial_DQ_beta | 4.866 | 1.428 | 1.822 |
| II_partial_DR_beta | 4.728 | 2.131 | 2.465 |
| II_partial_alpha | 3.675 | 2.314 | 2.487 |

**What this establishes**

1. **The reproducibility gate passed.** Arm D matches Stage 4 Point D within 1 % on
   `overall`, `cdr3` and `framework`. `pocket` came out 7 % better, plausibly seed noise in
   the smallest region; Stage 4 ran one seed, so this cannot be separated.
2. **Full MHC beats pocket even at a matched window.** D is −44.7 % vs A on `pocket_ppl`, so
   Stage 4's direction survives. Its size shrinks from −57.7 % because removing truncation
   alone improved A from 4.732 to 3.368. Roughly a quarter of the old headline was window.
3. **Adding CDR1/CDR2 hurts.** F is worse than D on every shared region: `pocket` +12 %,
   `cdr3` +44 %. This matches Stage 4's C-vs-D signal: under uniform 15 % masking, a longer
   TCR dilutes the MHC and CDR3 signal. That points at masking (Stage 4.5), not representation.
4. **Class I is still the weak regime** at 7.6–17.7 against 1.3–2.5 for class II. Balanced
   sampling (Stage 4.4) is the candidate fix.

**What it does not establish.** These are perplexities on each arm's own serialization of the
same rows. Pocket residues sit in different context per arm, which is the intended variable,
but perplexity is still not a downstream metric, and allele memorization is untested
(Stage 4.6). Stage 4.2's SFT judge decides whether D is actually better for specificity.

## 11. Post-run incident: the box idled 14.5 h (~USD 100)

Training ended at 2026-10-03 09:28:58 UTC. The box was stopped by hand at 2026-10-04 00:15 UTC.

**Cause, reproduced.** The watcher waited on `pgrep -f "[r]un_stage41_pipeline"`. The driver
was launched with `tmux new-session -d -s stage41 "… python scripts/training/run_stage41_pipeline.py"`.
The tmux **server** process keeps that whole command line as its argv for life, so the
pattern still matched pid 14572 after the Python driver exited (observed on the box,
2026-10-03 23:58). The watcher ran in its own tmux session on the same server, which kept the
server alive, so the condition could never clear. It never fired and the box never stopped.
Reproduced locally with a fake process whose argv mimics the tmux server: the old pattern
matches it, the new one does not.

**Second bug, found on the manual publish.** `build_stage41_summary.py` crashed with
`KeyError: 'F'`, and a later block read Stage 4's `payloads["B"/"C"/"E"]`. Both were Stage 4
leftovers keyed by letter. Fixed to derive everything from `POINTS`; the shipped summary is from
the fixed generator. Had the watcher fired, this would still have aborted the publish step,
although the stop would have gone through.

**Prevention.** `scripts/training/autostop_when_done.sh` replaces the ad-hoc watcher. It
waits on the driver **PID** (`kill -0`), never a name pattern. Two backup guards fire
independently: GPUs idle with no anchored training process for 20 min, and a hard wall-clock
ceiling `MAX_HOURS`. It refuses to arm on a dead PID, and it publishes before stopping.
`tests/test_autostop_when_done.sh` covers each guard plus the 2026-10-03 wrapper shape (8/8
pass; a failure in each is defined in the test). CloudWatch idle alarms would be a better
outer backstop, but `cloudwatch:PutMetricAlarm` is denied to our SSO role (tested 2026-10-03),
so it would need IT.
