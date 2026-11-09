# Judge diagnostic pilot — 2026-11-09

Status: preparation authorized; no EC2 inference or training launched. This is a versioned,
evaluation-only diagnostic. The V04a negative result and original candidate sets remain unchanged.

## Purpose and limits

Check whether the existing checkpoints produce stable rankings and use paired receptor context.
The pilot measures feasibility, coverage and inference time. It does not produce a confirmatory
CPT-benefit verdict, and a favorable pilot cannot reverse the locked result. CPU checks identify
known-positive collisions and allele-name evidence; unknown MHC sequences remain unknown until
sequence-aware validation. CPT training overlap remains unverified without the original corpus
crosswalk. Existing ranks cannot substitute for inference on receptor replacements.

## Fixed choices

- Build a SHA256-addressed manifest from the five downloaded paired-I splits and the candidate pool.
- Preserve exact legacy draws, including length fallback; annotate alternative positives and
  training exposure instead of silently removing candidates.
- Select a deterministic bounded sample with peptide-cluster coverage. Record selection and
  unshuffleable coverage. Results on this sample do not estimate the full original benchmark.
- Keep alpha/beta CDR3s paired when replacing receptors. Match both full MHC sequences and both
  CDR3 lengths; exclude the original receptor and known positive donor/truth contexts. Independent
  donor replacement is not a bijective permutation and is not a permutation significance test.
- Score original, identical repeat, reversed candidate order restored to canonical order, and
  fixed donor replacements. Preserve query peptide, MHC, serialization order and candidate list.
- Save all logits, exact truth ties, pessimistic rank and tie-aware midrank. Refuse nonfinite
  values, empty inputs and truncation. Publish only a complete output and refuse overwrite.
- Record manifest/checkpoint hashes, software/revision, precision and batch settings, and elapsed
  inference time. A precision sensitivity must actually disable encoder bf16 autocast, not merely
  cast the head or disable an outer context.
- No seed sweep, optimizer changes, new CPT, benchmark filtering or revised primary endpoint here.

## How results will be used

First inspect repeat/order differences and ties. Numerical instability must be understood before
interpreting donor effects. Report donor coverage and differences for both arms on exactly the same
eligible queries. Failure to detect a difference is not equivalence. Donor-associated performance
loss is not proof of binding recognition: inappropriate replacement distributions can also cause it.

A later confirmatory diagnostic may use original-minus-donor normalized rank with peptide-cluster
resampling and a 0.02 practical-equivalence margin. That proposed margin is about two candidate
positions, not a biological threshold. Before such a test, freeze its population, cluster count,
number of replacement replicates, treatment of donor reuse, interval construction and precision
adequacy. The pilot must not automatically emit a pass/fail label under that rule.

## Execution order

1. Build and inspect the CPU manifest and coverage report; record unresolved class/CPT overlap.
2. Exercise scorer failure cases and order alignment with fake models on CPU.
3. Measure a tiny real-checkpoint run under an approved capped inference visit; extrapolate runtime
   separately for both arms and any higher-precision sensitivity before expanding.
4. Diagnose numerical stability, then receptor dependence; decide whether a fresh matched base and
   seed replication can answer a scientifically meaningful question.

Evaluation-only is cheaper than training, not free. The scorer's between-batch time check cannot
kill a hung CUDA kernel or stop an EC2 instance. A future EC2 visit needs its external timeout and
verified shutdown watchdog; the module alone is not a safe unattended launcher.

## Local checks and invocation

The 300-query coverage run (100 per condition) has eligible receptor replacements for 26 IID,
75 novel-peptide and 33 level-4 queries. This selection is deliberately cluster-diverse, not a
representative estimate. The frozen 15-query smoke manifest (five per condition) has replacements
for one IID and five novel-peptide queries; all five level-4 smoke queries are unshuffleable. Use
it only for loading, score stability and runtime, not a level-4 biological conclusion.

From the code repository, with pandas/NumPy/pyarrow installed:

```sh
python -m track_a.esm_sft.diagnostic_data \
  --splits-dir ../quest-cpt/audits/2026-11-09/artifacts/splits \
  --out /tmp/new-diagnostic-manifest.json --max-queries 5
python -m track_a.esm_sft.diagnostic_eval --help
```

Future inference command template (requires real checkpoint/adapter paths and the approved visit):

```sh
python -m track_a.esm_sft.diagnostic_eval \
  --manifest /path/to/diagnostic-pilot.json --ckpt /path/to/best.pt \
  --adapter none --task as_paired_i --tcr_variant cdr3 --mhc_variant full \
  --max_len 1024 --batch-size 100 --max-seconds 600 \
  --out /path/to/new-base-diagnostic.json
```

Run D separately with its adapter and SFT checkpoint, using the SAME manifest. `--precision fp32`
currently refuses before loading: higher-precision encoder inference is not implemented.
The 600-second argument is a requested between-batch budget, not a measured duration or external
kill watchdog. Keep the full 100 candidates in each batch for the first legacy-comparable check.

Verification (local `python3.12`; Homebrew `python3` lacks torch): 78 targeted tests passed across manifest, diagnostic scorer, paired comparison,
rank scoring and fp32-head regressions. Fake scoring exercised all 300 rows with batch size 37.
The full test suite cannot collect locally without duckdb, stitchr, safetensors and peft.
Real checkpoint loading, GPU behavior and throughput remain untested.
