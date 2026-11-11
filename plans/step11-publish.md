# Corpus repair step 11 — build what is missing, audit the built files, publish

Opened 2026-11-11. Not a stage. Step text and amendments: ACS
`quest-cpt/audits/2026-11-09/corpus-repair-plan.md`. Contract: `sampling-contract.json`,
revision 4. Lightweight plan.

## Status 2026-11-11

Done. Both versions published and re-checked from the laptop (visit CR2). Results: ACS
`repair-status.md`, "Step 11". The preflight is `quest/data/corpus_preflight.py`; wiring it
into a launcher is step 12.

## Goal

Both corpus versions the contract fixes, published to one new S3 prefix with everything needed
to trust them later without this conversation: the audits run on the files as published, their
hashes, and a completion marker written last. A training preflight that refuses a corpus
without them.

## Where things stand

- `task_enriched_cap1000`: built and verified on the box (visit CR1, run 4, V1-V11 pass),
  unpublished at `/home/ubuntu/cpt_repair/cr1-20261111T013453Z/dataset`. It is not rebuilt: the
  same code and contract would produce the same rows, and what gets published is checked by hash.
- `repaired_proportional_10M`: never built. Its sampling is the builder's original row path,
  which still hands the 1.38B-row beta partition to the filter (the 37-minute join that the
  contract path no longer does).

## Approach

- **Proportional path.** Give the two sampling queries the same split filter the contract path
  has. The rows sampled must not change: the laptop build before and after has to be
  row-identical. Then build on the box with `--distribution proportional --sampling-contract …
  --contract-version repaired_proportional_10M`, so the computed targets are checked against the
  contract's, and verify (V11 holds its split and stratum counts to the stated tolerance).
- **Leakage audit on the built files**, both versions: `cpt-benchmark-overlap.py` generalized
  from "run05" to any built dataset, plus the minimum edit distance of any built peptide to a
  held-out one. Required: zero for the four held-out parts, minimum distance 2.
- **Coverage report**, both versions, from the built files: per split and stratum, rows,
  distinct examples, peptides and classical class-I chains; `coverage-by-split.csv`.
- **Publish** to the prefix reserved at step 1,
  `s3://fh-pi-warren-h-eco/quest/cpt/repair-20261109T-eligibility/<version>/`: the dataset,
  `_build_artifacts/` (manifest, eligibility audit, V1-V11), `leakage-audit.json`,
  `coverage-by-split.csv`, `sampling-contract.json`, and `PUBLISHED.json` written LAST with the
  sha256 of every object. A second pass downloads nothing but re-lists the prefix and compares
  sizes and the manifest; the prefix must be empty before the first write.
- **Preflight.** A small function the trainer can call: given a dataset directory, refuse unless
  the completion marker, a PASS for V1-V11, a zero leakage audit, a non-zero class-I training
  count and the contract hash are all present and consistent. Not wired into any launcher here;
  step 12 does that.

Not in scope: `token_exact`, any training, pair-level isolation of the three unisolated
benchmark parts, changing the contract.

## Steps

1. Split filter in the proportional sampling queries; laptop builds before and after are
   row-identical; tests.
2. Leakage audit and coverage scripts over a built dataset; run both on the laptop test build.
3. Preflight function and tests (each missing or inconsistent piece refuses).
4. Box visit CR2 (r6i.8xlarge, CPU, watchdog as in CR1): build and verify the proportional
   version; run the leakage audit and coverage report on both built datasets; publish both;
   re-list and compare. Stop, switch the type back.
5. Record: repair-status, plan, NEXT, LOG; one combined review of steps 1-3 before the visit.

## Verification

- Step 1: row hashes identical before and after, on the laptop.
- Step 4: proportional V1-V11 pass; leakage audit zero and minimum distance 2 for both; every
  published object's size and sha256 equals its local value; `PUBLISHED.json` is the newest
  object under each version prefix.
- Failure looks like: a proportional stratum count outside its tolerance, any held-out peptide
  or association in a built file, an object missing from or differing in the listing.

## Cost

One visit, about an hour: the proportional build is expected near the enriched one's 20 minutes
but has not been measured; the audits and uploads are minutes. Roughly USD 2-3 at list price,
hard ceiling 4 h ≈ USD 8. S3: about 3 GB, cents per month.
