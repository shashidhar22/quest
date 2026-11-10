# NEXT — quest (updated 2026-11-09) · history: `docs/LOG.md`

Judge-first stands unchanged: legacy arm D shows no benefit, locked primary −0.023. Corpus
eligibility repair is now through step 8 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. First move: decide step 9's open questions (per-peptide cap on the paired class-I stratum; one
   corpus version or two; a residue budget for held-out splits or not), then write
   `sampling-contract.json`. The candidate sizing is corrected and passes 201 checks, but nothing
   is frozen; detail and how to re-run in ACS `repair/contract/README.md` and `docs/LOG.md`.
2. Three builder flags are now required and have no defaults: `--eligibility-policy
   reference-aware-v1`, `--benchmark-exclusion heldout-peptides-v2` (+ `--exclusion-list`),
   `--split-routing receptor-hash-v2`. `cluster-v0` / `none` reproduce past builds.
3. Nothing has been built or run on a box under the new policies. The box is first needed at
   step 10; steps 9 and the rest of the analysis run on the laptop.
4. Not isolated from CPT and not safe to score against a CPT model: benchmark parts
   `test_novel_tcr`, `test_novel_allele`, `test_mixed` (their peptides overlap train).
5. EC2 last verified stopped. Preserve colleague edits in `/home/ubuntu/quest`; deploy by bundle
   + worktree to `/home/ubuntu/quest-jf`.
