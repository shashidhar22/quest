# NEXT — quest (updated 2026-11-09) · history: `docs/LOG.md`

Judge-first stands unchanged: legacy arm D shows no benefit, locked primary −0.023. Corpus
eligibility repair is now through step 9 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. First move: step 10's builder work, on the laptop. The contract is frozen (ACS
   `repair/contract/sampling-contract.json`, two versions) but the builder cannot meet it yet: it
   needs example-level sampling for the task-enriched version, and `mhc_class` (plus manifest
   field and verifier V7) taken from the eligibility table for both. Then the small build.
2. Three builder flags are now required and have no defaults: `--eligibility-policy
   reference-aware-v1`, `--benchmark-exclusion heldout-peptides-v2` (+ `--exclusion-list`),
   `--split-routing receptor-hash-v2`. `cluster-v0` / `none` reproduce past builds.
3. Nothing has been built or run on a box under the new policies. The box is first needed for
   step 10's build itself. The contract's counts for strata 3-9 are a proposal, not derived.
4. Not isolated from CPT and not safe to score against a CPT model: benchmark parts
   `test_novel_tcr`, `test_novel_allele`, `test_mixed` (their peptides overlap train).
5. EC2 last verified stopped. Preserve colleague edits in `/home/ubuntu/quest`; deploy by bundle
   + worktree to `/home/ubuntu/quest-jf`.
