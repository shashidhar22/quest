# NEXT — quest (updated 2026-11-11) · history: `docs/LOG.md`

Judge-first stands unchanged: legacy arm D shows no benefit, locked primary −0.023. Corpus
eligibility repair is now through step 10 of 12 (task-enriched version); the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. First move: decide step 11 — what to build and publish (the task-enriched corpus is built
   and verified on the box, unpublished, at `/home/ubuntu/cpt_repair/cr1-20261111T013453Z/dataset`;
   the proportional version has never been built), then the leakage audit on the built files.
   Results: ACS `repair-status.md` "Step 10, box half"; visit `box-jobs/visits/cr1-2026-11-10.md`.
2. Contract build: `scripts/data_processing/contract_build_box/run_contract_build.sh` (CPU box,
   r6i.8xlarge, 20 min). Contract revision 4, `--scale 7445573`. The box's own
   `data/molecule_clusters/` is the WRONG cluster set (`neurips_two`); the contract's is in
   `/home/ubuntu/cr1_in/molecule_clusters/` and on S3 `quest/tcrbench2/molecule_clusters/`.
3. No CPT run can use the new corpus until `token_exact` lands in the training loop (step 12).
   The contract's counts for strata 3-9 are a proposal, not derived.
4. Not isolated from CPT and not safe to score against a CPT model: benchmark parts
   `test_novel_tcr`, `test_novel_allele`, `test_mixed` (their peptides overlap train).
5. EC2 last verified stopped. Preserve colleague edits in `/home/ubuntu/quest`; deploy by bundle
   + worktree to `/home/ubuntu/quest-jf`.
