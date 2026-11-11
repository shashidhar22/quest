# NEXT — quest (updated 2026-11-11) · history: `docs/LOG.md`

Judge-first stands unchanged: legacy arm D shows no benefit, locked primary −0.023. Corpus
eligibility repair is through step 11 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. First move: step 12 — `token_exact` in the training loop, the preflight
   (`quest/data/corpus_preflight.py: corpus_problems`) wired into the launcher, then price the
   first CPT run. Both corpus versions are published, each with `PUBLISHED.json`:
   `s3://fh-pi-warren-h-eco/quest/cpt/repair-20261109T-eligibility/{task_enriched_cap1000,repaired_proportional_10M}/`.
   Results: ACS `repair-status.md` "Step 11"; visit `box-jobs/visits/cr2-2026-11-11.md`.
2. Rebuilding: `scripts/data_processing/contract_build_box/run_contract_build.sh` then
   `audit_and_publish.sh` (CPU box, r6i.8xlarge; enriched 20 min, proportional 31 min and
   243 GB peak). Contract revision 4. The box's own `data/molecule_clusters/` is the WRONG
   cluster set (`neurips_two`); the contract's is in `/home/ubuntu/cr1_in/molecule_clusters/`
   and on S3 `quest/tcrbench2/molecule_clusters/`.
3. No CPT run can use the new corpus until `token_exact` lands in the training loop (step 12).
   The contract's counts for strata 3-9 are a proposal, not derived.
4. Not isolated from CPT and not safe to score against a CPT model: benchmark parts
   `test_novel_tcr`, `test_novel_allele`, `test_mixed` (their peptides overlap train).
5. EC2 last verified stopped. Preserve colleague edits in `/home/ubuntu/quest`; deploy by bundle
   + worktree to `/home/ubuntu/quest-jf`.
