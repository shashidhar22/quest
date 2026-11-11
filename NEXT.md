# NEXT — quest (updated 2026-11-11) · history: `docs/LOG.md`

Judge-first stands unchanged: legacy arm D shows no benefit, locked primary −0.023. Corpus
eligibility repair is through step 11 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. First move: approve and run visit DG1 (checkpoint diagnostic, ≈ USD 1, ceiling 2 h), then
   PL1 (10% two-arm pilot, ≈ USD 20, ceiling 24 h). Both are prepared and dry-run on the
   laptop, neither has touched a GPU: `plans/step12-diagnostic-and-pilot.md`, visit notes in
   ACS `quest-cpt/box-jobs/visits/`. Before launch: commit, bundle to a NEW worktree, upload the
   two manifests, and regenerate `experiment-lock.json` if any of its 57 pinned files changed (every repo module the trainer and judge import).
   Open: whether to widen the pilot's 2% masked-target bounds (set from 45 smoke steps).
2. Rebuilding: `scripts/data_processing/contract_build_box/run_contract_build.sh` then
   `audit_and_publish.sh` (CPU box, r6i.8xlarge; enriched 20 min, proportional 31 min and
   243 GB peak). Contract revision 4. The box's own `data/molecule_clusters/` is the WRONG
   cluster set (`neurips_two`); the contract's is in `/home/ubuntu/cr1_in/molecule_clusters/`
   and on S3 `quest/tcrbench2/molecule_clusters/`.
3. The contract's counts for strata 3-9 are a proposal, not derived.
4. Not isolated from CPT and not safe to score against a CPT model: benchmark parts
   `test_novel_tcr`, `test_novel_allele`, `test_mixed` (their peptides overlap train).
5. EC2 last verified stopped. Preserve colleague edits in `/home/ubuntu/quest`; deploy by bundle
   + worktree (recent visits used `/home/ubuntu/quest-cr`).
