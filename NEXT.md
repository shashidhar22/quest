# NEXT — quest (updated 2026-11-12) · history: `docs/LOG.md`

Judge-first's −0.023 for legacy arm D is correctly computed, but both of its SFT judges sit at
the training base rate (DG1, 2026-11-11), so it compares what is left over, not two working judges. Corpus
eligibility repair is through step 11 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. DG1 judge repair is implemented and locally checked (2026-11-12): explicit peptide pool,
   feature normalization, mask wiring and checkpoint persistence; original Track A stays compatible.
   Cached-feature validation beats the prior at two epochs across four seeds (ACS findings 9.14).
   First move: validate one freshly trained repaired base judge end to end, measuring held-out BCE,
   within-query ranking and receptor-swap sensitivity. No repaired ESM-C checkpoint exists yet.
2. PL1 and DG1 stage B remain on hold. Existing V04a checkpoints cannot load into the repaired
   architecture; use the historical revision to reproduce them. Before a new box visit, refresh
   the experiment lock/recipe, choose fresh checkpoint paths and revise diagnostic time limits
   (stage B measured ~63 min per checkpoint). No EC2 launch in this repair session.
3. Rebuilding: `scripts/data_processing/contract_build_box/run_contract_build.sh` then
   `audit_and_publish.sh` (CPU box, r6i.8xlarge; enriched 20 min, proportional 31 min and
   243 GB peak). Contract revision 4. The box's own `data/molecule_clusters/` is the WRONG
   cluster set (`neurips_two`); the contract's is in `/home/ubuntu/cr1_in/molecule_clusters/`
   and on S3 `quest/tcrbench2/molecule_clusters/`.
4. The contract's counts for strata 3-9 are a proposal, not derived.
5. Not isolated from CPT and not safe to score against a CPT model: benchmark parts
   `test_novel_tcr`, `test_novel_allele`, `test_mixed` (their peptides overlap train).
6. EC2 last verified stopped (2026-11-11 23:55 UTC). Preserve colleague edits in `/home/ubuntu/quest`; deploy by bundle
   + worktree (recent visits used `/home/ubuntu/quest-cr`).
7. Repair commits in QUEST and ACS are local only: automatic approval review blocked the
   GitHub push as an unverified export. User approval of the existing remote destinations is pending.
