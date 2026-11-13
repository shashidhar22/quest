# NEXT — quest (updated 2026-11-12) · history: `docs/LOG.md`

Judge-first's −0.023 for legacy arm D is correctly computed, but both of its SFT judges sit at
the training base rate (DG1, 2026-11-11), so it compares what is left over, not two working judges. Corpus
eligibility repair is through step 11 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. DG1 code repairs are done; scientific gate remains uncleared (ACS findings 9.15).
   GPU BCE: trainable 0.5453, frozen 0.4531; independent train-only calibration gives 0.4990/0.4522,
   still above prior 0.4506. Corrected receptor swaps show no reliable true-receptor advantage.
2. First move: test a fixed-feature/train-only-standardized head baseline (the earlier review's
   minimal proposal), using the corrected known-positive-excluding controls. Numerical collapse
   and control bugs are repaired; neither current neural recipe is promoted. PL1 stays on hold.
   All three EC2 jobs completed/published; EC2 stopped confirmed 2026-11-13 03:21 UTC.
3. Rebuilding: `scripts/data_processing/contract_build_box/run_contract_build.sh` then
   `audit_and_publish.sh` (CPU box, r6i.8xlarge; enriched 20 min, proportional 31 min and
   243 GB peak). Contract revision 4. The box's own `data/molecule_clusters/` is the WRONG
   cluster set (`neurips_two`); the contract's is in `/home/ubuntu/cr1_in/molecule_clusters/`
   and on S3 `quest/tcrbench2/molecule_clusters/`.
4. The contract's counts for strata 3-9 are a proposal, not derived.
5. Not isolated from CPT and not safe to score against a CPT model: benchmark parts
   `test_novel_tcr`, `test_novel_allele`, `test_mixed` (their peptides overlap train).
6. EC2 shutdown requested after completed readout (state above). Preserve colleague edits in `/home/ubuntu/quest`; deploy by bundle
   + worktree (recent visits used `/home/ubuntu/quest-cr`).
7. Saved runs: trainable 20261112T160549Z-1857, frozen 20261113T024720Z-2701, corrected readout
   20261113T030649Z-1684 (Nov 12 local). S3 quest/tcrbench2/pilot12/ prefixes dg1-repair,
   dg1-repair-frozen, dg1-repair-readout respectively. ACS preserves reports/raw evaluation scores;
   model checkpoints remain on S3 and the EC2 volume. No jobs are left intentionally running.
   The experiment lock and checkpoint paths still need revision before any pilot.
