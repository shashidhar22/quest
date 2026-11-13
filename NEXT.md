# NEXT — quest (updated 2026-11-12) · history: `docs/LOG.md`

Judge-first's −0.023 for legacy arm D is correctly computed, but both of its SFT judges sit at
the training base rate (DG1, 2026-11-11), so it compares what is left over, not two working judges. Corpus
eligibility repair is through step 11 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. DG1 repair removes constant scores, but the first full ESM-C run fails the scientific gate:
   validation BCE 0.5453 vs prior 0.4506; receptor substitutions barely change ranking. A receptor-free
   composition baseline gets BCE 0.4250 / AUROC 0.7513 on matching split hashes (ACS findings 9.15).
2. Frozen-encoder comparison RUNNING: 20261113T024720Z-2701 (Nov 12 local), same rows/seed/recipe,
   /home/ubuntu/dg1-repair-frozen-out/latest on i-0030c1a9c845b3714. Retrieve validation.json and
   scores from S3 quest/tcrbench2/pilot12/dg1-repair-frozen/<run>; verify stopped after completion.
   PL1 remains on hold; experiment lock/checkpoint paths need revision before any later pilot.
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
7. Initial repair and first-run continuation commits are pushed. First GPU run completed/published;
   watchdog stopped it. Two restart attempts refused capacity; third succeeded for the frozen run.
   Current limits: 90-minute job, two-hour watchdog; boot shutdown scheduled 05:47 UTC Nov 13.
