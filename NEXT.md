# NEXT — quest (updated 2026-11-17) · history: `docs/LOG.md`

Judge-first's −0.023 for legacy arm D is correctly computed, but both of its SFT judges sit at
the training base rate (DG1, 2026-11-11), so it compares what is left over, not two working judges. Corpus
eligibility repair is through step 11 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. DG1 code repairs are done; scientific gate remains uncleared (ACS findings 9.15).
   GPU BCE: trainable 0.5453, frozen 0.4531; independent train-only calibration gives 0.4990/0.4522,
   still above prior 0.4506. Corrected receptor swaps show no reliable true-receptor advantage.
2. Fixed-feature MPS run completed; independent replay passes. C=1: validation BCE 0.7378.
   Training-cluster CV (best logistic 0.6644 vs prior 0.4506) is NOT a valid selection: held-out
   peptides stay in the fit folds as negatives, OOF AUROC 0.06-0.12 (ACS cv-leak-check.py, Nov 17).
   Best logistic C=0.0001 gives held-out BCE 0.4290 but receptor delta +0.0016,
   CI [-0.0185,+0.0216]; exploratory, no recipe promoted. PL1 stays on hold.
   First move: revise receptor-sensitive validation before further paid training; retain the
   receptor-free comparator; draw negatives within each fold for any training-only selection.
   Fold-local CV rerun ACTIVE (Nov 17): local MPS, 6000 rows, folds624/188/188;
   /private/tmp/dg1-foldwise-20261117/stage-0.log and job-status.json.
   Bounded wrapper runs selection automatically after embedding; review results before promotion.
3. Rebuilding: `scripts/data_processing/contract_build_box/run_contract_build.sh` then
   `audit_and_publish.sh` (CPU box, r6i.8xlarge; enriched 20 min, proportional 31 min and
   243 GB peak). Contract revision 4. The box's own `data/molecule_clusters/` is the WRONG
   cluster set (`neurips_two`); the contract's is in `/home/ubuntu/cr1_in/molecule_clusters/`
   and on S3 `quest/tcrbench2/molecule_clusters/`.
4. The contract's counts for strata 3-9 are a proposal, not derived.
5. Not isolated from CPT and not safe to score against a CPT model: benchmark parts
   `test_novel_tcr`, `test_novel_allele`, `test_mixed` (their peptides overlap train).
6. EC2 restored to g6.2xlarge and stopped after capacity refusals. Preserve colleague edits in `/home/ubuntu/quest`; deploy by bundle
   + worktree (recent visits used `/home/ubuntu/quest-cr`).
7. Saved runs: trainable 20261112T160549Z-1857, frozen 20261113T024720Z-2701, corrected readout
   20261113T030649Z-1684 (Nov 12 local). S3 quest/tcrbench2/pilot12/ prefixes dg1-repair,
   dg1-repair-frozen, dg1-repair-readout respectively. ACS preserves reports/raw evaluation scores;
   model checkpoints remain on S3 and the EC2 volume. Only the bounded local foldwise rerun above is active.
   Fixed-feature cache: pilot12/dg1-repair-fixed/local-20261113T1422Z/ (S3 SHA-256 verified Nov 17).
   The experiment lock and checkpoint paths still need revision before any pilot.
