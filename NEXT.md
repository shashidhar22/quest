# NEXT — quest (updated 2026-11-11) · history: `docs/LOG.md`

Judge-first's −0.023 for legacy arm D is correctly computed, but both of its SFT judges sit at
the training base rate (DG1, 2026-11-11), so it compares what is left over, not two working judges. Corpus
eligibility repair is through step 11 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. First move: repair the SFT judge. Visit DG1 (2026-11-11) found both V04a judges at the
   training base rate: every logit within 1e-04 of −1.669, arm D's training loss 0.451 against
   a prior entropy of 0.4506 (ACS `quest-cpt/findings.md` 9.13). Start on the laptop: the
   judge's training loop, loss and learning rate on a few hundred molecules, and why the
   pooled vector moves by under 1e-03 between peptides. No box needed to begin.
2. On hold until the judge learns: PL1 (10% two-arm pilot, ≈ USD 20) and DG1's stage B
   (≈ USD 2.4 at the measured 2.4 s per scoring; its 50-minute step limit and 2 h ceiling
   are too small). Both are prepared: `plans/step12-diagnostic-and-pilot.md`, ACS
   `quest-cpt/box-jobs/visits/` and `ops/`. Regenerate `experiment-lock.json` if any pinned
   file changes, which a judge repair will do.
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
