# NEXT — quest (updated 2026-11-10) · history: `docs/LOG.md`

Judge-first stands unchanged: legacy arm D shows no benefit, locked primary −0.023. Corpus
eligibility repair is now through step 9 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. First move: the box half of step 10 — build `task_enriched_cap1000` (contract revision 2)
   against the real single-receptor partitions and run the verifier WITHOUT
   `--allow-background-subset`. Untested there: background supply over 1.42B rows, its 1% residue
   tolerance, and the build time (the background query sorts each split by key). Needs an EC2
   visit; plan in `plans/step10-contract-build.md`, laptop results in ACS `repair-status.md`.
2. Builder flags, none with a default: `--eligibility-policy reference-aware-v1`,
   `--benchmark-exclusion heldout-peptides-v2` (+ `--exclusion-list`), `--split-routing
   receptor-hash-v2`; for a contract build also `--distribution contract --sampling-contract
   <json> --contract-version task_enriched_cap1000 --scale 7571616`.
3. Nothing has run on a box under the new policies. The proportional contract version has
   not been built anywhere. The contract's counts for strata 3-9 are a proposal, not derived.
4. Not isolated from CPT and not safe to score against a CPT model: benchmark parts
   `test_novel_tcr`, `test_novel_allele`, `test_mixed` (their peptides overlap train).
5. EC2 last verified stopped. Preserve colleague edits in `/home/ubuntu/quest`; deploy by bundle
   + worktree to `/home/ubuntu/quest-jf`.
