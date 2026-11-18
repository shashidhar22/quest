# NEXT — quest (updated 2026-11-17) · history: `docs/LOG.md`

Judge-first's −0.023 for legacy arm D is correctly computed, but both of its SFT judges sit at
the training base rate (DG1, 2026-11-11), so it compares what is left over, not two working judges. Corpus
eligibility repair is through step 11 of 12; the plan and all results are in ACS
`quest-cpt/audits/2026-11-09/` (`corpus-repair-plan.md`, `repair-status.md`).
1. DG1 code repairs are done; scientific gate remains uncleared (ACS findings 9.15).
   GPU BCE: trainable 0.5453, frozen 0.4531; independent train-only calibration gives 0.4990/0.4522,
   still above prior 0.4506. Corrected receptor swaps show no reliable true-receptor advantage.
2. Fold-local CV repair and rerun COMPLETE (Nov 17; ACS findings 9.16). All 17 tests and
   independent replay of 15 CV fits pass; zero input-cluster overlap. Prior BCE 0.45056 beats
   best logistic 0.45323 (C=0.001, OOF AUROC 0.58059). Original contaminated CV is superseded.
   Exploratory full-panel refit: validation BCE 0.40305; receptor delta +0.01725,
   CI [-0.06512,+0.02695]. No recipe promoted; PL1 stays blocked.
   Paired 100-candidate test complete: frozen judge 194/255 Recall@10 with either receptor.
   Paired interaction readout improves mean rank by 28.19, but Recall@10 gain CI crosses zero.
   Independent 43-cluster/43-donor test complete: 10 true vs 8 donor hits, peptide-only 9.
   Rank gain +5.21 (95% interval −8.30 to +18.14). PL1 stays on hold. See revised step-12 plan.
   Evidence: ACS judge-head/paired-100. All local jobs complete.
   CPU baseline-plus-correction and joint training completed: both selected zero receptor
   weight on training folds. No PL1 promotion. Results: ACS judge-head/joint-readout.
   Priority 3 complete: chain pooling won 68/128 vs 69/128 matched peptide-only pooling
   on familiar peptides with held-out receptor clusters. PL1 remains on hold. See chain-pooling/.
   Reciprocal assignment diagnostic: chain head 63/128, same receptor wins both peptides
   in 114/128. Similarity 64.1%. Training roles are strongly imbalanced. Reciprocal
   training loss comparison completed: 85 single-use groups, familiar-target assignment
   70.3% directional / 69.5% reciprocal. Both heads give 4/43 true hits on unseen targets
   versus 9/43 peptide-only. PL1 remains on hold. See ACS judge-head/reciprocal-training/. Evidence: ACS judge-head/reciprocal-control/.
   Evidence: ACS judge-head/local-foldwise; full cache S3 prefix
   pilot12/dg1-repair-fixed/local-20261117-foldwise-v1/ (see feature-cache.json).
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
   model checkpoints remain on S3 and the EC2 volume. No experiment jobs remain active.
   Fixed-feature cache: pilot12/dg1-repair-fixed/local-20261113T1422Z/ (S3 SHA-256 verified Nov 17).
   The experiment lock and checkpoint paths still need revision before any pilot.
