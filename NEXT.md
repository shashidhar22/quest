# NEXT — quest (updated 2026-11-09) · history: `docs/LOG.md` · stages: `plans/README.md` · result + order: `~/Documents/ACS/quest-cpt/audits/2026-11-06/session-summary.md`

**Judge-first DONE (2026-11-07):** legacy arm D does NOT beat base. Locked primary −0.023 [−0.044, −0.002] at the peptide unit; the pep-cluster read spans zero; test_level4 R@10 0.401 → 0.228. Box `i-0030c1a9c845b3714` stopped. Results: `s3://fh-pi-warren-h-eco/quest/tcrbench2/jf_out/20261106T225852Z-9828/`.
1. **First move:** decide on the seed band. Retrain base under current code + 2 more SFT seeds per arm (≈USD 15); one seed per arm can't separate CPT from SFT noise (arm-D val AUC 0.630 vs base 0.702).
2. Free: explain why BOTH arms are below chance on test_novel_pep (norm_rank 0.457 / 0.434 vs 0.5), from the eval files.
3. Then decide whether stage 4.3 (`token_exact` loop) is still worth buying.
4. Box: `/home/ubuntu/quest` is a colleague's branch with uncommitted edits — never check out there; use the worktree `/home/ubuntu/quest-jf`. Push both remotes (`git push`, `git push mine`). AWS SSO expires ~8 h.
