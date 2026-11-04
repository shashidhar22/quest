# NEXT — quest (updated 2026-11-04) · history: `docs/LOG.md` · stages: `plans/README.md` · knowledge: `~/Documents/ACS/quest-cpt/`

**No box is running.** `i-0030c1a9c845b3714` is stopped and sized **g6.2xlarge**; V04 needs
**g6.12xlarge** on the same volume `vol-08f5626ccf18fdd4c`. Decision gates and all Stage 4.2/4.3
detail: `quest-cpt/box-jobs/README.md` §3.2–3.4.

1. **`aws sso login --profile DataUserAccess-953915750371`** — the token expires; nothing touching
   the box or S3 runs without it.
2. **`git push`** the one local commit. Claude's sandbox refuses push as a public-surface action, so
   Yuexin runs it.
3. **First minutes on the box, before any job writes:** read-only copy of `/home/ubuntu/v2_full.sh`
   and any surviving uncommitted `track_a/` off `vol-08f5626ccf18fdd4c`, hashed and committed. It is
   the only record of what computed the two failed V03 verdicts.
4. **Resize to g6.12xlarge, then launch `scripts/training/v04_box/run_v04.sh`** and arm
   `autostop_when_done.sh` on its PID with `MAX_HOURS`. It encodes the chain: P1 arm A × 3 seeds +
   base re-score under the length-matched draw → **G1** locked-read readability gate on the NEW-draw
   ranks → P2 arm D × 3 seeds, comparison cells **only on a G1 pass**. ≈10 h, ≈USD 46 on-demand.

**Open:** ρ is assumed 0.8 against a measured breakeven **0.765** — the first comparison cell must
report measured ρ against both · every readability number is power under the **uniform** draw until
G1 recomputes it · `4.3-noise-det` not run, so the flash-attn share of the 1.31 % is unapportioned ·
arm A has never been judge-scored · §9.6's length finding still needs restating per peptide in the
findings text itself.
