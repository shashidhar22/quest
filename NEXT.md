# NEXT — quest (updated 2026-11-06 23:20 UTC) · history: `docs/LOG.md` · stages: `plans/README.md` · status + order: `~/Documents/ACS/quest-cpt/audits/2026-11-06/session-summary.md`

**MID-FLIGHT: judge-first run `20261106T225852Z-9828` is RUNNING** on `i-0030c1a9c845b3714` (started 22:58 UTC, gate ~00:20, done ~04:00, ceiling 8 h; the watchdog stops the box).
1. **First move:** confirm the box is `stopped`. Then read `s3://fh-pi-warren-h-eco/quest/tcrbench2/jf_out/20261106T225852Z-9828/`: `status.json` (complete or refused), `comparison.json`, and `launch_status.json`, whose `driver_rc` should equal the real exit code.
2. If the gate refused, the judge moves to `as_trb_i`; stages 4.4–4.7 continue. If it was interrupted, rerun with `RESUME=20261106T225852Z-9828` (see the session summary).
3. Then stage 4.3: build `token_exact` into the training loop (`plans/v04-audit-repair.md` step 5). On the first re-baseline, check that a TERM stops torchrun's workers too.
4. Box rule: `/home/ubuntu/quest` is a colleague's branch with uncommitted edits. Never check out there; run from the worktree `/home/ubuntu/quest-jf`.
5. Push both remotes: `git push` and `git push mine`.
