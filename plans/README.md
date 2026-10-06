# CPT stage index — the only place stage status is tracked

| Stage | Plan | Status | Result or gate |
|---|---|---|---|
| 4.1 | `stage4-1.md` | **DONE** 2026-10-03 | `pocket_ppl` A 3.368 / D 1.862 / F 2.090, all 7,800 steps; `docs/cpt_stage4.1/README.md` §10 |
| 4.2 | `stage4-2.md` | **wave 1 launcher built** (2026-10-06, `stage42_box/wave1.sh`, dry-run 5/5); runs after 4.3 in the same box visit | downstream SFT judge; wave 1 (base vs arm D, 4 cells, decision order, harness gate after cell 1) is **the gate for 4.4–4.7** |
| 4.3 | `stage4-3.md` | **GPU results 2026-10-06:** `EVAL_AB` PASS, ≈ 3×/GPU speed, **`TRAIN_EQUIV` FAIL** (v1 sampler: ordering effect, findings §9.3); sampler v2 built + verified on laptop, GPU re-test (`TRAIN_EQUIV_V2`) queued as phase 3 of the running box visit; `RESUME` pending | trainer efficiency + mid-run resume; the v2 equivalence run becomes 4.4's control |
| 4.6 | `stage4-6.md` | step 1 free, run with 4.2 | allele-frequency diagnostic; decides whether `pocket_ppl` means biochemistry or HLA recall |
| 4.4 | `stage4-4.md` | after 4.2 + 4.3 | balanced vs proportional; step 1 needs **no build** (run06 vs 4.1 arm A) |
| 4.5 | `stage4-5.md` | after 4.4 | pocket-upweighted masking on the 4.4 winner |
| 4.7 | `stage4-7.md` | last | scaling 10M → 100M at the final recipe |

Order rationale: 4.2 and 4.3 cost nothing to start and gate everything; 4.6 step 1 is free and
changes how every `pocket_ppl` is read; 4.4 and 4.5 train new adapters and so wait for the judge;
4.7 is the only step that must be repeated if the recipe changes, so it goes last.

Update this table when a stage starts or ends. Nothing else tracks status.
Knowledge (state, findings, data map): `~/Documents/ACS/quest-cpt/`. History: `docs/LOG.md`.
