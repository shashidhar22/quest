# CPT stage index — the only place stage status is tracked

| Stage | Plan | Status | Result or gate |
|---|---|---|---|
| 4.1 | `stage4-1.md` | **DONE** 2026-10-03 | `pocket_ppl` A 3.368 / D 1.862 / F 2.090, all 7,800 steps; `docs/cpt_stage4.1/README.md` §10 |
| 4.2 | `stage4-2.md` | planned | downstream SFT judge; **gate for 4.4 onward** |
| 4.3 | `stage4-3.md` | planned | trainer efficiency + mid-run resume; runs in parallel with 4.2 |
| 4.4 | `stage4-4.md` | planned | balanced sampling at cdr3 × full; needs 4.2 and 4.3 |
| 4.5 | `stage4-5.md` | planned | pocket-upweighted masking on the 4.4 winner |
| 4.6 | `stage4-6.md` | planned | allele-frequency diagnostic; cheap form needs no GPU |
| 4.7 | `stage4-7.md` | planned | scaling 10M → 100M at the final recipe; last |

Update this table when a stage starts or ends. Nothing else tracks status.
Knowledge (state, findings, data map): `~/Documents/ACS/quest-cpt/`. History: `docs/LOG.md`.
