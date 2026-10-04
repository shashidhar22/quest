# TCRBench v3 — CPT Stage 4.1 SUMMARY (Stage 4 re-run at one fixed max_length)

Generated: 2026-10-04T00:13:56.494432Z

ESM-C 300M LoRA MLM trained on three serialization arms with `max_length=1024` and effective batch 1024 held fixed across all of them, so serialization is the only variable — the control Stage 4 lacked. Canonical evaluation is region-restricted on `data/cpt_canonical_eval_v2`; primary decision metric is `pocket_ppl` (MHC-bearing rows only, by construction).

## 1. Setup

| Field | Value |
|---|---|
| Base model | `esmc_300m` (EvolutionaryScale ESM Cambrian) |
| LoRA | r=32, α=64, dropout=0.05; auto-detected ESM-C targets |
| Canonical eval | `/home/ubuntu/quest/data/cpt_canonical_eval_v2/` (19,708 rows, 9 strata) |
| EOS strategy | `raw` |
| Early stop | `pocket_ppl` plateau, 3-eval patience, 1% rel improvement |
| WandB project | `tcrbench-v3-cpt-stage41` |

## 2. Per-point execution

| Point | TCR | MHC | Dataset | Status | Wall (hr) |
|---|---|---|---|---|---:|
| A | cdr3 | pocket | run01 | ✓ | 1.58 |
| D | cdr3 | full | run05 | ✓ | 5.98 |
| F | cdr123 | full | run11 | ✓ | 6.05 |

## 3. Region-restricted canonical eval (centerpiece)

Perplexity per region, overall. Lower is better. `pocket_ppl` is the primary decision metric.

| Region | A (cdr3 + pocket) | D (cdr3 + full MHC) | F (cdr123 + full MHC) |
|---|:---:|:---:|:---:|
| overall_ppl | 4.405 | 1.726 | 1.966 |
| cdr3_ppl | 3.849 | 4.024 | 5.786 |
| cdr12_ppl | — | — | 2.260 |
| framework_ppl | — | 1.275 | 1.376 |
| peptide_ppl | 14.908 | 15.214 | 16.649 |
| pocket_ppl (primary) | 3.368 | 1.862 | 2.090 |
| pocket_contact_only_ppl | — | 4.162 | 4.023 |

## 4. Per-mhc_class breakdown for primary metric (`pocket_ppl`)

| mhc_class | A | D | F |
|---|:---:|:---:|:---:|
| I | 17.690 | 7.555 | 8.538 |
| II_complete | 3.083 | 1.800 | 2.006 |
| II_partial_DP_beta | 1.608 | 1.350 | 1.316 |
| II_partial_DQ_beta | 4.866 | 1.428 | 1.822 |
| II_partial_DR_beta | 4.728 | 2.131 | 2.465 |
| II_partial_alpha | 3.675 | 2.314 | 2.487 |

## 5. Serialization comparison (one fixed max_length across all arms)

| Arm | serialization | `pocket_ppl` | change vs A |
|---|---|---:|---:|
| A (baseline) | cdr3 + pocket | 3.368 | — |
| D | cdr3 + full MHC | 1.862 | -44.7% |
| F | cdr123 + full MHC | 2.090 | -37.9% |

**Best `pocket_ppl`**: Point D (cdr3 + full MHC) = 1.862

## 6. Cross-stage synthesis

- Stage 2 (TCR-axis sweep, pocket MHC held): CDR3 won on `cdr3_ppl`; full TCR lost on multi-segment rows.
- Stage 3 (MHC-axis sweep, CDR3 TCR held): full MHC pushed `pocket_ppl` down dramatically vs pocket-only (Run 05 vs Stage 2 Run 01).
- Stage 4 reported those gains, but set `max_length` per point (64/256/1024/1024), so representation and context window moved together and each point was scored on its own truncated view of the eval set.
- **Stage 4.1 (this run) holds `max_length` at 1024 and effective batch at 1024 across every arm**, so the serialization comparison above is the first one in this programme that attributes its own result. Arm D is a reproducibility control against Stage 4 Point D (`pocket_ppl` 2.001).

## 7. Diagnostics

- **Point A** (cdr3 + pocket): tcr=cdr3 mhc=pocket eos_strategy=raw rows=19708 empty=0 annot_fail=0
- **Point D** (cdr3 + full MHC): tcr=cdr3 mhc=full eos_strategy=raw rows=19708 empty=0 annot_fail=0
- **Point F** (cdr123 + full MHC): tcr=cdr123 mhc=full eos_strategy=raw rows=19708 empty=0 annot_fail=0
