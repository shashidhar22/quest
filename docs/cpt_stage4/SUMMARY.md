# TCRBench v3 — CPT Stage 4 SUMMARY (Coupled TCR×MHC Representation Sweep)

Generated: 2026-05-29T16:41:56.002694Z

ESM-C 300M LoRA MLM trained on five coupling-scope points to test whether jointly enriching the TCR and MHC representations beats the Stage 2 CDR3-only winner. Canonical evaluation is region-restricted on `data/cpt_canonical_eval_v2`; primary decision metric is `pocket_ppl` (MHC-bearing rows only, by construction).

## 1. Setup

| Field | Value |
|---|---|
| Base model | `esmc_300m` (EvolutionaryScale ESM Cambrian) |
| LoRA | r=32, α=64, dropout=0.05; auto-detected ESM-C targets |
| Canonical eval | `/home/ubuntu/quest/data/cpt_canonical_eval_v2/` (19,708 rows, 9 strata) |
| EOS strategy | `raw` |
| Early stop | `pocket_ppl` plateau, 3-eval patience, 1% rel improvement |
| WandB project | `tcrbench-v3-cpt-stage4` |

## 2. Per-point execution

| Point | TCR | MHC | Dataset | Status | Wall (hr) |
|---|---|---|---|---|---:|
| A | cdr3 | pocket | run01 | ✓ | 1.44 |
| B | cdr123 | pocket_contact | run10 | ✓ | 1.12 |
| C | full | full | run09 | ✓ | 14.07 |
| D | cdr3 | full | run05 | ✓ | 5.67 |
| E | full | pocket | run03 (reused ckpt) | ✓ | — |

## 3. Region-restricted canonical eval (centerpiece)

Perplexity per region, overall. Lower is better. `pocket_ppl` is the primary decision metric.

| Region | A (cdr3 + pocket) | B (cdr123 + pocket+contact) | C (full + full) | D (cdr3 + full MHC) | E (full + pocket (reused)) |
|---|:---:|:---:|:---:|:---:|:---:|
| overall_ppl | 4.968 | 6.590 | 4.936 | 1.746 | 3.328 |
| cdr3_ppl | 3.968 | 6.172 | 6.526 | 4.002 | 6.342 |
| cdr12_ppl | — | 2.436 | 2.976 | — | 2.995 |
| framework_ppl | — | — | 4.699 | 1.283 | 2.977 |
| peptide_ppl | 15.951 | 15.862 | 17.611 | 15.354 | 16.895 |
| pocket_ppl (primary) | 4.732 | 14.198 | 14.836 | 2.001 | 15.499 |
| pocket_contact_only_ppl | — | 15.217 | 14.956 | 4.487 | — |

## 4. Per-mhc_class breakdown for primary metric (`pocket_ppl`)

| mhc_class | A | B | C | D | E |
|---|:---:|:---:|:---:|:---:|:---:|
| I | 18.822 | 16.060 | 17.302 | 11.448 | 16.315 |
| II_complete | 4.574 | 13.929 | 14.622 | 1.945 | 15.185 |
| II_partial_DP_beta | 1.551 | 13.399 | 15.658 | 1.301 | 17.999 |
| II_partial_DQ_beta | 6.404 | 14.870 | 15.118 | 1.466 | 17.445 |
| II_partial_DR_beta | 5.300 | 15.506 | 15.452 | 2.282 | 16.191 |
| none | 6.243 | 13.320 | 15.609 | 2.317 | 16.722 |

## 5. Coupling decision matrix

| Comparison | `pocket_ppl` change vs A | interpretation |
|---|---:|---|
| A (baseline) | 4.732 | reference (cdr3+pocket) |
| B vs A | +200.0% | cdr123 + pocket_contact effect |
| C vs A | +213.5% | full + full effect (max scope) |
| D vs A | -57.7% | MHC-side scope only (cdr3 + full MHC) |
| E vs A | +227.5% | TCR-side scope only (full + pocket) |

**Best `pocket_ppl`**: Point D (cdr3 + full MHC) = 2.001

## 6. Cross-stage synthesis

- Stage 2 (TCR-axis sweep, pocket MHC held): CDR3 won on `cdr3_ppl`; full TCR lost on multi-segment rows.
- Stage 3 (MHC-axis sweep, CDR3 TCR held): full MHC pushed `pocket_ppl` down dramatically vs pocket-only (Run 05 vs Stage 2 Run 01).
- Stage 4 tests whether _coupled_ enrichment (rich TCR ∧ rich MHC) compounds the gains or saturates. See decision matrix above.

## 7. Diagnostics

- **Point A** (cdr3 + pocket): tcr=cdr3 mhc=pocket eos_strategy=raw rows=19708 empty=0 annot_fail=0
- **Point B** (cdr123 + pocket+contact): tcr=cdr123 mhc=pocket_contact eos_strategy=raw rows=19708 empty=0 annot_fail=0
- **Point C** (full + full): tcr=full mhc=full eos_strategy=raw rows=19708 empty=0 annot_fail=0
- **Point D** (cdr3 + full MHC): tcr=cdr3 mhc=full eos_strategy=raw rows=19708 empty=0 annot_fail=0
- **Point E** (full + pocket (reused)): tcr=full mhc=pocket eos_strategy=raw rows=19708 empty=0 annot_fail=0
