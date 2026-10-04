# TCRBench v3 — CPT Stage 2 SUMMARY (TCR Representation Sweep)

ESM-C 300M LoRA MLM trained on three 10M-row CPT datasets that vary the TCR representation (CDR3 / CDR1+2+3 / Full chain), all on the proportional distribution and the MHC pocket variant. Canonical evaluation is region-restricted on `data/cpt_canonical_eval_v2`; primary decision metric is `cdr3_ppl`.

## 1. Setup

| Field | Value |
|---|---|
| Base model | `esmc_300m` (EvolutionaryScale ESM Cambrian) |
| LoRA | r=32, α=64, dropout=0.05; targets: layernorm_qkv.1, out_proj, ffn.1, ffn.3 |
| Canonical eval | `/home/ubuntu/quest/data/cpt_canonical_eval_v2/` (19,708 rows, 9 strata) |
| EOS strategy | `raw` (ESM-C tokenizer natively recognizes `<eos>`) |
| Early stop | CdrPplPlateauCallback on `cdr3_ppl`, 3-eval patience, 1% rel improvement |

## 2. Per-run summary

| Run | TCR | MHC | Max len | Batch × Accum | Best `cdr3_ppl` | Final step | Stop trigger |
|---|---|---|---:|---:|---:|---:|---|
| Run 01 (CDR3) | cdr3 | pocket | 64 | 192×2 | 3.966 | 5200 | early-stop |
| Run 02 (CDR1+2+3) | cdr123 | pocket | 96 | 128×2 | 6.171 | 3000 | early-stop |
| Run 03 (Full TCR) | full | pocket | 1024 | 16×16 | 6.342 | 7500 | early-stop |

## 3. Region-restricted canonical eval (centerpiece)

Perplexity per region, overall. Lower is better.

| Region | Run 01 (CDR3) | Run 02 (CDR1+2+3) | Run 03 (Full TCR) |
|---|---|---|---|
| overall | 4.967 | 6.378 | 3.328 |
| **cdr3** (primary) | 3.966 | 6.171 | 6.342 |
| cdr12 | — | 2.516 | 2.995 |
| pocket | 4.733 | 14.130 | 15.499 |
| pocket_contact_only | — | — | — |
| peptide | 15.954 | 16.313 | 16.895 |
| framework (sanity) | — | — | 2.977 |

## 4. Per-stratum `cdr3_ppl` breakdown

| stratum_key | Run 01 (CDR3) | Run 02 (CDR1+2+3) | Run 03 (Full TCR) | nrows |
|---|---|---|---|---|
| II_complete | 4.496 | 8.870 | 15.754 | 270 |
| II_partial_DP_beta | 4.429 | 10.635 | 12.044 | 26 |
| II_partial_DQ_beta | 5.319 | 9.925 | 15.034 | 56 |
| II_partial_DR_beta | 4.238 | 9.781 | 12.549 | 91 |
| none_nseg1 | 3.405 | 3.656 | 3.963 | 9121 |
| none_nseg2 | 4.809 | 12.051 | 10.661 | 4742 |
| none_nseg3 | 4.378 | 8.523 | 10.563 | 309 |
| none_nseg4+ | 3.466 | 9.955 | 13.366 | 16 |

## 5. Cross-stage comparison (model-size effect at fixed CDR3 rep)

| Checkpoint | overall_ppl | cdr3_ppl |
|---|---:|---:|
| Stage 1 Run 01 (ESM-2 150M proportional) | 4.786 | 3.859 |
| Stage 1 Run 06 (ESM-2 150M balanced) | 3.781 | 3.210 |
| Stage 2 Run 01 (ESM-C 300M, CDR3) | 4.967 | 3.966 |

## 6. Decision gates

- Overall `cdr3_ppl` improvement Run 03 vs Run 01: **-59.9%**
- `nseg2_cdr3_ppl` (multi-segment rows) Run 03 vs Run 01: **-121.5%**
- Run 02 vs Run 03 `cdr3_ppl`: -2.7% (negative means Run 02 wins)

**Recommendation** (apply the plan's decision rules):
- Improvement only **-121.5%** on `nseg2_cdr3_ppl` (threshold: <5%). **Stick with CDR3 for production; drop Run 09 (~$50 saved).**

## 7. Diagnostics

- **Run 01 (CDR3)**: tcr=cdr3 mhc=pocket eos_strategy=raw rows=19708/19708 empty=0 annot_fail=0
- **Run 02 (CDR1+2+3)**: tcr=cdr123 mhc=pocket eos_strategy=raw rows=19708/19708 empty=0 annot_fail=0
- **Run 03 (Full TCR)**: tcr=full mhc=pocket eos_strategy=raw rows=19708/19708 empty=0 annot_fail=0
