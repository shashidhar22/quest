# Stage 4.6 step 1 — does pocket perplexity track closeness to a training MHC?

Generated 2026-10-15 12:01 by `scripts/analysis/stage46_allele_freq.py`.

**Measure.** For each eval row, the identity of its MHC chain (rarest/farthest chain for two-chain rows) to the nearest MHC chain in that arm's own training set, by global alignment. Pocket perplexity is token-weighted within each bin, from a per-row re-score that reproduces the stored Stage 4.1 result (check per arm below).

**Why not frequency deciles as planned.** Only 214 of 4,531 MHC eval rows have an exact training match (none of the 23 class I), so ~95 % of rows share frequency 0.

**Rule, fixed before results:** within a class, ppl(farthest bin) / ppl(exact) > 2.0 = largely recall. Bins under 30 rows are shown but not judged. Ratios carry a row-level bootstrap CI (added 2026-10-15) and are judged on the whole interval.
## Summary

| arm   | mhc_class                |   rows |   tokens |   ratio |   ci_lo |   ci_hi | verdict      | repro   |
|:------|:-------------------------|-------:|---------:|--------:|--------:|--------:|:-------------|:--------|
| A     | I                        |     23 |      131 | nan     | nan     | nan     | not judged   | PASS    |
| A     | II_complete              |   2371 |   19,687 |   2.608 |   2.349 |   2.893 | recall       | PASS    |
| A     | II_partial_DP_beta       |    160 |      496 | nan     | nan     | nan     | not judged   | PASS    |
| A     | II_partial_DQ_beta       |    199 |      598 |   4.437 |   3.579 |   5.462 | recall       | PASS    |
| A     | II_partial_DR_beta       |   1547 |    4,704 | nan     | nan     | nan     | not judged   | PASS    |
| A     | II_partial_alpha         |    157 |      857 | nan     | nan     | nan     | not judged   | PASS    |
| A     | ALL (class-adjusted rho) |   4457 |          |  -0.393 | nan     | nan     |              | PASS    |
| D     | I                        |     23 |      125 | nan     | nan     | nan     | not judged   | PASS    |
| D     | II_complete              |   2370 |   19,662 |   3.167 |   2.824 |   3.531 | recall       | PASS    |
| D     | II_partial_DP_beta       |    161 |      490 | nan     | nan     | nan     | not judged   | PASS    |
| D     | II_partial_DQ_beta       |    191 |      574 |   1.458 |   1.281 |   1.673 | not recall   | PASS    |
| D     | II_partial_DR_beta       |   1526 |    4,574 | nan     | nan     | nan     | not judged   | PASS    |
| D     | II_partial_alpha         |    157 |      862 | nan     | nan     | nan     | not judged   | PASS    |
| D     | ALL (class-adjusted rho) |   4428 |          |  -0.329 | nan     | nan     |              | PASS    |
| F     | I                        |     23 |      141 | nan     | nan     | nan     | not judged   | PASS    |
| F     | II_complete              |   2369 |   20,051 |   2.914 |   2.570 |   3.289 | recall       | PASS    |
| F     | II_partial_DP_beta       |    161 |      458 | nan     | nan     | nan     | not judged   | PASS    |
| F     | II_partial_DQ_beta       |    197 |      593 |   1.462 |   1.195 |   1.771 | not recall   | PASS    |
| F     | II_partial_DR_beta       |   1558 |    4,668 |   1.498 |   0.917 |   2.278 | inconclusive | PASS    |
| F     | II_partial_alpha         |    157 |      854 | nan     | nan     | nan     | not judged   | PASS    |
| F     | ALL (class-adjusted rho) |   4465 |          |  -0.304 | nan     | nan     |              | PASS    |


**Two limits on how this table may be read** (2026-10-15):

1. **The class-I stratum is 23 rows and ~130 masked tokens**, 11 alleles, 11 of them HLA-B\*08:03, and the eval holds **no HLA-A\*02:01 at all**. No class-I perplexity here is quotable without its n, and none of it speaks to an A\*02:01 benchmark such as IMMREP25. See `quest-cpt/findings.md` §9.7.

2. **Ratios are not comparable across arms.** Each arm's identity is measured against its own training set in its own serialization — pocket pseudo-sequences for A, full chains for D and F — so the identity bins mean different things per arm. Read each arm against the rule, never against another arm.

## Arm A — cdr3 x pocket, trained on `run01_cdr3_pocket_proportional_10M`

Reproduction of the stored Stage 4.1 pocket result: **PASS** — masked tokens 26473 vs 26473, ppl 3.3672 vs 3.3680 (0.03 %), mps.

| class | <0.93 | 0.93-0.97 | 0.97-<1 | exact | far/exact |
|---|---|---|---|---|---|
| I | 17.653 (n=20) | 18.705 (n=2) | 16.154 (n=1) | — | not judged |
| II_complete | 3.761 (n=1898) | 1.410 (n=385) | 1.421 (n=2) | 1.442 (n=86) | 2.61 [2.35, 2.89] recall |
| II_partial_DP_beta | — | 1.705 (n=134) | — | 1.186 (n=26) | not judged |
| II_partial_DQ_beta | 7.321 (n=146) | — | — | 1.650 (n=53) | 4.44 [3.58, 5.46] recall |
| II_partial_DR_beta | 4.882 (n=1428) | 3.917 (n=90) | — | 1.817 (n=29) | not judged |
| II_partial_alpha | 4.053 (n=142) | — | — | 1.511 (n=15) | not judged |

Class-adjusted Spearman(per-row pocket loss, identity) = **-0.393** over 4,457 rows (negative = closer to training, lower loss).

## Arm D — cdr3 x full, trained on `run05_cdr3_fullmhc_proportional_10M`

Reproduction of the stored Stage 4.1 pocket result: **PASS** — masked tokens 26287 vs 26287, ppl 1.8632 vs 1.8623 (0.05 %), mps.

| class | <0.93 | 0.93-0.97 | 0.97-<1 | exact | far/exact |
|---|---|---|---|---|---|
| I | 3.569 (n=1) | 7.684 (n=19) | 8.816 (n=3) | — | not judged |
| II_complete | 3.654 (n=244) | 1.855 (n=1611) | 1.178 (n=429) | 1.153 (n=86) | 3.17 [2.82, 3.53] recall |
| II_partial_DP_beta | — | — | 1.381 (n=135) | 1.263 (n=26) | not judged |
| II_partial_DQ_beta | 1.496 (n=22) | 1.672 (n=95) | 1.163 (n=23) | 1.147 (n=51) | 1.46 [1.28, 1.67] not recall |
| II_partial_DR_beta | 1.001 (n=2) | 3.415 (n=335) | 1.884 (n=1161) | 1.364 (n=28) | not judged |
| II_partial_alpha | 3.826 (n=9) | 2.436 (n=133) | — | 1.107 (n=15) | not judged |

Class-adjusted Spearman(per-row pocket loss, identity) = **-0.329** over 4,428 rows (negative = closer to training, lower loss).

## Arm F — cdr123 x full, trained on `run11_cdr123_fullmhc_proportional_10M`

Reproduction of the stored Stage 4.1 pocket result: **PASS** — masked tokens 26765 vs 26765, ppl 2.0936 vs 2.0900 (0.17 %), mps.

| class | <0.93 | 0.93-0.97 | 0.97-<1 | exact | far/exact |
|---|---|---|---|---|---|
| I | 3.226 (n=1) | 8.959 (n=19) | 8.613 (n=3) | — | not judged |
| II_complete | 4.118 (n=244) | 2.101 (n=1611) | 1.208 (n=428) | 1.413 (n=86) | 2.91 [2.57, 3.29] recall |
| II_partial_DP_beta | — | — | 1.326 (n=135) | 1.276 (n=26) | not judged |
| II_partial_DQ_beta | 1.896 (n=22) | 2.160 (n=98) | 1.555 (n=23) | 1.478 (n=54) | 1.46 [1.19, 1.77] not recall |
| II_partial_DR_beta | 1.003 (n=2) | 3.967 (n=342) | 2.160 (n=1184) | 2.648 (n=30) | 1.50 [0.92, 2.28] inconclusive |
| II_partial_alpha | 4.812 (n=9) | 2.326 (n=133) | — | 2.874 (n=15) | not judged |

Class-adjusted Spearman(per-row pocket loss, identity) = **-0.304** over 4,465 rows (negative = closer to training, lower loss).
