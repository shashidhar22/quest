# QUEST Standardized Data Manifest

This is the catalog of **standardized output** under `data/standardized_again/` — Stage 1 of the [data curation pipeline](../../../CLAUDE.md#data-curation-pipeline-overview). Each "bucket" is the per-source output of a standardizer in `scripts/data_processing/standardize/`, written as one or more Parquet shards using the unified 25-column [`TARGET_COLUMNS` schema](../../../quest/data/standardization.py).

The companion **raw-data** manifest at [`../raw_data/`](../raw_data/) covers the *input* sources (citations, methodology, fidelity scores). This wiki covers what comes out of the standardization step: how many rows, which columns are populated, what got dropped, and how that maps back to the raw input.

## How to read this manifest

- **The master summary table below** is the at-a-glance overview, rendered from [`inventory.json`](inventory.json).
- **Per-bucket detail pages** under [`sources/`](sources/) cover the standardization narrative (what the standardizer does, drop semantics, caveats) plus an auto-generated inventory block.
- **Fidelity** is *inherited* from the raw-data manifest — no separate scoring here. The standardization step doesn't change evidence quality.
- **Reviews**: this manifest was audited at creation by an [independent code review](CODE_REVIEW.md) of the inventory script and an [independent verifier](VERIFIER_REPORT.md).

## Master summary table

<!-- BEGIN: AUTO-MASTER-TABLE (build_standardized_manifest.py) -->

**Last regenerated**: 2026-04-27T07:09:01Z · **Buckets**: 18 · **Total standardized rows**: 5,666,341,543 · **Total dropped (during validation)**: 1,329,792,390

| Bucket | Rows | Dropped | Drop rate | Parquet size | Files | Coverage (trb/tra/pep/mhc1) | Yield vs raw |
|--------|-----:|--------:|----------:|-------------:|------:|-----------------------------|-------------:|
| [`adc`](sources/adc.md) | 2,187,853,453 | 126,625,882 | 5.47% | 98.1G | 2188 | trb=91% tra=3% pep=0% mhc1=0% | 106.4% |
| [`batman`](sources/batman.md) | 12,731 | 0 | 0.00% | 190.9K | 1 | trb=90% tra=90% pep=100% mhc1=88% | 55.8% |
| [`cedar_pmhc`](sources/cedar_pmhc.md) | 4,316,254 | 224,804 | 4.95% | 31.5M | 1 | trb=0% tra=0% pep=95% mhc1=36% | 66.6% |
| [`cedar`](sources/cedar.md) | 105,825 | 385 | 0.36% | 5.2M | 1 | trb=76% tra=48% pep=100% mhc1=95% | 1.6% |
| [`iedb_pmhc`](sources/iedb_pmhc.md) | 516,483 | 6,176 | 1.18% | 5.2M | 1 | trb=0% tra=0% pep=100% mhc1=60% | 16.3% |
| [`iedb`](sources/iedb.md) | 226,270 | 2,270 | 0.99% | 11.2M | 1 | trb=85% tra=28% pep=99% mhc1=55% | 7.1% |
| [`imgthla`](sources/imgthla.md) | 24,005 | 0 | 0.00% | 865.2K | 1 | trb=0% tra=0% pep=0% mhc1=75% | 0.3% |
| [`immuneaccess`](sources/immuneaccess.md) | 2,768,339,213 | 1,064,722,302 | 27.78% | 139.6G | 2769 | trb=97% tra=3% pep=0% mhc1=0% | 113.6% |
| [`immunecode`](sources/immunecode.md) | 251,274,283 | 55,603,532 | 18.12% | 14.9G | 1526 | trb=100% tra=0% pep=11% mhc1=11% | 85.2% |
| [`mcpas`](sources/mcpas.md) | 36,433 | 244 | 0.67% | 1.9M | 1 | trb=94% tra=34% pep=38% mhc1=12% | 89.3% |
| [`netmhcpan`](sources/netmhcpan.md) | 34,034,376 | 27,332 | 0.08% | 491.5M | 35 | trb=0% tra=0% pep=100% mhc1=14% | 100.0% |
| [`ots`](sources/ots.md) | 6,968,867 | 384 | 0.01% | 676.0M | 7 | trb=100% tra=100% pep=0% mhc1=0% | 107.5% |
| [`rcc_atlas`](sources/rcc_atlas.md) | 82 | 6 | 6.82% | 15.6K | 1 | trb=82% tra=9% pep=93% mhc1=91% | 100.0% |
| [`studies`](sources/studies.md) | 92,142,845 | 15,025,806 | 14.02% | 1.5G | 93 | trb=69% tra=22% pep=0% mhc1=0% | 36.0% |
| [`tadb`](sources/tadb.md) | 1,147 | 2 | 0.17% | 32.6K | 1 | trb=0% tra=0% pep=100% mhc1=44% | 100.0% |
| [`tcrdb`](sources/tcrdb.md) | 316,284,375 | 67,552,409 | 17.60% | 16.3G | 317 | trb=100% tra=0% pep=0% mhc1=0% | 121.5% |
| [`trait`](sources/trait.md) | 3,967,323 | 177 | 0.00% | 335.3M | 4 | trb=100% tra=100% pep=100% mhc1=100% | 100.0% |
| [`vdjdb`](sources/vdjdb.md) | 237,578 | 679 | 0.28% | 9.5M | 3 | trb=48% tra=39% pep=53% mhc1=45% | 64.9% |

*Rows column*: count from `manifest.json:row_count` (verified against parquet metadata). *Yield* = standardized rows / raw input rows (where the raw counterpart is unambiguous; some standardizers — e.g. vdjdb — emit >1 row per input record by design, so >100% is expected). *Coverage* shows the percent of rows where the column is non-null AND non-empty.

<!-- END: AUTO-MASTER-TABLE -->

## What a "bucket" contains

Every directory under `data/standardized_again/{name}/` has exactly three artifact types:

- `manifest.json` — the ground-truth metadata for the bucket: `row_count`, `dropped_count`, `elapsed_seconds`, `timestamp`, and `source_checksums` (md5 per input file). The standardizer writes this on completion.
- `dropped.tsv` — rows dropped during validation. Columns: `reason | source_file | row_index | field | raw_value`. Used to debug upstream data-quality issues.
- `part_XXXX.parquet` — sharded standardized output in the 25-column TARGET_COLUMNS schema.

The standardizer for each bucket is at `scripts/data_processing/standardize/{name}.py`; the `BaseStandardizer` contract is defined at `scripts/data_processing/standardize/_base.py`.

## TARGET_COLUMNS (the 25-column unified schema)

| Group | Columns |
|-------|---------|
| TCR alpha | `tra`, `trav_gene`, `trad_gene`, `traj_gene`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `tra_full` |
| TCR beta | `trb`, `trbv_gene`, `trbd_gene`, `trbj_gene`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `trb_full` |
| Peptide–MHC | `peptide`, `mhc_one`, `mhc_two`, `mhc_one_allele`, `mhc_two_allele` |
| Metadata | `binding`, `score`, `source`, `study_id` |

Authoritative definition: [`quest/data/standardization.py:18-44`](../../../quest/data/standardization.py).

## Column-population semantics differ across buckets

The standardizers don't agree on what to put into `binding` and `score` (this is a real downstream-friction point — surfaced by the Context Writer review):

- **`binding`** — populated explicitly by `mcpas`, `batman`, `trait`, `rcc_atlas`, `cedar_pmhc`, `iedb_pmhc`. Empty (implicit positive) for `vdjdb`, `cedar`, `iedb`. Downstream consumers cannot infer "is this a positive?" from `binding != ""` alone — they must use the `source` column to disambiguate.
- **`score`** — different per source: `batman` carries `peptide_activity` (0–1), `vdjdb` carries the 0–3 evidence score, `iedb_pmhc` carries IC50/Kd binding affinity, `netmhcpan` is uniformly empty despite the inputs carrying target values.

Each bucket page documents its own conventions.

## Dedup happens *next*, not here

Standardization does not deduplicate within or across buckets. Cross-bucket duplicates (e.g., `vdjdb` and `iedb` and `mcpas` all curate the same A\*02:01-restricted CMV pp65 epitopes) are removed by the Stage 2 streaming dedup at `scripts/data_processing/deduplicate_streaming.py`. The row counts in this manifest are pre-dedup.

## Regenerating

```bash
# Full refresh (re-reads every parquet's metadata, re-aggregates dropped.tsv, re-embeds)
python scripts/analysis/build_standardized_manifest.py --embed

# Single bucket (writes to alternate paths to protect the full inventory):
python scripts/analysis/build_standardized_manifest.py --bucket vdjdb \
    --out /tmp/vdjdb_check.json --csv /tmp/vdjdb_check.csv
```

The full rebuild is currently ~19 min; the long pole is `immuneaccess/dropped.tsv` (62 GB streamed line-by-line for reason aggregation) and column coverage on adc/immuneaccess (sampled to 30 shards each).

## Known caveats

The reviews flagged the following — addressed where possible, otherwise documented:

| # | Issue | Status |
|---|-------|--------|
| 1 | `--bucket NAME` partially overwrote `inventory.csv` (the `--out` guard didn't cover `--csv`) | **Fixed** — guard now requires explicit `--out` AND `--csv` when `--bucket` is set |
| 2 | Yield rate > 100% for tcrdb/immuneaccess/ots/adc (sampled-raw + standardizer row multiplication) | **Documented** — `yield_exceeds_100` flag and explanatory note now attached in JSON on next regeneration; reader should not interpret > 100% as a literal proportion |
| 3 | `binding` semantics differ across buckets (explicit vs implicit positive) | **Documented** — see the section above; downstream consumers must use `source` + `binding` together |
| 4 | `score` semantics differ across buckets (vdjdb 0–3, batman peptide_activity, iedb_pmhc IC50, others empty) | **Documented** — each per-source page declares its convention |
| 5 | Adaptive-format sources (immuneaccess/immunecode/tcrdb) drop ~17–28% of rows due to `TCRBV*`/`TCRBD*` prefix vs IMGT `TRBV*`/`TRBD*` | **Open** — flagged by Context Writer; candidate for upstream fix in the gene normalizer |
| 6 | `imgthla` source-checksum count includes `.git/` files (548 checksums) | **Acknowledged** — informational only, doesn't affect output |
| 7 | `dropped_count` from `manifest.json` not cross-checked against `dropped.tsv` row count in JSON output | **Open** — Code Review M2; today they agree on all 18 buckets |
| 8 | Trait has 175-row gap (`row_count + dropped` vs raw input by 0.004%) | **Acknowledged** — Verifier flagged; within tolerance but worth a spot-check if 1-to-1 mapping is required |

## Files in this directory

- `README.md` — this file
- `inventory.json` / `inventory.csv` — machine-readable manifest
- `CODE_REVIEW.md` — independent code review of the manifest script
- `VERIFIER_REPORT.md` — independent verification of counts and coverage
- `sources/*.md` — per-bucket detail pages for the 18 standardized buckets
