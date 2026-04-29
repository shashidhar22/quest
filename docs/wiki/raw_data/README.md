# QUEST Raw Data Manifest

This is the **single source of truth** for what raw immunology data lives under `data/raw_data/` in this repository, where each source came from, how it was generated, and how much we trust it. The manifest is read by humans (orientation, audits) and by automation (counts via `inventory.json`).

The intent of this wiki: a new collaborator should be able to learn within ~15 minutes (a) what's on disk, (b) which sources are appropriate for which training task, and (c) where to read more.

## How to read this manifest

- **The master summary table below** is the at-a-glance overview. It is rendered from [`inventory.json`](inventory.json) and refreshed by re-running `python scripts/analysis/build_raw_data_manifest.py --embed`.
- **Per-source detail pages** under [`databases/`](databases/) and [`studies/`](studies/) carry the lit-review content (citations, methodology, caveats) plus an auto-generated inventory block.
- **Fidelity scores (1–5)** follow the [Fidelity Rubric](FIDELITY_RUBRIC.md). Fidelity reflects evidence quality for *TCR–pMHC interaction supervision*; it is not a general "data goodness" score.
- **Reviews**: this manifest was audited at creation by an [independent code review](CODE_REVIEW.md) of the inventory script and an [independent verifier](VERIFIER_REPORT.md) cross-checking counts and citations. Both passed with caveats; remaining caveats are documented in this file.

## Master summary table

<!-- BEGIN: AUTO-MASTER-TABLE (build_raw_data_manifest.py) -->

**Last regenerated**: 2026-04-27T01:22:07Z · **Sources**: 82 · **Total bytes**: 3.1T · **Total records (sum across sources, deduped)**: 5,366,160,737

### Databases

| Source | Kind | Records | Size | Files | Fidelity | Standardizer |
|--------|------|---------|------|-------|----------|--------------|
| [`BATMAN`](databases/batman.md) | database | 22,827 | 1.6M | 3 | 5 | [`batman.py`](../../scripts/data_processing/standardize/batman.py) |
| [`CEDAR`](databases/cedar.md) | database | 6,481,168 | 8.1G | 12 | 3 | [`cedar.py`](../../scripts/data_processing/standardize/cedar.py); [`cedar_pmhc.py`](../../scripts/data_processing/standardize/cedar_pmhc.py) |
| [`IEDB`](databases/iedb.md) | database | 3,168,890 | 2.8G | 12 | 3 | [`iedb.py`](../../scripts/data_processing/standardize/iedb.py); [`iedb_pmhc.py`](../../scripts/data_processing/standardize/iedb_pmhc.py) |
| [`IMGTHLA`](databases/imgthla.md) | database | 7,500,331 | 4.1G | 549 | 5 | [`imgthla.py`](../../scripts/data_processing/standardize/imgthla.py) |
| [`McPAS-TCR`](databases/mcpas.md) | database | 40,779 | 8.2M | 2 | 3 | [`mcpas.py`](../../scripts/data_processing/standardize/mcpas.py) |
| [`NetMHCPan`](databases/netmhcpan.md) | database | 34,040,203 | 1.7G | 36 | 1 | [`netmhcpan.py`](../../scripts/data_processing/standardize/netmhcpan.py) |
| [`OTS`](databases/ots.md) | database | ~6,482,299 (CI 4,882,239–8,082,358) | 60.1G | 2166 | 2 | [`ots.py`](../../scripts/data_processing/standardize/ots.py) |
| [`RCC_ATLAS`](databases/rcc_atlas.md) | database | 82 | 10.3K | 1 | 4 | [`rcc_atlas.py`](../../scripts/data_processing/standardize/rcc_atlas.py) |
| [`TaDB`](databases/tadb.md) | database | 1,147 | 54.2K | 1 | 3 | [`tadb.py`](../../scripts/data_processing/standardize/tadb.py) |
| [`adc`](databases/adc.md) | database | ~2,055,697,412 (CI 185,162,871–3,926,231,882) | 1.5T | 9296 | 2 | [`adc.py`](../../scripts/data_processing/standardize/adc.py) |
| [`immuneACCESS`](databases/immuneaccess.md) | database | ~2,437,152,912 (CI 1,046,309,348–3,827,995,102) | 1.2T | 29233 | 2 | [`immuneaccess.py`](../../scripts/data_processing/standardize/immuneaccess.py) |
| [`immuneCODE`](databases/immunecode.md) | database | ~294,756,159 (CI 208,497,358–380,358,513) | 228.4G | 1430 | — | [`immunecode.py`](../../scripts/data_processing/standardize/immunecode.py) |
| [`tcrdb`](databases/tcrdb.md) | database | ~260,304,089 (CI 24,652,130–495,953,565) | 38.1G | 9366 | 2 | [`tcrdb.py`](../../scripts/data_processing/standardize/tcrdb.py) |
| [`trait`](databases/trait.md) | database | 3,967,325 | 65.1M | 36 | 3 | [`trait.py`](../../scripts/data_processing/standardize/trait.py) |
| [`vdjdb`](databases/vdjdb.md) | database | 366,238 | 802.0M | 19 | 4 | [`vdjdb.py`](../../scripts/data_processing/standardize/vdjdb.py) |

### Studies

| Source | Kind | Records | Size | Files | Fidelity | Standardizer |
|--------|------|---------|------|-------|----------|--------------|
| [`GSE101660`](studies/GSE101660.md) | study | 12,438,482 | 2.6G | 43 | 3 | — |
| [`GSE108694`](studies/GSE108694.md) | study | 28,593 | 2.2M | 1 | 1 | — |
| [`GSE110684`](studies/GSE110684.md) | study | 0 | 1.1G | 10 | 2 | — |
| [`GSE111360`](studies/GSE111360.md) | study | 5,467,343 | 1.3G | 84 | 4 | — |
| [`GSE113194`](studies/GSE113194.md) | study | 301,212 | 5.1G | 12 | 1 | — |
| [`GSE114135`](studies/GSE114135.md) | study | 18,865 | 21.4M | 1 | 1 | — |
| [`GSE114724`](studies/GSE114724.md) | study | 102,582 | 282.5M | 23 | 4 | — |
| [`GSE121810`](studies/GSE121810.md) | study | 37,036 | 5.3M | 1 | 1 | — |
| [`GSE123813`](studies/GSE123813.md) | study | 110,656 | 58.0M | 5 | 4 | — |
| [`GSE126310`](studies/GSE126310.md) | study | 24,889 | 128.8M | 25 | 3 | — |
| [`GSE134520`](studies/GSE134520.md) | study | 297,843 | 2.4G | 13 | 1 | — |
| [`GSE145370`](studies/GSE145370.md) | study | 129,034 | 553.5M | 29 | 4 | — |
| [`GSE145926`](studies/GSE145926.md) | study | 14,602 | 227.2M | 22 | 4 | — |
| [`GSE151928`](studies/GSE151928.md) | study | 0 | 195.3M | 11 | 1 | — |
| [`GSE158036`](studies/GSE158036.md) | study | 4,740 | 6.1M | 4 | 3 | — |
| [`GSE162500`](studies/GSE162500.md) | study | 7,108,065 | 594.0M | 30 | 4 | — |
| [`GSE165080`](studies/GSE165080.md) | study | 0 | 282.1M | 5 | 1 | — |
| [`GSE166489`](studies/GSE166489.md) | study | 253,095 | 864.6M | 54 | 4 | — |
| [`GSE167118`](studies/GSE167118.md) | study | 425,373 | 1.0G | 128 | 4 | — |
| [`GSE169086`](studies/GSE169086.md) | study | 18,727 | 46.8M | 11 | 2 | — |
| [`GSE169441`](studies/GSE169441.md) | study | 202,613 | 37.1M | 54 | 3 | — |
| [`GSE173351`](studies/GSE173351.md) | study | 1,975,344 | 4.3G | 325 | 4 | — |
| [`GSE182109`](studies/GSE182109.md) | study | 0 | 2.3G | 132 | 1 | — |
| [`GSE182861`](studies/GSE182861.md) | study | 46,854 | 441.7M | 21 | 4 | — |
| [`GSE184320`](studies/GSE184320.md) | study | 224,634 | 1.5G | 51 | 4 | — |
| [`GSE185386`](studies/GSE185386.md) | study | 58,836 | 477.1M | 54 | 4 | — |
| [`GSE186344`](studies/GSE186344.md) | study | 80,377 | 3.0G | 94 | 1 | — |
| [`GSE188737`](studies/GSE188737.md) | study | 53,459 | 1.4G | 2 | 1 | — |
| [`GSE193745`](studies/GSE193745.md) | study | 161,219 | 390.5M | 4 | 1 | — |
| [`GSE194189`](studies/GSE194189.md) | study | 520,909 | 3.7G | 116 | 4 | — |
| [`GSE195486`](studies/GSE195486.md) | study | 290,875 | 1.3G | 97 | 4 | — |
| [`GSE200996`](studies/GSE200996.md) | study | 8,983,507 | 3.6G | 215 | 4 | — |
| [`GSE201425`](studies/GSE201425.md) | study | 86,156 | 486.5M | 65 | 4 | — |
| [`GSE205393`](studies/GSE205393.md) | study | 11,800 | 3.4G | 2 | 4 | — |
| [`GSE206325`](studies/GSE206325.md) | study | 305,131 | 69.1M | 1 | 4 | — |
| [`GSE208262`](studies/GSE208262.md) | study | 482,093 | 91.3M | 39 | 3 | — |
| [`GSE216005`](studies/GSE216005.md) | study | 0 | 807.0M | 66 | 1 | — |
| [`GSE216914`](studies/GSE216914.md) | study | 60,707 | 212.7M | 20 | 3 | — |
| [`GSE218743`](studies/GSE218743.md) | study | 34,194 | 551.4M | 2 | 1 | — |
| [`GSE222011`](studies/GSE222011.md) | study | 90,107 | 169.0M | 10 | 5 | — |
| [`GSE224028`](studies/GSE224028.md) | study | 2,839 | 11.8M | 6 | 3 | — |
| [`GSE231590`](studies/GSE231590.md) | study | 310,675 | 975.8M | 49 | 4 | — |
| [`GSE232447`](studies/GSE232447.md) | study | 152,856 | 663.5M | 8 | 5 | — |
| [`GSE233149`](studies/GSE233149.md) | study | 3,475,743 | 111.1M | 50 | 1 | — |
| [`GSE233869`](studies/GSE233869.md) | study | 61,057 | 89.1M | 19 | 3 | — |
| [`GSE234129`](studies/GSE234129.md) | study | 19,488 | 388.2M | 2 | 1 | — |
| [`GSE235080`](studies/GSE235080.md) | study | 19,156 | 68.6M | 6 | 4 | — |
| [`GSE79338`](studies/GSE79338.md) | study | 4,395,689 | 679.1M | 31 | 3 | — |
| [`GSE93777`](studies/GSE93777.md) | study | 0 | 4.8G | 450 | 1 | — |
| [`GSE94968`](studies/GSE94968.md) | study | 20,764 | 1.6M | 13 | 2 | — |
| [`GSM6911623`](studies/GSM6911623.md) | study | 10,594 | 6.2M | 1 | 4 | — |
| [`ZEN14010377`](studies/ZEN14010377.md) | study | 166,635,658 | 18.5G | 1768 | 5 | — |
| [`ZEN7555405`](studies/ZEN7555405.md) | study | 20,129 | 3.8M | 5 | 5 | — |
| [`ZEN8140861`](studies/ZEN8140861.md) | study | 9,104,969 | 2.3G | 282 | 5 | — |
| [`acsr_arks_two`](studies/acsr_arks_two.md) | study | 2,679,842 | 738.8M | 137 | 2 | — |
| [`acsr_arks`](studies/acsr_arks.md) | study | 389 | 118.3K | 14 | 2 | — |
| [`acsr_southafrica`](studies/acsr_southafrica.md) | study | 346,681 | 96.4M | 60 | 2 | — |
| [`acsr_texas`](studies/acsr_texas.md) | study | 421,643 | 116.6M | 34 | 2 | — |
| [`acsr`](studies/acsr.md) | study | 158,609 | 43.8M | 10 | 2 | — |
| [`aku_breast_pilot`](studies/aku_breast_pilot.md) | study | 5,288 | 1.5M | 12 | 2 | — |
| [`bgi_pilot`](studies/bgi_pilot.md) | study | 0 | 3.3G | 1 | 1 | — |
| [`bl_uganda`](studies/bl_uganda.md) | study | 803,616 | 222.1M | 52 | 2 | — |
| [`cfar`](studies/cfar.md) | study | 8,412,417 | 2.3G | 240 | 2 | — |
| [`geo`](studies/geo.md) | study | 21 | 2.1G | 97 | 3 | — |
| [`kobs`](studies/kobs.md) | study | 2,219,979 | 2.4G | 32 | 2 | — |
| [`kstme`](studies/kstme.md) | study | 5,099,803 | 5.0G | 849 | 3 | — |
| [`lung_va`](studies/lung_va.md) | study | 11,351,019 | 1.8G | 320 | 3 | — |

*Records column*: exact counts for fully-counted sources; `~N (CI lo–hi)` for sampled sources. `—` means the source has no row-shaped data or counting was not possible. *Fidelity* uses the [1–5 rubric](FIDELITY_RUBRIC.md). *Standardizer* links to the Python module that maps this source to the 25-column TARGET_COLUMNS schema; sources without a standardizer are not (yet) ingested into training datasets.

<!-- END: AUTO-MASTER-TABLE -->

## How the data flows downstream

```
data/raw_data/databases/{name}/                                                  data/raw_data/studies/{name}/
data/raw_data/studies/{name}/                                                              │
        │                                                                                  │
        │ scripts/data_processing/standardize/{name}.py                                     │
        ▼                                                                                  ▼
quest/data/standardization.py: 25-column TARGET_COLUMNS (TCR α/β, peptide, MHC-I/II, source/study_id, score, binding)
        │
        │ scripts/data_processing/deduplicate_streaming.py (3-stage: exact hash → CDR3 near-dedup → cross-source)
        ▼
docs/BENCHMARK_*.md  (foundation MLM splits, clustering, train/val/test splits, tokenized parquet)
        │
        ▼
quest-train / scripts/training/*  (ESM2 native MLM, TCR seq2seq, contrastive, cross-encoder, pMHC seq2seq)
```

The 5-stage curation pipeline (Standardize → Deduplicate → Validate → Splits → Tokenize) is described in [CLAUDE.md](../../../CLAUDE.md) under *Data Curation Pipeline Overview*. Each per-source page in this manifest links to the corresponding standardizer script under `scripts/data_processing/standardize/`.

## Conventions used in this manifest

- **Records vs. rows.** Per-source "Records" counts use the most natural unit for each format: rows for CSV/TSV, FASTA entries (`^>`) for sequence files, JSON list elements, parquet rows, etc. Each per-source page declares its record definition explicitly.
- **Dedup.** When a source ships an archive (`.zip`, `.tar.gz`) **plus** the unpacked files on disk, the inventory script counts the unpacked side only. When it ships multiple alternate dumps of the same data (vdjdb), only canonical files contribute to the total. See [CODE_REVIEW.md](CODE_REVIEW.md) for the full rule list.
- **Sampling.** For directory-of-many-files sources >50 GB (adc, immuneACCESS, immuneCODE, OTS, tcrdb), totals are extrapolated from a 30-file random sample (seed=42). The 95% confidence interval is reported alongside the point estimate. Re-run with `--exact` to count every file (slow).
- **Doc artifacts (READMEs, LICENSE files, download_*.sh scripts) are listed under "Non-data artifacts"** in each per-source page and excluded from record totals.

## Regenerating

```bash
# Full refresh: re-counts every source, re-embeds inventory blocks into wiki pages
python scripts/analysis/build_raw_data_manifest.py --embed

# Single-source refresh (writes to alternate paths to protect the full inventory):
python scripts/analysis/build_raw_data_manifest.py --source vdjdb --exact \
    --out /tmp/vdjdb_check.json --csv /tmp/vdjdb_check.csv

# Skip sampling (count every file in the huge sources too — slow, but exact):
python scripts/analysis/build_raw_data_manifest.py --exact --embed
```

The output `inventory.json` is checked into git as the canonical source. The wiki pages are partially auto-generated: the `<!-- BEGIN: AUTO-INVENTORY ... -->` block in each per-source page is replaced on `--embed`; everything outside that block is hand-curated lit review and is preserved across re-runs.

## Known caveats

The reviews flagged the following — addressed where possible, otherwise documented:

| # | Issue | Status |
|---|-------|--------|
| 1 | NetMHCpan headerless `c*_ba`/`c*_el` files not counted | **Fixed** — `noext`-format files in `NetMHCpan_train/` are now counted via `wc -l` |
| 2 | Archives + unpacked siblings double-counted (NetMHCpan, vdjdb, CEDAR, trait, IMGTHLA) | **Fixed** — dedup pass suppresses redundant archive contributions |
| 3 | `--source NAME` mode silently overwrote `inventory.json` | **Fixed** — refuses to overwrite default outputs without explicit `--out` |
| 4 | TRAIT primary citation had hallucinated authors | **Fixed** — corrected to Wei et al. 2025 (PMID 40257421) |
| 5 | Sampling extrapolates over `>2 GB` files excluded from sample (latent bias) | **Documented** — see Code Review §M4; CIs reported but bias direction not auto-corrected |
| 6 | CRC-corrupted IEDB zips silently swallowed | **Documented** — Code Review §M1; loose CSVs are intact, so headline counts are correct by accident |
| 7 | mcpas record count (~40K) vs. paper's "~5K sequences" | **Footnoted** — standardizer explodes `/`-separated peptides into multiple rows; both counts are correct for their interpretation |
| 8 | rcc_atlas has no peer-reviewed citation (in-house collection) | **Acknowledged** — `{citation needed}` flag retained; per-row references in the CSV's `reference` column provide row-level provenance |
| 9 | tadb has no `Epitope type` validation filter in the standardizer | **Open** — flagged for the Standardization Specialist; not a manifest issue |

## Files in this directory

- `README.md` — this file
- `FIDELITY_RUBRIC.md` — definition of the 1–5 fidelity scoring scale
- `inventory.json` — machine-readable manifest (full per-source detail)
- `inventory.csv` — flattened per-source summary
- `CODE_REVIEW.md` — independent code review of the manifest script
- `VERIFIER_REPORT.md` — independent verification of counts and citations
- `databases/*.md` — per-source detail pages for the 15 databases
- `studies/README.md` + `studies/*.md` — per-study pages for the GEO/Zenodo/cohort entries

## Phase 2 (deferred)

This manifest covers raw data only. A separate wiki for the **training protocol** (which model architectures get trained on which subsets, with which hyperparameters and infra) will be added in a follow-up pass alongside the existing [BENCHMARK_*.md](../../) docs.
