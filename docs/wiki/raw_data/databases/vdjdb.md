# VDJdb

## Identity
- **Source path**: `data/raw_data/databases/vdjdb/`
- **Primary citation**: Shugay M, Bagaev DV, Zvyagin IV, et al. (2018). "VDJdb: a curated database of T-cell receptor sequences with known antigen specificity." *Nucleic Acids Research* 46(D1):D419-D427, [doi:10.1093/nar/gkx760](https://doi.org/10.1093/nar/gkx760) — [PubMed 28977646](https://pubmed.ncbi.nlm.nih.gov/28977646/)
- **2019 update**: Bagaev DV, Vroomans RMA, Samir J, et al. (2020). "VDJdb in 2019: database extension, new analysis infrastructure and a T-cell receptor motif compendium." *Nucleic Acids Research* 48(D1):D1057-D1062, [doi:10.1093/nar/gkz874](https://doi.org/10.1093/nar/gkz874) — [PubMed 31588507](https://pubmed.ncbi.nlm.nih.gov/31588507/)
- **Version / release**: `2025-12-29` (per `latest-version.txt`); downloaded `2026-01-29`
- **License**: Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0) — see `LICENSE.txt`
- **Project URL**: <https://vdjdb.cdr3.net/> ; source repo: <https://github.com/antigenomics/vdjdb-db>

## Data shape
- **Chains present**: TRA, TRB, paired alpha/beta, with explicit V/D/J gene calls. Wide format (`vdjdb_full.txt`) keeps an alpha row and beta row of the same clonotype on a single record.
- **Epitope/MHC labelling**: present and required. Every record carries `antigen.epitope`, `mhc.a` (class-I or class-II alpha), `mhc.b` (class-II beta), and `mhc.class`.
- **Single-cell vs. bulk**: heterogeneous. Records originate from tetramer/dextramer sorting, single-cell paired sequencing, MIRA, and yeast display, depending on the underlying study.
- **Species filter**: standardizer keeps only `species == "HomoSapiens"` (`scripts/data_processing/standardize/vdjdb.py:80-83`); upstream DB also includes mouse and macaque.

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/vdjdb`
- **Size**: 802.0M (840,999,437 bytes)
- **Files**: 19
- **Format breakdown**: `.txt`=16, `.html`=1, `.zip`=1, `.md`=1
- **Records**: 366,238 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: sum of records in archived csv/tsv/fasta members; tab-delimited row excluding header
- **Standardizer**: `scripts/data_processing/standardize/vdjdb.py` (`VdjdbStandardizer`)
- **Expected input files**: `vdjdb_full.txt`, `vdjdb.txt`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `vdjdb.scored.txt` | txt | 187.8M | — | wc -l minus 1 (header) |
| `vdjdb.txt` | txt | 187.3M | 226,494 | wc -l minus 1 (header) |
| `vdjdb_full_scored.txt` | txt | 91.6M | — | wc -l minus 1 (header) |
| `vdjdb_full.txt` | txt | 91.3M | 139,744 | wc -l minus 1 (header) |
| `vdjdb_full_filtered.txt` | txt | 90.6M | — | wc -l minus 1 (header) |
| `vdjdb-2025-12-29.zip` | zip | 69.4M | — | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `vdjdb.slim.scored.txt` | txt | 33.4M | — | wc -l minus 1 (header) |
| `vdjdb.slim.txt` | txt | 33.1M | — | wc -l minus 1 (header) |
| `cluster_members.txt` | txt | 7.3M | — | non-tabular metadata txt |
| `motif_pwms.txt` | txt | 4.9M | — | non-tabular metadata txt |
| `vdjdb_summary_embed.html` | html | 4.5M | — | unsupported format: html |
| `vdjdb_full_cdr3aa_broken.txt` | txt | 644.0K | — | wc -l minus 1 (header) |

**Non-data artifacts**: `LICENSE.txt`, `README.md`

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: VDJdb is an **aggregator** of upstream studies; methods reflect the source publication for each row. Most rows come from peptide-MHC tetramer / dextramer staining followed by single-cell or bulk TCR sequencing; recent additions include 10x Genomics paired single-cell data and yeast display screens.
- **Original cohort**: thousands of donors across hundreds of studies — VDJdb 2019 alone reported a 5x record increase over 2018, primarily driven by EBV, CMV, Influenza-A, HIV-1, SARS-CoV-2, and self-antigen tetramer studies plus 10x Genomics public CD8+ single-cell panels.
- **Curation pipeline**: Manual curation of published studies into a Git-versioned schema (<https://github.com/antigenomics/vdjdb-db>). Each record carries a `vdjdb.score` (0-3 confidence flag), `meta.study.id`, and provenance fields. Confidence accounts for: (i) frequency of clone in the antigen-specific sample, (ii) reliability of TCR sequencing method (Sanger vs HTS vs single-cell), and (iii) presence of additional validation (e.g. functional assays).
- **Updates / versioning history**: 2017 launch (v0.x); 2019 5x extension and motif compendium ([doi:10.1093/nar/gkz874](https://doi.org/10.1093/nar/gkz874)); 2020-2024 substantial COVID-19 additions; 2023 structural-modeling adjunct ([Shcherbinin et al. 2023, doi:10.3389/fimmu.2023.1224969](https://doi.org/10.3389/fimmu.2023.1224969), [PubMed 37649481](https://pubmed.ncbi.nlm.nih.gov/37649481/)). Database releases are continuous on GitHub master.

## Caveats & known issues
- **Severe epitope imbalance.** Roughly 70% of TCR-epitope pairs concentrate around ~100 epitopes; HLA-A*02:01-restricted records dominate. Public benchmarks consistently report that prediction performance collapses on held-out epitopes/alleles ([IMMREP22 workshop report](https://www.sciencedirect.com/science/article/pii/S2667119023000046); [arXiv 2312.16594](https://arxiv.org/html/2312.16594v1)).
- **Score 0 records are unreliable.** Score-0 rows lack at least one of (sequencing reliability, functional validation, or clonal frequency evidence) and are widely excluded from downstream training. The QUEST standardizer splits score-0 into a TCR-only branch (MLM-only) and a pMHC-only branch (no joint TCR-pMHC training).
- **Class-II is sparse and noisy.** Class-II records (mhc.b populated) are a minority and skew toward DRB1*15:01/DRB1*04:01.
- **Cross-database overlap.** VDJdb aggregates many records that also appear in IEDB and McPAS-TCR. Cross-source dedup in QUEST's pipeline is required to avoid amplifying these duplicates.
- **Mouse and macaque rows must be filtered.** Species filter is applied in standardizer; pre-2019 versions did not consistently flag this.
- **Method-specific bias.** Tetramer-derived records overrepresent high-avidity TCRs; MIRA-derived records (e.g. ImmuneCODE re-imports) are CD8-biased. See [TCRMatch (Chronister et al. 2021, doi:10.3389/fimmu.2021.640725)](https://doi.org/10.3389/fimmu.2021.640725) for a discussion of imbalance when using IEDB/VDJdb-derived datasets to train similarity-based predictors.

### Note on the inventory record count

The vdjdb release ships **nine alternate dumps of the same underlying data** (`vdjdb.txt`, `vdjdb.scored.txt`, `vdjdb.slim.txt`, `vdjdb.slim.scored.txt`, `vdjdb_full.txt`, `vdjdb_full_scored.txt`, `vdjdb_full_filtered.txt`, plus a versioned `.zip` that bundles all of them). The auto-inventory above only counts the **canonical files**: `vdjdb.txt` (~226K rows — superset of all entries, paired and unpaired) and `vdjdb_full.txt` (~140K rows — the paired clonotype view). The other seven files are flagged as redundant alternate dumps. If you sum the inventory's primary-data file rows naïvely, you will see a number ~10× the true unique-record count — the dedup pass corrects this in `total_records`. The downstream QUEST standardizer reads `vdjdb_full.txt` (paired view).

## Fidelity assessment
- **Confidence (1–5)**: **4 (High)** — peer-reviewed, manually curated, with an explicit per-record confidence score; experimentally validated TCR-pMHC interactions when filtered to score >= 1, but heavy epitope/HLA imbalance prevents a Gold rating.
- **Recommended QUEST integration tier**: **quality-scored** (per CLAUDE.md framework). High-score (>=1) → expand into individual chains AND all interaction permutations for both MLM and contrastive/cross-encoder/seq2seq training. Score-0 → individual chains only (MLM) and pMHC-only (peptide-MHC training only).
- **Score / quality threshold**: `vdjdb.score >= 1` triggers full TCR-pMHC integration; score 0 is split into TCR-only + pMHC-only branches in `scripts/data_processing/standardize/vdjdb.py:99-137`.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/vdjdb.py`
- **Class**: `VdjdbStandardizer`
<!-- TODO: filled by code-cross-reference -->
