# TCRdb

## Identity
- **Source path**: `data/raw_data/databases/tcrdb/`
- **Primary citation**: Chen S-Y, Yue T, Lei Q, Guo A-Y (2021). "TCRdb: a comprehensive database for T-cell receptor sequences with powerful search function." *Nucleic Acids Research* 49(D1):D468-D474, [doi:10.1093/nar/gkaa796](https://doi.org/10.1093/nar/gkaa796) — [PubMed 32990749](https://pubmed.ncbi.nlm.nih.gov/32990749/)
- **Version / release**: ~316M rows across category-organized CSV/TSV files. QUEST snapshot organized into `cancer/`, `autoimmunity/`, `healthy/`, `inflammation/`, `viral/`, `transplantation/` subtrees plus `tcrdb_all_metadata_combined.csv`.
- **License**: free academic-use; data downloaded from <http://bioinfo.life.hust.edu.cn/TCRdb/>. Re-publication terms unspecified beyond citation.
- **Project URL**: <http://bioinfo.life.hust.edu.cn/TCRdb/>

## Data shape
- **Chains present**: predominantly TRB. The QUEST standardizer detects alpha records via either an explicit `Chain` column or a `Vregion` prefix (`scripts/data_processing/standardize/tcrdb.py:124-136`); when found, the columns are remapped to `tra`/`trav_gene`/`traj_gene`. Pairing is not provided.
- **Epitope/MHC labelling**: **none** at the row level. Per-sample disease/category metadata in `tcrdb_metadata*.csv` indicates the cohort context (e.g. "viral", "cancer").
- **Single-cell vs. bulk**: bulk TCR-seq, re-processed uniformly by the TCRdb pipeline.
- **Species filter**: human-only per upstream curation.

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/tcrdb`
- **Size**: 38.1G (40,856,795,364 bytes)
- **Files**: 9366
- **Format breakdown**: `.csv`=9360, `.json`=3, `.xlsx`=1, `.sh`=1, `.py`=1
- **Records**: ~260,304,089 (sampled) (95% CI: 24,652,130–495,953,565)
- **Counting method**: sampled (30 of 9360 csv files, mean=27810.1 records/file, extrapolated)
- **Record definition**: json record (list element or document); tab/comma-delimited row excluding header; xlsx data row excluding header
- **Standardizer**: `scripts/data_processing/standardize/tcrdb.py` (`TcrdbStandardizer`)
- **Expected input files**: `*.csv`, `*.tsv`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `<9330 more csv files (not sampled)>` | csv | 38.0G | 259,468,544 | extrapolated from sample |
| `healthy/immunoSEQ67/T1D6_TN.csv` | csv | 39.8M | 361,938 | wc -l minus 1 (header) |
| `healthy/PRJEB31057/ERR3137091.csv` | csv | 10.3M | 151,439 | wc -l minus 1 (header) |
| `cancer/immunoSEQ24/MDA-3238-C.csv` | csv | 7.0M | 63,825 | wc -l minus 1 (header) |
| `cancer/immunoSEQ92/2300114934-90.csv` | csv | 5.7M | 52,150 | wc -l minus 1 (header) |
| `healthy/immunoSEQ67/T1D7_Treg.csv` | csv | 5.7M | 51,978 | wc -l minus 1 (header) |
| `cancer/PRJNA544699/SRR9160553.csv` | csv | 5.5M | 43,574 | wc -l minus 1 (header) |
| `cancer/PRJNA642967/SRR12145147.csv` | csv | 2.9M | 22,473 | wc -l minus 1 (header) |
| `cancer/immunoSEQ92/2300553175-48.csv` | csv | 2.1M | 19,396 | wc -l minus 1 (header) |
| `cancer/immunoSEQ01/043.csv` | csv | 1.6M | 15,090 | wc -l minus 1 (header) |
| `cancer/immunoSEQ129/MD043-003_D+121_32018_13.csv` | csv | 880.0K | 7,947 | wc -l minus 1 (header) |
| `healthy/PRJEB31057/ERR3137020.csv` | csv | 661.7K | 9,565 | wc -l minus 1 (header) |

**Non-data artifacts**: `download_tcrdb_metadata.py`, `tcrdb_download.sh`

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: TCRdb re-processes raw FASTQ from public TCR-seq deposits (SRA / GEO) through a uniform pipeline: alignment, V/D/J calling, CDR3 extraction, clone aggregation. Per the 2021 paper, the database contains "more than 277 million highly reliable TCR sequences from over 8,265 TCR-seq samples across hundreds of tissues/clinical conditions/cell types."
- **Original cohort**: 8,265+ samples spanning autoimmunity (RA, MS, T1D, IBD, etc.), oncology (lung, breast, colorectal, melanoma, etc.), inflammation, healthy donors, viral infection (CMV, EBV, HIV, influenza, COVID-19), and solid-organ transplantation.
- **Curation pipeline**: pulled raw data → uniform TCRdb pipeline → categorized output. The schema is `NNSeq, AASeq, Vregion, Dregion, Jregion, cloneCount, cloneFraction` per row; metadata for each sample carries disease/condition/tissue annotation.
- **Updates / versioning history**: 2021 launch in *NAR Database Issue*. The TCRdb maintainers indicate continued updates on the project page; specific release versions are not formally stamped. QUEST snapshot reflects the post-publication accumulated state (well beyond the 277M reported in 2021).

## Caveats & known issues
- **No antigen specificity.** TCRdb is a repertoire resource; it does not annotate per-clone epitope-binding. The `category` directory (cancer/autoimmunity/etc.) gives only cohort-level context.
- **Re-processing pipeline introduces homogenization artifacts.** Different upstream studies use different chemistries (Adaptive immunoSEQ vs. Roche Capture vs. 10x V(D)J vs. RACE-PCR); re-aligning all to one pipeline can shift V/D/J-call distributions.
- **Public-vs-private TCR mixing.** Highly expanded clones from chronic-infection / cancer cohorts can dominate; uniform-row weighting partially mitigates this.
- **Chain detection is heuristic.** The standardizer first looks for a `Chain` column, then falls back to V-gene prefix matching; novel V-gene formats default to TRB. New file formats may silently mislabel chains.
- **Cohort labels are coarse.** "viral" includes everything from acute COVID-19 to chronic HIV; "cancer" mixes melanoma TILs with bulk peripheral blood from breast cancer patients.
- **Cross-source overlap with immuneACCESS, ADC, immuneCODE.** TCRdb pulls SRA-deposited data that often coexists in iReceptor and immuneACCESS. Cross-source dedup is essential.
- **Massive scale (~316M rows).** Streaming required; standardizer chunks at 1M rows per chunk (`scripts/data_processing/standardize/tcrdb.py:23`).
- **No peer-reviewed audit of post-2021 quality.** The original NAR paper described 277M rows; the current QUEST snapshot may include data added without formal release notes.

## Fidelity assessment
- **Confidence (1–5)**: **2 (Bulk/Indirect)** for TCR-pMHC purposes. As a TCR-β MLM corpus, fidelity is **3**: uniform pipeline, broad disease coverage, but heterogeneous source assays.
- **Recommended QUEST integration tier**: **bulk-repertoire** (per CLAUDE.md framework). Use for MLM and TCR-only training (predominantly TRB; small TRA subset where detected). Do NOT use for TCR-pMHC interaction training.
- **Score / quality threshold**: chain detection + standard CDR3 length / amino-acid validation in `quest/data/standardization.py`. Metadata files are read but disease-state filtering is not applied at standardize time.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/tcrdb.py`
- **Class**: `TcrdbStandardizer` (streaming = True due to ~316M rows)
<!-- TODO: filled by code-cross-reference -->
