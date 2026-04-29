# OTS (Observed TCR Space)

## Identity
- **Source path**: `data/raw_data/databases/OTS/`
- **Primary citation**: Raybould MIJ, Greenshields-Watson A, Agarwal P, Aguilar-Sanjuan B, Olsen TH, Turnbull OM, Quast NP, Deane CM (2024). "The Observed T Cell Receptor Space database enables paired-chain repertoire mining, coherence analysis, and language modeling." *Cell Reports* 43(9):114704, [doi:10.1016/j.celrep.2024.114704](https://doi.org/10.1016/j.celrep.2024.114704) — [PubMed 39216000](https://pubmed.ncbi.nlm.nih.gov/39216000/)
- **Earlier preprint**: bioRxiv 2024.05.20.594960, [doi:10.1101/2024.05.20.594960](https://doi.org/10.1101/2024.05.20.594960)
- **Version / release**: 2,166 SRR-keyed paired CSV files in QUEST snapshot (`SRR{ID}_1_Paired_All.csv`); ~7M rows total. Per the published paper, OTS contains 5.35M redundant / 1.63M non-redundant paired-chain TCR sequences across 50 studies and ≥75 individuals.
- **License**: Free academic use; downloads from Oxford OPIG website.
- **Project URL**: <https://opig.stats.ox.ac.uk/webapps/ots/>

## Data shape
- **Chains present**: paired alpha-beta only, full-length annotation. Naming `SRR{ID}_1_Paired_All.csv` implies SRA-derived single-cell sequencing runs.
- **Epitope/MHC labelling**: **none**. OTS catalogues observed TCR pairs only; no antigen-specificity metadata.
- **Single-cell vs. bulk**: single-cell paired exclusively. Pairings are derived from cell-barcoded scTCR-seq.
- **Species filter**: predominantly human; some non-human studies are included upstream. Standardizer does not impose species filtering (relies on metadata absence to drop non-paired rows).

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/OTS`
- **Size**: 60.1G (64,483,842,489 bytes)
- **Files**: 2166
- **Format breakdown**: `.csv`=2165, `.sh`=1
- **Records**: ~6,482,299 (sampled) (95% CI: 4,882,239–8,082,358)
- **Counting method**: sampled (30 of 2165 csv files, mean=2994.1 records/file, extrapolated)
- **Record definition**: paired-chain TCR sequence row
- **Standardizer**: `scripts/data_processing/standardize/ots.py` (`OtsStandardizer`)
- **Expected input files**: `*.csv`, `*_Paired_All.csv`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `<2135 more csv files (not sampled)>` | csv | 59.3G | 6,392,475 | extrapolated from sample |
| `SRR20215432_1_Paired_All.csv` | csv | 73.0M | 8,279 | wc -l minus 2 (json metadata + header) |
| `SRR18175906_1_Paired_All.csv` | csv | 64.9M | 7,362 | wc -l minus 2 (json metadata + header) |
| `SRR25409186_1_Paired_All.csv` | csv | 53.5M | 6,170 | wc -l minus 2 (json metadata + header) |
| `SRR13113816_1_Paired_All.csv` | csv | 40.2M | 4,556 | wc -l minus 2 (json metadata + header) |
| `SRR13112910_1_Paired_All.csv` | csv | 39.3M | 4,456 | wc -l minus 2 (json metadata + header) |
| `SRR26157789_1_Paired_All.csv` | csv | 37.9M | 4,287 | wc -l minus 2 (json metadata + header) |
| `SRR13112645_1_Paired_All.csv` | csv | 36.2M | 4,103 | wc -l minus 2 (json metadata + header) |
| `SRR13112882_1_Paired_All.csv` | csv | 36.1M | 4,094 | wc -l minus 2 (json metadata + header) |
| `SRR13112835_1_Paired_All.csv` | csv | 34.7M | 3,936 | wc -l minus 2 (json metadata + header) |
| `SRR13113598_1_Paired_All.csv` | csv | 33.6M | 3,802 | wc -l minus 2 (json metadata + header) |
| `SRR13112741_1_Paired_All.csv` | csv | 33.3M | 3,777 | wc -l minus 2 (json metadata + header) |

**Non-data artifacts**: `bulk_download_ots.sh`

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: 10x Genomics V(D)J kits and similar paired single-cell TCR-seq platforms. Sources include CD8/CD4-sorted samples, peripheral blood, tumor-infiltrating lymphocytes, and disease cohorts deposited to SRA.
- **Original cohort**: ≥75 individuals across 50 studies. Cohorts span healthy donors, oncology, autoimmunity, and infectious-disease patients. Each `SRR*` ID corresponds to one SRA run; donor metadata comes from the upstream BioSample.
- **Curation pipeline**: Oxford Protein Informatics Group consistently re-processes raw FASTQ from public sources through a uniform AIRR-compliant pipeline; first row of each CSV is a JSON header with study-level metadata (the QUEST standardizer skips this row at `scripts/data_processing/standardize/ots.py:60-62`); subsequent rows contain ~270 per-cell columns including paired alpha+beta CDR3, V/D/J calls, and full-length sequences.
- **Updates / versioning history**: 2024 launch (Cell Reports + bioRxiv preprint). Authors stated OTS will be updated as a central community resource. The QUEST snapshot reflects the original 2024 release.

## Caveats & known issues
- **No epitope or MHC information.** OTS is purely a paired-chain TCR repertoire — useful for MLM and dual-encoder pre-training but not for direct TCR-pMHC modeling.
- **Donor / study heterogeneity.** Sequencing chemistry (10x v1 vs v2 vs v3), cell-sort strategy, and bulk pre-amplification differ across studies. Pairing accuracy varies.
- **Pairing accuracy depends on cell barcoding quality.** OTS retains only confidently paired alpha-beta sequences but doublets and ambient mRNA contamination still produce a small fraction of erroneous pairings.
- **Public TCRs are abundant.** OTS authors observe extensive TCR overlap across donors (public TCRs); without dedup, public TCRs will dominate training and reduce diversity.
- **Disease-cohort skew.** Many SRA-deposited paired TCR runs come from oncology and COVID-19 cohorts; healthy-donor representation is modest.
- **JSON metadata header.** Each CSV file has a JSON metadata first row that must be skipped (`skiprows=1`); failure to do so misaligns columns. The standardizer handles this.
- **Cross-source overlap with VDJdb 10x records.** Some SRA-deposited 10x runs are also re-curated into VDJdb with epitope labels; cross-source dedup must compare CDR3+V+J.

## Fidelity assessment
- **Confidence (1–5)**: **2 (Bulk/Indirect — repertoire only)** for TCR-pMHC purposes. As a paired-chain MLM corpus, fidelity is **4**: peer-reviewed, uniformly processed, single-cell paired data.
- **Recommended QUEST integration tier**: **bulk-repertoire** (per CLAUDE.md framework). Use for MLM and TCR-only training (tra, trb, tra+trb). Do NOT use as evidence of TCR-pMHC interaction.
- **Score / quality threshold**: standardizer skips the JSON metadata first line and treats all paired rows as MLM-quality (`scripts/data_processing/standardize/ots.py:60-67`). No additional filtering.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/ots.py`
- **Class**: `OtsStandardizer` (streaming = True due to ~7M rows across 2,166 files)
<!-- TODO: filled by code-cross-reference -->
