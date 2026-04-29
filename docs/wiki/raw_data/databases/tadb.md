# TaDB / TANTIGEN

## Identity
- **Source path**: `data/raw_data/databases/TaDB/`
- **Primary citation**: Zhang G, Chitkushev L, Olsen LR, Keskin DB, Brusic V (2021). "TANTIGEN 2.0: a knowledge base of tumor T cell antigens and epitopes." *BMC Bioinformatics* 22(Suppl 8):40, [doi:10.1186/s12859-021-03962-7](https://doi.org/10.1186/s12859-021-03962-7) — [PubMed 33849445](https://pubmed.ncbi.nlm.nih.gov/33849445/)
- **Earlier version**: TANTIGEN 1.0 (Olsen et al., 2017) — a precursor cataloging ~1,000 T-cell epitopes from 292 tumor antigens.
- **Version / release**: `tadb_t_cell_epitopes.csv` — ~1,100 rows of `ACCESSION, Epitope sequence, HLA allele, Epitope type`. Per the 2021 publication, TANTIGEN 2.0 contains 4,296 antigen variants from 403 unique tumor antigens with >1,500 T-cell epitopes and HLA ligands; the QUEST snapshot is a TCR-relevant subset.
- **License**: free for academic / research use; <http://projects.met-hilab.org/tadb> (mirror at Boston University Metropolitan College).
- **Project URL**: <http://projects.met-hilab.org/tadb> (the QUEST file uses the alternate filename convention `tadb_t_cell_epitopes.csv`)

## Data shape
- **Chains present**: **none** — no TCR sequences. This is a peptide-MHC / tumor-antigen catalog.
- **Epitope/MHC labelling**: each row has an epitope amino-acid sequence, an HLA allele, and an epitope-type tag (e.g. "Overexpressed antigen", "Cancer-testis antigen", "Mutated antigen", "Differentiation antigen", "Viral antigen-derived").
- **Single-cell vs. bulk**: not applicable — the source data is biochemical / bioinformatic catalog of T-cell-relevant tumor antigens.
- **Species filter**: human-only by upstream design.

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/TaDB`
- **Size**: 54.2K (55,535 bytes)
- **Files**: 1
- **Format breakdown**: `.csv`=1
- **Records**: 1,147 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header
- **Standardizer**: `scripts/data_processing/standardize/tadb.py` (`TadbStandardizer`)
- **Expected input files**: `*.csv`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tadb_t_cell_epitopes.csv` | csv | 54.2K | 1,147 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: TANTIGEN aggregates published T-cell epitopes from tumor antigens. Sources include: (1) published immunological assays (tetramer, ELISpot, ICS) from the literature; (2) MHC-binding measurements from IEDB; (3) algorithmically predicted neoepitopes confirmed in functional assays. Some entries carry validated TCR sequences; the QUEST extract is the **epitope side only** (no TCR columns in `tadb_t_cell_epitopes.csv`).
- **Original cohort**: not directly applicable — TANTIGEN is curated from the literature rather than being tied to a specific patient cohort. Tumor antigen categories include cancer-testis (e.g. NY-ESO-1, MAGE-A family), differentiation (e.g. MART-1 / Melan-A, gp100, tyrosinase), overexpressed (e.g. WT1, PRAME, survivin), mutated (KRAS hotspots, TP53 hotspots), and viral-tumor-related (HPV E6/E7, EBV LMP-1).
- **Curation pipeline**: Brusic / Zhang lab manually curates published tumor immunology papers, validates entries against IEDB and HLA-binding predictions, and exposes data through a Boston-University-hosted web portal. Tools include T-cell epitope prediction analytics integrated with the catalog.
- **Updates / versioning history**: TANTIGEN 1.0 (2017) → TANTIGEN 2.0 (2021). Subsequent updates not formally versioned.

## Caveats & known issues
- **No TCR data.** TaDB is **peptide-MHC only** in the QUEST extract; do NOT treat it as a TCR-specificity source.
- **Mixed evidence levels.** Some epitopes are experimentally validated tetramer hits; others are predicted-then-confirmed via ELISpot; a few entries are predicted-only neoepitopes. The CSV does not preserve the validation method per row in the QUEST extract — only `Epitope type` and HLA allele are kept.
- **Heavy class-I bias.** Cancer immunology is class-I-dominated (CD8 T cells against tumor); class-II coverage is minimal.
- **HLA-A*02 dominance.** Tumor-immunology literature heavily focuses on HLA-A*02:01 due to widespread availability of A*02:01 reagents; this introduces an HLA bias into any predictor trained on TaDB.
- **Cross-source overlap with IEDB.** TANTIGEN 2.0 itself cites IEDB-derived MHC-binding records; entries also appear in CEDAR. Cross-source dedup must compare on (peptide, HLA).
- **No binding-affinity values.** The QUEST extract carries only categorical epitope-type tags, not numeric IC50 or %Rank values.
- **Standardizer drops scores.** The QUEST standardizer maps only `Epitope sequence`, `HLA allele` → `peptide`, `mhc_one`/`mhc_two` (`scripts/data_processing/standardize/tadb.py:28-32`); all other metadata (epitope type, ACCESSION) is dropped at standardize time.

## Fidelity assessment
- **Confidence (1–5)**: **3 (Medium)** — peer-reviewed catalog, manual curation by an experienced lab, but mixed evidence levels (tetramer-validated to predicted-only) collapsed into one schema; small (~1.1K rows in QUEST snapshot).
- **Recommended QUEST integration tier**: **pMHC-only** (per CLAUDE.md framework). Use for peptide-MHC modeling (pMHC seq2seq, contrastive pMHC learning) only; do NOT use for TCR interaction training.
- **Score / quality threshold**: no explicit threshold. `Epitope sequence` and `HLA allele` columns are taken as-is. {note for Verifier: confirm whether standardizer should apply a minimum-validation filter against `Epitope type`}

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/tadb.py`
- **Class**: `TadbStandardizer`
<!-- TODO: filled by code-cross-reference -->
