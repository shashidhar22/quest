# RCC_ATLAS

## Identity
- **Source path**: `data/raw_data/databases/RCC_ATLAS/`
- **Primary citation**: in-house literature-curated collection. The dataset is a small, manually compiled CSV (`rcc_tcells.csv`, ~81 rows) drawn from published renal-cell-carcinoma T-cell studies and viral-control reference TCRs. There is no single canonical publication for the file itself; per-row references are stored in the `reference` column. {citation needed for an in-house manuscript that describes this collection}
- **Major source publications referenced inline (typical contents)**:
  - Krishna C, DiNatale RG, Kuo F, et al. (2021). "Single-cell sequencing links multiregional immune landscapes and tissue-resident T cells in ccRCC to tumor topology and therapy efficacy." *Cancer Cell* 39(5):662-677.e6, [doi:10.1016/j.ccell.2021.03.007](https://doi.org/10.1016/j.ccell.2021.03.007) — [PubMed 33861994](https://pubmed.ncbi.nlm.nih.gov/33861994/)
  - Chevrier S, Levine JH, Zanotelli VRT, et al. (2017). "An Immune Atlas of Clear Cell Renal Cell Carcinoma." *Cell* 169(4):736-749.e18, [doi:10.1016/j.cell.2017.04.016](https://doi.org/10.1016/j.cell.2017.04.016) — [PubMed 28475899](https://pubmed.ncbi.nlm.nih.gov/28475899/)
  - VDJdb / IEDB consensus references for viral controls (CMV pp65, EBV BMLF1, Influenza M1, SARS-CoV-2)
- **Version / release**: single CSV; QUEST snapshot dated per pull. ~81 rows.
- **License**: in-house — the file is hand-curated by QUEST contributors; per-row citations preserved.
- **Project URL**: not applicable (in-house). Underlying primary studies have their own DOIs.

## Data shape
- **Chains present**: TRA and TRB CDR3 sequences (paired where available). Columns: `antigen, category, epitope, MHC, CDR3A, CDR3B, clone_id, TRAV, TRBV, source, reference, note`.
- **Epitope/MHC labelling**: every row carries `epitope` and `MHC`; covers viral antigens (CMV pp65, EBV, Influenza, SARS-CoV-2) and tumor-associated antigens (NY-ESO-1, WT1, PRAME, MART-1, etc.).
- **Single-cell vs. bulk**: mixed — single-cell for ccRCC TIL studies (Krishna 2021), bulk/literature-curated for viral controls.
- **Species filter**: human-only by curation.

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/RCC_ATLAS`
- **Size**: 10.3K (10,542 bytes)
- **Files**: 1
- **Format breakdown**: `.csv`=1
- **Records**: 82 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header
- **Standardizer**: `scripts/data_processing/standardize/rcc_atlas.py` (`RccAtlasStandardizer`)

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `rcc_tcells.csv` | csv | 10.3K | 82 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: literature-curated collection. Categories include:
  - **RCC TILs** — paired-chain TCRs from single-cell ccRCC studies (e.g. Krishna 2021).
  - **Viral controls** — well-characterized public TCRs against CMV, EBV, Influenza, SARS-CoV-2 epitopes (VDJdb/IEDB consensus entries).
  - **Tumor-associated antigens (TAAs)** — TCRs against NY-ESO-1 / WT1 / PRAME / MART-1 etc. from published TCR-engineering / tetramer studies.
- **Original cohort**: small panel — every row is hand-picked from a primary study; donor counts vary.
- **Curation pipeline**: manual compilation; the `_binding` column is hard-coded to `"pos"` in the standardizer (`scripts/data_processing/standardize/rcc_atlas.py:64-65`) since all entries are positive associations.
- **Updates / versioning history**: in-house, no formal versioning.

## Caveats & known issues
- **Tiny dataset.** 81 rows; statistical power is minimal. RCC_ATLAS's value is curatorial — high-confidence positive examples for sanity-checking models against well-known RCC and viral epitopes.
- **Heavy reliance on upstream databases.** Many viral-control rows are flagged as `VDJdb/IEDB consensus`; these will match VDJdb / IEDB records if cross-source dedup is run on CDR3+epitope+MHC. RCC TIL rows are more likely unique.
- **TIL clones may be polyspecific.** Tumor-infiltrating T cells from ccRCC have been shown to often recognize multiple antigens (or antigens not yet identified); annotated antigen may be tentative for some rows.
- **MHC restriction inferred, not always validated.** Some entries record HLA restriction inferred from the cohort's HLA typing rather than direct experimental confirmation.
- **No formal publication.** Without an in-house manuscript, the curation criteria and inclusion/exclusion rules are not documented outside the file itself. {citation needed}
- **Class-II coverage is minimal.** Most rows are class-I; class-II MHC is rarely populated.
- **Hard-coded positive labels.** `_binding="pos"` is applied unconditionally in the standardizer — do NOT use this dataset to train negative-class examples.

## Fidelity assessment
- **Confidence (1–5)**: **4 (High)** for individual rows that come from peer-reviewed single-cell or tetramer studies; **3 (Medium)** at the dataset level due to mixed provenance and lack of in-house curation manuscript.
- **Recommended QUEST integration tier**: **experimentally validated** (per CLAUDE.md framework). Treat all rows as positive TCR-pMHC associations; expand into individual chains and interaction permutations. Cross-source dedup against VDJdb/IEDB will collapse most viral-control rows.
- **Score / quality threshold**: all rows assigned `binding="pos"` in `scripts/data_processing/standardize/rcc_atlas.py:64-65`. No additional threshold.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/rcc_atlas.py`
- **Class**: `RccAtlasStandardizer`
<!-- TODO: filled by code-cross-reference -->
