# CEDAR

## Identity
- **Source path**: `data/raw_data/databases/CEDAR/`
- **Primary citation**: Kosaloglu-Yalcin Z, Blazeska N, Vita R, Carter H, Nielsen M, Schoenberger S, Sette A, Peters B (2023). "The Cancer Epitope Database and Analysis Resource (CEDAR)." *Nucleic Acids Research* 51(D1):D845-D852, [doi:10.1093/nar/gkac902](https://doi.org/10.1093/nar/gkac902) — [PubMed 36250634](https://pubmed.ncbi.nlm.nih.gov/36250634/)
- **Earlier blueprint**: Kosaloglu-Yalcin Z, et al. (2021). "The Cancer Epitope Database and Analysis Resource: A Blueprint for the Establishment of a New Bioinformatics Resource for Use by the Cancer Immunology Community." *Frontiers in Immunology* 12:735609, [doi:10.3389/fimmu.2021.735609](https://doi.org/10.3389/fimmu.2021.735609)
- **Sibling-database methodology**: Blazeska N, et al. (2023). "IEDB and CEDAR: Two Sibling Databases…" *Methods Mol Biol* 2673:133-149, [doi:10.1007/978-1-0716-3239-0_9](https://doi.org/10.1007/978-1-0716-3239-0_9) — [PubMed 37258911](https://pubmed.ncbi.nlm.nih.gov/37258911/)
- **Version / release**: Full database export pulled via `download_cedar.sh` and `download_cedar_api.py`; ~106K TCR rows in the receptor file.
- **License**: Public-access, citation requested (NIH/NIAID-funded resource, La Jolla Institute).
- **Project URL**: <https://cedar.iedb.org/>

## Data shape
- **Chains present**: TRA, TRB, paired alpha-beta with V/D/J gene annotations. Same multi-level header schema as IEDB.
- **Epitope/MHC labelling**: epitope and MHC fields are populated directly on the receptor row in CEDAR (unlike IEDB, where the standardizer joins to a tcell file).
- **Single-cell vs. bulk**: heterogeneous; cancer epitope assays span tetramer staining, single-cell paired sequencing, MHC ligandomics by mass-spectrometry, and TIL co-culture screens.
- **Species filter**: human-only by upstream curation policy (CEDAR scope is human cancer); the standardizer does not need a species filter.

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/CEDAR`
- **Size**: 8.1G (8,683,260,966 bytes)
- **Files**: 12
- **Format breakdown**: `.csv`=5, `.zip`=5, `.sh`=1, `.py`=1
- **Records**: 6,481,168 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: sum of records in archived csv/tsv/fasta members; tab/comma-delimited row excluding header
- **Standardizer**: `scripts/data_processing/standardize/cedar.py` (`CedarStandardizer`)
- **Standardizer**: `scripts/data_processing/standardize/cedar_pmhc.py` (`CedarPmhcStandardizer`)

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `mhc_ligand/mhc_ligand_full_v3.csv` | csv | 7.1G | 4,618,251 | wc -l minus 1 (header) |
| `epitope/epitope_full_v3.csv` | csv | 531.2M | 1,497,440 | wc -l minus 1 (header) |
| `tcell/tcell_full_v3.csv` | csv | 331.9M | 151,479 | wc -l minus 1 (header) |
| `epitope/epitope_full_v3.zip` | zip | 58.0M | — | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `receptor/tcr_full_v3.csv` | csv | 39.8M | 105,846 | wc -l minus 1 (header) |
| `tcell/tcell_full_v3.zip` | zip | 13.2M | — | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `receptor/receptor_full_v3.zip` | zip | 2.1M | 106,999 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `receptor/bcr_full_v3.csv` | csv | 824.8K | 1,153 | wc -l minus 1 (header) |
| `bcell/bcell_full_v3.zip` | zip | 0B | — | archive error: BadZipFile |
| `mhc_ligand/mhc_ligand_full_v3.zip` | zip | 0B | — | archive error: BadZipFile |

**Non-data artifacts**: `download_cedar.sh`, `download_cedar_api.py`

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: All cancer-specific epitope data — T-cell assays, B-cell assays, MHC binding assays, and MHC ligandomics by MS — curated from the literature. The QUEST standardizer uses only the receptor file (paired TCR + epitope + MHC), as the tcell-assay file's IDs do not consistently overlap with the receptor file in current CEDAR exports (see `scripts/data_processing/standardize/cedar.py:6-12` comment).
- **Original cohort**: tumor antigens (neoantigens, cancer-testis antigens, overexpressed self-antigens), tumor-infiltrating lymphocyte (TIL) studies, adoptive cell therapy products, and immune-checkpoint-inhibitor cohorts. Donor counts inherit from upstream studies.
- **Curation pipeline**: Same workflow as IEDB. Manual curation by PhD-level curators using the IEDB Curation Application; controlled vocabularies (NCBI Taxonomy, MRO, Cell Ontology); validated against published positive AND negative results. Cancer-specific curation rules at <https://curationwiki.iedb.org/wiki/index.php/CEDAR/Cancer_Specific_Rules>.
- **Updates / versioning history**: 2021 blueprint paper; 2023 launch as the cancer-focused sibling of IEDB ([doi:10.1093/nar/gkac902](https://doi.org/10.1093/nar/gkac902)). Continuous updates as new cancer immunology papers are published.

## Caveats & known issues
- **Cancer-specific rules differ from IEDB.** Neoantigens (mutated self) are curated with mutation context that the QUEST standardizer does NOT currently propagate — only the mutated peptide is captured.
- **Receptor-tcell ID mismatch.** Per the standardizer comment, CEDAR's tcell-assay file IDs do not overlap with the receptor file IDs in the current export. The standardizer therefore reads only the receptor file and pulls epitope + MHC directly from receptor columns.
- **Same multi-level header complexity as IEDB.** The flattening logic in `_flatten_header` is shared; header drift across releases can silently break column mapping.
- **Limited paired-chain coverage.** Most cancer-cohort TCR studies sequence beta only; paired alpha-beta records remain a minority.
- **Neoantigen specificity is HLA-restricted and patient-specific.** Many records are unique to one tumor and one HLA, limiting transfer-learning utility unless aggregated by epitope motif.
- **Cross-source overlap with IEDB.** Some records co-exist with their IEDB versions (cancer-related records were partly migrated to CEDAR but referenced records can still appear via IEDB exports).

## Fidelity assessment
- **Confidence (1–5)**: **3 (Medium)** — peer-reviewed methodology, manual curation, but cancer-specific records carry the same heterogeneity issues as IEDB. Single-cell paired+epitope cancer records can be 4-5 individually.
- **Recommended QUEST integration tier**: **mixed**. Records with explicit T-cell binding evidence → experimentally validated. CEDAR also has a peptide-MHC-only branch (`scripts/data_processing/standardize/cedar_pmhc.py`) that is treated as pMHC-only training data.
- **Score / quality threshold**: no explicit score field. Standardizer keeps all receptor rows that have at least one CDR3 (downstream length filter applied in `quest/data/standardization.py`).

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/cedar.py` (TCR-receptor view) and `scripts/data_processing/standardize/cedar_pmhc.py` (pMHC-only view)
- **Class**: `CedarStandardizer`
<!-- TODO: filled by code-cross-reference -->
