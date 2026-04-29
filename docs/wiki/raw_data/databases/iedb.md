# IEDB

## Identity
- **Source path**: `data/raw_data/databases/IEDB/`
- **Primary citation**: Vita R, Blazeska N, Marrama D, et al. (2025). "The Immune Epitope Database (IEDB): 2024 update." *Nucleic Acids Research* 53(D1):D436-D443, [doi:10.1093/nar/gkae1092](https://doi.org/10.1093/nar/gkae1092) — [PubMed 39558162](https://pubmed.ncbi.nlm.nih.gov/39558162/)
- **Companion methodology paper**: Blazeska N, Kosaloglu-Yalcin Z, Vita R, Peters B, Sette A (2023). "IEDB and CEDAR: Two Sibling Databases to Serve the Global Scientific Community." *Methods Mol Biol* 2673:133-149, [doi:10.1007/978-1-0716-3239-0_9](https://doi.org/10.1007/978-1-0716-3239-0_9) — [PubMed 37258911](https://pubmed.ncbi.nlm.nih.gov/37258911/)
- **Version / release**: Full database export pulled via `download_iedb.sh`; downloaded 2026-01 (per QUEST sibling agents). Approximately 226K TCR-bearing rows after filtering.
- **License**: Public; "free of charge to anyone" with citation requested. Manual export terms at <https://www.iedb.org/>
- **Project URL**: <https://www.iedb.org/>

## Data shape
- **Chains present**: TRA, TRB, paired alpha-beta. Curated and calculated CDR3 columns plus full V/D/J gene calls.
- **Epitope/MHC labelling**: present in both `receptor/tcr_full_v3.csv` (epitope name and MHC allele on the receptor row) and `tcell/tcell_full_v3.csv` (T-cell assay outcome with peptide and MHC restriction). The QUEST standardizer joins on assay IDs to enrich binding labels and back-fill missing peptide/MHC.
- **Single-cell vs. bulk**: heterogeneous — IEDB ingests data from any T-cell assay format including tetramer single-cell, ELISpot, ICS, in-vitro proliferation, and structural studies.
- **Species filter**: standardizer keeps records whose linked T-cell assay is in `Homo sapiens` (`scripts/data_processing/standardize/iedb.py:175-192`); also excludes non-TCR receptors (`bcr`, `ig`, `immunoglobulin`).

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/IEDB`
- **Size**: 2.8G (3,003,590,844 bytes)
- **Files**: 12
- **Format breakdown**: `.zip`=5, `.csv`=4, `.sql.gz`=1, `.sh`=1
- **Records**: 3,168,890 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: sum of records in archived csv/tsv/fasta members; tab/comma-delimited row excluding header
- **Standardizer**: `scripts/data_processing/standardize/iedb.py` (`IedbStandardizer`)
- **Standardizer**: `scripts/data_processing/standardize/iedb_pmhc.py` (`IedbPmhcStandardizer`)

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcell/tcell_full_v3.csv` | csv | 1.2G | 567,634 | wc -l minus 1 (header) |
| `epitope/epitope_full_v3.csv` | csv | 810.8M | 2,366,205 | wc -l minus 1 (header) |
| `full_database/iedb_public.sql.gz` | sql.gz | 571.1M | — | sql dump not counted |
| `receptor/tcr_full_v3.csv` | csv | 84.5M | 226,303 | wc -l minus 1 (header) |
| `epitope/epitope_full_v3.zip` | zip | 83.3M | — | zip member inspection; no tabular members |
| `tcell/tcell_full_v3.zip` | zip | 42.3M | — | zip member inspection; no tabular members |
| `receptor/bcr_full_v3.csv` | csv | 7.0M | 8,748 | wc -l minus 1 (header) |
| `receptor/receptor_full_v3.zip` | zip | 5.7M | — | zip member inspection; no tabular members |
| `bcell/bcell_full_v3.zip` | zip | 0B | — | archive error: BadZipFile |
| `mhc_ligand/mhc_ligand_full_v3.zip` | zip | 0B | — | archive error: BadZipFile |

**Non-data artifacts**: `download_iedb.sh`

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: IEDB is a meta-curator. Each record is back-linked to a published assay and the underlying experimental method (tetramer, MHC binding, T-cell activation, structure determination, etc.). The QUEST standardizer filters to assays whose `Qualitative Measure` contains "positive".
- **Original cohort**: thousands of donors across thousands of publications, spanning infectious diseases (the bulk), allergies, autoimmunity, and transplantation. Cancer-specific records moved to the sibling CEDAR database.
- **Curation pipeline**: Manual curation by a team of PhD-level immunologists at La Jolla Institute, governed by the IEDB Curation Manual ([curationwiki.iedb.org](https://curationwiki.iedb.org/)). Workflow: (1) automated PubMed query retrieves candidate publications; (2) curators extract assays and epitopes per publication into a structured form; (3) controlled vocabularies enforce ontology (NCBI Taxonomy, MRO for MHC). All assays — positive AND negative — are captured.
- **Updates / versioning history**: Established 2004 (NIH/NIAID Contract HHSN272201200010C). Major redesign in 2016 with the v3 export; 2018 update ([Vita et al., NAR 2019](https://doi.org/10.1093/nar/gky1006)); 2024 update with improved interoperability ([doi:10.1093/nar/gkae1092](https://doi.org/10.1093/nar/gkae1092)). >23,000 records reviewed during the validation cycles described in the 2023 sibling-databases paper.

## Caveats & known issues
- **Heterogeneous assay quality.** Records range from gold-standard tetramer staining to weak ELISpot positives. IEDB exposes the experimental detail but does NOT collapse this into a single confidence score — downstream consumers must filter on `Method/Technique` if they want a quality cutoff. The QUEST standardizer uses `Qualitative Measure == "positive"` only, so positive-but-weak signals are retained.
- **Same epitope, multiple curations.** Common viral epitopes (NLVPMVATV, GLCTLVAML, GILGFVFTL, etc.) appear across hundreds of papers; downstream dedup is essential.
- **Multi-level CSV header complexity.** IEDB v3 exports use a two-row header (`Chain 1` / `Chain 2` / `Epitope` group rows on top of field-name rows). The standardizer (`scripts/data_processing/standardize/cedar.py:30-86`, shared with CEDAR) flattens these into single-level column names — header drift across releases will silently break this mapping.
- **Receptor-tcell join is by assay-ID, not by clone.** The standardizer extracts numeric assay IDs from the IRI URLs and joins receptor rows to the t-cell table via these IDs. If a receptor row has multiple assay IDs (comma-separated), only the first matching tcell row is kept (`scripts/data_processing/standardize/iedb.py:218-223`).
- **Heavy class-I, HLA-A*02 bias.** Acknowledged in IMMREP22 and downstream benchmarks; underrepresented HLA-C, HLA-DP, and non-European HLA alleles ([Frontiers in Immunology, NetMHCpan non-European HLA evaluation](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2023.1288105/full)).
- **Cross-source overlap.** IEDB is upstream of VDJdb, McPAS-TCR, TRAIT, and many TCR-focused databases. Cross-source dedup in QUEST's pipeline is mandatory or duplicate records will inflate effective sample sizes.
- **Some non-TCR receptors slip through type filtering.** The standardizer relies on string matching on a `Receptor_Type`-like column; novel terminology can cause BCR records to appear in TCR exports.

## Fidelity assessment
- **Confidence (1–5)**: **3 (Medium)** at the database level. Individual records can range from 5 (single-cell paired tetramer) to 2 (weak ELISpot positives) — fidelity is record-dependent. Curation is gold-standard; the heterogeneity is inherited from the source literature.
- **Recommended QUEST integration tier**: **mixed**. Positive T-cell assay records → experimentally validated tier (chains + interactions). The QUEST standardizer also produces a peptide-MHC-only view (`scripts/data_processing/standardize/iedb_pmhc.py`) — peptide-MHC records without TCRs feed pMHC-only training.
- **Score / quality threshold**: assay outcome must contain "positive" in `tcell_full_v3.csv` (`scripts/data_processing/standardize/iedb.py:168-172`); receptor type filtered to non-BCR (`scripts/data_processing/standardize/iedb.py:71-74`); organism limited to human (`scripts/data_processing/standardize/iedb.py:186-192`).

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/iedb.py` (TCR-receptor view) and `scripts/data_processing/standardize/iedb_pmhc.py` (pMHC-only view)
- **Class**: `IedbStandardizer`
<!-- TODO: filled by code-cross-reference -->
