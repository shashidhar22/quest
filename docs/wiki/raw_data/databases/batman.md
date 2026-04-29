# BATMAN / BATCAVE

## Identity
- **Source path**: `data/raw_data/databases/BATMAN/`
- **Primary citation (preprint)**: Banerjee A, Pattinson DJ, Wincek CL, Bunk P, Axhemi A, Chapin SR, Navlakha S, Meyer HV (2025). "Comprehensive epitope mutational scan database enables accurate T cell receptor cross-reactivity prediction." *bioRxiv* 2024.01.22.576714, [doi:10.1101/2024.01.22.576714](https://doi.org/10.1101/2024.01.22.576714) — [PubMed 38370810](https://pubmed.ncbi.nlm.nih.gov/38370810/)
- **Peer-reviewed**: Banerjee A, et al. (2025). "T cell receptor cross-reactivity prediction improved by a comprehensive mutational scan database." *Cell Systems* (2025), [doi:10.1016/j.cels.2025.101178](https://doi.org/10.1016/j.cels.2025.101178) — see [Cell Systems landing page](https://www.cell.com/cell-systems/abstract/S2405-4712(25)00178-4)
- **Version / release**: two Excel workbooks — `TCR_pMHCI_mutational_scan_database.xlsx` and `TCR_pMHCII_mutational_scan_database.xlsx`. Per the publication: ~22,000 TCR-peptide pairs across 25 immunogenic human and mouse epitopes against 151 TCRs; both class-I and class-II MHC.
- **License**: Open data per the bioRxiv preprint and the GitHub release at <https://github.com/meyer-lab-cshl/BATMAN-paper>. Cite the Cell Systems paper.
- **Project URL**: <https://github.com/meyer-lab-cshl/BATMAN> (model) ; <https://batman.cshl.edu/> (web app) ; <https://github.com/meyer-lab-cshl/BATMAN-paper> (paper artifacts and BATCAVE database)

## Data shape
- **Chains present**: TRA + TRB CDR3 plus full V/D/J gene calls (`trav, traj, trbv, trbd, trbj`); paired alpha-beta in nearly every row.
- **Epitope/MHC labelling**: every row has an `index_peptide` (the wild-type epitope), the `mhc` allele, and a mutated `peptide`. The `peptide_type` column distinguishes single-AA mutational scans from multi-AA mutations. `peptide_activity` is a continuous scalar measuring TCR activation against the mutated peptide (typically 0-1, with strong activation ≥ 0.5).
- **Single-cell vs. bulk**: not applicable. The data are TCR transduction / activation assays — engineered T cells are stimulated with mutated peptide variants and activation is read out by a reporter.
- **Species filter**: standardizer filters to `tcr_source_organism == "human"` (`scripts/data_processing/standardize/batman.py:96-99`); upstream BATCAVE includes mouse TCRs too.

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/BATMAN`
- **Size**: 1.6M (1,687,590 bytes)
- **Files**: 3
- **Format breakdown**: `.xlsx`=2
- **Records**: 22,827 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: xlsx data row excluding header
- **Standardizer**: `scripts/data_processing/standardize/batman.py` (`BatmanStandardizer`)

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `TCR_pMHCI_mutational_scan_database.xlsx` | xlsx | 1.3M | 17,097 | openpyxl read-only max_row minus 1 per sheet |
| `TCR_pMHCII_mutational_scan_database.xlsx` | xlsx | 322.4K | 5,730 | openpyxl read-only max_row minus 1 per sheet |

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: comprehensive single-amino-acid mutational scans of immunogenic peptides. For each (TCR, index peptide) pair, ~190 single-AA-mutated variants are tested for TCR activation. Multi-AA mutations are also included for some TCRs. Activation is measured by a TCR reporter (e.g. NFAT-GFP / Jurkat NFAT-luciferase) and normalized to the wild-type response.
- **Original cohort**: 151 TCRs across 25 immunogenic human and mouse epitopes (both MHC classes). TCRs come from published structural / functional studies; mutational data is aggregated from multiple labs and re-uniformized into BATCAVE.
- **Curation pipeline**: Meyer lab (CSHL Simons Center for Quantitative Biology) collected published mutational-scan datasets, normalized activation values to the wild-type response (so each TCR has its own scale), and stored both raw and normalized values per row. Excel workbooks are split by MHC class.
- **Updates / versioning history**: bioRxiv v1 (January 2024) → v3 (2025) → Cell Systems publication (2025). The BATCAVE database has continued to grow with new mutational-scan studies.

## Caveats & known issues
- **Mutational scans, not natural repertoires.** Every record is an engineered TCR + designed peptide variant. The dataset is excellent for cross-reactivity / off-target prediction but not for training general TCR-pMHC binding (which needs natural-repertoire diversity).
- **Index peptides are heavily-studied immunodominant epitopes.** NLVPMVATV (CMV pp65), GILGFVFTL (Influenza M1), GLCTLVAML (EBV BMLF1), MART-1 ELAGIGILTV / ALA-MART, and a small number of mouse epitopes. Coverage is concentrated; cross-epitope generalization is the open problem the BATMAN paper benchmarks.
- **Gene-name format requires preprocessing.** BATMAN stores genes as bare numeric identifiers (`12-3`, `35*02`, `1*00(80)`) without the `TRAV`/`TRBV` prefix; the standardizer prepends the prefix and strips parenthetical confidence scores (`scripts/data_processing/standardize/batman.py:42-60`).
- **Continuous `peptide_activity` thresholded into binary.** The QUEST standardizer derives `binding="pos"` for activity ≥ 0.1 and `binding="neg"` for < 0.1 (`scripts/data_processing/standardize/batman.py:124-131`). The 0.1 threshold is lenient; the BATMAN paper itself uses 0.5 for "strong" activation. Treat the QUEST `binding` column as a coarse proxy and prefer `score` (= raw `peptide_activity`) for downstream weighting.
- **Mouse TCRs filtered out.** The QUEST standardizer keeps only `tcr_source_organism == "human"`; ~half of BATCAVE may be lost.
- **Replicate aggregation upstream.** The Cell Systems paper aggregates multiple replicates per (TCR, peptide) — the published row is typically the mean activation. Per-replicate noise is not preserved.
- **Class-II rows are minority.** Class-I records dominate; class-II mutational scans exist for only a few TCRs.
- **PMID column.** Per-row references are stored in `pmid`; cross-source dedup with the upstream studies (which may also appear in IEDB / VDJdb) is required.

## Fidelity assessment
- **Confidence (1–5)**: **5 (Gold)** for cross-reactivity / mutational-scan use cases — peer-reviewed, multi-replicate, quantitative activation measurements with both positives and negatives. **4 (High)** for general TCR-pMHC binding because peptide diversity is centered on a few index epitopes.
- **Recommended QUEST integration tier**: **experimentally validated**. Use for chain expansion and TCR-pMHC interaction training. Treat the continuous `peptide_activity` (mapped to QUEST `score`) as primary; the binary `binding` column is a coarse derivative. Particularly valuable for cross-encoder training because BATMAN provides explicit negatives (low-activity peptide variants).
- **Score / quality threshold**: human-organism filter (`scripts/data_processing/standardize/batman.py:96-99`); binary binding from `peptide_activity` threshold 0.1 (`scripts/data_processing/standardize/batman.py:127-131`); raw activity preserved in `score`.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/batman.py`
- **Class**: `BatmanStandardizer`
<!-- TODO: filled by code-cross-reference -->
