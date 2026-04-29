# TRAIT

## Identity
- **Source path**: `data/raw_data/databases/trait/`
- **Primary citation**: Wei M, Wu J, Bai S, Zhou Y, Chen Y, Zhang X, Zhao W, Chi Y, Pan G, Zhu F, Chen S, Zhou Z (2025). "TRAIT: A multi-omics database for T-cell receptor–antigen interactions." *Genomics, Proteomics & Bioinformatics* 23(3):qzaf033, [doi:10.1093/gpbjnl/qzaf033](https://doi.org/10.1093/gpbjnl/qzaf033) — [PubMed 40257421](https://pubmed.ncbi.nlm.nih.gov/40257421/) (also as bioRxiv preprint [doi:10.1101/2024.11.20.624436](https://doi.org/10.1101/2024.11.20.624436))
- **Version / release**: Files dated `20250312` (e.g. `Interactive_TCR-pMHC_Pairs.zip_20250312.zip`); ~4M rows after expansion of per-epitope ZIPs.
- **License**: Free academic-use; specific terms on the database website. {citation needed for explicit license string}
- **Project URL**: <https://pgx.zju.edu.cn/traitdb> (Zhejiang University; also referenced through the Genomics, Proteomics & Bioinformatics journal landing page)

## Data shape
- **Chains present**: TRA, TRB, paired alpha-beta. Filename-encoded metadata follows the pattern `{MHC}_{epitope}_{protein}_{disease}_binder_{pos|neg}.zip` (e.g. `A0201_GLCTLVAML_BMLF1_EBV_binder_pos.zip`). Inner files are TSV/CSV per epitope.
- **Epitope/MHC labelling**: epitope and MHC are encoded into the filename and propagated as `peptide` and `mhc_one` (class-I focus) for every row. Includes both binders and non-binders.
- **Single-cell vs. bulk**: heterogeneous — TRAIT integrates single-cell omics (10x Genomics, MERIT, etc.), tetramer-sorted bulk TCRs, and structural data from PDB. The Omics archive also contains bulk repertoire data with epitope assignments.
- **Species filter**: TRAIT is human-focused; standardizer does not impose an explicit species filter (relies on upstream).

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/trait`
- **Size**: 65.1M (68,295,572 bytes)
- **Files**: 36
- **Format breakdown**: `.zip`=32, `.txt`=2, `.sh`=1
- **Records**: 3,967,325 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: line; sum of records in archived csv/tsv/fasta members
- **Standardizer**: `scripts/data_processing/standardize/trait.py` (`TraitStandardizer`)
- **Expected input files**: `epitopes`, `main`, `*.zip`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `main/Omics.zip` | zip | 43.6M | 3,362,476 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `epitopes/non_binding/A0301_RIAAWMATY_BCL-2L1_Cancer_binder_neg.txt` | txt | 8.1M | 67,174 | wc -l (assumed no header) |
| `main/Interactive_TCR-pMHC_Pairs.zip_20250312.zip` | zip | 4.8M | — | zip member inspection; no tabular members |
| `epitopes/non_binding/A0301_RIAAWMATY_BCL-2L1_Cancer_binder_neg.zip` | zip | 965.8K | — | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `epitopes/non_binding/A0201_ELAGIGILTV_MART-1_Cancer_binder_neg.zip` | zip | 962.3K | 66,862 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `epitopes/non_binding/B0801_FLRGRAYGL_EBNA-3A_EBV_binder_neg.zip` | zip | 956.8K | 67,150 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `epitopes/non_binding/A0101_VTEHDTLLY_IE-1_CMV_binder_neg.zip` | zip | 953.0K | 67,174 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `epitopes/non_binding/B3501_IPSINVHHY_pp65_CMV_binder_neg.zip` | zip | 953.0K | 67,169 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `epitopes/non_binding/A0201_NLVPMVATV_pp65_CMV_binder_neg.zip` | zip | 953.0K | 67,172 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `epitopes/non_binding/A2402_QYDPVAALF_pp65_CMV_binder_neg.zip` | zip | 952.9K | 67,169 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `epitopes/non_binding/A0201_GLCTLVAML_BMLF1_EBV_binder_neg.zip` | zip | 952.2K | 67,060 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `epitopes/non_binding/B0801_RAKFKQLL_BZLF1_EBV_binder_neg.zip` | zip | 940.3K | 65,982 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |

**Non-data artifacts**: `download_trait.sh`

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: aggregation across literature, public datasets, and omics platforms. TRAIT specifically emphasizes single-cell omics as a primary reliable data source for TCR-antigen interactions and includes "millions of reliable non-interactive TCRs" as negatives (per the [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.11.20.624436v1.full)).
- **Original cohort**: 7,979,361 antigen-specificity-validated TCR sequences for 1,127 unique epitopes and 112 MHC alleles; 220 PDB structures and 759 binding-affinity records also collected.
- **Curation pipeline**: The TRAIT team performs PubMed keyword searches plus public-dataset and omics-platform crawls. Sequences are normalized to a single schema; both interactive and reliable non-interactive TCRs are catalogued. Mutational scan data is curated to support cross-reactivity studies.
- **Updates / versioning history**: bioRxiv preprint November 2024; published in *Genomics, Proteomics & Bioinformatics* in 2025. The QUEST snapshot is dated 2025-03-12.

## Caveats & known issues
- **Heavy overlap with IEDB / VDJdb / 10x Genomics single-cell data.** TRAIT explicitly aggregates from these upstream sources, so any pipeline using TRAIT alongside VDJdb or IEDB will see heavy duplication unless cross-source dedup is run.
- **Filename-encoded metadata is brittle.** The standardizer parses filenames to extract MHC and epitope. Files that don't match the `{MHC}_{epitope}_{protein}_{disease}_binder_{pos|neg}` pattern are silently skipped (`scripts/data_processing/standardize/trait.py:43-47`). Any future filename-format change will silently shrink the ingest.
- **Negative-binder set construction is opaque.** TRAIT's "non-interactive" TCRs are typically donor-paired distractors or random repertoire backgrounds. These negatives can leak signals (e.g. donor-specific V-gene biases) — see [IMMREP23 lessons learned (Hudson et al.)](https://www.immunoinformaticsjournal.com/article/S2667-1190(24)00015-6/fulltext) on the perils of poorly-controlled negatives.
- **MHC-class-II coverage is limited compared to class-I.** Filename pattern records `mhc_one` only; class-II alpha/beta heterodimer information is not preserved through the standardizer.
- **Recent and not yet broadly benchmarked.** TRAIT is too new (2025) to have appeared in the major IMMREP benchmarking efforts; downstream evaluation of its specific quality is still emerging.
- **Some inner ZIPs use different separators.** The standardizer auto-detects `\t` vs `,` per file (`scripts/data_processing/standardize/trait.py:125-128`) but the structure is heterogeneous across the four top-level archives (`Interactive_TCR-pMHC_Pairs`, `Mutation`, `Omics`, `Therapeutics`).

## Fidelity assessment
- **Confidence (1–5)**: **3 (Medium)** at the database level. Individual single-cell-omics-derived records can be 4-5; literature-aggregated records inherit upstream quality. Heavy overlap with already-included sources reduces marginal information value.
- **Recommended QUEST integration tier**: **multi-source aggregation with experimentally validated data** (per CLAUDE.md framework). Treat positive binders as experimentally validated; treat negatives as informative-but-method-dependent. Cross-source dedup is critical.
- **Score / quality threshold**: filename `binder_pos` → `binding="pos"`; `binder_neg` → `binding="neg"` (`scripts/data_processing/standardize/trait.py:55-65`). No additional confidence threshold applied in code.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/trait.py`
- **Class**: `TraitStandardizer` (streaming = True due to ~4M rows)
<!-- TODO: filled by code-cross-reference -->
