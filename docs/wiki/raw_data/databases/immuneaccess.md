# immuneACCESS

## Identity
- **Source path**: `data/raw_data/databases/immuneACCESS/`
- **Primary citation**: there is no single canonical citation for the immuneACCESS portal itself; it is the public-data arm of Adaptive Biotechnologies' immunoSEQ platform and hosts hundreds of independently published cohorts. The most-cited foundational paper for the immunoSEQ technology is: Robins HS, Campregher PV, Srivastava SK, et al. (2009). "Comprehensive assessment of T-cell receptor beta-chain diversity in alphabeta T cells." *Blood* 114(19):4099-4107, [doi:10.1182/blood-2009-04-217604](https://doi.org/10.1182/blood-2009-04-217604) — [PubMed 19706884](https://pubmed.ncbi.nlm.nih.gov/19706884/)
- **immuneSEQ multiplexed antigen-identification (MIRA)**: Klinger M, Kong K, Moorhead M, Weng L, Zheng J, Faham M (2013). "Combining next-generation sequencing and immune assays: a novel method for identification of antigen-specific T cells." *PLoS ONE* 8(9):e74231, [doi:10.1371/journal.pone.0074231](https://doi.org/10.1371/journal.pone.0074231) — [PubMed 24069285](https://pubmed.ncbi.nlm.nih.gov/24069285/)
- **Repository description**: immuneACCESS is registered in re3data — <https://www.re3data.org/repository/r3d100013228>
- **Version / release**: rolling. QUEST snapshot includes `existing_data/`, `immunoseq_data/`, `FHCRC-Warren-Updated_datasets/`, and `iseq/` subtrees pulled via Adaptive's immunoSEQ Data Assistant. Last-modified dates inherit from individual project deposits.
- **License**: per-project. Adaptive applies a "freely available for non-commercial research" framework to most projects but specific licenses vary (e.g. some projects require user registration). Several individual studies cite their own DOIs.
- **Project URL**: <https://clients.adaptivebiotech.com/immuneaccess>

## Data shape
- **Chains present**: predominantly TRB (Adaptive's bulk immunoSEQ panel is beta-only by default). Smaller subsets of TRA exist in `bulk_survey_tra/`. Paired alpha-beta is rare.
- **Epitope/MHC labelling**: not at the rearrangement level. Some projects have associated phenotype metadata (HLA typing, disease state) at the sample level.
- **Single-cell vs. bulk**: predominantly bulk TCR-β-seq using Adaptive's multiplex PCR + UMI chemistry. Paired-chain projects exist but are minority.
- **Species filter**: human-only by upstream design (immunoSEQ products target human TRB).

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/immuneACCESS`
- **Size**: 1.2T (1,318,342,382,550 bytes)
- **Files**: 29233
- **Format breakdown**: `.tsv`=29209, `.jar`=9, `.noext`=5, `.sql`=2, `.txt`=2, `.tar.gz`=1, `.zip`=1, `.py`=1, `.log`=1, `.properties`=1
- **Records**: ~2,437,152,912 (sampled) (95% CI: 1,046,309,348–3,827,995,102)
- **Counting method**: sampled (30 of 29209 tsv files, mean=83438.4 records/file, extrapolated; 5 files >2.0G excluded from sample)
- **Record definition**: line; sum of records in archived csv/tsv/fasta members; tab/comma-delimited row excluding header
- **Standardizer**: `scripts/data_processing/standardize/immuneaccess.py` (`ImmuneaccessStandardizer`)
- **Expected input files**: `*.tsv`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `<29179 more tsv files (not sampled)>` | tsv | 1.2T | 2,434,649,074 | extrapolated from sample |
| `existing_data/bulk_survey_trb/03858003620965_TCRB.tsv` | tsv | 166.9M | 463,513 | wc -l minus 1 (header) |
| `existing_data/bulk_survey_trb/03855000013644_TCRB.tsv` | tsv | 144.9M | 403,626 | wc -l minus 1 (header) |
| `existing_data/bulk_survey_trb/ADIRP0001325_TCRB.tsv` | tsv | 126.2M | 352,496 | wc -l minus 1 (header) |
| `FHCRC-Warren-Updated_datasets/HIP13769_GAMMA.tsv` | tsv | 89.8M | 272,913 | wc -l minus 1 (header) |
| `existing_data/bulk_survey_trb/03855000011740_TCRB.tsv` | tsv | 84.9M | 237,057 | wc -l minus 1 (header) |
| `FHCRC-Warren-Updated_datasets/03855000013287_TCRB.tsv` | tsv | 81.3M | 226,156 | wc -l minus 1 (header) |
| `existing_data/bulk_survey_trb/subj92_PB28.tsv` | tsv | 44.1M | 123,338 | wc -l minus 1 (header) |
| `existing_data/bulk_survey_trb/88b55f27-a_TCRB.tsv` | tsv | 42.5M | 118,802 | wc -l minus 1 (header) |
| `existing_data/bulk_survey_trb/P01_Tissue_mild_disease.tsv` | tsv | 19.3M | 53,696 | wc -l minus 1 (header) |
| `existing_data/bulk_survey_trb/ST-00020236-B.tsv` | tsv | 15.2M | 42,698 | wc -l minus 1 (header) |
| `FHCRC-Warren-Updated_datasets/2134040_ARKS.tsv` | tsv | 13.2M | 36,867 | wc -l minus 1 (header) |

**Non-data artifacts**: `FHCRC-Warren-Updated_datasets/download_progress.log`, `download_immunoseq_api.py`, `immunoSEQDataAssistant/commons-codec-1.10.jar`, `immunoSEQDataAssistant/commons-compress-1.7.jar`, `immunoSEQDataAssistant/commons-io-2.4.jar`, `immunoSEQDataAssistant/commons-lang3-3.1.jar`, `immunoSEQDataAssistant/commons-logging-1.1.3.jar`, `immunoSEQDataAssistant/commons-vfs2-2.0.jar`, `immunoSEQDataAssistant/dataTransfer.jar`, `immunoSEQDataAssistant/immunoSEQDataAssistant-15.05.27.1308.jar` (+2 more)

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: bulk multiplex-PCR-based TCR-β sequencing on the immunoSEQ assay (Robins et al. 2009 chemistry; later UMI-controlled versions). Standard output is a per-sample TSV with per-clone abundance, V/D/J calls, and frame-type (`In`, `Out`, `Stop`).
- **Original cohort**: hundreds of cohorts across decades of Adaptive collaborations — oncology (Snyder/Riaz melanoma, MSI-H tumors), autoimmunity, transplantation, T1D, infectious disease (CMV, EBV, HCV), and large healthy-donor surveys (Emerson/DeWitt CMV cohort, the Britanova aging cohort references, etc.). Donor counts often >1,000 per major study.
- **Curation pipeline**: Adaptive standardizes raw sequencing data through their internal pipeline (V/D/J calling, UMI-based error correction, clone clustering); deposits TSVs to immuneACCESS with per-project metadata. The QUEST standardizer (`scripts/data_processing/standardize/immuneaccess.py`) reads `*.tsv` recursively, filters to `frame_type == "In"` (or `productive == true/t/1` if AIRR-compliant), and uses `v_resolved`/`j_resolved` when `v_gene`/`j_gene` are absent.
- **Updates / versioning history**: immuneACCESS launched mid-2010s as Adaptive's open-data portal. Major data deposits triggered by individual papers; ImmuneCODE (COVID-19) is the largest single deposit (treated as separate database in QUEST — see `immunecode.md`).

## Caveats & known issues
- **TRB-bias.** ~95%+ of records are beta-chain only. Alpha-chain coverage exists in select projects but is sparse.
- **No antigen specificity at the row level.** Project-level metadata may indicate the cohort context (e.g. "CMV-seropositive donor") but per-clone antigen labels are absent unless the project deposits an explicit MIRA-style mapping.
- **Donor-level HLA typing variable.** Not every project provides full HLA typing; pMHC analysis from immuneACCESS alone is rarely possible.
- **Frame-type semantics.** Adaptive uses `frame_type` (`In`, `Out`, `Stop`) instead of the AIRR-standard `productive`. Both fields exist in newer projects; the standardizer prefers `frame_type` then falls back to `productive`.
- **Chain detection from path.** The standardizer infers TRA vs. TRB from directory/filename patterns (`bulk_survey_tra`, `_tcra`, `_tra.`); if a new project uses different naming conventions, records default to TRB and a warning is emitted (`scripts/data_processing/standardize/immuneaccess.py:121-135`).
- **Clonal-frequency skew.** Large clones from chronic-infection / cancer cohorts dominate raw counts. Treating each row as one record (uniform weighting) reduces but does not remove the bias.
- **Cross-source overlap with ADC, immuneCODE, tcrdb.** Adaptive frequently mirrors data across portals; cross-source dedup is essential.
- **Massive scale (~3.5B rows).** Streaming required; standardizer reads only the columns it needs (`scripts/data_processing/standardize/immuneaccess.py:55-72`).

## Fidelity assessment
- **Confidence (1–5)**: **2 (Bulk/Indirect)** for TCR-pMHC purposes. As a TCR-β language-model corpus, fidelity is **3-4**: peer-reviewed assay, uniform processing, but heavy clonal/donor skew.
- **Recommended QUEST integration tier**: **bulk-repertoire** (per CLAUDE.md framework). Use for MLM and TCR-only training (predominantly TRB; full_trb where stitching is feasible). Do NOT use for TCR-pMHC interaction training.
- **Score / quality threshold**: in-frame / productive filter (`scripts/data_processing/standardize/immuneaccess.py:83-92`); chain inferred from path; no additional quality threshold.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/immuneaccess.py`
- **Class**: `ImmuneaccessStandardizer` (streaming = True due to ~3.5B rows)
<!-- TODO: filled by code-cross-reference -->
