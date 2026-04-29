# ImmuneCODE

## Identity
- **Source path**: `data/raw_data/databases/immuneCODE/`
- **Primary citation**: Nolan S, Vignali M, Klinger M, Dines JN, Kaplan IM, Svejnoha E, Craft T, Boland K, Pesesky MW, Gittelman RM, Snyder TM, Gooley CJ, Semprini S, Cerchione C, Nicolini F, Mazza M, Delmonte OM, Dobbs K, Carreño-Tarragona G, Barrio S, Sambri V, Martinelli G, Goldman JD, Heath JR, Notarangelo LD, Martinez-Lopez J, Howie B, Carlson JM, Robins HS (2025). "A large-scale database of T-cell receptor beta sequences and binding associations from natural and synthetic exposure to SARS-CoV-2." *Frontiers in Immunology* 16:1488851, [doi:10.3389/fimmu.2025.1488851](https://doi.org/10.3389/fimmu.2025.1488851) — [PubMed 40034696](https://pubmed.ncbi.nlm.nih.gov/40034696/)
- **Earlier preprint**: Snyder TM, Gittelman RM, Klinger M, et al. (2020). "Magnitude and dynamics of the T-cell response to SARS-CoV-2 infection at both individual and population levels." medRxiv 2020.07.31.20165647, [doi:10.1101/2020.07.31.20165647](https://doi.org/10.1101/2020.07.31.20165647)
- **MIRA technology citation**: Klinger M, Kong K, Moorhead M, Weng L, Zheng J, Faham M (2013). *PLoS ONE* 8(9):e74231, [doi:10.1371/journal.pone.0074231](https://doi.org/10.1371/journal.pone.0074231) — [PubMed 24069285](https://pubmed.ncbi.nlm.nih.gov/24069285/)
- **Version / release**: `ImmuneCODE-MIRA-Release002.1` and `ImmuneCODE-Repertoires-002.2.tgz` / `ImmuneCODE-Review-002`. Per the Frontiers in Immunology 2025 publication, the database contains hundreds of millions of TCR sequences from >1,400 subjects and >160,000 high-confidence SARS-CoV-2-associated TCRs.
- **License**: "Free to use for research purposes" with citation requested. Specific terms at <https://www.adaptivebiotech.com/immunecode/>.
- **Project URL**: <https://www.adaptivebiotech.com/immunecode/> ; data on immuneACCESS at <https://clients.adaptivebiotech.com/pub/covid-2020>

## Data shape
- **Chains present**: TRB only.
- **Epitope/MHC labelling**: present in MIRA `peptide-detail-ci.csv` (Class I) and `peptide-detail-cii.csv` (Class II) — each row carries a TCR BioIdentity and one or more SARS-CoV-2 peptide(s) with the experiment ID; HLA is back-filled from `subject-metadata.csv`. Repertoire/Review TSVs (`*_TCRB.tsv`) carry no antigen labels.
- **Single-cell vs. bulk**: bulk immunoSEQ TCR-β with MIRA antigen-mapping experiments. MIRA peptides are detected by sorting peptide-stimulated T cells and sequencing their expanded TCRs.
- **Species filter**: human-only by design.

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/immuneCODE`
- **Size**: 228.4G (245,189,595,083 bytes)
- **Files**: 1430
- **Format breakdown**: `.tsv`=1416, `.csv`=8, `.tar.gz`=2, `.zip`=1, `.xlsx`=1, `.txt`=1, `.sh`=1
- **Records**: ~294,756,159 (sampled) (95% CI: 208,497,358–380,358,513)
- **Counting method**: sampled (30 of 1416 tsv files, mean=207929.3 records/file, extrapolated)
- **Record definition**: line; sum of records in archived csv/tsv/fasta members; tab/comma-delimited row excluding header; xlsx data row excluding header
- **Standardizer**: `scripts/data_processing/standardize/immunecode.py` (`ImmunecodeStandardizer`)
- **Expected input files**: `*_TCRB.tsv`, `peptide-detail-ci.csv`, `subject-metadata.csv`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `<1386 more tsv files (not sampled)>` | tsv | 199.3G | 288,190,056 | extrapolated from sample |
| `immunecode_covid/ImmuneCODE-Repertoires-002.tgz` | tar.gz | 23.8G | — | archive >1.0G; not extracted |
| `immunecode_covid/ImmuneCODE-Repertoires-002.2.tgz` | tar.gz | 1.1G | — | archive >1.0G; not extracted |
| `immunecode_covid/ImmuneCODE-Review-002/INCOV077-AC-3_TCRB.tsv` | tsv | 425.7M | 627,597 | wc -l minus 1 (header) |
| `immunecode_covid/ImmuneCODE-Review-002/1337122BW_TCRB.tsv` | tsv | 376.2M | 549,728 | wc -l minus 1 (header) |
| `immunecode_covid/ImmuneCODE-Review-002/KH20-09729_TCRB.tsv` | tsv | 292.3M | 438,297 | wc -l minus 1 (header) |
| `immunecode_covid/ImmuneCODE-Review-002/234129BW_TCRB.tsv` | tsv | 279.5M | 416,971 | wc -l minus 1 (header) |
| `immunecode_covid/ImmuneCODE-Review-002/ADIRP0000392_TCRB.tsv` | tsv | 270.3M | 399,438 | wc -l minus 1 (header) |
| `immunecode_covid/ImmuneCODE-Review-002/860011254_TCRB.tsv` | tsv | 267.5M | 387,789 | wc -l minus 1 (header) |
| `immunecode_covid/ImmuneCODE-Review-002/ADIRP0001893_TCRB.tsv` | tsv | 263.9M | 386,897 | wc -l minus 1 (header) |
| `immunecode_covid/ImmuneCODE-Review-002/860011351_TCRB.tsv` | tsv | 246.3M | 370,852 | wc -l minus 1 (header) |
| `immunecode_covid/ImmuneCODE-Review-002/INCOV026-BL-3_TCRB.tsv` | tsv | 246.1M | 364,723 | wc -l minus 1 (header) |

**Non-data artifacts**: `download_immunecode.sh`

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**:
  - **MIRA (Multiplex Identification of T-cell Receptor Antigen specificity)** — see Klinger 2013. Subjects' T cells are stimulated with pools of peptides; expanded clones are sequenced and deconvoluted to specific peptide-pool memberships across multiple replicates, yielding TCR→peptide(s) assignments. Subject HLA typing maps experiments to MHC alleles.
  - **Repertoire / Review** — bulk immunoSEQ TCR-β sequencing of natural-infection and vaccinated-cohort subjects, no antigen mapping.
  - **MIRA panels** — four panels described in the Frontiers paper: `minigene_Set1` and `minigene_Set2` target large protein sequences (long peptides up to ~25 AA); `C19_cI` and `C19_cII` target individual class-I and class-II peptides respectively.
- **Original cohort**: >1,400 subjects exposed to or infected with SARS-CoV-2 plus mRNA-vaccinated controls. Includes cohorts from Italy, Spain, US, with a multi-site design coordinated by Adaptive Biotechnologies and Microsoft Research.
- **Curation pipeline**: Adaptive runs the MIRA experiment, deconvolutes peptide hits, applies a "high-confidence" filter (multiple-replicate hits, frequency thresholds), and publishes the results to immuneACCESS as ImmuneCODE-MIRA. Repertoire data is published as ImmuneCODE-Review TSVs. The QUEST standardizer parses the Adaptive `bio_identity` triple-format string `CDR3+TCRBV-gene+TCRBJ-gene` (`scripts/data_processing/standardize/immunecode.py:34-50`).
- **Updates / versioning history**: First public release September 2020 (medRxiv 2020.07.31). Iterations through 2021-2024 (Releases 001 / 002 / 002.1 / 002.2). Formal peer-reviewed publication in *Frontiers in Immunology* February 2025.

## Caveats & known issues
- **TRB-only.** No alpha-chain information, limiting paired-chain modeling.
- **Class-I CD8 bias.** Per the 2025 paper: "the majority of the MIRA data are from CD8+ T cells binding peptides presented by MHC class I molecules"; CD4/MHC-II coverage is much weaker. The QUEST standardizer uses only the **class-I peptide-detail file** for MIRA epitope mapping (`scripts/data_processing/standardize/immunecode.py:240-243`); class-II minigene constructs are excluded.
- **Long-peptide minigenes (25+ AA).** Some MIRA experiments stimulate with minigenes that exceed natural epitope length. The standardizer filters peptides >25 AA away (`scripts/data_processing/standardize/immunecode.py:120-122`).
- **HLA from subject metadata is bag-of-alleles.** A subject typically has 2-12 HLA alleles; the QUEST standardizer assigns the **first** allele in the subject's list to a given peptide (`scripts/data_processing/standardize/immunecode.py:127-132`), which is an approximation. The actual restricting allele is unknown without further deconvolution.
- **Only SARS-CoV-2 antigens.** ImmuneCODE focuses exclusively on COVID-19; transferring patterns to non-CoV antigens may be limited.
- **Cross-reactivity with influenza.** Multiple papers document SARS-CoV-2-associated TCRs that cross-react with the immunodominant influenza M1 epitope GILGFVFTL ([Sidhom et al. 2020 bioRxiv](https://www.biorxiv.org/content/10.1101/2020.06.20.160499v1)); some MIRA-mapped TCRs may not be CoV-specific.
- **MIRA experiment design.** Peptide pools mean that the antigen attribution per TCR is at the **pool** level; deconvolution to single peptide depends on pool design and frequency at each replicate. Confidence is binarized into "high-confidence" — there is no per-record numeric score.
- **Cross-source overlap with immuneACCESS.** ImmuneCODE is hosted on the immuneACCESS portal; both sources should not be ingested without dedup.
- **Repertoire TSVs are bulk and unlabeled.** The Review/Repertoire TSVs (~38M rows across ~1,400 files) carry no epitope information and are bulk-repertoire only.

## Fidelity assessment
- **Confidence (1–5)**:
  - MIRA epitope-specific records: **4 (High)** — experimentally mapped TCR→peptide associations with multi-replicate confidence filtering. Caveat: HLA assignment is bag-of-alleles approximation.
  - Repertoire / Review records: **2 (Bulk/Indirect)** — no epitope labels, bulk TCR-β.
- **Recommended QUEST integration tier**:
  - MIRA records → **experimentally validated** (use for chain expansion + TCR-pMHC interaction training; treat HLA assignment with caution).
  - Repertoire / Review records → **bulk-repertoire** (use for MLM only).
- **Score / quality threshold**: peptide length ≤25 AA (`scripts/data_processing/standardize/immunecode.py:120-122`); class-I MIRA file only (`scripts/data_processing/standardize/immunecode.py:240-243`); productive frame filter on Review TSVs.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/immunecode.py`
- **Class**: `ImmunecodeStandardizer` (streaming = True due to ~38M Review rows)
<!-- TODO: filled by code-cross-reference -->
