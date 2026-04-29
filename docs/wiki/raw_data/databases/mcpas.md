# McPAS-TCR

## Identity
- **Source path**: `data/raw_data/databases/McPAS-TCR/`
- **Primary citation**: Tickotsky N, Sagiv T, Prilusky J, Shifrut E, Friedman N (2017). "McPAS-TCR: a manually curated catalogue of pathology-associated T cell receptor sequences." *Bioinformatics* 33(18):2924-2929, [doi:10.1093/bioinformatics/btx286](https://doi.org/10.1093/bioinformatics/btx286) — [PubMed 28481982](https://pubmed.ncbi.nlm.nih.gov/28481982/)
- **Version / release**: Single CSV (`McPAS-TCR.csv`); the upstream DB updates continuously without explicit versioning. Last public Friedman-lab update reported as 2022-09-10 prior to QUEST download.
- **License**: Free for academic use; no formal license file shipped (Friedman lab terms of use on the website).
- **Project URL**: <https://friedmanlab.weizmann.ac.il/McPAS-TCR/>

## Data shape
- **Chains present**: TRA, TRB, paired alpha-beta when available. Wide format (`CDR3.alpha.aa`, `CDR3.beta.aa`, `TRAV`, `TRAJ`, `TRBV`, `TRBD`, `TRBJ`).
- **Epitope/MHC labelling**: `Epitope.peptide` and `MHC` are present for most rows. Pathology and category metadata (`Pathology`, `Category`) are also captured.
- **Single-cell vs. bulk**: heterogeneous, depending on source publication. Mostly tetramer-sorted bulk TCR-seq with a small fraction of single-cell paired data.
- **Species filter**: standardizer keeps only `Species == "human"` (`scripts/data_processing/standardize/mcpas.py:62-66`); upstream DB includes mouse.

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/McPAS-TCR`
- **Size**: 8.2M (8,607,789 bytes)
- **Files**: 2
- **Format breakdown**: `.csv`=1, `.sh`=1
- **Records**: 40,779 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header
- **Standardizer**: `scripts/data_processing/standardize/mcpas.py` (`McpasStandardizer`)
- **Expected input files**: `McPAS-TCR.csv`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `McPAS-TCR.csv` | csv | 8.2M | 40,779 | wc -l minus 1 (header) |

**Non-data artifacts**: `download_mcpas.sh`

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: aggregation of published TCR-antigen associations. Methods recorded in `Antigen.identification.method` include tetramer/dextramer staining, in-vitro stimulation, ELISpot, ICS, and cytotoxicity assays. The QUEST standardizer treats only this whitelist as `binding="pos"`; rows with weaker or missing method annotations are kept but left unflagged.
- **Original cohort**: ~5,000 sequences spanning autoimmunity (T1D, MS, RA, etc.), oncology (various neoantigens), pathogen infections (CMV, EBV, Influenza, HIV, HCV, SARS-CoV-2), and allergy. Donor counts inherit from each published study.
- **Curation pipeline**: manual curation by the Friedman lab; the website provides a free-text search and downloadable CSV. No automated submission interface; the curators read the literature and transcribe records.
- **Updates / versioning history**: 2017 launch in *Bioinformatics*; ad-hoc updates (no semantic versioning) typically once or twice per year. The TCRMatch paper ([Chronister et al. 2021, doi:10.3389/fimmu.2021.640725](https://doi.org/10.3389/fimmu.2021.640725), [PubMed 33777034](https://pubmed.ncbi.nlm.nih.gov/33777034/)) treats McPAS as one of three primary input sources for TCR specificity prediction.

## Caveats & known issues
- **No confidence scoring.** Unlike VDJdb, McPAS does not stamp records with a confidence level — every entry is presented as equally trustworthy regardless of upstream method. The QUEST standardizer compensates by deriving `binding="pos"` only when `Antigen.identification.method` is in a whitelist of validated assays (`scripts/data_processing/standardize/mcpas.py:30-41`).
- **Encoding quirks.** The CSV ships in `latin-1` and contains stray non-ASCII characters; the standardizer reads it with `encoding="latin-1"`.
- **Multi-epitope rows.** `Epitope.peptide` sometimes contains `/`-separated peptides (epitope mapping ambiguity); QUEST's standardizer explodes these into separate rows.
- **Heavy viral skew and HLA-A*02:01 dominance.** Independent benchmarking notes that ~97% of antigens reported as binding to a TCR in McPAS/VDJdb-class databases are viral, with 70% of pairs concentrated in ~100 antigens; class-I HLA-A*02:01 is over-represented. See [Hudson et al., "T-cell receptor binding prediction: A machine learning revolution"](https://www.sciencedirect.com/science/article/pii/S2667119024000107).
- **Limited paired-chain coverage.** McPAS predominantly contains CDR3β-only entries; <4% of records have full paired alpha+beta CDR3 information ([T-cell receptor binding prediction review](https://www.sciencedirect.com/science/article/pii/S2667119024000107)).
- **Models trained on McPAS underperform VDJdb-trained equivalents.** A repeated finding across IMMREP benchmarks ([Sidhom et al., IMMREP22](https://www.sciencedirect.com/science/article/pii/S2667119023000046)) — likely a combination of smaller training set and looser curation.
- **Cross-source overlap.** McPAS heavily overlaps with VDJdb and IEDB curation; deduplication in QUEST's pipeline is required.

### Note on the inventory record count

The original 2017 paper reports "~5,000 sequences." The auto-inventory above counts ~40K records — both numbers are correct for their interpretation. McPAS's source CSV uses `/`-separated lists in the `Epitope.peptide` column to encode multi-epitope ambiguity (one TCR observed against several peptides). The QUEST standardizer ([mcpas.py](../../../scripts/data_processing/standardize/mcpas.py)) explodes these lists into one row per (TCR, peptide), which is the natural unit for downstream training. So **rows in the standardized output ≈ 40K**, **unique source TCRs ≈ 5K**.

## Fidelity assessment
- **Confidence (1–5)**: **3 (Medium)** — manually curated and peer-reviewed, but no per-record confidence score, modest paired-chain coverage, and substantial epitope/HLA bias. Records derived from tetramer/multimer/in-vitro stimulation are higher quality (effective fidelity 4); the rest are lower.
- **Recommended QUEST integration tier**: **mixed**. Records with validated antigen-identification methods (tetramer, dextramer, multimer, stimulation, ELISpot, ICS, cytotoxicity) → experimentally validated tier (full chain + interaction expansion); other records → use chains for MLM only.
- **Score / quality threshold**: validated-method whitelist applied in `scripts/data_processing/standardize/mcpas.py:30-41`; rows in whitelist receive `binding="pos"`, others left empty.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/mcpas.py`
- **Class**: `McpasStandardizer`
<!-- TODO: filled by code-cross-reference -->
