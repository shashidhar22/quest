# NetMHCpan

## Identity
- **Source path**: `data/raw_data/databases/NetMHCPan/`
- **Primary citation**: Reynisson B, Alvarez B, Paul S, Peters B, Nielsen M (2020). "NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation by concurrent motif deconvolution and integration of MS MHC eluted ligand data." *Nucleic Acids Research* 48(W1):W449-W454, [doi:10.1093/nar/gkaa379](https://doi.org/10.1093/nar/gkaa379) — [PubMed 32406916](https://pubmed.ncbi.nlm.nih.gov/32406916/)
- **NetMHCpan 4.0**: Jurtz V, Paul S, Andreatta M, Marcatili P, Peters B, Nielsen M (2017). "NetMHCpan-4.0…" *J Immunol* 199(9):3360-3368, [doi:10.4049/jimmunol.1700893](https://doi.org/10.4049/jimmunol.1700893)
- **Version / release**: Local archives `NetMHCpan_train.tar.gz` and `NetMHCIIpan_train.tar.gz` correspond to the public training data of NetMHCpan-4.1 (class I) and NetMHCIIpan-4.0 (class II). ~34M peptide-allele pairs.
- **License**: Academic-use; the NetMHCpan suite is distributed under the DTU HealthTech "non-commercial academic" license (<https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/>). Re-distribution of the training data is conditioned on citation.
- **Project URL**: <https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/> ; <https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.0/>

## Data shape
- **Chains present**: none. This is **peptide-MHC binding data only**, no TCR.
- **Epitope/MHC labelling**: every row is a (peptide, HLA-allele) pair. Two assay types are encoded:
  - **`_ba`** files: quantitative binding-affinity measurements (1 - log50k(IC50nM) target)
  - **`_el`** files: mass-spectrometry-eluted ligand (positive presentation events) plus computational decoys
- **Single-cell vs. bulk**: not applicable. Source data are biochemical assays (IC50) and MHC-immunoprecipitation MS experiments.
- **Species filter**: standardizer filters to human alleles only (`mhc_one` / `mhc_two` must start with `HLA-`); BoLA, DLA, SLA, H-2, Mamu, etc. are dropped (`scripts/data_processing/standardize/netmhcpan.py:79-84`).

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/NetMHCPan`
- **Size**: 1.7G (1,818,965,719 bytes)
- **Files**: 36
- **Format breakdown**: `.txt`=21, `.noext`=11, `.tar.gz`=2, `.dat`=2
- **Records**: 34,040,203 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: line; peptide-MHC binding/eluted-ligand row; sum of records in archived csv/tsv/fasta members
- **Standardizer**: `scripts/data_processing/standardize/netmhcpan.py` (`NetmhcpanStandardizer`)
- **Expected input files**: `NetMHCpan_train`, `NetMHCIIpan_train`, `c*_ba`, `c*_el`, `train_*.txt`, `test_*.txt`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `NetMHCIIpan_train.tar.gz` | tar.gz | 371.3M | — | tar member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `NetMHCIIpan_train/train_EL2.txt` | txt | 156.2M | 3,269,040 | wc -l (no header) |
| `NetMHCIIpan_train/train_EL1.txt` | txt | 156.1M | 3,266,452 | wc -l (no header) |
| `NetMHCIIpan_train/train_EL5.txt` | txt | 156.0M | 3,266,371 | wc -l (no header) |
| `NetMHCIIpan_train/train_EL3.txt` | txt | 156.0M | 3,265,194 | wc -l (no header) |
| `NetMHCIIpan_train/train_EL4.txt` | txt | 155.9M | 3,263,499 | wc -l (no header) |
| `NetMHCpan_train.tar.gz` | tar.gz | 87.1M | — | tar member inspection; no tabular members |
| `NetMHCpan_train/c003_el` | noext | 53.5M | 2,579,638 | wc -l (no header) |
| `NetMHCpan_train/c002_el` | noext | 53.5M | 2,577,697 | wc -l (no header) |
| `NetMHCpan_train/c000_el` | noext | 53.4M | 2,574,792 | wc -l (no header) |
| `NetMHCpan_train/c004_el` | noext | 53.3M | 2,569,713 | wc -l (no header) |
| `NetMHCpan_train/c001_el` | noext | 53.2M | 2,566,453 | wc -l (no header) |

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: aggregation across hundreds of published binding-affinity assays (radiolabeled peptide competition, fluorescence polarization) and MS eluted-ligand datasets from immunoprecipitated MHC complexes. Eluted ligands include both **single-allele** (SA) cell lines (where the allele is known) and **multi-allele** (MA) cell lines (where the allele must be deconvoluted via NNAlign_MA).
- **Original cohort**: more than 850,000 peptide measurements covering hundreds of class-I and class-II alleles. Class-I corpus is much larger and more allele-balanced than class-II.
- **Curation pipeline**: Nielsen lab (DTU) integrates published BA and EL datasets; assigns sample IDs to MA samples and uses NNAlign_MA to learn allele-specific motifs; releases training data alongside trained model weights. Data files are space-separated (`c000_ba`, `c000_el`, …) for class I and tab-separated (`train_BA*.txt`, `test_BA*.txt`) for class II.
- **Updates / versioning history**: NetMHCpan-1.0 (2007); 4.0 (2017, [PubMed 28978689](https://pubmed.ncbi.nlm.nih.gov/28978689/)) integrated EL data; 4.1 (2020) added concurrent motif deconvolution. NetMHCIIpan parallel evolution. As of 2026, NetMHCpan-4.1 / NetMHCIIpan-4.0 training data remains the publicly released corpus.

## Caveats & known issues
- **Multi-allele rows are ambiguous.** EL data from MA cell lines lists multiple HLA alleles per peptide; QUEST's standardizer detects comma-separated allele lists and stores the peptide with empty `mhc_one`/`mhc_two`, treating them as peptide-only data (`scripts/data_processing/standardize/netmhcpan.py:67-71`). Without deconvolution, attributing a single allele to a multi-allele peptide is unreliable.
- **HLA allele coverage gaps.** Per [Frontiers in Immunology, "Evaluating NetMHCpan performance on non-European HLA alleles"](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2023.1288105/full), seven HLA alleles (A*24:07, A*34:01, A*34:02, A*36:01, C*03:02, C*04:03, C*14:03) have no representation in the training data. African and Asian alleles are systematically underrepresented.
- **Decoys are computationally generated negatives.** `_el` files include random-protein-derived negatives at ~1:99 positive-to-decoy ratio. These are not experimentally validated non-binders — they are length-matched random peptides from the proteome.
- **No TCR data.** Despite being a core "immunology" resource, NetMHCpan training data is purely peptide-MHC. It must NOT be used as evidence of TCR specificity.
- **Cross-source overlap with IEDB.** The NetMHCpan training corpus is largely sourced from IEDB MHC-binding records; downstream pipelines that ingest both datasets must dedup to avoid double-counting.
- **Train/test definition is intrinsic to the dataset.** Files named `c000`, `c001`, … are the published 5-fold cross-validation splits. Re-using these files without honoring the original splits will leak training data into evaluation.
- **Class-II training data uses tab-separated TSVs** with a different schema from class-I (`train_BA*.txt`); peptide is column 0, allele is column 2 (`scripts/data_processing/standardize/netmhcpan.py:177-184`).

### Note on the inventory record count

The NetMHCpan source ships **both** the `.tar.gz` archives **and** the unpacked training-data directories alongside them (`NetMHCpan_train.tar.gz` + `NetMHCpan_train/`, `NetMHCIIpan_train.tar.gz` + `NetMHCIIpan_train/`). The auto-inventory only counts the unpacked side; the archives are flagged as redundant. Additionally, the class-I training files (`c000_ba`, `c000_el`, …) are extension-less plain-text tabular files — they are now counted via `wc -l` rather than being treated as opaque binary as in the first pass of the inventory.

## Fidelity assessment
- **Confidence (1–5)**: **1-2 (Predicted/Synthetic blended with Experimental)** for TCR purposes — but **4** for peptide-MHC modeling. The BA portion is experimentally measured (high quality); the EL portion is experimentally measured but allele-deconvoluted via ML; the decoys are synthetic.
- **Recommended QUEST integration tier**: **pMHC-only** (per CLAUDE.md framework). Use for peptide-MHC modeling (pMHC seq2seq, contrastive pMHC learning, MHC binding prediction). Do NOT use for TCR interaction training.
- **Score / quality threshold**: filter to single, unambiguous human HLA allele (`scripts/data_processing/standardize/netmhcpan.py:67-87`); rows with multi-allele labels are kept as peptide-only entries; non-human alleles are dropped.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/netmhcpan.py`
- **Class**: `NetmhcpanStandardizer` (streaming = True due to ~34M rows)
<!-- TODO: filled by code-cross-reference -->
