# IPD-IMGT/HLA

## Identity
- **Source path**: `data/raw_data/databases/IMGTHLA/`
- **Primary citation**: Robinson J, Barker DJ, Marsh SGE (2024). "25 years of the IPD-IMGT/HLA Database." *HLA* 103(6):e15549, [doi:10.1111/tan.15549](https://doi.org/10.1111/tan.15549) — [PubMed 38936817](https://pubmed.ncbi.nlm.nih.gov/38936817/)
- **2026 update**: Barker DJ, Natarajan RHL, Cooper MA, Hopper SJF, Yates AD, Parham P, Marsh SGE, Robinson J (2026). "The IPD-IMGT/HLA database: recent developments in sequence submission." *Nucleic Acids Research* 54(D1):D1152-D1158, [doi:10.1093/nar/gkaf1218](https://doi.org/10.1093/nar/gkaf1218) — [PubMed 41251166](https://pubmed.ncbi.nlm.nih.gov/41251166/)
- **Foundational citation**: Robinson J, Malik A, Parham P, Bodmer JG, Marsh SGE (2000). "IMGT/HLA - a sequence database for the human major histocompatibility complex." *Tissue Antigens* 55:280-287
- **Version / release**: `Alignments_Rel_3580.zip` — IPD-IMGT/HLA Release **3.58.0** (April 2024). Per `README.md` line 124, version is encoded as `<DB Name> <nomenclature>.<quarterly release>.<sequence version> <date> <commit>`.
- **License**: Creative Commons Attribution-NoDerivs (CC BY-ND) per `LICENCE.md` and README copyright notice. Mirroring is explicitly discouraged; modified data redistribution requires permission from hla@alleles.org.
- **Project URL**: <https://www.ebi.ac.uk/ipd/imgt/hla/> ; GitHub mirror <https://github.com/ANHIG/IMGTHLA>

## Data shape
- **Chains present**: none (this is an MHC reference, not a TCR source). Provides protein sequences for HLA class-I (A, B, C, E, F, G) and class-II (DRA, DQA1, DPA1, DRB1-5, DQB1, DPB1) alleles.
- **Epitope/MHC labelling**: HLA allele names at full 4-digit (and higher) resolution, e.g. `HLA-A*02:01`. The QUEST standardizer collapses to 4-digit allele resolution and stores the protein sequence in `mhc_one` (class-I and class-II alpha) or `mhc_two` (class-II beta).
- **Single-cell vs. bulk**: not applicable — reference allele sequences only.
- **Species filter**: human-only by definition (HLA = human leukocyte antigen).

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/IMGTHLA`
- **Size**: 4.1G (4,362,083,014 bytes)
- **Files**: 549
- **Format breakdown**: `.txt`=189, `.fasta`=110, `.pir`=105, `.msf`=105, `.zip`=6, `.md`=5, `.xsd`=2, `.csv`=1, `.noext`=1
- **Records**: 7,500,331 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: FASTA entry (>header); line; sum of records in archived csv/tsv/fasta members; tab/comma-delimited row excluding header
- **Standardizer**: `scripts/data_processing/standardize/imgthla.py` (`ImgthlaStandardizer`)
- **Expected input files**: `hla_prot.fasta`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `alignments/ClassI_nuc.txt` | txt | 80.4M | 571,828 | wc -l (assumed no header) |
| `msf/B_gen.msf` | msf | 38.8M | — | binary/non-tabular; not counted |
| `msf/C_gen.msf` | msf | 36.6M | — | binary/non-tabular; not counted |
| `hla_nuc.fasta` | fasta | 35.3M | 41,428 | grep -c '^>' |
| `fasta/hla_nuc.fasta` | fasta | 35.3M | — | grep -c '^>' |
| `alignments/B_gen.txt` | txt | 35.0M | 281,079 | wc -l (assumed no header) |
| `alignments/C_gen.txt` | txt | 33.0M | 266,027 | wc -l (assumed no header) |
| `hla.dat.zip` | zip | 29.8M | — | zip member inspection; no tabular members |
| `msf/A_gen.msf` | msf | 28.5M | — | binary/non-tabular; not counted |
| `alignments/A_gen.txt` | txt | 25.8M | 208,609 | wc -l (assumed no header) |
| `xml/hla_ambigs.xml.zip` | zip | 24.4M | — | zip member inspection; no tabular members |
| `Allelelist_history.txt` | txt | 21.9M | — | non-tabular metadata txt |

**Non-data artifacts**: `.gitattributes`, `LICENCE.md`, `Manual.md`, `README.md`, `ihiw/README.md`, `wmda/README.md`

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: HLA alleles are curated from sequencing submissions to the WHO Nomenclature Committee for Factors of the HLA System. Sequences come from Sanger sequencing, NGS, and increasingly from long-read full-gene sequencing. Curators verify each submission against published references.
- **Original cohort**: As of the 2026 update, >43,000 unique alleles from 47 genes recognized by the WHO Nomenclature Committee. Submissions have been tracked through >28,000 records over 26 years.
- **Curation pipeline**: Submissions go through the IPD-IMGT/HLA Submission Tool (replaced after 26 years per the 2026 NAR paper); each is validated against existing alleles, given a unique allele name following WHO nomenclature, and assigned a sequence version. ENA brokering is now offered to reduce duplicate submissions.
- **Updates / versioning history**: Founded 1998 ("HLA Informatics Group of the Anthony Nolan Research Institute"). Quarterly major releases since 2010 (January, April, July, October). Move from Git-LFS to ZIP archives for files >100MB at Release 3.56.0 (April 2024) per `README.md`. Hosted at EBI / Anthony Nolan Research Institute.

## Caveats & known issues
- **Common alleles dominate well-characterized data; rare alleles are partial sequences.** Many class-I alleles have full genomic sequences while uncommon alleles have only the cDNA / coding-region sequence. The standardizer keeps the longest available sequence per 4-digit allele (`scripts/data_processing/standardize/imgthla.py:67-69`).
- **Allele name ambiguity at low resolution.** Records like `HLA-A*02:01` actually correspond to many high-resolution alleles (`A*02:01:01:01`, `A*02:01:01:02L`, etc.) that may differ at silent positions or signal-peptide residues. Collapsing to 4-digit gives a single representative protein but masks rare variants.
- **Class-II beta chains require pairing with alpha.** The standardizer stores DRB/DQB/DPB sequences in `mhc_two` and DRA/DQA/DPA in `mhc_one`, but a complete class-II MHC requires the cognate heterodimer; pairing is not enforced at this stage.
- **Sequence updates can change content of an allele name.** Per the 2024 25-year-anniversary paper, naming committees occasionally split an allele into two when new variants are found — old training data may end up referring to a deprecated name.
- **Deleted alleles.** `Deleted_alleles.txt` is present and lists alleles withdrawn from the database; the standardizer does not currently filter against this list.
- **No T-cell or peptide-binding data.** Use of IMGTHLA is solely for MHC-sequence reference; do NOT use it as evidence of TCR-MHC binding.

### Note on the inventory record count

IMGTHLA distributes some FASTA files at **two paths** for compatibility (`hla_nuc.fasta` at the source root and `fasta/hla_nuc.fasta` inside the `fasta/` subdir; same for `hla_prot.fasta` and several gene-specific FASTAs). The auto-inventory dedups these by basename + size — only the first occurrence is counted toward `total_records`. Records reported reflect unique FASTA entries, not on-disk file copies.

## Fidelity assessment
- **Confidence (1–5)**: **5 (Gold)** as a reference of HLA protein sequences — peer-reviewed, WHO-sanctioned nomenclature, continuously curated since 1998. Not applicable as TCR-pMHC training data.
- **Recommended QUEST integration tier**: **MHC-reference**. Used to populate `mhc_one`/`mhc_two` protein sequences for downstream MHC-aware modeling (e.g. MHC pseudosequence construction, full-length MHC inputs). Not used for TCR or pMHC interaction training directly.
- **Score / quality threshold**: keep first or longest protein sequence per 4-digit allele (`scripts/data_processing/standardize/imgthla.py:67-69`); class-II beta → `mhc_two`, others → `mhc_one`.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/imgthla.py`
- **Class**: `ImgthlaStandardizer`
<!-- TODO: filled by code-cross-reference -->
