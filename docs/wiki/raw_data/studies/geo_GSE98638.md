# geo/GSE98638

## Identity
- **Source path**: `data/raw_data/studies/geo/GSE98638/`
- **Accession**: GEO GSE98638
- **Primary citation**: Zheng C et al., 2017, "Landscape of Infiltrating T Cells in Liver Cancer Revealed by Single-Cell Sequencing", Cell, doi:10.1016/j.cell.2017.05.035 — [PubMed](https://pubmed.ncbi.nlm.nih.gov/28622514/)
- **Submission date**: 2017-05-08
- **License**: GEO public domain

## Data shape
- **Organism**: Homo sapiens
- **Assay types present in directory**: scRNA-seq (Smart-seq2; TCR reconstructed from full-length reads)
- **Platform**: Illumina HiSeq 2500
- **Sample N**: 6 patients (~5,000 T cells)
- **Disease / condition**: Hepatocellular carcinoma

## Raw inventory
<!-- TODO: filled by Quantifier agent from inventory.json -->

## Study summary
- **Goal**: Landscape of HCC tumor-infiltrating T cells.
- **Method**: Smart-seq2 scRNA-seq of T cells from blood, normal, and tumor tissues.
- **Findings**: Identified exhausted CD8 and Treg dynamics; TCR clonotype tracking via full-length reads.

## Caveats / notes
- TCR is reconstructed from Smart-seq2 reads (not native VDJ kit).

## Fidelity assessment
- **Confidence (1-5)**: 4 — paired scTCR/GEX (reconstructed) with HCC labels.
- **QUEST integration tier**: single-cell-paired
