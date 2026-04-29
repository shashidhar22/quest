# geo

## Identity
- **Source path**: `data/raw_data/studies/geo/`
- **Accession**: container directory (multiple GEO accessions, see below)
- **Primary citation**: per-accession (see individual files)
- **Submission date**: download attempts run 2026-02-12 (see `download_summary.txt`)
- **License**: GEO public domain

## Contents

This directory contains GEO data from the legacy `download_dataset.py` orchestrator (see `studies/download_dataset.py`, `studies/download_summary.txt`, `studies/manifest.tsv`). It is a separate workspace from the per-study top-level directories.

GEO accessions present (each documented as individual `geo_GSE*.md` files):

| Accession | Citation (short) |
|-----------|------------------|
| GSE108989 | Zhang 2018 — CRC T-cell lineage tracking |
| GSE114727 | Azizi 2018 — Breast TME (SuperSeries of GSE114724) |
| GSE114944 | Duhen 2018 — CD39+CD103+ CD8 TILs |
| GSE115978 | Jerby-Arnon 2018 — Melanoma scRNA + ICI resistance |
| GSE121082 | Tu 2019 — TCR-beta in HAM/TSP and MS |
| GSE123139 | Li 2019 — Dysfunctional CD8 T cells in melanoma |
| GSE123814 | Yost 2019 — PD-1 blockade BCC/SCC SuperSeries |
| GSE132810 | Kim 2019 — 4-1BB+PD-1-high CD8 TILs in HCC |
| GSE141878 | Yang 2020 — Methylation in CRC tumor-reactive CD8 |
| GSE145596 | Liu 2020 — FucoID antigen-reactive TIL identification (mouse) |
| GSE215120 | Zhang 2022 — Acral melanoma scRNA + TCR |
| GSE222448 | Barras 2024 — TIL adoptive therapy in melanoma |
| GSE225984 | Zhang 2023 — SEQTR TCR-seq method paper |
| GSE59455 | Brown 2016 — FFPE melanoma transcriptome |
| GSE72056 | Tirosh 2016 — Smart-seq2 melanoma scRNA |
| GSE98638 | Zheng 2017 — HCC TIL scRNA |
| GSE99254 | Guo 2018 — NSCLC TIL scRNA |

## Caveats
- Several accessions overlap with the top-level studies/ directory (e.g. GSE114724/GSE114727 are paired).
- Many of the matrix/suppl downloads were partial or failed — see `manifest.tsv`.
- Multi-organism series (e.g. GSE145596 mouse-only, GSE225984 mixed) require filtering before TCR ingestion.

## Fidelity assessment
- **Confidence (1-5)**: 3 — directory contains a mix of bulk and single-cell TCR/GEX from peer-reviewed studies; per-accession quality varies.
- **QUEST integration tier**: mixed (per-accession).
