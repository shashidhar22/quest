# Fidelity Rubric for QUEST Raw Data Sources

Every source in the [Raw Data Manifest](README.md) carries a numeric fidelity score from **1 (lowest)** to **5 (highest)**. The score reflects how confident we should be when using the source for *TCR–pMHC interaction* training, which is the most demanding training mode in QUEST.

The score is intentionally one number — sources with mixed-quality subsets get the score of their *best usable subset*, with caveats listed in the per-source page.

## Scale

| Score | Label | Criteria |
|-------|-------|----------|
| **5** | **Gold** | Direct experimental TCR–pMHC validation: paired-chain TCR sequence + epitope + MHC, identified via tetramer/dextramer staining, MIRA, or single-cell paired sequencing with antigen capture. Peer-reviewed primary publication. Multi-replicate or large cohort. *Examples:* VDJdb records with `score = 3`, MIRA-derived ImmuneCODE pairs, ZEN7555405 (vaccine dextramer-paired). |
| **4** | **High** | Experimentally validated TCR–pMHC interaction with at least one method-specific caveat (e.g., bulk MHC tetramer enrichment without single-cell pairing, ELISpot stimulation without direct binding measurement), **OR** a quality-scored database with a strict threshold applied. *Examples:* VDJdb `score >= 1`, McPAS-TCR validated-method records, BATMAN mutational scans, GSE114724-class single-cell paired TIL repertoires with disease metadata. |
| **3** | **Medium** | Curated/aggregated mixed-evidence sets: positive T-cell assays without direct binding measurement, computational labels with experimental backing, or multi-source aggregations that lump assay types. *Examples:* IEDB `outcome = Positive` T-cell assays, CEDAR cancer records, TRAIT integrated-omics records. |
| **2** | **Bulk / Indirect** | Large-N bulk repertoires with no epitope/MHC linkage. Useful only for masked-language modelling (MLM) and TCR-only training modes. *Examples:* Observed TCR Space (OTS), immuneACCESS, AIRR Data Commons (ADC), tcrdb, repertoire-only studies. |
| **1** | **Predicted / Synthetic** | Computationally generated, NN training sets with intrinsic-prediction labels, or synthetic data. Useful only for the modeling task they were designed for (e.g., NetMHCpan training data is valid for pMHC-binding prediction but **not** for TCR interaction training). |

## How the score is applied in QUEST

The fidelity score maps to one of the integration tiers defined in [`CLAUDE.md`](../../../CLAUDE.md) under "Integration Decision Framework":

| Fidelity | QUEST integration tier | Used for MLM | Used for TCR-pMHC interaction training |
|----------|------------------------|--------------|----------------------------------------|
| 5 | experimentally validated | ✅ | ✅ — expand into all permutations (TRA, TRB, paired, peptide, MHC, full interaction) |
| 4 | quality-scored / experimentally validated | ✅ | ✅ above the score threshold; below threshold falls to MLM-only |
| 3 | mixed | ✅ | ✅ for the validated subset; weighted lower in loss or used for finetuning only |
| 2 | bulk repertoire | ✅ | ❌ — no epitope to supervise interaction |
| 1 | pMHC-only / predicted | ✅ for peptide-MHC modelling | ❌ — never used as TCR-pMHC supervision |

## What the score does *not* capture

- **Statistical power**: a Gold (5) source with N=20 is still small. Sample size is reported separately in each per-source page.
- **Bias**: most TCR–pMHC datasets concentrate on HLA-A\*02:01 and a small set of viral epitopes. The score reflects *evidence quality per record*, not coverage.
- **Reproducibility of upstream curation**: scores are not adjusted for downstream-paper criticisms; those are listed in the "Caveats & known issues" section of each per-source page.
- **License**: orthogonal to fidelity; recorded separately.

## Reviewing the rubric

The rubric is intended to be applied conservatively: when in doubt, score lower. To revise a score for an existing source, edit its per-source page and update the master table in [README.md](README.md). Score changes should be justified in the page's "Caveats" section.

When a *new* source is added to `data/raw_data/`, the [Input Data Curator role in CLAUDE.md](../../../CLAUDE.md) describes the workflow for assigning a score.
