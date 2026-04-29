# ADC (AIRR Data Commons / iReceptor)

## Identity
- **Source path**: `data/raw_data/databases/adc/`
- **Primary citation**: Corrie BD, Marthandan N, Zimonja B, et al. (2018). "iReceptor: A platform for querying and analyzing antibody/B-cell and T-cell receptor repertoire data across federated repositories." *Immunological Reviews* 284(1):24-41, [doi:10.1111/imr.12666](https://doi.org/10.1111/imr.12666) — [PubMed 29944754](https://pubmed.ncbi.nlm.nih.gov/29944754/)
- **ADC API specification**: Christley S, Aguiar A, Blanck G, et al. (2020). "The ADC API: A Web API for the Programmatic Query of the AIRR Data Commons." *Frontiers in Big Data* 3:22, [doi:10.3389/fdata.2020.00022](https://doi.org/10.3389/fdata.2020.00022) — [PubMed 33693395](https://pubmed.ncbi.nlm.nih.gov/33693395/)
- **AIRR standard**: <https://docs.airr-community.org/en/stable/api/adc_api.html> — AIRR Community Rearrangement Schema v1.x
- **Version / release**: Federated query result (per `checkpoint.json`); QUEST mirror dated 2026-01 spans ~20 federated repositories including iReceptor IPA, COVID-19 nodes, T1D, HPAP, VDJBase, VDJServer, DKFZ scireptor, Münster.
- **License**: Per-repository; most sources require citation of the original study DOI (recorded in `repertoires.json` study metadata). iReceptor itself is open-access for research.
- **Project URL**: <https://gateway.ireceptor.org/> ; <https://docs.airr-community.org/>

## Data shape
- **Chains present**: TRA, TRB, TRG, TRD (the standardizer drops gamma-delta — `_get_column_map_for_locus` returns `None` for non-α/β at `scripts/data_processing/standardize/adc.py:117-119`). Pairing via `cell_id` is preserved when present in source.
- **Epitope/MHC labelling**: **none in the rearrangements file**. Some studies provide phenotype/sample metadata in `repertoires.json` (e.g. cell sort, disease state) but no per-receptor epitope annotation.
- **Single-cell vs. bulk**: heterogeneous — some federated nodes carry bulk repertoires (e.g. Adaptive immunoSEQ-style), others carry single-cell (10x V(D)J).
- **Species filter**: ADC is human-focused; non-human nodes are excluded by repository selection.

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/databases/adc`
- **Size**: 1.5T (1,624,616,570,936 bytes)
- **Files**: 9296
- **Format breakdown**: `.tsv`=9259, `.json`=36, `.log`=1
- **Records**: ~2,055,697,412 (sampled) (95% CI: 185,162,871–3,926,231,882)
- **Counting method**: sampled (30 of 9259 tsv files, mean=222021.5 records/file, extrapolated; 175 files >2.0G excluded from sample)
- **Record definition**: json record (list element or document); tab/comma-delimited row excluding header
- **Standardizer**: `scripts/data_processing/standardize/adc.py` (`AdcStandardizer`)
- **Expected input files**: `*.tsv`

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `<9229 more tsv files (not sampled)>` | tsv | 1.5T | 2,049,036,731 | extrapolated from sample |
| `ipa5.ireceptor.org/rearrangements/96.tsv` | tsv | 1.6G | 2,987,213 | wc -l minus 1 (header) |
| `t1d-1.ireceptor.org/rearrangements/65b8098d866f210a329b2069.tsv` | tsv | 444.5M | 990,726 | wc -l minus 1 (header) |
| `covid19-4.ireceptor.org/rearrangements/5f491b246d45ecc67f6d343e.tsv` | tsv | 220.7M | 507,301 | wc -l minus 1 (header) |
| `covid19-1.ireceptor.org/rearrangements/613b7f45adf6058ccd84908e.tsv` | tsv | 208.9M | 136,060 | wc -l minus 1 (header) |
| `covid19-4.ireceptor.org/rearrangements/5f4c182ac7928edd1d536220.tsv` | tsv | 191.0M | 440,167 | wc -l minus 1 (header) |
| `t1d-2.ireceptor.org/rearrangements/68cdc80f2011973e4bfcdda6.tsv` | tsv | 147.8M | 294,459 | wc -l minus 1 (header) |
| `t1d-1.ireceptor.org/rearrangements/63641c36f5e323fae91fe0e1.tsv` | tsv | 139.1M | 279,515 | wc -l minus 1 (header) |
| `covid19-4.ireceptor.org/rearrangements/5f4c18cdc7928edd1d5362d5.tsv` | tsv | 129.0M | 297,092 | wc -l minus 1 (header) |
| `ipa1.ireceptor.org/rearrangements/72.tsv` | tsv | 124.2M | 175,140 | wc -l minus 1 (header) |
| `covid19-4.ireceptor.org/rearrangements/5f491b6c6d45ecc67f6d348f.tsv` | tsv | 70.8M | 163,124 | wc -l minus 1 (header) |
| `t1d-2.ireceptor.org/rearrangements/68cdc7fa2011973e4bfcdd0c.tsv` | tsv | 70.4M | 141,342 | wc -l minus 1 (header) |

**Non-data artifacts**: `checkpoint.json`, `download.log`

<!-- END: AUTO-INVENTORY -->

## Provenance & methodology
- **Experimental method**: aggregation of bulk and single-cell AIRR-seq experiments. Each repository node ingests AIRR-compliant Rearrangement TSVs and Repertoire metadata JSON. Productive vs. non-productive flag is preserved per AIRR schema.
- **Original cohort**: thousands of donors across hundreds of studies. Notable nodes:
  - **iReceptor Public Archive (IPA1-6)** — generic published-study deposits
  - **COVID-19 nodes (covid19-1..4)** — SARS-CoV-2 cohorts
  - **T1D nodes** — type-1 diabetes immune repertoires
  - **HPAP** — Human Pancreas Analysis Program
  - **scireptor.dkfz.de**, **agschwab.uni-muenster.de** — German repositories
  - **VDJserver.org**, **VDJBase** — academic mirrors
- **Curation pipeline**: AIRR Community-mandated data model (`Repertoire`, `Rearrangement`, `Receptor`, etc.). Data are uploaded by submitters with controlled vocabularies. iReceptor Gateway federates queries across nodes via the ADC API. The QUEST standardizer iterates `{repo}/rearrangements/{repertoire_id}.tsv` and resolves study IDs via `{repo}/repertoires.json`.
- **Updates / versioning history**: iReceptor launched 2018. AIRR Community has progressively expanded ADC participation; in 2023, ADC supported >338 unique users with >250,000 queries / >1.5 TB downloaded ([Antibody Society AIRR Data Commons page](https://www.antibodysociety.org/the-airr-community/airr-data-commons/)).

## Caveats & known issues
- **No epitope/MHC labels.** ADC is purely a repertoire resource — useful for MLM and TCR-only modeling but not for TCR-pMHC interaction training.
- **Productive-only filter.** The standardizer keeps only productive rearrangements (`productive == true/t/1` at `scripts/data_processing/standardize/adc.py:60-65`). Non-productive sequences are discarded — a sensible default for protein modeling but it removes biology that might be relevant for thymic selection / repertoire-shape modeling.
- **Repertoire metadata is uneven.** Some `repertoires.json` files carry rich phenotype annotation; others carry only de-identified subject IDs. Disease-state filtering at the per-receptor level is not always possible.
- **Massive scale (~3B rows).** This is by far the largest source in QUEST. Streaming with chunked CSV reading is mandatory; memory blow-ups are easy.
- **Locus column inconsistency.** Some submitters use `TRB`, others use `TCRB` or leave the field blank. The standardizer maps to upper-case `TRA`/`TRB` and skips other loci (`scripts/data_processing/standardize/adc.py:71-73`).
- **Cross-source overlap with immuneACCESS, immuneCODE, OTS.** Several Adaptive-deposited repertoires are mirrored across ADC and immuneACCESS; some 10x runs appear in both ADC and OTS. Cross-source dedup is required.
- **Federated availability changes over time.** Some nodes go offline or change URLs; the snapshot directory listing is the ground truth at QUEST's pull date.
- **Productive flag and frame_type semantics differ across nodes.** Some nodes use `frame_type == "In"` (Adaptive convention), others use `productive == "T"` (AIRR standard).

## Fidelity assessment
- **Confidence (1–5)**: **2 (Bulk/Indirect)** as a TCR-pMHC source (no epitope labels). As a TCR-language-model corpus, fidelity is **3-4**: heterogeneous quality but enormous scale and broad donor diversity.
- **Recommended QUEST integration tier**: **bulk-repertoire** (per CLAUDE.md framework). Use for MLM and TCR-only training (tra, trb, tra+trb where paired). Do NOT use for TCR-pMHC interaction training.
- **Score / quality threshold**: productive-only filter (`scripts/data_processing/standardize/adc.py:60-65`); locus restricted to TRA/TRB; no additional confidence threshold.

## Standardizer
- **Implementation**: `scripts/data_processing/standardize/adc.py`
- **Class**: `AdcStandardizer` (streaming = True due to ~3B rows)
<!-- TODO: filled by code-cross-reference -->
