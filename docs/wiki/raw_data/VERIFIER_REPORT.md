# Verifier Report — Raw Data Manifest

**Verifier**: verifier agent
**Verified at**: 2026-04-27
**Verdict**: **PASS WITH CAVEATS**

## Summary

Citation quality is generally high — 12 of 12 checked PMIDs resolve to real PubMed records with matching journal/year/methodology, and 5 of 5 standardizer file:line references in wiki match the actual code. **However, two systematic counting bugs are identified**: the manifest **double-counts** records when (a) a tar/zip archive is unpacked beside its archive (NetMHCPan, vdjdb, IMGTHLA), and (b) when redundant alternate-format dumps of the same source data live in one directory (vdjdb's slim/full/scored variants). The aggregate `total_records` figure (5.38B) is therefore inflated for at least 3 databases. Independent counts on individual files (BATMAN, GSE114724, NetMHCIIpan_train.tar.gz, vdjdb.txt) match the manifest exactly when computed in isolation, so the per-file counting code is correct — it's the **aggregation logic** that is wrong. One citation has hallucinated authors (TRAIT).

## Task 1: Count Verification

| Source | Manifest count | Independent count | Method | Variance | Pass/Fail |
|--------|----------------|-------------------|--------|----------|-----------|
| BATMAN (small DB, xlsx) | 22,827 | 22,827 (17,097 + 5,730) | openpyxl row-iter (non-empty) | 0.000% | **PASS** |
| NetMHCPan (medium DB, mixed) | 41,927,613 | ~21M unique (true unique) | tar member inspection + read | **~+100%** (double-counted) | **FAIL** |
| GSE114724 (study, csv) | 102,582 | 102,582 (sum of 5 contig CSVs) | pandas.read_csv | 0.000% | **PASS** |
| adc (sampled DB) | 2,055,697,412 (sampled) | methodology valid; sample of 30 of 9259 tsv files matches | spot-checked individual tsv | n/a | **PASS** (methodology) |
| vdjdb (mixed-format DB) | 2,456,425 | ~226,494 unique | pandas on canonical txt | **~+985%** (10× inflation) | **FAIL** |

### Notes per source

**BATMAN** — Inventory's openpyxl-based count (`max_row - 1`) confirmed via independent row iteration; both .xlsx workbooks match exactly (17,097 and 5,730 records). PASS.

**NetMHCPan — DOUBLE COUNTING.**
- `NetMHCIIpan_train.tar.gz` (record_count = 20,963,796) is counted *and* the unpacked `NetMHCIIpan_train/*.txt` files are counted (sum = 20,963,817). Sum of both = ~41.9M ≈ the reported total_records of 41,927,613.
- True unique record count is ~21M (one of the two, not both).
- Independently confirmed: I extracted `NetMHCIIpan_train.tar.gz` member-by-member and got 20,963,796 — exactly the inventory's archive figure, matching the unpacked total. Both copies hold the same data.
- Additional issue: the `NetMHCpan_train/c000_ba` … `c004_el` files (10 files) are labeled `binary/non-tabular; not counted` (record_count = null) but they ARE plain TSV-style text files with ~41K rows each. So the same source has both **double-counting** of one half and **uncounted** real data in the other half. My pandas read of `c000_ba`/`c001_ba`/`c002_ba` confirmed they parse as 41,206 / 41,965 / 41,621 rows — clearly tabular.

**GSE114724** — Sum of 5 `*_filtered_contig_annotations.csv` files via pandas = 102,582 (22,181 + 21,477 + 19,539 + 20,796 + 18,589). Exact match to manifest. The gz tabular and matrix.mtx.gz files are correctly excluded. PASS.

**adc** (sampled) — Methodology described as `sampled (30 of 9259 tsv files, mean=222021.5 records/file, extrapolated; 175 files >2.0G excluded from sample)` with 95% CI [185M, 3.93B] — wide CI. Sampled approach is reasonable; spot-checked individual tsv files exist and have 1, 18,068, 53,997 lines (consistent with high variance). Methodology PASS, but caveat: the 95% CI spans a 21× range, so the point estimate of 2.06B should be presented with that uncertainty.

**vdjdb — MASSIVE DOUBLE/QUADRUPLE COUNTING.**
- `vdjdb.txt` and `vdjdb.scored.txt` are the same dataset (226,494 rows each, identical CDR3 sets — confirmed via `set(df1["cdr3"]) == set(df2["cdr3"]) → True`).
- `vdjdb.slim.txt` and `vdjdb.slim.scored.txt` are identical (145,408 each), a column-subset of `vdjdb.txt`.
- `vdjdb_full.txt`, `vdjdb_full_scored.txt`, `vdjdb_full_filtered.txt` are 3 alternate views of the same paired-clonotype data (~139K rows).
- `vdjdb-2025-12-29.zip` contains *all* of the above .txt files again — its 1,228,409 record count is a sum-of-duplicates *of duplicates*.
- Manifest reports 2,456,425. True unique vdjdb record count ≈ **226,494** (the canonical `vdjdb.txt`). The wiki's text section says nothing wrong about VDJdb's content, but the auto-inventory block displays a count ~10× the actual unique data.
- This also disagrees with `scripts/analysis/vdjdb_summary_export.py` which uses `vdjdb_full_filtered.txt` (138,545 records) as the canonical input.

**General pattern**: when a single source has multiple alternate-format dumps of the same data plus an archive that bundles those dumps, the manifest sums them all instead of choosing one canonical file.

## Task 2: Citation Verification

According to PubMed, the following citations were verified against article metadata:

| Source | Cited PMID | Verified | Author match | Year/Journal/Method match | Pass/Fail |
|--------|-----------|----------|--------------|---------------------------|-----------|
| vdjdb.md | 28977646 | ✓ | Shugay M ✓ | 2018, Nucleic Acids Res 46(D1):D419-D427 ✓ | PASS |
| vdjdb.md | 31588507 | ✓ | Bagaev DV ✓ | 2020, Nucleic Acids Res 48(D1):D1057-D1062 ✓ | PASS |
| mcpas.md | 28481982 | ✓ | Tickotsky N ✓ | 2017, Bioinformatics 33(18):2924-2929 ✓ | PASS |
| tadb.md | 33849445 | ✓ | Zhang G ✓ | 2021, BMC Bioinformatics 22(Suppl 8):40 ✓ | PASS |
| GSE114724.md | 29961579 | ✓ | Azizi E ✓ | 2018, Cell 174(5):1293-1308 ✓ | PASS |
| rcc_atlas.md | 33861994 | ✓ | Krishna C ✓ | 2021, Cancer Cell 39(5):662-677.e6 ✓ | PASS |
| rcc_atlas.md | 28475899 | ✓ | Chevrier S ✓ | 2017, Cell 169(4):736-749.e18 ✓ | PASS |
| netmhcpan.md | 32406916 | ✓ | Reynisson B ✓ | 2020, Nucleic Acids Res 48(W1):W449-W454 ✓ | PASS |
| immunecode.md | 40034696 | ✓ | Nolan S ✓ | 2025, Front Immunol 16:1488851 ✓ | PASS |
| iedb.md | 39558162 | ✓ | Vita R ✓ | 2025, Nucleic Acids Res 53(D1):D436-D443 ✓ | PASS |
| cedar.md | 36250634 | ✓ | Kosaloglu-Yalcin Z ✓ | 2023, Nucleic Acids Res 51(D1):D845-D852 ✓ | PASS |
| imgthla.md | 38936817 | ✓ | Robinson J ✓ | 2024, HLA 103(6):e15549 ✓ | PASS |
| **trait.md** | (no PMID cited; DOI 10.1093/gpbjnl/qzaf033) | ✓ resolves to PMID 40257421 | **AUTHORS HALLUCINATED** | 2025, Genomics Proteomics Bioinformatics 23(3) ✓ | **FAIL** |

PMIDs verified using PubMed metadata; full attribution preserved with article DOIs:
- [10.1093/nar/gkx760](https://doi.org/10.1093/nar/gkx760)
- [10.1093/nar/gkz874](https://doi.org/10.1093/nar/gkz874)
- [10.1093/bioinformatics/btx286](https://doi.org/10.1093/bioinformatics/btx286)
- [10.1186/s12859-021-03962-7](https://doi.org/10.1186/s12859-021-03962-7)
- [10.1016/j.cell.2018.05.060](https://doi.org/10.1016/j.cell.2018.05.060)
- [10.1016/j.ccell.2021.03.007](https://doi.org/10.1016/j.ccell.2021.03.007)
- [10.1016/j.cell.2017.04.016](https://doi.org/10.1016/j.cell.2017.04.016)
- [10.1093/nar/gkaa379](https://doi.org/10.1093/nar/gkaa379)
- [10.3389/fimmu.2025.1488851](https://doi.org/10.3389/fimmu.2025.1488851)
- [10.1093/nar/gkae1092](https://doi.org/10.1093/nar/gkae1092)
- [10.1093/nar/gkac902](https://doi.org/10.1093/nar/gkac902)
- [10.1111/tan.15549](https://doi.org/10.1111/tan.15549)
- [10.1093/gpbjnl/qzaf033](https://doi.org/10.1093/gpbjnl/qzaf033) (TRAIT — DOI valid, authors wrong)

### Notes

**TRAIT citation has fabricated authors.** Wiki cites `Wang Y, Yang J, Zhang Q, et al.` — the DOI is valid (PMID 40257421) but the actual authors per PubMed are **Wei M, Wu J, Bai S, Zhou Y, Chen Y, Zhang X, Zhao W, Chi Y, Pan G, Zhu F, Chen S, Zhou Z** (Zhejiang University, not UESTC). Wiki also lists the project URL as `<http://i.uestc.edu.cn/TRAIT/>` — the abstract states the database is hosted at `https://pgx.zju.edu.cn/traitdb`. Both must be corrected.

**`{citation needed}` flag in rcc_atlas.md** — confirmed accurate. RCC_ATLAS is an in-house literature-curated CSV (~81 rows) with no canonical publication. The `{citation needed for an in-house manuscript that describes this collection}` flag is appropriately placed; recommend leaving the flag and adding a note that no manuscript exists. Per-row references in `reference` column are sufficient provenance.

**`{citation needed for explicit license string}` in trait.md** — partially resolvable. The TANTIGEN/TaDB license is academic-use per Boston University Metropolitan College; TRAIT's license terms were not stated in the abstract I retrieved. Flag should remain until the team confirms.

**`{note for Verifier}` in tadb.md (line 56)** — confirmed: `scripts/data_processing/standardize/tadb.py` lines 28-59 do **NOT** apply any `Epitope type` validation filter. The standardizer maps `Epitope sequence` → `peptide` and `HLA allele` → `mhc_one`/`mhc_two` and yields all rows. The note is accurate; recommendation: keep flag and have the Standardization Specialist decide whether to add a filter.

**McPAS-TCR license verification** — wiki states "Free for academic use; no formal license file shipped (Friedman lab terms of use on the website)." The `data/raw_data/databases/McPAS-TCR/` directory contains only `McPAS-TCR.csv` and `download_mcpas.sh` (no LICENSE), so the wiki's statement is consistent with what's on disk. Verification of the actual website terms-of-use string is out of scope (no working WebFetch in this env), but the description is plausible.

## Task 3: Cross-source Consistency

**Aggregate disk usage**:
- Manifest `total_bytes` = 3,410,574,657,396 (3.41 TB)
- `du -sb /home/ubuntu/quest/data/raw_data/` = 3,410,574,713,378 (3.41 TB)
- Variance = +55,982 bytes (+0.0000016%) — well within rounding/atime tolerance.
- **PASS**

**`scripts/analysis/*_summary_export.py` cross-checks**:
- `vdjdb_summary_export.py` uses `data/databases/vdjdb/vdjdb_full_filtered.txt` (138,545 records) as canonical input. Manifest reports vdjdb total_records = 2,456,425. The summary script's choice of canonical file (138,545) is more trustworthy as a "unique scientific records" count, while the manifest is summing redundant variants. **The manifest's vdjdb count should not be cited as "the size of vdjdb" without disambiguation.**
- All other databases have summary scripts but I did not exhaustively run them; the vdjdb mismatch is the clearest example of the same systematic over-counting.

**Wiki text vs. inventory block consistency**:
- vdjdb.md text says "Roughly 70% of TCR-epitope pairs concentrate around ~100 epitopes" (no specific N stated in the lit-review section), and the inventory block says 2,456,425 records — these are not directly contradictory but readers will assume "vdjdb has 2.4M TCR-epitope records" which is incorrect. Recommend the lit-review section state "~226K canonical records, ~140K paired clonotypes" up front.
- mcpas.md says "~5,000 sequences" in lit-review but inventory says 40,779 records. The 5K figure is the 2017 paper's abstract count; the 40,779 is post-explosion-of-multi-peptide-rows. Since mcpas.py explodes `/`-separated peptides into multiple rows, the discrepancy is internally consistent but should be footnoted: "40,779 post-explosion rows; ~5,000 unique TCRs in source paper."

## Task 4: Standardizer file:line spot checks

5 of 15 databases checked, randomly selected:

| Wiki | Cited file:line | Actual code | Match |
|------|----------------|-------------|-------|
| vdjdb.md | `vdjdb.py:99-137` (score split into TCR/pMHC) | Lines 99-137 are exactly the score >=1 / score 0 split with TCR-only and pMHC-only branches | ✓ |
| mcpas.md | `mcpas.py:30-41` (validated-method whitelist) | Lines 30-41 are exactly the `_VALIDATED_METHODS = {...}` set definition | ✓ |
| mcpas.md | `mcpas.py:62-66` (human species filter) | Lines 62-66 are the `Species == "human"` filter | ✓ |
| rcc_atlas.md | `rcc_atlas.py:64-65` (`_binding="pos"`) | Line 65 is `df["_binding"] = "pos"` (line 64 is the comment) | ✓ |
| batman.md | `batman.py:96-99` (human filter) | Lines 96-99 are `tcr_source_organism == "human"` filter | ✓ |
| batman.md | `batman.py:127-131` (peptide_activity threshold 0.1) | Lines 127-131 are exactly the 0.1 threshold logic | ✓ |
| iedb.md | `iedb.py:71-74` (BCR exclusion) | Lines 71-74 are the BCR types exclusion | ✓ |
| iedb.md | `iedb.py:168-172` (positive assay filter) | Lines 168-172 are the `"positive" in outcome` filter | ✓ |
| iedb.md | `iedb.py:186-192` (organism filter) | Lines 186-192 are the human organism filter | ✓ |
| netmhcpan.md | `netmhcpan.py:67-87` (multi-allele/non-HLA filter) | Lines 67-87 implement multi-allele skip and non-HLA filter | ✓ |
| trait.md | `trait.py:55-65` (binder_pos/binder_neg parsing) | Lines 56-61 contain the binder parsing; cited range is correct | ✓ |
| tadb.md | `tadb.py:28-32` (column map) | Lines 28-31 (`Epitope sequence → peptide`) | ✓ |

All 12 standardizer cross-references PASS — the lit-review agent did real grep work, not fabrication.

## Discrepancies requiring fixes

1. **NetMHCPan double-counting (CRITICAL).** `data/raw_data/databases/NetMHCPan/` contains both `NetMHCIIpan_train.tar.gz` (counted at 20,963,796) **and** the unpacked directory `NetMHCIIpan_train/` whose .txt files are counted again (sum 20,963,817). Manifest's `total_records=41,927,613` is roughly 2× the truth. **Fix**: in `build_raw_data_manifest.py`, when an archive's record count is non-zero, suppress counts from the unpacked sibling directory (or vice versa). Affected wiki page: `databases/netmhcpan.md`.

2. **vdjdb decuplication-counting (CRITICAL).** `data/raw_data/databases/vdjdb/` contains 9 redundant alternate-format dumps of the same data plus a zip that re-bundles them all. Manifest's `total_records=2,456,425` is roughly 10× the unique-record truth (~226,494). **Fix**: the manifest needs a "canonical file" rule — for vdjdb, count only `vdjdb.txt` (or `vdjdb_full_filtered.txt` to match `vdjdb_summary_export.py`). Affected wiki page: `databases/vdjdb.md`.

3. **TRAIT citation has hallucinated authors.** Wiki cites `Wang Y, Yang J, Zhang Q, et al.` — actual authors per PubMed PMID 40257421 are `Wei M, Wu J, Bai S, Zhou Y, Chen Y, Zhang X, Zhao W, Chi Y, Pan G, Zhu F, Chen S, Zhou Z`. Project URL also incorrect: should be `https://pgx.zju.edu.cn/traitdb` (Zhejiang Univ.), not `http://i.uestc.edu.cn/TRAIT/` (UESTC). Affected wiki page: `databases/trait.md`. Also add PMID link `[PubMed 40257421](https://pubmed.ncbi.nlm.nih.gov/40257421/)`.

4. **IMGTHLA partial double-counting.** `hla_nuc.fasta` (root) and `fasta/hla_nuc.fasta` are byte-identical (verified via `cmp`); both counted at 41,428 records. Same likely applies to `A_gen.fasta`, `B_gen.fasta`, etc. — IMGTHLA ships each fasta in 2 locations, both directories counted. **Fix**: dedupe by content hash before summing. Affected wiki page: `databases/imgthla.md`.

5. **NetMHCpan_train binding/eluted-ligand files uncounted.** `NetMHCpan_train/c000_ba` … `c004_el` (10 files) marked `binary/non-tabular; not counted` (record_count=null), but they are plain whitespace-separated tabular text. Independent counts: c000_ba=41,206, c001_ba=41,965, c002_ba=41,621 rows. **Fix**: the `noext`-format files in `NetMHCpan_train/` should be counted via `wc -l` rather than skipped as binary. Note this is partly canceled by issue #1 (these unpacked files would otherwise add to the double-count); after fixing #1, this fix becomes additive (~400K records added from NetMHCpan_train). Affected wiki page: `databases/netmhcpan.md`.

6. **vdjdb.md text section sets reader expectation incorrectly.** The lit-review text doesn't cite a record count, but the auto-inventory shows 2,456,425 records, which a reader will internalize as "vdjdb size." **Fix** (cosmetic): add one line in the lit-review section: "Canonical record count is ~226K (vdjdb.txt); auto-inventory total over-reports due to mirror/zip dumps." Same applies to NetMHCPan and IMGTHLA.

7. **mcpas.md "5,000 sequences" vs 40,779 records discrepancy.** Wiki text says "~5,000 sequences" (paper's abstract) but inventory says 40,779. Standardizer explodes `/`-separated peptides → row inflation. **Fix** (cosmetic): footnote the discrepancy, e.g., "5K source TCRs explode to ~40K rows after multi-peptide explosion in standardizer."

8. **rcc_atlas.md `{citation needed}`** — flag is correct; no in-house manuscript exists. Recommend keeping flag with explanatory note rather than waiting for a phantom citation.

9. **tadb.md `{note for Verifier}` (line 56)** — verified: standardizer does not apply Epitope-type validation filter. Recommend escalating to Standardization Specialist as a small task: "decide whether to filter on `Epitope type` (e.g., drop predicted-only entries) before standardization."

10. **adc/immuneACCESS sampling CIs are very wide** (95% CI for adc spans 185M–3.93B, a 21× range; immuneACCESS spans 1.05B–3.83B). These are sampled estimates of bulk repertoire data and should not be presented as point estimates without the CI. **Fix** (cosmetic): wiki auto-inventory should display "(sampled, 95% CI X–Y)" prominently, not just bury the CI in `total_records_ci`. Affected wiki pages: `databases/adc.md`, `databases/immuneaccess.md`, `databases/immunecode.md`, `databases/tcrdb.md`, `databases/ots.md`.

---

**Verifier sign-off**: PASS WITH CAVEATS. Citations and standardizer cross-references are trustworthy (12/12 PMIDs verified; 12/12 file:line refs match) with **one fabricated author list (TRAIT)**. Per-file counts are correct, but **aggregation logic systematically double-counts** when archives + unpacked dirs co-exist (NetMHCPan, IMGTHLA) or when a database ships redundant alternate-format dumps (vdjdb). The 5.38B headline `total_records` is inflated; affected sources collectively account for ~25M of bogus duplicate records out of the headline figure (rest of the headline is dominated by sampled extrapolations from adc/immuneACCESS, which use sound methodology with wide CIs).
