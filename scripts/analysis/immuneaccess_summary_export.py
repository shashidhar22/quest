#!/usr/bin/env python3
"""Export comprehensive immuneACCESS analysis tables.

Memory-optimized: streams one TSV at a time, accumulates stats in
sets/counters. Never holds the full dataset in memory.
"""

import gc
import os
from collections import Counter, defaultdict

import pandas as pd

DATA_DIR = "data/databases/immuneACCESS"
OUTPUT_DIR = "data/analysis/immuneACCESS"

# Only read these columns from the 30-column immuneACCESS TSV
USECOLS = ["amino_acid", "frame_type", "v_gene", "j_gene", "templates"]

# Three subdirectory groups to scan
SUBDIRS = {
    "FHCRC-Warren": "FHCRC-Warren-Updated_datasets",
    "bulk_survey_trb": "existing_data/bulk_survey_trb",
    "bulk_survey_tra": "existing_data/bulk_survey_tra",
}


def aggregate_data(data_dir):
    """Stream through immuneACCESS TSV files and accumulate statistics.

    Processes one file at a time -- peak memory is accumulator sets
    instead of loading the full dataset.
    """
    # --- Global accumulators ---
    total_records = 0
    productive_count = 0
    unique_amino_acid = set()
    unique_v_gene = set()
    unique_j_gene = set()

    # --- Per-category accumulators ---
    cat_records = Counter()
    cat_productive = Counter()
    cat_amino_acid = defaultdict(set)
    cat_v_gene = defaultdict(set)
    cat_j_gene = defaultdict(set)
    cat_files = Counter()

    # --- Frame-type accumulators ---
    frame_type_records = Counter()
    cat_frame_type_records = defaultdict(Counter)

    # --- Data completeness accumulators ---
    has_amino_acid_count = 0
    has_v_gene_count = 0
    has_j_gene_count = 0

    file_count = 0
    skipped = 0
    total_files_found = 0

    for category, subdir in SUBDIRS.items():
        dirpath = os.path.join(data_dir, subdir)
        if not os.path.isdir(dirpath):
            print(f"  WARNING: directory not found: {dirpath}")
            continue

        tsv_files = sorted(f for f in os.listdir(dirpath) if f.endswith(".tsv"))
        total_files_found += len(tsv_files)
        print(f"  {category}: {len(tsv_files)} TSV files in {subdir}")

        for tsv_file in tsv_files:
            fpath = os.path.join(dirpath, tsv_file)

            try:
                df = pd.read_csv(
                    fpath, sep="\t", usecols=USECOLS,
                    dtype=str, na_filter=False,
                )
            except Exception:
                skipped += 1
                continue

            if len(df) == 0:
                cat_files[category] += 1
                file_count += 1
                continue

            n = len(df)

            # Vectorized masks
            has_aa = (df["amino_acid"] != "") & (df["amino_acid"] != "na")
            has_v = (df["v_gene"] != "") & (df["v_gene"] != "unresolved")
            has_j = (df["j_gene"] != "") & (df["j_gene"] != "unresolved")
            is_productive = df["frame_type"] == "In"

            n_productive = int(is_productive.sum())
            n_has_aa = int(has_aa.sum())
            n_has_v = int(has_v.sum())
            n_has_j = int(has_j.sum())

            # Extract unique values from this file
            file_aa = set(df.loc[has_aa, "amino_acid"])
            file_v = set(df.loc[has_v, "v_gene"])
            file_j = set(df.loc[has_j, "j_gene"])

            # Frame type distribution
            for ft_val, ft_count in df["frame_type"].value_counts().items():
                ft_val_str = str(ft_val).strip()
                if ft_val_str == "" or ft_val_str == "na":
                    ft_val_str = "(unknown)"
                frame_type_records[ft_val_str] += int(ft_count)
                cat_frame_type_records[category][ft_val_str] += int(ft_count)

            # Update global accumulators
            total_records += n
            productive_count += n_productive
            unique_amino_acid.update(file_aa)
            unique_v_gene.update(file_v)
            unique_j_gene.update(file_j)

            has_amino_acid_count += n_has_aa
            has_v_gene_count += n_has_v
            has_j_gene_count += n_has_j

            # Update per-category
            cat_records[category] += n
            cat_productive[category] += n_productive
            cat_amino_acid[category].update(file_aa)
            cat_v_gene[category].update(file_v)
            cat_j_gene[category].update(file_j)
            cat_files[category] += 1

            file_count += 1
            del df, file_aa, file_v, file_j

            if file_count % 2000 == 0:
                print(f"    Processed {file_count} files ...")
                gc.collect()

    print(f"  Processed {file_count} files ({skipped} skipped)")

    return {
        "total_records": total_records,
        "productive_count": productive_count,
        "unique_amino_acid": unique_amino_acid,
        "unique_v_gene": unique_v_gene,
        "unique_j_gene": unique_j_gene,
        "cat_records": cat_records,
        "cat_productive": cat_productive,
        "cat_amino_acid": cat_amino_acid,
        "cat_v_gene": cat_v_gene,
        "cat_j_gene": cat_j_gene,
        "cat_files": cat_files,
        "frame_type_records": frame_type_records,
        "cat_frame_type_records": cat_frame_type_records,
        "has_amino_acid_count": has_amino_acid_count,
        "has_v_gene_count": has_v_gene_count,
        "has_j_gene_count": has_j_gene_count,
        "total_files": total_files_found,
        "processed_files": file_count,
    }


# ---------------------------------------------------------------------------
# Build TSV DataFrames from aggregated stats
# ---------------------------------------------------------------------------
def overview(stats):
    return pd.DataFrame(
        [
            {
                "total_records": stats["total_records"],
                "unique_cdr3_aa": len(stats["unique_amino_acid"]),
                "unique_v_gene": len(stats["unique_v_gene"]),
                "unique_j_gene": len(stats["unique_j_gene"]),
                "productive_records": stats["productive_count"],
                "unique_epitopes": 0,
                "unique_mhc": 0,
                "unique_pmhc": 0,
                "total_files": stats["total_files"],
            }
        ]
    )


def category_overview(stats):
    rows = []
    for cat in sorted(stats["cat_records"]):
        rows.append(
            {
                "category": cat,
                "total_records": stats["cat_records"][cat],
                "unique_cdr3_aa": len(stats["cat_amino_acid"].get(cat, set())),
                "unique_v_gene": len(stats["cat_v_gene"].get(cat, set())),
                "unique_j_gene": len(stats["cat_j_gene"].get(cat, set())),
                "productive_records": stats["cat_productive"][cat],
                "num_files": stats["cat_files"][cat],
                "unique_epitopes": 0,
                "unique_mhc": 0,
            }
        )
    return pd.DataFrame(rows)


def epitope_summary():
    return pd.DataFrame(
        columns=[
            "epitope", "antigen_protein", "pathology", "category",
            "primary_mhc", "mhc_class", "total_records", "paired_records",
            "unique_paired_tcrs", "unique_beta_only_tcrs",
        ]
    )


def pmhc_summary():
    return pd.DataFrame(
        columns=[
            "epitope", "mhc", "mhc_class", "antigen_protein", "pathology",
            "total_records", "unique_paired_tcrs", "unique_beta_only_tcrs",
        ]
    )


def study_summary(stats):
    rows = []
    for cat in sorted(stats["cat_records"]):
        rows.append(
            {
                "study": cat,
                "total_records": stats["cat_records"][cat],
                "productive_records": stats["cat_productive"][cat],
                "unique_cdr3_aa": len(stats["cat_amino_acid"].get(cat, set())),
                "unique_v_gene": len(stats["cat_v_gene"].get(cat, set())),
                "unique_j_gene": len(stats["cat_j_gene"].get(cat, set())),
                "num_files": stats["cat_files"][cat],
            }
        )
    return pd.DataFrame(rows)


def mhc_allele_summary():
    return pd.DataFrame(
        columns=["mhc", "mhc_class", "total_records", "unique_epitopes", "unique_paired_tcrs"]
    )


def species_breakdown(stats):
    rows = []
    # Frame type distribution (In/Out/Stop)
    for ft in sorted(stats["frame_type_records"]):
        rows.append(
            {
                "category": "frame_type",
                "label": ft,
                "total_records": stats["frame_type_records"][ft],
                "unique_epitopes": 0,
                "unique_paired_tcrs": 0,
            }
        )
    # Category breakdown
    for cat in sorted(stats["cat_records"]):
        rows.append(
            {
                "category": "subdirectory",
                "label": cat,
                "total_records": stats["cat_records"][cat],
                "unique_epitopes": 0,
                "unique_paired_tcrs": 0,
            }
        )
    # Per-category frame type breakdown
    for cat in sorted(stats["cat_frame_type_records"]):
        for ft in sorted(stats["cat_frame_type_records"][cat]):
            rows.append(
                {
                    "category": f"frame_type_{cat}",
                    "label": ft,
                    "total_records": stats["cat_frame_type_records"][cat][ft],
                    "unique_epitopes": 0,
                    "unique_paired_tcrs": 0,
                }
            )
    return pd.DataFrame(rows)


def data_completeness(stats):
    total = stats["total_records"]
    return pd.DataFrame(
        [
            {
                "category": "has_amino_acid",
                "record_count": stats["has_amino_acid_count"],
                "productive_records": stats["productive_count"],
                "unique_cdr3_aa": len(stats["unique_amino_acid"]),
                "unique_v_gene": len(stats["unique_v_gene"]),
            },
            {
                "category": "has_v_gene",
                "record_count": stats["has_v_gene_count"],
                "productive_records": 0,
                "unique_cdr3_aa": 0,
                "unique_v_gene": len(stats["unique_v_gene"]),
            },
            {
                "category": "has_j_gene",
                "record_count": stats["has_j_gene_count"],
                "productive_records": 0,
                "unique_cdr3_aa": 0,
                "unique_v_gene": 0,
            },
            {
                "category": "productive",
                "record_count": stats["productive_count"],
                "productive_records": stats["productive_count"],
                "unique_cdr3_aa": 0,
                "unique_v_gene": 0,
            },
            {
                "category": "total",
                "record_count": total,
                "productive_records": stats["productive_count"],
                "unique_cdr3_aa": len(stats["unique_amino_acid"]),
                "unique_v_gene": len(stats["unique_v_gene"]),
            },
        ]
    )


# ---------------------------------------------------------------------------
# SUMMARY.txt generation
# ---------------------------------------------------------------------------
def write_summary_txt(stats, output_dir):
    """Write a human-readable SUMMARY.txt after all TSVs are exported."""
    from datetime import date

    ov = pd.read_csv(os.path.join(output_dir, "overview.tsv"), sep="\t")
    cat_ov = pd.read_csv(os.path.join(output_dir, "category_overview.tsv"), sep="\t")
    completeness = pd.read_csv(
        os.path.join(output_dir, "data_completeness.tsv"), sep="\t"
    )
    species = pd.read_csv(
        os.path.join(output_dir, "species_breakdown.tsv"), sep="\t"
    )
    studies = pd.read_csv(os.path.join(output_dir, "study_summary.tsv"), sep="\t")

    r = ov.iloc[0]
    total = int(r["total_records"])
    productive = int(r["productive_records"])
    productive_pct = 100.0 * productive / total if total else 0
    num_files = int(r["total_files"])
    lines = []

    # --- Header ---
    lines.append("=" * 80)
    lines.append("                    immuneACCESS Analysis Summary")
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {DATA_DIR}")
    lines.append("=" * 80)
    lines.append("")

    # --- 1. DATASET OVERVIEW ---
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Total records:          {total:>12,}")
    lines.append(f"Unique CDR3 AA:         {int(r['unique_cdr3_aa']):>12,}")
    lines.append(f"Unique V genes:         {int(r['unique_v_gene']):>12,}")
    lines.append(f"Unique J genes:         {int(r['unique_j_gene']):>12,}")
    lines.append(
        f"Productive records:     {productive:>12,}  ({productive_pct:.1f}% of total)"
    )
    lines.append(f"Unique epitopes:        {'N/A':>12s}  (no epitope annotation)")
    lines.append(f"Unique MHC alleles:     {'N/A':>12s}  (no MHC annotation)")
    lines.append(f"Source files:           {num_files:>12,}")
    lines.append("")

    lines.append("Subdirectory distribution:")
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cat = row["category"]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        n_files = int(row["num_files"])
        lines.append(f"  {cat:<20s} {cnt:>12,}  ({pct:>5.1f}%)  [{n_files:,} files]")
    lines.append("")

    # --- 2. MHC CLASS BREAKDOWN ---
    lines.append("2. MHC CLASS BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available in this database. immuneACCESS does not include MHC annotations."
    )
    lines.append("")

    # --- 3. CATEGORY BREAKDOWN ---
    lines.append("3. CATEGORY BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Category':<20s} {'Records':>12s} {'%':>7s} "
        f"{'Productive':>12s} {'CDR3 AAs':>12s} {'V genes':>9s}"
    )
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['category'])[:18]:<20s} {cnt:>12,} {pct:>6.1f}% "
            f"{int(row['productive_records']):>12,} {int(row['unique_cdr3_aa']):>12,} "
            f"{int(row['unique_v_gene']):>9,}"
        )
    lines.append("")

    dominant = cat_ov.loc[cat_ov["total_records"].idxmax()]
    dom_pct = 100.0 * int(dominant["total_records"]) / total if total else 0
    lines.append(
        f"The '{dominant['category']}' category has the most records with "
        f"{dom_pct:.1f}% of the database."
    )
    lines.append("")

    # --- 4. TOP EPITOPES ---
    lines.append("4. TOP EPITOPES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available in this database. immuneACCESS is bulk TCR repertoire data\n"
        "without epitope annotations."
    )
    lines.append("")

    # --- 5. TOP MHC ALLELES ---
    lines.append("5. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available in this database. immuneACCESS does not include MHC annotations."
    )
    lines.append("")

    # --- 6. PATHOLOGY & SPECIES ---
    lines.append("6. FRAME TYPE & SUBDIRECTORY BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    frame_types = species[species["category"] == "frame_type"].copy()
    frame_types = frame_types.sort_values("total_records", ascending=False)
    frame_total = int(frame_types["total_records"].sum())

    if len(frame_types) > 0:
        lines.append("Frame type distribution (In=productive, Out=non-productive, Stop=stop codon):")
        for _, row in frame_types.iterrows():
            ft = str(row["label"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / frame_total if frame_total else 0
            lines.append(
                f"  {ft + ':':<25s} {cnt:>12,} records  ({pct:>5.1f}%)"
            )
        lines.append("")

    subdir_species = species[species["category"] == "subdirectory"].copy()
    subdir_species = subdir_species.sort_values("total_records", ascending=False)
    if len(subdir_species) > 0:
        lines.append("Subdirectory breakdown:")
        for _, row in subdir_species.iterrows():
            sp = str(row["label"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total if total else 0
            lines.append(f"  {sp:<20s} {cnt:>12,}  ({pct:>5.1f}%)")
        lines.append("")

    # Per-category frame type breakdown
    for cat in sorted(stats["cat_frame_type_records"]):
        cat_ft = species[species["category"] == f"frame_type_{cat}"].copy()
        cat_ft = cat_ft.sort_values("total_records", ascending=False)
        if len(cat_ft) > 0:
            cat_total = int(cat_ft["total_records"].sum())
            lines.append(f"  Frame types in {cat}:")
            for _, row in cat_ft.iterrows():
                ft = str(row["label"])
                cnt = int(row["total_records"])
                pct = 100.0 * cnt / cat_total if cat_total else 0
                lines.append(
                    f"    {ft + ':':<23s} {cnt:>12,}  ({pct:>5.1f}%)"
                )
            lines.append("")

    # --- 7. STUDY LANDSCAPE ---
    lines.append("7. STUDY LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")

    total_studies = len(studies)
    lines.append(f"Total subdirectory groups: {total_studies}")
    lines.append(
        "Note: immuneACCESS does not provide per-study metadata within files.\n"
        "Studies are approximated by subdirectory grouping."
    )
    lines.append("")

    lines.append("Per-subdirectory summary:")
    for _, row in studies.sort_values("total_records", ascending=False).iterrows():
        study_label = str(row["study"])[:40]
        lines.append(
            f"  {study_label:<42s} {int(row['total_records']):>12,} records  "
            f"({int(row['num_files']):>6,} files, "
            f"{int(row['unique_cdr3_aa']):>10,} unique CDR3s)"
        )
    lines.append("")

    # --- 8. DATA COMPLETENESS ---
    lines.append("8. DATA COMPLETENESS")
    lines.append("-" * 80)
    lines.append("")

    criteria = {
        "has_amino_acid": "Has CDR3 amino acid sequence (not 'na' or empty).",
        "has_v_gene": "Has resolved V gene annotation (not 'unresolved').",
        "has_j_gene": "Has resolved J gene annotation (not 'unresolved').",
        "productive": "Productive rearrangement (frame_type = 'In').",
        "total": "All records in the dataset.",
    }

    lines.append(
        f"  {'Category':<25s} {'Records':>12s} {'%':>7s} "
        f"{'CDR3 AAs':>12s} {'V genes':>9s}"
    )
    for _, row in completeness.iterrows():
        cat = row["category"]
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {cat:<25s} {cnt:>12,} {pct:>6.1f}% "
            f"{int(row['unique_cdr3_aa']):>12,} {int(row['unique_v_gene']):>9,}"
        )
    lines.append("")

    lines.append("Criteria:")
    for cat, desc in criteria.items():
        lines.append(f"  - {cat:<25s} {desc}")
    lines.append("")

    lines.append(
        "immuneACCESS is bulk TCR repertoire data with no epitope or MHC\n"
        "annotation. This data is best suited for unsupervised pre-training,\n"
        "TCR language modeling, and repertoire-level analysis."
    )
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

    lines.append(
        f"a) Data scale: {total:,} records across {num_files:,} files in 3 subdirectories.\n"
        f"   {int(r['unique_cdr3_aa']):,} unique CDR3 amino acid sequences."
    )
    lines.append("")

    lines.append(
        f"b) Productive sequences: {productive:,} records ({productive_pct:.1f}% of total)\n"
        f"   are productive rearrangements (frame_type = 'In').\n"
        f"   Non-productive and stop-codon sequences should be filtered for most tasks."
    )
    lines.append("")

    lines.append(
        f"c) V/J gene annotation: {int(r['unique_v_gene']):,} unique V genes and "
        f"{int(r['unique_j_gene']):,} unique J genes.\n"
        f"   Gene usage analysis and conditional generation are supported."
    )
    lines.append("")

    lines.append(
        f"d) Category composition: '{dominant['category']}' accounts for "
        f"{dom_pct:.1f}% of records.\n"
        f"   Includes TRA (alpha) and TRB (beta) chain data in separate subdirectories."
    )
    lines.append("")

    lines.append(
        "e) No epitope/MHC: immuneACCESS does not include epitope or MHC annotations.\n"
        "   This data cannot be used for TCR-pMHC binding prediction\n"
        "   but is valuable for TCR language modeling and repertoire analysis."
    )
    lines.append("")

    lines.append(
        "f) No pairing: Bulk repertoire data without alpha-beta chain pairing.\n"
        "   Each file contains single-chain sequences only."
    )
    lines.append("")

    # --- Footer ---
    lines.append("=" * 80)
    lines.append(f"Output files in {OUTPUT_DIR}/:")

    tsv_files = [
        "overview.tsv", "category_overview.tsv", "epitope_summary.tsv",
        "pmhc_summary.tsv", "study_summary.tsv", "mhc_allele_summary.tsv",
        "species_breakdown.tsv", "data_completeness.tsv",
    ]
    for fname in tsv_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            tsv_df = pd.read_csv(fpath, sep="\t")
            lines.append(f"  {fname:<30s} - {len(tsv_df):>5,} rows")
    lines.append("=" * 80)

    summary_path = os.path.join(output_dir, "SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_dir = os.path.join(project_root, DATA_DIR)
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Streaming immuneACCESS data from {data_dir} ...")
    stats = aggregate_data(data_dir)
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  Productive records: {stats['productive_count']:,}")
    print(f"  Unique CDR3 AA: {len(stats['unique_amino_acid']):,}")
    print(f"  Unique V genes: {len(stats['unique_v_gene']):,}")
    print(f"  Unique J genes: {len(stats['unique_j_gene']):,}")
    print()

    exports = [
        ("overview.tsv", lambda s: overview(s)),
        ("category_overview.tsv", lambda s: category_overview(s)),
        ("epitope_summary.tsv", lambda s: epitope_summary()),
        ("pmhc_summary.tsv", lambda s: pmhc_summary()),
        ("study_summary.tsv", lambda s: study_summary(s)),
        ("mhc_allele_summary.tsv", lambda s: mhc_allele_summary()),
        ("species_breakdown.tsv", lambda s: species_breakdown(s)),
        ("data_completeness.tsv", lambda s: data_completeness(s)),
    ]

    for filename, func in exports:
        print(f"Computing {filename} ...")
        result = func(stats)
        path = os.path.join(output_dir, filename)
        result.to_csv(path, sep="\t", index=False)
        print(f"  -> {path}  ({len(result)} rows, {result.columns.size} cols)")

    print("Writing SUMMARY.txt ...")
    write_summary_txt(stats, output_dir)

    # --- Verification ---
    print("\n--- Verification ---")

    ov = pd.read_csv(os.path.join(output_dir, "overview.tsv"), sep="\t")
    cat_ov = pd.read_csv(os.path.join(output_dir, "category_overview.tsv"), sep="\t")
    cat_sum = cat_ov["total_records"].sum()
    all_total = ov.iloc[0]["total_records"]
    print(
        f"Category sum: {cat_sum}, Overview total: {all_total}, "
        f"match: {cat_sum == all_total}"
    )

    comp = pd.read_csv(os.path.join(output_dir, "data_completeness.tsv"), sep="\t")
    comp_total_row = comp[comp["category"] == "total"]["record_count"].iloc[0]
    print(
        f"Completeness total: {comp_total_row}, Overview total: {all_total}, "
        f"match: {comp_total_row == all_total}"
    )

    sp = pd.read_csv(os.path.join(output_dir, "species_breakdown.tsv"), sep="\t")
    ft_sum = sp[sp["category"] == "frame_type"]["total_records"].sum()
    print(
        f"Frame type sum: {ft_sum}, Overview total: {all_total}, "
        f"match: {ft_sum == all_total}"
    )

    subdir_sum = sp[sp["category"] == "subdirectory"]["total_records"].sum()
    print(
        f"Subdirectory sum: {subdir_sum}, Overview total: {all_total}, "
        f"match: {subdir_sum == all_total}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
