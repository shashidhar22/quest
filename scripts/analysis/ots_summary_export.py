#!/usr/bin/env python3
"""Export comprehensive OTS analysis tables.

Memory-optimized: streams one CSV at a time, accumulates stats in
sets/counters. Never holds the full 61 GB dataset in memory.
"""

import csv
import gc
import json
import os
from collections import Counter, defaultdict

import pandas as pd

DATA_DIR = "data/databases/OTS"
OUTPUT_DIR = "data/analysis/ots"

# Only read these columns from the 192-column AIRR CSV
OTS_USECOLS = [
    "cdr3_aa_beta",
    "cdr3_aa_alpha",
    "v_call_beta",
    "v_call_alpha",
    "j_call_beta",
    "j_call_alpha",
    "productive_beta",
    "productive_alpha",
]


def parse_ots_metadata(filepath):
    """Parse JSON metadata from the first line of an OTS CSV file."""
    with open(filepath, "r") as f:
        first_line = f.readline()
    try:
        reader = csv.reader([first_line])
        fields = next(reader)
        return json.loads(fields[0])
    except (json.JSONDecodeError, StopIteration, IndexError):
        return {}


def aggregate_data(data_dir):
    """Stream through OTS CSV files and accumulate statistics.

    Processes one file at a time — peak memory is accumulator sets
    (~1-3 GB) instead of 61+ GB for full concat.
    """
    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    print(f"  Found {len(csv_files)} OTS CSV files")

    # --- Global accumulators ---
    total_records = 0
    total_paired = 0
    beta_cdr3s = set()
    alpha_cdr3s = set()
    paired_tcrs = set()

    # --- Per-category (Disease) accumulators ---
    cat_records = Counter()
    cat_paired = Counter()
    cat_beta_cdr3s = defaultdict(set)
    cat_alpha_cdr3s = defaultdict(set)
    cat_paired_tcrs = defaultdict(set)

    # --- Per-study accumulators (counters + limited sets) ---
    study_records = Counter()
    study_paired = Counter()
    study_files = Counter()
    study_beta_cdr3s = defaultdict(set)
    study_categories = defaultdict(set)
    study_species = defaultdict(set)

    # --- Per-species accumulators ---
    species_records = Counter()
    species_paired = Counter()

    # --- Disease breakdown (same as category) ---

    meta_rows = []
    file_count = 0
    skipped = 0

    for csv_file in csv_files:
        fpath = os.path.join(data_dir, csv_file)

        meta = parse_ots_metadata(fpath)
        if not meta:
            skipped += 1
            continue

        try:
            df = pd.read_csv(
                fpath, skiprows=1, usecols=OTS_USECOLS,
                dtype=str, na_filter=False,
            )
        except Exception:
            skipped += 1
            continue

        if len(df) == 0:
            continue

        meta["_file"] = csv_file
        meta_rows.append(meta)

        category = (meta.get("Disease") or "").strip() or "(unknown)"
        author = (meta.get("Author") or "").strip()
        link = (meta.get("Link") or "").strip()
        study = f"{author} | {link}".strip(" |") or "(unknown)"
        sp = (meta.get("Species") or "").strip()

        # Vectorized boolean masks
        has_beta = df["cdr3_aa_beta"] != ""
        has_alpha = df["cdr3_aa_alpha"] != ""
        is_paired = has_beta & has_alpha

        n = len(df)
        n_paired = int(is_paired.sum())

        # Extract unique values from this file
        file_beta = set(df.loc[has_beta, "cdr3_aa_beta"])
        file_alpha = set(df.loc[has_alpha, "cdr3_aa_alpha"])
        file_paired = set()
        if n_paired > 0:
            paired_df = df[is_paired]
            file_paired = set(
                paired_df["cdr3_aa_alpha"] + "|" + paired_df["cdr3_aa_beta"]
            )
            del paired_df

        # Update global accumulators
        total_records += n
        total_paired += n_paired
        beta_cdr3s.update(file_beta)
        alpha_cdr3s.update(file_alpha)
        paired_tcrs.update(file_paired)

        # Update per-category
        cat_records[category] += n
        cat_paired[category] += n_paired
        cat_beta_cdr3s[category].update(file_beta)
        cat_alpha_cdr3s[category].update(file_alpha)
        cat_paired_tcrs[category].update(file_paired)

        # Update per-study
        study_records[study] += n
        study_paired[study] += n_paired
        study_files[study] += 1
        study_beta_cdr3s[study].update(file_beta)
        study_categories[study].add(category)
        if sp:
            study_species[study].add(sp)

        # Update per-species
        if sp:
            species_records[sp] += n
            species_paired[sp] += n_paired

        file_count += 1
        del df, file_beta, file_alpha, file_paired

        if file_count % 100 == 0:
            print(f"    Processed {file_count} files ...")
            gc.collect()

    print(f"  Processed {file_count} files ({skipped} skipped)")

    return {
        "total_records": total_records,
        "total_paired": total_paired,
        "beta_cdr3s": beta_cdr3s,
        "alpha_cdr3s": alpha_cdr3s,
        "paired_tcrs": paired_tcrs,
        "cat_records": cat_records,
        "cat_paired": cat_paired,
        "cat_beta_cdr3s": cat_beta_cdr3s,
        "cat_alpha_cdr3s": cat_alpha_cdr3s,
        "cat_paired_tcrs": cat_paired_tcrs,
        "study_records": study_records,
        "study_paired": study_paired,
        "study_files": study_files,
        "study_beta_cdr3s": study_beta_cdr3s,
        "study_categories": study_categories,
        "study_species": study_species,
        "species_records": species_records,
        "species_paired": species_paired,
        "meta_rows": meta_rows,
    }


# ---------------------------------------------------------------------------
# Build TSV DataFrames from aggregated stats
# ---------------------------------------------------------------------------
def overview(stats):
    return pd.DataFrame(
        [
            {
                "total_records": stats["total_records"],
                "unique_beta_cdr3": len(stats["beta_cdr3s"]),
                "unique_alpha_cdr3": len(stats["alpha_cdr3s"]),
                "paired_records": stats["total_paired"],
                "unique_paired_tcrs": len(stats["paired_tcrs"]),
                "unique_epitopes": 0,
                "unique_mhc": 0,
                "unique_pmhc": 0,
                "unique_tcr_pmhc_paired": 0,
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
                "unique_beta_cdr3": len(stats["cat_beta_cdr3s"].get(cat, set())),
                "unique_alpha_cdr3": len(stats["cat_alpha_cdr3s"].get(cat, set())),
                "paired_records": stats["cat_paired"][cat],
                "unique_paired_tcrs": len(stats["cat_paired_tcrs"].get(cat, set())),
                "unique_epitopes": 0,
                "unique_mhc": 0,
                "unique_pmhc": 0,
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
    for study in sorted(stats["study_records"]):
        rows.append(
            {
                "study": study,
                "total_records": stats["study_records"][study],
                "paired_records": stats["study_paired"][study],
                "unique_beta_cdr3": len(stats["study_beta_cdr3s"].get(study, set())),
                "unique_paired_tcrs": 0,
                "num_files": stats["study_files"][study],
                "categories": ";".join(sorted(stats["study_categories"].get(study, set()))),
                "species": ";".join(sorted(stats["study_species"].get(study, set()))),
            }
        )
    return pd.DataFrame(rows)


def mhc_allele_summary():
    return pd.DataFrame(
        columns=["mhc", "mhc_class", "total_records", "unique_epitopes", "unique_paired_tcrs"]
    )


def species_breakdown(stats):
    rows = []
    for sp in sorted(stats["species_records"]):
        rows.append(
            {
                "category": "host_species",
                "species": sp,
                "total_records": stats["species_records"][sp],
                "unique_epitopes": 0,
                "unique_paired_tcrs": 0,
            }
        )
    for cat in sorted(stats["cat_records"]):
        rows.append(
            {
                "category": "disease",
                "species": cat,
                "total_records": stats["cat_records"][cat],
                "unique_epitopes": 0,
                "unique_paired_tcrs": len(stats["cat_paired_tcrs"].get(cat, set())),
            }
        )
    return pd.DataFrame(rows)


def data_completeness(stats):
    return pd.DataFrame(
        [
            {"category": "has_epitope_and_mhc", "record_count": 0,
             "paired_tcrs": 0, "unique_beta_cdr3": 0, "unique_epitopes": 0},
            {"category": "has_epitope_only", "record_count": 0,
             "paired_tcrs": 0, "unique_beta_cdr3": 0, "unique_epitopes": 0},
            {"category": "has_mhc_only", "record_count": 0,
             "paired_tcrs": 0, "unique_beta_cdr3": 0, "unique_epitopes": 0},
            {"category": "no_epitope_no_mhc",
             "record_count": stats["total_records"],
             "paired_tcrs": len(stats["paired_tcrs"]),
             "unique_beta_cdr3": len(stats["beta_cdr3s"]),
             "unique_epitopes": 0},
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
    paired = int(r["paired_records"])
    paired_pct = 100.0 * paired / total if total else 0
    num_files = len(stats["meta_rows"])
    lines = []

    # --- Header ---
    lines.append("=" * 80)
    lines.append("                    OTS Analysis Summary")
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {DATA_DIR}")
    lines.append("=" * 80)
    lines.append("")

    # --- 1. DATASET OVERVIEW ---
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Total records:          {total:>12,}")
    lines.append(f"Unique beta CDR3s:      {r['unique_beta_cdr3']:>12,}")
    lines.append(f"Unique alpha CDR3s:     {r['unique_alpha_cdr3']:>12,}")
    lines.append(
        f"Paired records (a+b):   {paired:>12,}  ({paired_pct:.1f}% of total)"
    )
    lines.append(f"Unique paired TCRs:     {r['unique_paired_tcrs']:>12,}")
    lines.append(f"Unique epitopes:        {'N/A':>12s}  (no epitope annotation)")
    lines.append(f"Unique MHC alleles:     {'N/A':>12s}  (no MHC annotation)")
    lines.append(f"Source files:           {num_files:>12,}")
    lines.append("")

    lines.append("Disease category distribution:")
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cat = row["category"]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cat:<20s} {cnt:>12,}  ({pct:>5.1f}%)")
    lines.append("")

    # --- 2. MHC CLASS BREAKDOWN ---
    lines.append("2. MHC CLASS BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available in this database. OTS does not include MHC annotations."
    )
    lines.append("")

    # --- 3. CATEGORY BREAKDOWN ---
    lines.append("3. CATEGORY BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Category':<20s} {'Records':>12s} {'%':>7s} "
        f"{'Paired TCRs':>13s} {'Beta CDR3s':>12s}"
    )
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['category'])[:18]:<20s} {cnt:>12,} {pct:>6.1f}% "
            f"{int(row['unique_paired_tcrs']):>13,} {int(row['unique_beta_cdr3']):>12,}"
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
        "Not available in this database. OTS does not include epitope annotations."
    )
    lines.append("")

    # --- 5. TOP MHC ALLELES ---
    lines.append("5. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available in this database. OTS does not include MHC annotations."
    )
    lines.append("")

    # --- 6. PATHOLOGY & SPECIES ---
    lines.append("6. PATHOLOGY & SPECIES")
    lines.append("-" * 80)
    lines.append("")

    host_species = species[species["category"] == "host_species"].copy()
    host_species = host_species.sort_values("total_records", ascending=False)
    host_total = int(host_species["total_records"].sum())

    if len(host_species) > 0:
        lines.append("Host species breakdown:")
        for _, row in host_species.iterrows():
            sp = str(row["species"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / host_total if host_total else 0
            lines.append(
                f"  {sp + ':':<25s} {cnt:>12,} records  ({pct:>5.1f}%)"
            )
        lines.append("")

    disease_species = species[species["category"] == "disease"].copy()
    disease_species = disease_species.sort_values("total_records", ascending=False)
    if len(disease_species) > 0:
        lines.append("Disease breakdown:")
        for _, row in disease_species.head(10).iterrows():
            sp = str(row["species"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total if total else 0
            lines.append(f"  {sp:<20s} {cnt:>12,}  ({pct:>5.1f}%)")
        lines.append("")

    # --- 7. STUDY LANDSCAPE ---
    lines.append("7. STUDY LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")

    total_studies = len(studies)
    studies_with_paired = int((studies["paired_records"] > 0).sum())
    studies_with_paired_pct = (
        100.0 * studies_with_paired / total_studies if total_studies else 0
    )

    lines.append(f"Total studies (Author | Link): {total_studies}")
    lines.append(
        f"Studies with paired TCR data: {studies_with_paired} "
        f"({studies_with_paired_pct:.0f}%)"
    )
    lines.append("")

    lines.append("Largest contributors:")
    top_studies = studies.nlargest(5, "total_records")
    for _, row in top_studies.iterrows():
        study_label = str(row["study"])[:40]
        lines.append(
            f"  {study_label:<42s} {int(row['total_records']):>10,} records  "
            f"({int(row['paired_records']):>8,} paired)"
        )
    lines.append("")

    top5_records = int(top_studies["total_records"].sum())
    top5_pct = 100.0 * top5_records / total if total else 0
    lines.append(
        f"The top 5 studies account for ~{top5_pct:.0f}% of all records."
    )
    lines.append("")

    # --- 8. DATA COMPLETENESS ---
    lines.append("8. DATA COMPLETENESS")
    lines.append("-" * 80)
    lines.append("")

    criteria = {
        "has_epitope_and_mhc": "Has both epitope sequence and MHC annotation.",
        "has_epitope_only": "Has epitope but missing MHC allele.",
        "has_mhc_only": "Has MHC annotation but no epitope sequence.",
        "no_epitope_no_mhc": "Missing both epitope and MHC (TCR-only records).",
    }

    lines.append(
        f"  {'Category':<25s} {'Records':>12s} {'%':>7s} "
        f"{'Paired TCRs':>13s} {'Beta CDR3s':>12s}"
    )
    for _, row in completeness.iterrows():
        cat = row["category"]
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {cat:<25s} {cnt:>12,} {pct:>6.1f}% "
            f"{int(row['paired_tcrs']):>13,} {int(row['unique_beta_cdr3']):>12,}"
        )
    lines.append("")

    lines.append("Criteria:")
    for cat, desc in criteria.items():
        lines.append(f"  - {cat:<25s} {desc}")
    lines.append("")

    lines.append(
        "All OTS records are TCR-only (no epitope or MHC annotation).\n"
        "This data is best suited for unsupervised pre-training, paired TCR\n"
        "modeling, and repertoire analysis."
    )
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

    lines.append(
        f"a) Data scale: {total:,} records across {num_files} paired repertoire files.\n"
        f"   {int(r['unique_beta_cdr3']):,} unique beta CDR3s and "
        f"{int(r['unique_alpha_cdr3']):,} unique alpha CDR3s."
    )
    lines.append("")

    lines.append(
        f"b) Paired TCR availability: {int(r['unique_paired_tcrs']):,} unique paired TCRs\n"
        f"   ({paired_pct:.1f}% of records have both alpha+beta chains).\n"
        f"   OTS is a rich source of paired TCR data for alpha-beta modeling."
    )
    lines.append("")

    lines.append(
        f"c) Category bias: '{dominant['category']}' accounts for {dom_pct:.1f}% "
        f"of records."
    )
    lines.append("")

    lines.append(
        "d) No epitope/MHC: OTS does not include epitope or MHC annotations.\n"
        "   This data cannot be used for TCR-pMHC binding prediction\n"
        "   but is valuable for TCR language modeling and paired TCR analysis."
    )
    lines.append("")

    lines.append(
        "e) V/J gene annotation: Full V and J gene calls are available for\n"
        "   both alpha and beta chains, enabling gene usage analysis."
    )
    lines.append("")

    if len(host_species) >= 2:
        top_host = host_species.iloc[0]
        top_host_pct = (
            100.0 * int(top_host["total_records"]) / host_total
            if host_total
            else 0
        )
        lines.append(
            f"f) Species coverage: {top_host['species']} accounts for "
            f"{top_host_pct:.1f}% of records\n"
            f"   with species annotation."
        )
    elif len(host_species) == 1:
        lines.append(
            f"f) Species coverage: All data is from {host_species.iloc[0]['species']} samples."
        )
    else:
        lines.append(
            "f) Species coverage: Species metadata available per file."
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

    print(f"Streaming OTS data from {data_dir} ...")
    stats = aggregate_data(data_dir)
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  Paired records: {stats['total_paired']:,}")
    print(f"  Unique beta CDR3s: {len(stats['beta_cdr3s']):,}")
    print(f"  Unique alpha CDR3s: {len(stats['alpha_cdr3s']):,}")
    print(f"  Unique paired TCRs: {len(stats['paired_tcrs']):,}")
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
    comp_sum = comp["record_count"].sum()
    print(
        f"Completeness sum: {comp_sum}, Overview total: {all_total}, "
        f"match: {comp_sum == all_total}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
