#!/usr/bin/env python3
"""Export comprehensive tcrdb analysis tables.

Memory-optimized: streams one CSV at a time, accumulates stats in
sets/counters. Never holds the full 39 GB dataset in memory.
"""

import gc
import os
import resource
from collections import Counter, defaultdict

import pandas as pd


def _mem_mb():
    """Return current RSS in MB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

DATA_DIR = "data/databases/tcrdb"
METADATA_FILE = "data/databases/tcrdb/tcrdb_all_metadata_combined.csv"
OUTPUT_DIR = "data/analysis/tcrdb"

CATEGORIES = [
    "cancer", "viral", "autoimmunity", "healthy", "inflammation", "transplantation"
]


def _chain_type(vregion, chain_col):
    """Determine chain type from Chain column or Vregion prefix."""
    if chain_col:
        ch = chain_col.strip("[]' ").upper()
        if ch in ("TRB", "TCRB"):
            return "beta"
        if ch in ("TRA", "TCRA"):
            return "alpha"
    vr = str(vregion)
    if vr.startswith("TRBV"):
        return "beta"
    if vr.startswith("TRAV"):
        return "alpha"
    return "unknown"


def _vectorized_chain_types(df):
    """Vectorized chain type assignment for a single file DataFrame."""
    n = len(df)
    result = ["unknown"] * n

    has_chain = "Chain" in df.columns
    has_vregion = "Vregion" in df.columns

    if has_chain:
        chain_upper = df["Chain"].str.strip("[]' ").str.upper()
        for i, val in enumerate(chain_upper):
            if val in ("TRB", "TCRB"):
                result[i] = "beta"
            elif val in ("TRA", "TCRA"):
                result[i] = "alpha"

    if has_vregion:
        for i, vr in enumerate(df["Vregion"]):
            if result[i] == "unknown" and vr:
                if vr.startswith("TRBV"):
                    result[i] = "beta"
                elif vr.startswith("TRAV"):
                    result[i] = "alpha"

    return result


def aggregate_data(data_dir, metadata_path):
    """Stream through tcrdb CSV files and accumulate statistics.

    Memory-optimized for 16 GB machines:
    - Global uniqueness tracked via hash(seq) (8-byte ints, ~4x smaller)
    - Categories processed sequentially (only 1 category's string sets alive)
    - Species uniqueness tracked via hash(seq)
    """
    # Load lightweight metadata lookup (11K rows)
    meta = pd.read_csv(metadata_path, dtype=str, na_filter=False)
    print(f"  Metadata: {len(meta):,} rows")
    meta_lookup = {}
    if "RunId" in meta.columns:
        for _, row in meta.iterrows():
            meta_lookup[row["RunId"]] = {
                "Species": row.get("Species", ""),
                "Condition": row.get("Condition", ""),
            }

    # --- Global accumulators (hash-based for memory) ---
    total_records = 0
    global_beta_hashes = set()
    global_alpha_hashes = set()
    chain_counts = Counter()

    # --- Per-category: counts only (sets built per-category below) ---
    cat_records = Counter()
    cat_beta_counts = {}
    cat_alpha_counts = {}

    # --- Per-species accumulators (hash-based) ---
    species_records = Counter()
    species_beta_hashes = defaultdict(set)

    # --- Per-project rows ---
    project_rows = []

    file_count = 0
    skipped = 0

    print(f"  RSS before processing: {_mem_mb():.0f} MB")

    # Process one category at a time to limit peak memory
    for category in CATEGORIES:
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            print(f"  Skipping missing category dir: {cat_dir}")
            continue

        # Per-category string sets (freed after this category)
        cat_beta_set = set()
        cat_alpha_set = set()

        for project_id in sorted(os.listdir(cat_dir)):
            proj_dir = os.path.join(cat_dir, project_id)
            if not os.path.isdir(proj_dir):
                continue

            csv_files = sorted(
                f for f in os.listdir(proj_dir) if f.endswith(".csv")
            )
            if not csv_files:
                continue

            # Per-project accumulators (cleared after each project)
            proj_records = 0
            proj_beta = set()
            proj_alpha = set()
            proj_files = 0
            proj_chain_types = set()
            proj_species = set()

            for csv_file in csv_files:
                fpath = os.path.join(proj_dir, csv_file)
                try:
                    df = pd.read_csv(fpath, dtype=str, na_filter=False)
                except Exception:
                    skipped += 1
                    continue

                if len(df) == 0:
                    continue

                if "AASeq" not in df.columns:
                    skipped += 1
                    continue

                # Filter out rows with empty AASeq
                df = df[df["AASeq"] != ""]
                if len(df) == 0:
                    continue

                # Determine RunId for metadata lookup
                run_id = (
                    df["RunId"].iloc[0]
                    if "RunId" in df.columns
                    else os.path.splitext(csv_file)[0]
                )
                meta_info = meta_lookup.get(run_id, {})
                sp = meta_info.get("Species", "")

                # Vectorized chain type assignment
                chain_types = _vectorized_chain_types(df)
                aaseqs = df["AASeq"].values
                n = len(df)

                # Count chain types
                for ct_val in chain_types:
                    chain_counts[ct_val] += 1

                # Update accumulators
                for i in range(n):
                    ct = chain_types[i]
                    seq = aaseqs[i]
                    if ct == "beta":
                        global_beta_hashes.add(hash(seq))
                        cat_beta_set.add(seq)
                        proj_beta.add(seq)
                        if sp:
                            species_beta_hashes[sp].add(hash(seq))
                    elif ct == "alpha":
                        global_alpha_hashes.add(hash(seq))
                        cat_alpha_set.add(seq)
                        proj_alpha.add(seq)

                total_records += n
                cat_records[category] += n
                proj_records += n
                proj_chain_types.update(chain_types)

                if sp:
                    species_records[sp] += n
                    proj_species.add(sp)

                proj_files += 1
                file_count += 1
                del df, chain_types, aaseqs

                if file_count % 500 == 0:
                    print(f"    Processed {file_count} files ... RSS={_mem_mb():.0f} MB")
                    gc.collect()

            # Save project summary, then free project sets
            if proj_files > 0:
                project_rows.append(
                    {
                        "project_id": project_id,
                        "total_records": proj_records,
                        "unique_beta_cdr3": len(proj_beta),
                        "unique_alpha_cdr3": len(proj_alpha),
                        "num_files": proj_files,
                        "categories": category,
                        "species": ";".join(sorted(proj_species)),
                        "chain_types": ";".join(sorted(proj_chain_types)),
                    }
                )
            del proj_beta, proj_alpha
            gc.collect()

        # Record category counts, then free category string sets
        cat_beta_counts[category] = len(cat_beta_set)
        cat_alpha_counts[category] = len(cat_alpha_set)
        print(
            f"  Category '{category}': {cat_records[category]:,} records, "
            f"{cat_beta_counts[category]:,} unique beta, "
            f"{cat_alpha_counts[category]:,} unique alpha, "
            f"RSS={_mem_mb():.0f} MB"
        )
        del cat_beta_set, cat_alpha_set
        gc.collect()

    print(f"  Processed {file_count} files ({skipped} skipped)")
    print(f"  RSS after processing: {_mem_mb():.0f} MB")

    return {
        "total_records": total_records,
        "global_beta_count": len(global_beta_hashes),
        "global_alpha_count": len(global_alpha_hashes),
        "chain_counts": chain_counts,
        "cat_records": cat_records,
        "cat_beta_counts": cat_beta_counts,
        "cat_alpha_counts": cat_alpha_counts,
        "species_records": species_records,
        "species_beta_counts": {sp: len(s) for sp, s in species_beta_hashes.items()},
        "project_rows": project_rows,
    }


# ---------------------------------------------------------------------------
# Build TSV DataFrames from aggregated stats
# ---------------------------------------------------------------------------
def overview(stats):
    return pd.DataFrame(
        [
            {
                "total_records": stats["total_records"],
                "unique_beta_cdr3": stats["global_beta_count"],
                "unique_alpha_cdr3": stats["global_alpha_count"],
                "paired_records": 0,
                "unique_paired_tcrs": 0,
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
                "unique_beta_cdr3": stats["cat_beta_counts"].get(cat, 0),
                "unique_alpha_cdr3": stats["cat_alpha_counts"].get(cat, 0),
                "paired_records": 0,
                "unique_paired_tcrs": 0,
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
    return pd.DataFrame(stats["project_rows"])


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
                "unique_beta_cdr3": stats["species_beta_counts"].get(sp, 0),
            }
        )
    for cat in sorted(stats["cat_records"]):
        rows.append(
            {
                "category": "condition_category",
                "species": cat,
                "total_records": stats["cat_records"][cat],
                "unique_epitopes": 0,
                "unique_beta_cdr3": stats["cat_beta_counts"].get(cat, 0),
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
             "paired_tcrs": 0,
             "unique_beta_cdr3": stats["global_beta_count"],
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
    lines = []

    # --- Header ---
    lines.append("=" * 80)
    lines.append("                    tcrdb Analysis Summary")
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {DATA_DIR}")
    lines.append("=" * 80)
    lines.append("")

    # --- 1. DATASET OVERVIEW ---
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Total records:          {total:>10,}")
    lines.append(f"Unique beta CDR3s:      {r['unique_beta_cdr3']:>10,}")
    lines.append(f"Unique alpha CDR3s:     {r['unique_alpha_cdr3']:>10,}")
    lines.append(f"Paired records (a+b):   {'N/A':>10s}  (single-chain data)")
    lines.append(f"Unique epitopes:        {'N/A':>10s}  (no epitope annotation)")
    lines.append(f"Unique MHC alleles:     {'N/A':>10s}  (no MHC annotation)")
    lines.append("")

    # Chain type distribution
    chain_counts = stats["chain_counts"]
    lines.append("Chain type distribution:")
    for chain in sorted(chain_counts, key=chain_counts.get, reverse=True):
        cnt = chain_counts[chain]
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {chain:<12s} {cnt:>10,}  ({pct:>5.1f}%)")
    lines.append("")

    # Category distribution
    lines.append("Category distribution:")
    for _, row in cat_ov.iterrows():
        cat = row["category"]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cat:<20s} {cnt:>10,}  ({pct:>5.1f}%)")
    lines.append("")

    # --- 2. MHC CLASS BREAKDOWN ---
    lines.append("2. MHC CLASS BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available in this database. tcrdb does not include MHC annotations."
    )
    lines.append("")

    # --- 3. CATEGORY BREAKDOWN ---
    lines.append("3. CATEGORY BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Category':<20s} {'Records':>10s} {'%':>7s} "
        f"{'Beta CDR3s':>12s} {'Alpha CDR3s':>13s}"
    )
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {row['category']:<20s} {cnt:>10,} {pct:>6.1f}% "
            f"{int(row['unique_beta_cdr3']):>12,} {int(row['unique_alpha_cdr3']):>13,}"
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
        "Not available in this database. tcrdb does not include epitope annotations."
    )
    lines.append("")

    # --- 5. TOP MHC ALLELES ---
    lines.append("5. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available in this database. tcrdb does not include MHC annotations."
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
                f"  {sp + ':':<25s} {cnt:>10,} records  ({pct:>5.1f}%)"
            )
        lines.append("")

    cond_species = species[species["category"] == "condition_category"].copy()
    cond_species = cond_species.sort_values("total_records", ascending=False)
    if len(cond_species) > 0:
        lines.append("Condition category breakdown:")
        for _, row in cond_species.iterrows():
            sp = str(row["species"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total if total else 0
            lines.append(f"  {sp:<20s} {cnt:>10,}  ({pct:>5.1f}%)")
        lines.append("")

    # --- 7. STUDY LANDSCAPE ---
    lines.append("7. STUDY LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")

    total_projects = len(studies)
    lines.append(f"Total projects: {total_projects}")
    lines.append(f"Total data files: {int(studies['num_files'].sum()):,}")
    lines.append("")

    lines.append("Largest projects:")
    top_studies = studies.nlargest(10, "total_records")
    for _, row in top_studies.iterrows():
        proj = str(row["project_id"])
        lines.append(
            f"  {proj:<25s} {int(row['total_records']):>10,} records  "
            f"({int(row['num_files']):>4,} files, "
            f"{int(row['unique_beta_cdr3']):>7,} beta CDR3s)"
        )
    lines.append("")

    top5_records = int(top_studies.head(5)["total_records"].sum())
    top5_pct = 100.0 * top5_records / total if total else 0
    lines.append(
        f"The top 5 projects account for ~{top5_pct:.0f}% of all records."
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
        f"  {'Category':<25s} {'Records':>10s} {'%':>7s} "
        f"{'Beta CDR3s':>12s}"
    )
    for _, row in completeness.iterrows():
        cat = row["category"]
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {cat:<25s} {cnt:>10,} {pct:>6.1f}% "
            f"{int(row['unique_beta_cdr3']):>12,}"
        )
    lines.append("")

    lines.append("Criteria:")
    for cat, desc in criteria.items():
        lines.append(f"  - {cat:<25s} {desc}")
    lines.append("")

    lines.append(
        "All tcrdb records are TCR-only (no epitope or MHC annotation).\n"
        "This data is best suited for unsupervised pre-training and repertoire analysis."
    )
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

    lines.append(
        f"a) Data scale: {total:,} clonotype records across {total_projects} projects.\n"
        f"   {int(r['unique_beta_cdr3']):,} unique beta CDR3s and "
        f"{int(r['unique_alpha_cdr3']):,} unique alpha CDR3s.\n"
        f"   Large-scale repertoire data suitable for unsupervised pre-training."
    )
    lines.append("")

    lines.append(
        "b) Chain coverage: Data is predominantly single-chain per row.\n"
        "   No paired alpha+beta data is available at the row level.\n"
        "   V/D/J gene segment annotation is included for most records."
    )
    lines.append("")

    lines.append(
        f"c) Category bias: '{dominant['category']}' accounts for {dom_pct:.1f}% "
        f"of records.\n"
        f"   The 6 categories provide disease-context labels for repertoire-level\n"
        f"   classification tasks."
    )
    lines.append("")

    lines.append(
        "d) No epitope/MHC: tcrdb does not include epitope or MHC annotations.\n"
        "   This data cannot be used for TCR-pMHC binding prediction\n"
        "   but is valuable for TCR language modeling and repertoire analysis."
    )
    lines.append("")

    lines.append(
        "e) Clone counts: Most records include cloneCount and cloneFraction,\n"
        "   enabling frequency-weighted analysis and expanded clone identification."
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
    else:
        lines.append(
            "f) Species coverage: Species metadata is available via joined metadata."
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
    metadata_path = os.path.join(project_root, METADATA_FILE)
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Streaming tcrdb data from {data_dir} ...")
    stats = aggregate_data(data_dir, metadata_path)
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  Unique beta CDR3s: {stats['global_beta_count']:,}")
    print(f"  Unique alpha CDR3s: {stats['global_alpha_count']:,}")
    print(f"  Chain counts: {dict(stats['chain_counts'])}")
    print(f"  Category counts: {dict(stats['cat_records'])}")
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
