#!/usr/bin/env python3
"""Export comprehensive TaDB analysis tables.

TaDB is a database of tumor-associated T cell epitopes with HLA allele
restrictions. Single CSV file with 1,147 records.
"""

import os
import re

import pandas as pd

DATA_DIR = "data/databases/TaDB"
OUTPUT_DIR = "data/analysis/TaDB"

DATA_FILE = "tadb_t_cell_epitopes.csv"


def classify_mhc(allele):
    """Classify an HLA allele as MHCI or MHCII."""
    a = allele.upper().strip()
    # Class II patterns: DR, DQ, DP, HLA-D, "HLA class II"
    if any(
        x in a
        for x in ["DR", "DQ", "DP", "HLA CLASS II", "HLA-D"]
    ):
        return "MHCII"
    # Class I patterns: A, B, C, HLA-A, HLA-B, HLA-C, or A*XXXX
    if re.match(r"^[ABC]\*?\d", a) or re.match(r"^HLA-[ABC]", a):
        return "MHCI"
    return "unknown"


def load_data(data_dir):
    """Load TaDB data."""
    fpath = os.path.join(data_dir, DATA_FILE)
    df = pd.read_csv(fpath, dtype=str)
    df = df.fillna("")

    # Derive columns
    df["has_epitope"] = df["Epitope sequence"] != ""
    df["has_mhc"] = df["HLA allele"] != ""
    df["mhc_class"] = df["HLA allele"].apply(classify_mhc)
    df["peptide_length"] = df["Epitope sequence"].str.len()

    # pMHC key
    df["pmhc"] = df["Epitope sequence"] + "|" + df["HLA allele"]

    # Category = Epitope type
    df["category"] = df["Epitope type"].replace("", "(unknown)")

    return df


# ---------------------------------------------------------------------------
# Report functions
# ---------------------------------------------------------------------------
def overview(df):
    return pd.DataFrame(
        [
            {
                "total_records": len(df),
                "unique_epitopes": df.loc[
                    df["has_epitope"], "Epitope sequence"
                ].nunique(),
                "unique_mhc": df.loc[df["has_mhc"], "HLA allele"].nunique(),
                "unique_pmhc": df["pmhc"].nunique(),
                "unique_beta_cdr3": 0,
                "unique_alpha_cdr3": 0,
                "paired_records": 0,
                "unique_paired_tcrs": 0,
                "unique_tcr_pmhc_paired": 0,
            }
        ]
    )


def category_overview(df):
    rows = []
    for cat, grp in df.groupby("category", sort=True):
        rows.append(
            {
                "category": cat,
                "total_records": len(grp),
                "unique_epitopes": grp.loc[
                    grp["has_epitope"], "Epitope sequence"
                ].nunique(),
                "unique_mhc": grp.loc[grp["has_mhc"], "HLA allele"].nunique(),
                "unique_pmhc": grp["pmhc"].nunique(),
                "paired_records": 0,
                "unique_paired_tcrs": 0,
                "unique_epitopes": grp["Epitope sequence"].nunique(),
                "unique_mhc": grp["HLA allele"].nunique(),
                "unique_pmhc": grp["pmhc"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def epitope_summary(df):
    rows = []
    for epitope, grp in df[df["has_epitope"]].groupby(
        "Epitope sequence", sort=True
    ):
        primary_mhc = grp["HLA allele"].mode()
        primary_mhc = primary_mhc.iloc[0] if not primary_mhc.empty else ""
        mhc_class = grp["mhc_class"].mode()
        mhc_class = mhc_class.iloc[0] if not mhc_class.empty else "unknown"
        etype = grp["Epitope type"].mode()
        etype = etype.iloc[0] if not etype.empty else ""

        rows.append(
            {
                "epitope": epitope,
                "epitope_type": etype,
                "primary_mhc": primary_mhc,
                "mhc_class": mhc_class,
                "peptide_length": len(epitope),
                "total_records": len(grp),
                "unique_mhc_alleles": grp["HLA allele"].nunique(),
                "paired_records": 0,
                "unique_paired_tcrs": 0,
                "unique_beta_only_tcrs": 0,
            }
        )
    return pd.DataFrame(rows)


def pmhc_summary(df):
    rows = []
    for (epitope, mhc), grp in df[df["has_epitope"] & df["has_mhc"]].groupby(
        ["Epitope sequence", "HLA allele"], sort=True
    ):
        mhc_class = grp["mhc_class"].mode()
        mhc_class = mhc_class.iloc[0] if not mhc_class.empty else "unknown"
        etype = grp["Epitope type"].mode()
        etype = etype.iloc[0] if not etype.empty else ""

        rows.append(
            {
                "epitope": epitope,
                "mhc": mhc,
                "mhc_class": mhc_class,
                "epitope_type": etype,
                "total_records": len(grp),
                "unique_paired_tcrs": 0,
                "unique_beta_only_tcrs": 0,
            }
        )
    return pd.DataFrame(rows)


def study_summary():
    """TaDB has no study/publication info."""
    return pd.DataFrame(
        columns=["study", "total_records", "paired_records", "unique_epitopes"]
    )


def mhc_allele_summary(df):
    rows = []
    valid = df[df["has_mhc"]]
    for (mhc, mhc_class), grp in valid.groupby(
        ["HLA allele", "mhc_class"], sort=True
    ):
        rows.append(
            {
                "mhc": mhc,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "unique_epitopes": grp["Epitope sequence"].nunique(),
                "unique_paired_tcrs": 0,
            }
        )
    return pd.DataFrame(rows)


def species_breakdown(df):
    """Epitope type and MHC class breakdown."""
    rows = []
    # MHC class breakdown
    for mc, grp in df.groupby("mhc_class", sort=True):
        rows.append(
            {
                "category": "mhc_class",
                "species": mc,
                "total_records": len(grp),
                "unique_epitopes": grp["Epitope sequence"].nunique(),
                "unique_paired_tcrs": 0,
            }
        )
    # Peptide length breakdown
    for plen, grp in df.groupby("peptide_length", sort=True):
        rows.append(
            {
                "category": "peptide_length",
                "species": str(plen),
                "total_records": len(grp),
                "unique_epitopes": grp["Epitope sequence"].nunique(),
                "unique_paired_tcrs": 0,
            }
        )
    return pd.DataFrame(rows)


def data_completeness(df):
    has_epitope = df["has_epitope"]
    has_mhc = df["has_mhc"]
    categories = {
        "has_epitope_and_mhc": has_epitope & has_mhc,
        "has_epitope_only": has_epitope & ~has_mhc,
        "has_mhc_only": ~has_epitope & has_mhc,
        "no_epitope_no_mhc": ~has_epitope & ~has_mhc,
    }
    rows = []
    for label, mask in categories.items():
        sub = df[mask]
        rows.append(
            {
                "category": label,
                "record_count": len(sub),
                "paired_tcrs": 0,
                "unique_beta_cdr3": 0,
                "unique_epitopes": sub["Epitope sequence"].nunique()
                if len(sub) > 0
                else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SUMMARY.txt
# ---------------------------------------------------------------------------
def write_summary_txt(df, output_dir):
    from datetime import date

    ov = pd.read_csv(os.path.join(output_dir, "overview.tsv"), sep="\t")
    cat_ov = pd.read_csv(
        os.path.join(output_dir, "category_overview.tsv"), sep="\t"
    )
    epi = pd.read_csv(os.path.join(output_dir, "epitope_summary.tsv"), sep="\t")
    completeness = pd.read_csv(
        os.path.join(output_dir, "data_completeness.tsv"), sep="\t"
    )
    mhc_alleles = pd.read_csv(
        os.path.join(output_dir, "mhc_allele_summary.tsv"), sep="\t"
    )
    species = pd.read_csv(
        os.path.join(output_dir, "species_breakdown.tsv"), sep="\t"
    )

    r = ov.iloc[0]
    total = int(r["total_records"])
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("                    TaDB Analysis Summary")
    lines.append("          Tumor-Associated T Cell Epitope Database")
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {DATA_DIR}")
    lines.append("=" * 80)
    lines.append("")

    # 1. DATASET OVERVIEW
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Total records:          {total:>10,}")
    lines.append(f"Unique epitopes:        {int(r['unique_epitopes']):>10,}")
    lines.append(f"Unique HLA alleles:     {int(r['unique_mhc']):>10,}")
    lines.append(f"Unique pMHC combos:     {int(r['unique_pmhc']):>10,}")
    lines.append(f"TCR sequences:          {'N/A':>10s}  (epitope-only database)")
    lines.append("")

    lines.append("Epitope type distribution:")
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['category']):<38s} {cnt:>6,}  ({pct:>5.1f}%)"
        )
    lines.append("")

    # 2. MHC CLASS BREAKDOWN
    lines.append("2. MHC CLASS BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    mhc_class_agg = (
        mhc_alleles.groupby("mhc_class")
        .agg(epitopes=("unique_epitopes", "sum"), records=("total_records", "sum"))
        .reset_index()
        .sort_values("records", ascending=False)
    )
    mhc_total = int(mhc_class_agg["records"].sum())
    lines.append(f"  {'Class':<12s} {'Epitopes':>10s} {'Records':>10s}")
    for _, row in mhc_class_agg.iterrows():
        lines.append(
            f"  {row['mhc_class']:<12s} {int(row['epitopes']):>10,} "
            f"{int(row['records']):>10,}"
        )
    lines.append("")

    # 3. CATEGORY BREAKDOWN
    lines.append("3. CATEGORY BREAKDOWN (BY EPITOPE TYPE)")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Epitope Type':<38s} {'Records':>7s} {'%':>7s} {'Epitopes':>10s}"
    )
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['category']):<38s} {cnt:>7,} {pct:>6.1f}% "
            f"{int(row['unique_epitopes']):>10,}"
        )
    lines.append("")

    dominant = cat_ov.loc[cat_ov["total_records"].idxmax()]
    dom_pct = 100.0 * int(dominant["total_records"]) / total if total else 0
    lines.append(
        f"'{dominant['category']}' has the most records with {dom_pct:.1f}% "
        f"of the database."
    )
    lines.append("")

    # 4. TOP EPITOPES
    lines.append("4. TOP EPITOPES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Epitope':<22s} {'Type':<30s} {'HLA':<15s} {'Records':>7s}"
    )
    top_epi = epi.nlargest(15, "total_records")
    for _, row in top_epi.iterrows():
        lines.append(
            f"  {str(row['epitope'])[:20]:<22s} "
            f"{str(row['epitope_type'])[:28]:<30s} "
            f"{str(row['primary_mhc'])[:13]:<15s} {int(row['total_records']):>7,}"
        )
    lines.append("")

    # 5. TOP MHC ALLELES
    lines.append("5. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Allele':<25s} {'Class':<10s} {'Records':>7s} {'Epitopes':>10s}"
    )
    top_mhc = mhc_alleles.nlargest(15, "total_records")
    for _, row in top_mhc.iterrows():
        lines.append(
            f"  {str(row['mhc'])[:23]:<25s} {row['mhc_class']:<10s} "
            f"{int(row['total_records']):>7,} {int(row['unique_epitopes']):>10,}"
        )
    lines.append("")

    if len(top_mhc) > 0:
        top_a = top_mhc.iloc[0]
        top_pct = 100.0 * int(top_a["total_records"]) / mhc_total if mhc_total else 0
        lines.append(
            f"{top_a['mhc']} is the most represented allele with "
            f"{top_pct:.1f}% of records."
        )
    lines.append("")

    # 6. PATHOLOGY & SPECIES
    lines.append("6. PEPTIDE LENGTH & MHC CLASS")
    lines.append("-" * 80)
    lines.append("")

    mhc_rows = species[species["category"] == "mhc_class"]
    lines.append("MHC class breakdown:")
    for _, row in mhc_rows.sort_values("total_records", ascending=False).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['species']):<15s} {cnt:>6,}  ({pct:>5.1f}%)"
        )
    lines.append("")

    plen_rows = species[species["category"] == "peptide_length"].copy()
    plen_rows["species"] = plen_rows["species"].astype(int)
    plen_rows = plen_rows.sort_values("total_records", ascending=False)
    lines.append("Top peptide lengths:")
    for _, row in plen_rows.head(10).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {int(row['species']):>3d} aa: {cnt:>6,} records  ({pct:>5.1f}%)"
        )
    lines.append("")

    # 7. STUDY LANDSCAPE
    lines.append("7. STUDY LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")
    lines.append("TaDB does not include publication or study metadata.")
    lines.append("")

    # 8. DATA COMPLETENESS
    lines.append("8. DATA COMPLETENESS")
    lines.append("-" * 80)
    lines.append("")
    criteria = {
        "has_epitope_and_mhc": "Has both epitope sequence and HLA allele.",
        "has_epitope_only": "Has epitope but missing HLA allele.",
        "has_mhc_only": "Has HLA allele but no epitope sequence.",
        "no_epitope_no_mhc": "Missing both epitope and HLA.",
    }
    lines.append(f"  {'Category':<25s} {'Records':>7s} {'%':>7s} {'Epitopes':>10s}")
    for _, row in completeness.iterrows():
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {row['category']:<25s} {cnt:>7,} {pct:>6.1f}% "
            f"{int(row['unique_epitopes']):>10,}"
        )
    lines.append("")
    lines.append("Criteria:")
    for cat, desc in criteria.items():
        lines.append(f"  - {cat:<25s} {desc}")
    lines.append("")

    # 9. KEY TAKEAWAYS
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"a) Tumor epitope catalog: {total:,} epitope-HLA records covering\n"
        f"   {int(r['unique_epitopes']):,} unique epitope sequences across "
        f"{int(r['unique_mhc']):,} HLA alleles."
    )
    lines.append("")
    lines.append(
        "b) No TCR data: TaDB is an epitope-only database. No TCR sequences\n"
        "   are included. Useful for MHC-peptide binding studies."
    )
    lines.append("")
    lines.append(
        f"c) Cancer focus: All epitopes are tumor-associated, with\n"
        f"   '{dominant['category']}' being the largest category ({dom_pct:.1f}%)."
    )
    lines.append("")
    lines.append(
        "d) Peptide lengths span 8-31 aa, with 9-mers most common\n"
        "   (standard MHC-I binding length)."
    )
    lines.append("")

    # Footer
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

    with open(os.path.join(output_dir, "SUMMARY.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {os.path.join(output_dir, 'SUMMARY.txt')}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_dir = os.path.join(project_root, DATA_DIR)
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading TaDB data from {data_dir} ...")
    df = load_data(data_dir)
    print(f"  Loaded {len(df):,} records")
    print(f"  Unique epitopes: {df['Epitope sequence'].nunique()}")
    print(f"  Unique HLA alleles: {df['HLA allele'].nunique()}")
    print()

    exports = [
        ("overview.tsv", lambda d: overview(d)),
        ("category_overview.tsv", lambda d: category_overview(d)),
        ("epitope_summary.tsv", lambda d: epitope_summary(d)),
        ("pmhc_summary.tsv", lambda d: pmhc_summary(d)),
        ("study_summary.tsv", lambda d: study_summary()),
        ("mhc_allele_summary.tsv", lambda d: mhc_allele_summary(d)),
        ("species_breakdown.tsv", lambda d: species_breakdown(d)),
        ("data_completeness.tsv", lambda d: data_completeness(d)),
    ]

    for filename, func in exports:
        print(f"Computing {filename} ...")
        result = func(df)
        path = os.path.join(output_dir, filename)
        result.to_csv(path, sep="\t", index=False)
        print(f"  -> {path}  ({len(result)} rows)")

    print("Writing SUMMARY.txt ...")
    write_summary_txt(df, output_dir)

    # Verification
    print("\n--- Verification ---")
    ov = pd.read_csv(os.path.join(output_dir, "overview.tsv"), sep="\t")
    cat_ov = pd.read_csv(os.path.join(output_dir, "category_overview.tsv"), sep="\t")
    print(f"Category sum: {cat_ov['total_records'].sum()}, Overview: {ov.iloc[0]['total_records']}, match: {cat_ov['total_records'].sum() == ov.iloc[0]['total_records']}")
    comp = pd.read_csv(os.path.join(output_dir, "data_completeness.tsv"), sep="\t")
    print(f"Completeness sum: {comp['record_count'].sum()}, match: {comp['record_count'].sum() == ov.iloc[0]['total_records']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
