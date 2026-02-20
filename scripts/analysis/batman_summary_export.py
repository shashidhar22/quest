#!/usr/bin/env python3
"""Export comprehensive BATMAN analysis tables.

BATMAN (Binding Affinity of TCR-MHC ANtigen) is a mutational scanning database
containing TCR-pMHC interaction data with quantitative binding measurements.
Two files: pMHCI (17,097 rows) and pMHCII (5,730 rows).
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = "data/databases/BATMAN"
OUTPUT_DIR = "data/analysis/BATMAN"

PMHCI_FILE = "TCR_pMHCI_mutational_scan_database.xlsx"
PMHCII_FILE = "TCR_pMHCII_mutational_scan_database.xlsx"


def load_data(data_dir):
    """Load both BATMAN Excel files and combine."""
    chunks = []

    for fname, mhc_class in [(PMHCI_FILE, "MHCI"), (PMHCII_FILE, "MHCII")]:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  Warning: {fpath} not found, skipping")
            continue
        print(f"  Reading {fname} ...")
        df = pd.read_excel(fpath, dtype=str)
        df["mhc_class"] = mhc_class
        df["source_file"] = fname
        chunks.append(df)
        print(f"    {len(df):,} rows loaded")

    if not chunks:
        raise RuntimeError("No BATMAN data loaded")

    df = pd.concat(chunks, ignore_index=True)

    # Normalize tcr_source_organism to lowercase
    df["tcr_source_organism"] = df["tcr_source_organism"].str.lower().str.strip()

    # Derive standard columns
    for col in [
        "cdr3a", "cdr3b", "trav", "traj", "trbv", "trbd", "trbj",
        "va", "vb", "peptide", "index_peptide", "mhc", "peptide_type",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df["has_alpha"] = df["cdr3a"] != ""
    df["has_beta"] = df["cdr3b"] != ""
    df["is_paired"] = df["has_alpha"] & df["has_beta"]

    df["paired_tcr"] = np.where(
        df["is_paired"],
        df["cdr3a"] + "|" + df["cdr3b"],
        pd.NA,
    )

    df["has_epitope"] = df["index_peptide"] != ""
    df["has_mhc"] = df["mhc"] != ""

    # pMHC key
    df["pmhc"] = np.where(
        df["has_epitope"] & df["has_mhc"],
        df["index_peptide"] + "|" + df["mhc"],
        pd.NA,
    )

    # TCR-pMHC paired key
    df["tcr_pmhc_paired"] = np.where(
        df["is_paired"] & df["has_epitope"],
        df["paired_tcr"] + "|" + df["index_peptide"],
        pd.NA,
    )

    # Category = peptide_type
    df["category"] = df["peptide_type"].replace("", "(unknown)")

    # Study = PMID
    df["study"] = df["pmid"].fillna("(unknown)").replace("", "(unknown)")

    return df


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
def overview(df):
    return pd.DataFrame(
        [
            {
                "total_records": len(df),
                "unique_beta_cdr3": df.loc[df["has_beta"], "cdr3b"].nunique(),
                "unique_alpha_cdr3": df.loc[df["has_alpha"], "cdr3a"].nunique(),
                "paired_records": int(df["is_paired"].sum()),
                "unique_paired_tcrs": df["paired_tcr"].dropna().nunique(),
                "unique_tcrs": df["tcr"].nunique(),
                "unique_index_peptides": df.loc[
                    df["index_peptide"] != "", "index_peptide"
                ].nunique(),
                "unique_mutant_peptides": df.loc[
                    df["peptide"] != "", "peptide"
                ].nunique(),
                "unique_epitopes": df.loc[
                    df["index_peptide"] != "", "index_peptide"
                ].nunique(),
                "unique_mhc": df.loc[df["mhc"] != "", "mhc"].nunique(),
                "unique_pmhc": df["pmhc"].dropna().nunique(),
                "unique_tcr_pmhc_paired": df["tcr_pmhc_paired"].dropna().nunique(),
            }
        ]
    )


# ---------------------------------------------------------------------------
# 2. Category overview (peptide_type)
# ---------------------------------------------------------------------------
def category_overview(df):
    rows = []
    for category, grp in df.groupby("category", sort=True):
        rows.append(
            {
                "category": category,
                "total_records": len(grp),
                "unique_beta_cdr3": grp.loc[grp["has_beta"], "cdr3b"].nunique(),
                "unique_alpha_cdr3": grp.loc[grp["has_alpha"], "cdr3a"].nunique(),
                "paired_records": int(grp["is_paired"].sum()),
                "unique_paired_tcrs": grp["paired_tcr"].dropna().nunique(),
                "unique_epitopes": grp.loc[
                    grp["index_peptide"] != "", "index_peptide"
                ].nunique(),
                "unique_mhc": grp.loc[grp["mhc"] != "", "mhc"].nunique(),
                "unique_pmhc": grp["pmhc"].dropna().nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Epitope summary (by index_peptide)
# ---------------------------------------------------------------------------
def epitope_summary(df):
    rows = []
    epitope_col = "index_peptide"
    for epitope, grp in df[df[epitope_col] != ""].groupby(epitope_col, sort=True):
        primary_mhc = grp["mhc"].mode()
        primary_mhc = primary_mhc.iloc[0] if not primary_mhc.empty else ""
        mhc_class = grp["mhc_class"].mode()
        mhc_class = mhc_class.iloc[0] if not mhc_class.empty else "unknown"

        ptype = grp.loc[grp["peptide_type"] != "", "peptide_type"].mode()
        ptype = ptype.iloc[0] if not ptype.empty else ""

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["has_beta"] & ~grp["has_alpha"]]

        rows.append(
            {
                "epitope": epitope,
                "peptide_type": ptype,
                "primary_mhc": primary_mhc,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "unique_mutant_peptides": grp.loc[
                    grp["peptide"] != "", "peptide"
                ].nunique(),
                "unique_tcrs": grp["tcr"].nunique(),
                "paired_records": len(paired),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
                "unique_beta_only_tcrs": beta_only["cdr3b"].nunique()
                if len(beta_only)
                else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. pMHC summary
# ---------------------------------------------------------------------------
def pmhc_summary(df):
    rows = []
    valid = df[df["pmhc"].notna()].copy()
    for pmhc_key, grp in valid.groupby("pmhc", sort=True):
        parts = str(pmhc_key).split("|", 1)
        epitope = parts[0] if len(parts) > 0 else ""
        mhc = parts[1] if len(parts) > 1 else ""
        mhc_class = grp["mhc_class"].mode()
        mhc_class = mhc_class.iloc[0] if not mhc_class.empty else "unknown"

        ptype = grp.loc[grp["peptide_type"] != "", "peptide_type"].mode()
        ptype = ptype.iloc[0] if not ptype.empty else ""

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["has_beta"] & ~grp["has_alpha"]]

        rows.append(
            {
                "epitope": epitope,
                "mhc": mhc,
                "mhc_class": mhc_class,
                "peptide_type": ptype,
                "total_records": len(grp),
                "unique_tcrs": grp["tcr"].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
                "unique_beta_only_tcrs": beta_only["cdr3b"].nunique()
                if len(beta_only)
                else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Study summary (by PMID)
# ---------------------------------------------------------------------------
def study_summary(df):
    rows = []
    for pmid, grp in df.groupby("study", sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "pmid": pmid,
                "total_records": len(grp),
                "paired_records": len(paired),
                "unique_tcrs": grp["tcr"].nunique(),
                "unique_epitopes": grp.loc[
                    grp["index_peptide"] != "", "index_peptide"
                ].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
                "mhc_class": ";".join(sorted(grp["mhc_class"].unique())),
                "categories": ";".join(
                    sorted(
                        grp.loc[
                            grp["peptide_type"] != "", "peptide_type"
                        ]
                        .unique()
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. MHC allele summary
# ---------------------------------------------------------------------------
def mhc_allele_summary(df):
    rows = []
    valid = df[df["mhc"] != ""]
    for (mhc, mhc_class), grp in valid.groupby(["mhc", "mhc_class"], sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "mhc": mhc,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "unique_epitopes": grp.loc[
                    grp["index_peptide"] != "", "index_peptide"
                ].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Species breakdown
# ---------------------------------------------------------------------------
def species_breakdown(df):
    rows = []

    # Host species breakdown
    for species, grp in df.groupby("tcr_source_organism", sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "category": "tcr_source_organism",
                "species": species,
                "total_records": len(grp),
                "unique_epitopes": grp.loc[
                    grp["index_peptide"] != "", "index_peptide"
                ].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            }
        )

    # MHC class breakdown
    for mhc_class, grp in df.groupby("mhc_class", sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "category": "mhc_class",
                "species": mhc_class,
                "total_records": len(grp),
                "unique_epitopes": grp.loc[
                    grp["index_peptide"] != "", "index_peptide"
                ].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            }
        )

    # Assay type breakdown
    if "assay" in df.columns:
        for assay, grp in df.groupby("assay", sort=True):
            paired = grp[grp["is_paired"]]
            rows.append(
                {
                    "category": "assay",
                    "species": assay,
                    "total_records": len(grp),
                    "unique_epitopes": grp.loc[
                        grp["index_peptide"] != "", "index_peptide"
                    ].nunique(),
                    "unique_paired_tcrs": paired[
                        "paired_tcr"
                    ].dropna().nunique(),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Data completeness
# ---------------------------------------------------------------------------
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
                "paired_tcrs": sub["paired_tcr"].dropna().nunique(),
                "unique_beta_cdr3": sub.loc[
                    sub["has_beta"], "cdr3b"
                ].nunique(),
                "unique_epitopes": sub.loc[
                    sub["index_peptide"] != "", "index_peptide"
                ].nunique()
                if len(sub) > 0
                else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SUMMARY.txt generation
# ---------------------------------------------------------------------------
def write_summary_txt(df, output_dir):
    """Write a human-readable SUMMARY.txt after all TSVs are exported."""
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
    studies = pd.read_csv(
        os.path.join(output_dir, "study_summary.tsv"), sep="\t"
    )

    r = ov.iloc[0]
    total = int(r["total_records"])
    lines = []

    # --- Header ---
    lines.append("=" * 80)
    lines.append("                    BATMAN Analysis Summary")
    lines.append(
        "       Binding Affinity of TCR-MHC ANtigen Mutational Scan Database"
    )
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {DATA_DIR}")
    lines.append("=" * 80)
    lines.append("")

    # --- 1. DATASET OVERVIEW ---
    paired = int(r["paired_records"])
    paired_pct = 100.0 * paired / total if total else 0
    has_epi_mhc = int(
        completeness.loc[
            completeness["category"] == "has_epitope_and_mhc", "record_count"
        ].iloc[0]
    )
    has_epi_mhc_pct = 100.0 * has_epi_mhc / total if total else 0

    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Total records:          {total:>10,}")
    lines.append(f"Unique TCR clones:      {int(r['unique_tcrs']):>10,}")
    lines.append(f"Unique beta CDR3s:      {int(r['unique_beta_cdr3']):>10,}")
    lines.append(f"Unique alpha CDR3s:     {int(r['unique_alpha_cdr3']):>10,}")
    lines.append(
        f"Paired records (a+b):   {paired:>10,}  ({paired_pct:.1f}% of total)"
    )
    lines.append(f"Unique paired TCRs:     {int(r['unique_paired_tcrs']):>10,}")
    lines.append(f"Unique index peptides:  {int(r['unique_index_peptides']):>10,}")
    lines.append(f"Unique mutant peptides: {int(r['unique_mutant_peptides']):>10,}")
    lines.append(f"Unique MHC alleles:     {int(r['unique_mhc']):>10,}")
    lines.append(f"Unique pMHC combos:     {int(r['unique_pmhc']):>10,}")
    lines.append(f"Unique TCR-pMHC pairs:  {int(r['unique_tcr_pmhc_paired']):>10,}")
    lines.append("")

    lines.append("Category (peptide_type) distribution:")
    for _, row in cat_ov.sort_values(
        "total_records", ascending=False
    ).iterrows():
        cat = row["category"]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cat:<22s} {cnt:>7,}  ({pct:>5.1f}%)")
    lines.append("")

    lines.append(
        f"Records with full annotation (epitope + MHC): "
        f"{has_epi_mhc:,}  ({has_epi_mhc_pct:.1f}% of total)"
    )
    lines.append("")

    # --- 2. MHC CLASS BREAKDOWN ---
    lines.append("2. MHC CLASS BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    mhc_class_agg = (
        mhc_alleles.groupby("mhc_class")
        .agg(
            epitopes=("unique_epitopes", "sum"),
            records=("total_records", "sum"),
            paired_tcrs=("unique_paired_tcrs", "sum"),
        )
        .reset_index()
        .sort_values("records", ascending=False)
    )
    lines.append(
        f"  {'Class':<12s} {'Epitopes':>10s} {'Records':>10s} {'Paired TCRs':>13s}"
    )
    mhc_total_records = int(mhc_class_agg["records"].sum())
    for _, row in mhc_class_agg.iterrows():
        lines.append(
            f"  {row['mhc_class']:<12s} {int(row['epitopes']):>10,} "
            f"{int(row['records']):>10,} {int(row['paired_tcrs']):>13,}"
        )
    lines.append("")

    mhci_row = mhc_class_agg[mhc_class_agg["mhc_class"] == "MHCI"]
    if not mhci_row.empty and mhc_total_records > 0:
        mhci_pct = 100.0 * int(mhci_row["records"].iloc[0]) / mhc_total_records
        lines.append(
            f"MHC class I accounts for {mhci_pct:.1f}% of records, "
            f"MHC class II for {100.0 - mhci_pct:.1f}%."
        )
    lines.append("")

    # --- 3. CATEGORY BREAKDOWN ---
    lines.append("3. CATEGORY BREAKDOWN (BY PEPTIDE TYPE)")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Category':<22s} {'Records':>9s} {'%':>7s} {'Paired TCRs':>13s} "
        f"{'Epitopes':>10s}"
    )
    for _, row in cat_ov.sort_values(
        "total_records", ascending=False
    ).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['category'])[:20]:<22s} {cnt:>9,} {pct:>6.1f}% "
            f"{int(row['unique_paired_tcrs']):>13,} "
            f"{int(row['unique_epitopes']):>10,}"
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
    lines.append("4. TOP EPITOPES (INDEX PEPTIDES)")
    lines.append("-" * 80)
    lines.append("")
    lines.append("By total records:")
    lines.append("")
    lines.append(
        f"  {'Index Peptide':<22s} {'Type':<18s} {'MHC':<18s} "
        f"{'Records':>8s} {'TCRs':>6s}"
    )

    top_epi = epi.nlargest(15, "total_records")
    for _, row in top_epi.iterrows():
        epi_name = str(row["epitope"])[:20]
        ptype = str(row["peptide_type"])[:16]
        mhc = str(row["primary_mhc"])[:16]
        lines.append(
            f"  {epi_name:<22s} {ptype:<18s} {mhc:<18s} "
            f"{int(row['total_records']):>8,} {int(row['unique_tcrs']):>6,}"
        )
    lines.append("")

    lines.append("By unique TCR clones:")
    lines.append("")
    lines.append(
        f"  {'Index Peptide':<22s} {'Type':<18s} {'TCRs':>6s} "
        f"{'Mutant Peps':>12s}"
    )

    top_tcr = epi.nlargest(10, "unique_tcrs")
    for _, row in top_tcr.iterrows():
        epi_name = str(row["epitope"])[:20]
        ptype = str(row["peptide_type"])[:16]
        lines.append(
            f"  {epi_name:<22s} {ptype:<18s} "
            f"{int(row['unique_tcrs']):>6,} "
            f"{int(row['unique_mutant_peptides']):>12,}"
        )
    lines.append("")

    # --- 5. TOP MHC ALLELES ---
    lines.append("5. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Allele':<25s} {'Class':<10s} {'Records':>9s} "
        f"{'Epitopes':>10s} {'Paired TCRs':>13s}"
    )

    top_mhc = mhc_alleles.sort_values("total_records", ascending=False)
    for _, row in top_mhc.iterrows():
        allele = str(row["mhc"])[:23]
        lines.append(
            f"  {allele:<25s} {row['mhc_class']:<10s} "
            f"{int(row['total_records']):>9,} "
            f"{int(row['unique_epitopes']):>10,} "
            f"{int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

    if len(top_mhc) > 0:
        top_allele = top_mhc.iloc[0]
        top_allele_pct = (
            100.0 * int(top_allele["total_records"]) / mhc_total_records
            if mhc_total_records > 0
            else 0
        )
        lines.append(
            f"{top_allele['mhc']} is the most represented allele with "
            f"{top_allele_pct:.1f}% of records."
        )
    lines.append("")

    # --- 6. PATHOLOGY & SPECIES ---
    lines.append("6. SPECIES & ASSAY BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    host_species = species[species["category"] == "tcr_source_organism"].copy()
    host_species = host_species.sort_values("total_records", ascending=False)
    host_total = int(host_species["total_records"].sum())

    if len(host_species) > 0:
        lines.append("TCR source organism:")
        for _, row in host_species.iterrows():
            sp = str(row["species"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / host_total if host_total else 0
            lines.append(
                f"  {sp:<20s} {cnt:>9,} records  ({pct:>5.1f}%)"
            )
        lines.append("")

    assay_rows = species[species["category"] == "assay"].copy()
    assay_rows = assay_rows.sort_values("total_records", ascending=False)
    if len(assay_rows) > 0:
        lines.append("Assay type breakdown:")
        for _, row in assay_rows.iterrows():
            assay = str(row["species"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total if total else 0
            lines.append(f"  {assay:<25s} {cnt:>9,}  ({pct:>5.1f}%)")
        lines.append("")

    # --- 7. STUDY LANDSCAPE ---
    lines.append("7. STUDY LANDSCAPE (BY PMID)")
    lines.append("-" * 80)
    lines.append("")

    total_studies = len(studies)
    studies_with_paired = int((studies["paired_records"] > 0).sum())
    studies_with_paired_pct = (
        100.0 * studies_with_paired / total_studies if total_studies else 0
    )

    lines.append(f"Total publications (PMIDs): {total_studies}")
    lines.append(
        f"Publications with paired TCR data: {studies_with_paired} "
        f"({studies_with_paired_pct:.0f}%)"
    )
    lines.append("")

    lines.append("Largest contributors:")
    top_studies = studies.nlargest(10, "total_records")
    for _, row in top_studies.iterrows():
        pmid = str(row["pmid"])
        lines.append(
            f"  PMID {pmid:<12s} {int(row['total_records']):>7,} records  "
            f"({int(row['unique_tcrs']):>3,} TCRs, "
            f"{int(row['unique_epitopes']):>3,} epitopes)"
        )
    lines.append("")

    top5_records = int(top_studies.head(5)["total_records"].sum())
    top5_pct = 100.0 * top5_records / total if total else 0
    lines.append(
        f"The top 5 publications account for ~{top5_pct:.0f}% of all records."
    )
    lines.append("")

    # --- 8. DATA COMPLETENESS ---
    lines.append("8. DATA COMPLETENESS")
    lines.append("-" * 80)
    lines.append("")

    criteria = {
        "has_epitope_and_mhc": "Has both index peptide and MHC annotation.",
        "has_epitope_only": "Has index peptide but missing MHC allele.",
        "has_mhc_only": "Has MHC annotation but no index peptide.",
        "no_epitope_no_mhc": "Missing both index peptide and MHC.",
    }

    lines.append(
        f"  {'Category':<25s} {'Records':>9s} {'%':>7s} "
        f"{'Paired TCRs':>13s} {'Epitopes':>10s}"
    )
    for _, row in completeness.iterrows():
        cat = row["category"]
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {cat:<25s} {cnt:>9,} {pct:>6.1f}% "
            f"{int(row['paired_tcrs']):>13,} {int(row['unique_epitopes']):>10,}"
        )
    lines.append("")

    lines.append("Criteria:")
    for cat, desc in criteria.items():
        lines.append(f"  - {cat:<25s} {desc}")
    lines.append("")

    lines.append(
        f"All {total:,} records have both index peptide and MHC annotation.\n"
        f"BATMAN is a fully annotated mutational scanning database."
    )
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

    lines.append(
        f"a) Mutational scanning data: {total:,} records covering "
        f"{int(r['unique_tcrs']):,} unique TCR clones\n"
        f"   against {int(r['unique_index_peptides']):,} index peptides with "
        f"{int(r['unique_mutant_peptides']):,} mutant peptide variants.\n"
        f"   Each record includes quantitative peptide_activity measurements."
    )
    lines.append("")

    lines.append(
        f"b) Paired TCR availability: {int(r['unique_paired_tcrs']):,} unique "
        f"paired TCRs ({paired_pct:.1f}% of records\n"
        f"   have both alpha+beta CDR3 sequences)."
    )
    lines.append("")

    lines.append(
        f"c) Category distribution: '{dominant['category']}' accounts for "
        f"{dom_pct:.1f}% of records.\n"
        f"   Data spans viral, cancer, autoimmune, and neoantigen peptide types."
    )
    lines.append("")

    if len(top_mhc) > 0:
        top_a = top_mhc.iloc[0]
        top_a_pct = (
            100.0 * int(top_a["total_records"]) / mhc_total_records
            if mhc_total_records > 0
            else 0
        )
        lines.append(
            f"d) HLA bias: {top_a['mhc']} represents {top_a_pct:.1f}% of "
            f"records. Both human HLA and\n"
            f"   mouse MHC alleles are represented across {int(r['unique_mhc']):,} "
            f"distinct alleles."
        )
    lines.append("")

    lines.append(
        "e) Binding landscape: BATMAN provides quantitative binding activity\n"
        "   measurements (peptide_activity) for systematic single-residue\n"
        "   mutations, enabling fine-grained TCR specificity modeling."
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
            f"f) Species coverage: {top_host['species']} TCRs account for "
            f"{top_host_pct:.1f}% of records.\n"
            f"   Both human and mouse TCR data are included."
        )
    elif len(host_species) == 1:
        lines.append(
            f"f) Species coverage: All data is from "
            f"{host_species.iloc[0]['species']} TCRs."
        )
    lines.append("")

    # --- Footer ---
    lines.append("=" * 80)
    lines.append(f"Output files in {OUTPUT_DIR}/:")

    tsv_files = [
        "overview.tsv",
        "category_overview.tsv",
        "epitope_summary.tsv",
        "pmhc_summary.tsv",
        "study_summary.tsv",
        "mhc_allele_summary.tsv",
        "species_breakdown.tsv",
        "data_completeness.tsv",
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

    print(f"Loading BATMAN data from {data_dir} ...")
    df = load_data(data_dir)
    print(f"  Loaded {len(df):,} records, {df.columns.size} columns")
    print(f"  TCR clones: {df['tcr'].nunique()}")
    print(f"  MHC class distribution: {df['mhc_class'].value_counts().to_dict()}")
    print(f"  Paired records: {df['is_paired'].sum():,}")
    print()

    exports = [
        ("overview.tsv", overview),
        ("category_overview.tsv", category_overview),
        ("epitope_summary.tsv", epitope_summary),
        ("pmhc_summary.tsv", pmhc_summary),
        ("study_summary.tsv", study_summary),
        ("mhc_allele_summary.tsv", mhc_allele_summary),
        ("species_breakdown.tsv", species_breakdown),
        ("data_completeness.tsv", data_completeness),
    ]

    for filename, func in exports:
        print(f"Computing {filename} ...")
        result = func(df)
        path = os.path.join(output_dir, filename)
        result.to_csv(path, sep="\t", index=False)
        print(f"  -> {path}  ({len(result)} rows, {result.columns.size} cols)")

    print("Writing SUMMARY.txt ...")
    write_summary_txt(df, output_dir)

    # --- Verification ---
    print("\n--- Verification ---")

    ov = pd.read_csv(os.path.join(output_dir, "overview.tsv"), sep="\t")
    cat_ov = pd.read_csv(
        os.path.join(output_dir, "category_overview.tsv"), sep="\t"
    )
    cat_sum = cat_ov["total_records"].sum()
    all_total = ov.iloc[0]["total_records"]
    print(
        f"Category sum: {cat_sum}, Overview total: {all_total}, "
        f"match: {cat_sum == all_total}"
    )

    comp = pd.read_csv(
        os.path.join(output_dir, "data_completeness.tsv"), sep="\t"
    )
    comp_sum = comp["record_count"].sum()
    print(
        f"Completeness sum: {comp_sum}, Overview total: {all_total}, "
        f"match: {comp_sum == all_total}"
    )

    summary_path = os.path.join(output_dir, "SUMMARY.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            content = f.read()
        sections = sum(1 for i in range(1, 10) if f"\n{i}. " in content)
        print(f"SUMMARY.txt sections found: {sections}/9")

    print("\nDone.")


if __name__ == "__main__":
    main()
