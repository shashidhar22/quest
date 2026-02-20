#!/usr/bin/env python3
"""Export comprehensive CEDAR analysis tables.

CEDAR (Curated Epitope Database for Adaptive Receptors) integrates three
data tables: receptor (TCR sequences), tcell (T cell assay data), and
epitope (epitope metadata). Files use multi-level CSV headers.

receptor/tcr_full_v3.csv  ~105K rows - primary TCR dataset
tcell/tcell_full_v3.csv   ~151K rows - assay/host/MHC enrichment
epitope/epitope_full_v3.csv ~1.5M rows - epitope source organism
"""

import os
import re

import numpy as np
import pandas as pd

DATA_DIR = "data/databases/CEDAR"
OUTPUT_DIR = "data/analysis/CEDAR"


def flatten_columns(df):
    """Flatten MultiIndex columns from multi-level CSV header to Group_Field."""
    cols = []
    for col in df.columns:
        parts = [str(c).strip() for c in col if "Unnamed" not in str(c)]
        cols.append("_".join(parts).strip("_"))
    df.columns = cols
    return df


def classify_mhc(allele):
    """Classify an HLA allele string as MHCI or MHCII."""
    if not allele or pd.isna(allele):
        return "unknown"
    a = str(allele).upper().strip()
    if any(x in a for x in ["DR", "DQ", "DP", "HLA-D", "CLASS II"]):
        return "MHCII"
    if re.match(r"^HLA-[ABC]", a) or re.match(r"^[ABC]\*?\d", a):
        return "MHCI"
    # Mouse MHC
    if any(x in a for x in ["H-2", "H2-"]):
        if any(x in a for x in ["-IA", "-IE", "H2-A", "H2-E"]):
            return "MHCII"
        return "MHCI"
    return "unknown"


def load_data(data_dir):
    """Load CEDAR receptor data and enrich from tcell table."""
    # --- Load receptor (primary) ---
    receptor_path = os.path.join(data_dir, "receptor", "tcr_full_v3.csv")
    print(f"  Reading {receptor_path} ...")
    df = pd.read_csv(receptor_path, header=[0, 1], dtype=str)
    df = flatten_columns(df)
    df = df.fillna("")
    print(f"    {len(df):,} rows, {len(df.columns)} columns")

    # --- Load tcell for enrichment ---
    tcell_path = os.path.join(data_dir, "tcell", "tcell_full_v3.csv")
    print(f"  Reading {tcell_path} (for enrichment) ...")
    tcell = pd.read_csv(tcell_path, header=[0, 1], dtype=str)
    tcell = flatten_columns(tcell)
    tcell = tcell.fillna("")

    # Extract numeric assay ID from IRI URL
    tcell["assay_id"] = tcell["Assay ID_CEDAR IRI"].str.extract(r"(\d+)$")[0]

    # Build lookup: assay_id -> enrichment fields
    tcell_lookup = (
        tcell.groupby("assay_id")
        .first()[
            [
                "Reference_PMID",
                "Host_Name",
                "1st in vivo Process_Disease",
                "MHC Restriction_Name",
                "MHC Restriction_Class",
                "Epitope_Source Organism",
            ]
        ]
        .reset_index()
    )
    print(f"    {len(tcell_lookup):,} unique assay IDs for enrichment")
    del tcell

    # --- Join receptor to tcell ---
    # Receptor Assay_CEDAR IDs can be comma-separated; take the first one
    df["primary_assay_id"] = (
        df["Assay_CEDAR IDs"].str.split(",").str[0].str.strip()
    )

    df = df.merge(
        tcell_lookup,
        left_on="primary_assay_id",
        right_on="assay_id",
        how="left",
        suffixes=("", "_tcell"),
    )
    df = df.fillna("")

    # --- Derive standard columns ---
    # CDR3 alpha (Chain 1): prefer Curated, fallback Calculated
    df["cdr3a"] = np.where(
        df["Chain 1_CDR3 Curated"] != "",
        df["Chain 1_CDR3 Curated"],
        df["Chain 1_CDR3 Calculated"],
    )

    # CDR3 beta (Chain 2): prefer Curated, fallback Calculated
    df["cdr3b"] = np.where(
        df["Chain 2_CDR3 Curated"] != "",
        df["Chain 2_CDR3 Curated"],
        df["Chain 2_CDR3 Calculated"],
    )

    # V genes
    df["va"] = np.where(
        df["Chain 1_Curated V Gene"] != "",
        df["Chain 1_Curated V Gene"],
        df["Chain 1_Calculated V Gene"],
    )
    df["vb"] = np.where(
        df["Chain 2_Curated V Gene"] != "",
        df["Chain 2_Curated V Gene"],
        df["Chain 2_Calculated V Gene"],
    )

    # Epitope
    df["epitope"] = df["Epitope_Name"].str.strip()

    # MHC: prefer receptor-level, fallback tcell enrichment
    df["mhc"] = np.where(
        df["Assay_MHC Allele Names"] != "",
        df["Assay_MHC Allele Names"],
        df["MHC Restriction_Name"],
    )

    # MHC class
    df["mhc_class"] = np.where(
        df["MHC Restriction_Class"].isin(["I", "II"]),
        np.where(df["MHC Restriction_Class"] == "I", "MHCI", "MHCII"),
        df["mhc"].apply(classify_mhc),
    )

    # Standard boolean columns
    df["has_alpha"] = df["cdr3a"] != ""
    df["has_beta"] = df["cdr3b"] != ""
    df["is_paired"] = df["has_alpha"] & df["has_beta"]

    df["paired_tcr"] = np.where(
        df["is_paired"],
        df["cdr3a"] + "|" + df["cdr3b"],
        pd.NA,
    )

    df["has_epitope"] = df["epitope"] != ""
    df["has_mhc"] = df["mhc"] != ""

    df["pmhc"] = np.where(
        df["has_epitope"] & df["has_mhc"],
        df["epitope"] + "|" + df["mhc"],
        pd.NA,
    )

    df["tcr_pmhc_paired"] = np.where(
        df["is_paired"] & df["has_epitope"],
        df["paired_tcr"].astype(str) + "|" + df["epitope"],
        pd.NA,
    )

    # Category = disease from tcell join
    df["category"] = (
        df["1st in vivo Process_Disease"].replace("", "(unknown)")
    )

    # Study = PMID
    df["study"] = df["Reference_PMID"].replace("", "(unknown)")

    # Host organism
    df["host_organism"] = df["Host_Name"].str.lower().str.strip()
    df.loc[df["host_organism"] == "", "host_organism"] = "(unknown)"

    # Epitope source organism (from receptor table first, then tcell)
    df["epitope_source_organism"] = np.where(
        df["Epitope_Source Organism"] != "",
        df["Epitope_Source Organism"],
        df.get("Epitope_Source Organism_tcell", ""),
    )
    df["epitope_source_organism"] = (
        df["epitope_source_organism"].str.lower().str.strip()
    )
    df.loc[
        df["epitope_source_organism"] == "", "epitope_source_organism"
    ] = "(unknown)"

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
                "unique_alpha_cdr3": df.loc[
                    df["has_alpha"], "cdr3a"
                ].nunique(),
                "paired_records": int(df["is_paired"].sum()),
                "unique_paired_tcrs": df["paired_tcr"].dropna().nunique(),
                "unique_epitopes": df.loc[
                    df["has_epitope"], "epitope"
                ].nunique(),
                "unique_mhc": df.loc[df["has_mhc"], "mhc"].nunique(),
                "unique_pmhc": df["pmhc"].dropna().nunique(),
                "unique_tcr_pmhc_paired": df[
                    "tcr_pmhc_paired"
                ].dropna().nunique(),
                "unique_studies": df["study"].nunique(),
            }
        ]
    )


# ---------------------------------------------------------------------------
# 2. Category overview (disease)
# ---------------------------------------------------------------------------
def category_overview(df):
    rows = []
    for category, grp in df.groupby("category", sort=True):
        rows.append(
            {
                "category": category,
                "total_records": len(grp),
                "unique_beta_cdr3": grp.loc[
                    grp["has_beta"], "cdr3b"
                ].nunique(),
                "unique_alpha_cdr3": grp.loc[
                    grp["has_alpha"], "cdr3a"
                ].nunique(),
                "paired_records": int(grp["is_paired"].sum()),
                "unique_paired_tcrs": grp["paired_tcr"].dropna().nunique(),
                "unique_epitopes": grp.loc[
                    grp["has_epitope"], "epitope"
                ].nunique(),
                "unique_mhc": grp.loc[grp["has_mhc"], "mhc"].nunique(),
                "unique_pmhc": grp["pmhc"].dropna().nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Epitope summary
# ---------------------------------------------------------------------------
def epitope_summary(df):
    rows = []
    for epitope, grp in df[df["has_epitope"]].groupby("epitope", sort=True):
        primary_mhc = grp.loc[grp["has_mhc"], "mhc"].mode()
        primary_mhc = primary_mhc.iloc[0] if not primary_mhc.empty else ""
        mhc_class = grp["mhc_class"].mode()
        mhc_class = mhc_class.iloc[0] if not mhc_class.empty else "unknown"

        source_org = grp["epitope_source_organism"].mode()
        source_org = source_org.iloc[0] if not source_org.empty else ""

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["has_beta"] & ~grp["has_alpha"]]

        rows.append(
            {
                "epitope": epitope,
                "source_organism": source_org,
                "primary_mhc": primary_mhc,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "paired_records": len(paired),
                "unique_paired_tcrs": paired[
                    "paired_tcr"
                ].dropna().nunique(),
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

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["has_beta"] & ~grp["has_alpha"]]

        rows.append(
            {
                "epitope": epitope,
                "mhc": mhc,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "unique_paired_tcrs": paired[
                    "paired_tcr"
                ].dropna().nunique(),
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
    for study, grp in df.groupby("study", sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "study": study,
                "total_records": len(grp),
                "paired_records": len(paired),
                "unique_paired_tcrs": paired[
                    "paired_tcr"
                ].dropna().nunique(),
                "unique_epitopes": grp.loc[
                    grp["has_epitope"], "epitope"
                ].nunique(),
                "unique_mhc": grp.loc[grp["has_mhc"], "mhc"].nunique(),
                "categories": ";".join(
                    sorted(
                        grp.loc[
                            grp["category"] != "(unknown)", "category"
                        ].unique()
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
    valid = df[df["has_mhc"]]
    for (mhc, mhc_class), grp in valid.groupby(
        ["mhc", "mhc_class"], sort=True
    ):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "mhc": mhc,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "unique_epitopes": grp.loc[
                    grp["has_epitope"], "epitope"
                ].nunique(),
                "unique_paired_tcrs": paired[
                    "paired_tcr"
                ].dropna().nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Species breakdown
# ---------------------------------------------------------------------------
def species_breakdown(df):
    rows = []

    # Host organism
    for host, grp in df.groupby("host_organism", sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "category": "host_organism",
                "species": host,
                "total_records": len(grp),
                "unique_epitopes": grp.loc[
                    grp["has_epitope"], "epitope"
                ].nunique(),
                "unique_paired_tcrs": paired[
                    "paired_tcr"
                ].dropna().nunique(),
            }
        )

    # Epitope source organism
    for org, grp in df.groupby("epitope_source_organism", sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "category": "epitope_source_organism",
                "species": org,
                "total_records": len(grp),
                "unique_epitopes": grp.loc[
                    grp["has_epitope"], "epitope"
                ].nunique(),
                "unique_paired_tcrs": paired[
                    "paired_tcr"
                ].dropna().nunique(),
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
                    grp["has_epitope"], "epitope"
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
                    sub["has_epitope"], "epitope"
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
    from datetime import date

    ov = pd.read_csv(os.path.join(output_dir, "overview.tsv"), sep="\t")
    cat_ov = pd.read_csv(
        os.path.join(output_dir, "category_overview.tsv"), sep="\t"
    )
    epi = pd.read_csv(
        os.path.join(output_dir, "epitope_summary.tsv"), sep="\t"
    )
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
    lines.append(
        "                    CEDAR Analysis Summary"
    )
    lines.append(
        "       Curated Epitope Database for Adaptive Receptors"
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
            completeness["category"] == "has_epitope_and_mhc",
            "record_count",
        ].iloc[0]
    )
    has_epi_mhc_pct = 100.0 * has_epi_mhc / total if total else 0

    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Total records:           {total:>10,}")
    lines.append(
        f"Unique beta CDR3s:       {int(r['unique_beta_cdr3']):>10,}"
    )
    lines.append(
        f"Unique alpha CDR3s:      {int(r['unique_alpha_cdr3']):>10,}"
    )
    lines.append(
        f"Paired records (a+b):    {paired:>10,}  ({paired_pct:.1f}% of total)"
    )
    lines.append(
        f"Unique paired TCRs:      {int(r['unique_paired_tcrs']):>10,}"
    )
    lines.append(
        f"Unique epitopes:         {int(r['unique_epitopes']):>10,}"
    )
    lines.append(f"Unique MHC alleles:      {int(r['unique_mhc']):>10,}")
    lines.append(f"Unique pMHC combos:      {int(r['unique_pmhc']):>10,}")
    lines.append(
        f"Unique TCR-pMHC pairs:   {int(r['unique_tcr_pmhc_paired']):>10,}"
    )
    lines.append(f"Unique studies (PMIDs):   {int(r['unique_studies']):>10,}")
    lines.append("")

    lines.append("Disease category distribution (top 15):")
    top_cats = cat_ov.nlargest(15, "total_records")
    for _, row in top_cats.iterrows():
        cat = str(row["category"])[:30]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cat:<32s} {cnt:>7,}  ({pct:>5.1f}%)")
    if len(cat_ov) > 15:
        lines.append(f"  ... and {len(cat_ov) - 15} more categories")
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

    if len(mhc_alleles) > 0:
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
            f"  {'Class':<12s} {'Epitopes':>10s} {'Records':>10s}"
            f" {'Paired TCRs':>13s}"
        )
        for _, row in mhc_class_agg.iterrows():
            lines.append(
                f"  {row['mhc_class']:<12s} {int(row['epitopes']):>10,}"
                f" {int(row['records']):>10,}"
                f" {int(row['paired_tcrs']):>13,}"
            )
    else:
        lines.append("No MHC allele data available.")
    lines.append("")

    # --- 3. CATEGORY BREAKDOWN ---
    lines.append("3. CATEGORY BREAKDOWN (BY DISEASE)")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Category':<32s} {'Records':>9s} {'%':>7s}"
        f" {'Paired TCRs':>13s} {'Epitopes':>10s}"
    )
    for _, row in cat_ov.nlargest(20, "total_records").iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['category'])[:30]:<32s} {cnt:>9,} {pct:>6.1f}%"
            f" {int(row['unique_paired_tcrs']):>13,}"
            f" {int(row['unique_epitopes']):>10,}"
        )
    if len(cat_ov) > 20:
        lines.append(f"  ... and {len(cat_ov) - 20} more categories")
    lines.append("")

    # --- 4. TOP EPITOPES ---
    lines.append("4. TOP EPITOPES")
    lines.append("-" * 80)
    lines.append("")
    if len(epi) > 0:
        lines.append(
            f"  {'Epitope':<22s} {'Source Organism':<20s}"
            f" {'MHC':<16s} {'Records':>8s}"
        )
        for _, row in epi.nlargest(15, "total_records").iterrows():
            epi_name = str(row["epitope"])[:20]
            org = str(row["source_organism"])[:18]
            mhc = str(row["primary_mhc"])[:14]
            lines.append(
                f"  {epi_name:<22s} {org:<20s} {mhc:<16s}"
                f" {int(row['total_records']):>8,}"
            )
    else:
        lines.append("No epitope data available.")
    lines.append("")

    # --- 5. TOP MHC ALLELES ---
    lines.append("5. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")
    if len(mhc_alleles) > 0:
        lines.append(
            f"  {'Allele':<25s} {'Class':<10s} {'Records':>9s}"
            f" {'Epitopes':>10s}"
        )
        for _, row in mhc_alleles.nlargest(15, "total_records").iterrows():
            allele = str(row["mhc"])[:23]
            lines.append(
                f"  {allele:<25s} {row['mhc_class']:<10s}"
                f" {int(row['total_records']):>9,}"
                f" {int(row['unique_epitopes']):>10,}"
            )
    else:
        lines.append("No MHC allele data available.")
    lines.append("")

    # --- 6. SPECIES BREAKDOWN ---
    lines.append("6. SPECIES & ORGANISM BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    host_species = species[species["category"] == "host_organism"].copy()
    host_species = host_species.sort_values(
        "total_records", ascending=False
    )
    host_total = int(host_species["total_records"].sum())

    if len(host_species) > 0:
        lines.append("Host organism:")
        for _, row in host_species.head(10).iterrows():
            sp = str(row["species"])[:30]
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / host_total if host_total else 0
            lines.append(
                f"  {sp:<32s} {cnt:>9,} records  ({pct:>5.1f}%)"
            )
        if len(host_species) > 10:
            lines.append(
                f"  ... and {len(host_species) - 10} more host organisms"
            )
        lines.append("")

    epi_org = species[
        species["category"] == "epitope_source_organism"
    ].copy()
    epi_org = epi_org.sort_values("total_records", ascending=False)
    if len(epi_org) > 0:
        lines.append("Epitope source organism (top 10):")
        for _, row in epi_org.head(10).iterrows():
            sp = str(row["species"])[:30]
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total if total else 0
            lines.append(
                f"  {sp:<32s} {cnt:>9,} records  ({pct:>5.1f}%)"
            )
        if len(epi_org) > 10:
            lines.append(
                f"  ... and {len(epi_org) - 10} more source organisms"
            )
        lines.append("")

    mhc_cls = species[species["category"] == "mhc_class"].copy()
    mhc_cls = mhc_cls.sort_values("total_records", ascending=False)
    if len(mhc_cls) > 0:
        lines.append("MHC class distribution:")
        for _, row in mhc_cls.iterrows():
            cls = str(row["species"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total if total else 0
            lines.append(
                f"  {cls:<12s} {cnt:>9,} records  ({pct:>5.1f}%)"
            )
        lines.append("")

    # --- 7. STUDY LANDSCAPE ---
    lines.append("7. STUDY LANDSCAPE (BY PMID)")
    lines.append("-" * 80)
    lines.append("")

    total_studies = len(studies)
    studies_with_paired = int((studies["paired_records"] > 0).sum())

    lines.append(f"Total publications (PMIDs): {total_studies}")
    lines.append(
        f"Publications with paired TCR data: {studies_with_paired}"
    )
    lines.append("")

    lines.append("Largest contributors:")
    for _, row in studies.nlargest(10, "total_records").iterrows():
        study_label = str(row["study"])[:12]
        lines.append(
            f"  PMID {study_label:<12s} {int(row['total_records']):>7,}"
            f" records  ({int(row['unique_epitopes']):>3,} epitopes,"
            f" {int(row['unique_mhc']):>3,} MHC alleles)"
        )
    lines.append("")

    # --- 8. DATA COMPLETENESS ---
    lines.append("8. DATA COMPLETENESS")
    lines.append("-" * 80)
    lines.append("")

    criteria = {
        "has_epitope_and_mhc": "Has both epitope and MHC annotation.",
        "has_epitope_only": "Has epitope but missing MHC allele.",
        "has_mhc_only": "Has MHC annotation but no epitope.",
        "no_epitope_no_mhc": "Missing both epitope and MHC.",
    }

    lines.append(
        f"  {'Category':<25s} {'Records':>9s} {'%':>7s}"
        f" {'Paired TCRs':>13s} {'Epitopes':>10s}"
    )
    for _, row in completeness.iterrows():
        cat = row["category"]
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {cat:<25s} {cnt:>9,} {pct:>6.1f}%"
            f" {int(row['paired_tcrs']):>13,}"
            f" {int(row['unique_epitopes']):>10,}"
        )
    lines.append("")

    lines.append("Criteria:")
    for cat, desc in criteria.items():
        lines.append(f"  - {cat:<25s} {desc}")
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

    lines.append(
        f"a) Data scale: {total:,} TCR records from {total_studies}"
        f" publications,\n"
        f"   with {int(r['unique_epitopes']):,} unique epitopes and"
        f" {int(r['unique_mhc']):,} MHC alleles."
    )
    lines.append("")

    lines.append(
        f"b) Paired TCR availability: {int(r['unique_paired_tcrs']):,}"
        f" unique paired TCRs ({paired_pct:.1f}% of records\n"
        f"   have both alpha+beta CDR3 sequences)."
    )
    lines.append("")

    lines.append(
        f"c) Annotation quality: {has_epi_mhc:,} records"
        f" ({has_epi_mhc_pct:.1f}%) have both\n"
        f"   epitope and MHC annotation. CEDAR integrates receptor,"
        f" assay, and epitope tables."
    )
    lines.append("")

    if len(host_species) > 0:
        top_host = host_species.iloc[0]
        top_host_pct = (
            100.0 * int(top_host["total_records"]) / host_total
            if host_total
            else 0
        )
        lines.append(
            f"d) Host organisms: {top_host['species']} accounts for"
            f" {top_host_pct:.1f}% of records.\n"
            f"   {len(host_species)} distinct host organisms represented."
        )
    lines.append("")

    lines.append(
        f"e) Disease categories: {len(cat_ov)} disease categories.\n"
        f"   Rich disease metadata from tcell assay integration."
    )
    lines.append("")

    lines.append(
        f"f) Three-table integration: receptor ({total:,} rows) joined with"
        f" tcell\n"
        f"   assay data for PMID, disease, host, and MHC enrichment."
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

    print(f"Loading CEDAR data from {data_dir} ...")
    df = load_data(data_dir)
    print(f"  Loaded {len(df):,} records, {df.columns.size} columns")
    print(f"  Paired records: {df['is_paired'].sum():,}")
    print(f"  MHC class distribution: {df['mhc_class'].value_counts().to_dict()}")
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
        print(
            f"  -> {path}  ({len(result)} rows, {result.columns.size} cols)"
        )

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
