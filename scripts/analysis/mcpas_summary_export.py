#!/usr/bin/env python3
"""Export comprehensive McPAS-TCR analysis tables."""

import os
import pandas as pd
import numpy as np

INPUT_FILE = "data/databases/McPAS-TCR/McPAS-TCR.csv"
OUTPUT_DIR = "data/analysis/mcpas"


def load_data(path):
    """Load McPAS-TCR CSV and derive analysis columns.

    McPAS uses "NA" as a string for missing values, so we read with
    na_filter=False and handle "NA" explicitly.
    """
    df = pd.read_csv(path, dtype=str, na_filter=False)

    # Treat "NA" and empty strings as missing for key columns
    for col in ["CDR3.alpha.aa", "CDR3.beta.aa", "Epitope.peptide", "MHC"]:
        df[col] = df[col].replace("NA", "")

    # Derived boolean columns
    df["has_alpha"] = df["CDR3.alpha.aa"] != ""
    df["has_beta"] = df["CDR3.beta.aa"] != ""
    df["is_paired"] = df["has_alpha"] & df["has_beta"]

    # Composite keys
    df["paired_tcr"] = np.where(
        df["is_paired"],
        df["CDR3.alpha.aa"] + "|" + df["CDR3.beta.aa"],
        pd.NA,
    )

    # MHC class derivation
    mhc_upper = df["MHC"].str.upper()
    df["mhc_class"] = np.where(
        mhc_upper.str.contains(r"HLA-[ABC]", regex=True, na=False),
        "MHCI",
        np.where(
            mhc_upper.str.contains(r"HLA-D", regex=True, na=False),
            "MHCII",
            "unknown",
        ),
    )

    # pMHC key (epitope + MHC)
    df["pmhc"] = df["Epitope.peptide"] + "|" + df["MHC"]

    # TCR-pMHC paired key
    df["tcr_pmhc_paired"] = np.where(
        df["is_paired"],
        df["paired_tcr"] + "|" + df["Epitope.peptide"],
        pd.NA,
    )

    # Clean Category and Pathology
    for col in ["Category", "Pathology", "Antigen.protein", "Species"]:
        df[col] = df[col].replace("NA", "")

    return df


# ---------------------------------------------------------------------------
# 1. Overview (single-row dataset summary)
# ---------------------------------------------------------------------------
def overview(df):
    rows = [
        {
            "total_records": len(df),
            "unique_beta_cdr3": df.loc[df["has_beta"], "CDR3.beta.aa"].nunique(),
            "unique_alpha_cdr3": df.loc[df["has_alpha"], "CDR3.alpha.aa"].nunique(),
            "paired_records": int(df["is_paired"].sum()),
            "unique_paired_tcrs": df["paired_tcr"].dropna().nunique(),
            "unique_epitopes": df.loc[df["Epitope.peptide"] != "", "Epitope.peptide"].nunique(),
            "unique_mhc": df.loc[df["MHC"] != "", "MHC"].nunique(),
            "unique_pmhc": df.loc[
                (df["Epitope.peptide"] != "") & (df["MHC"] != ""), "pmhc"
            ].nunique(),
            "unique_tcr_pmhc_paired": df["tcr_pmhc_paired"].dropna().nunique(),
        }
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Category overview (one row per Category)
# ---------------------------------------------------------------------------
def category_overview(df):
    rows = []
    for category, grp in df.groupby("Category", sort=True):
        if category == "":
            category = "(empty)"
        rows.append(
            {
                "category": category,
                "total_records": len(grp),
                "unique_beta_cdr3": grp.loc[grp["has_beta"], "CDR3.beta.aa"].nunique(),
                "unique_alpha_cdr3": grp.loc[grp["has_alpha"], "CDR3.alpha.aa"].nunique(),
                "paired_records": int(grp["is_paired"].sum()),
                "unique_paired_tcrs": grp["paired_tcr"].dropna().nunique(),
                "unique_epitopes": grp.loc[grp["Epitope.peptide"] != "", "Epitope.peptide"].nunique(),
                "unique_mhc": grp.loc[grp["MHC"] != "", "MHC"].nunique(),
                "unique_pmhc": grp.loc[
                    (grp["Epitope.peptide"] != "") & (grp["MHC"] != ""), "pmhc"
                ].nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Epitope summary
# ---------------------------------------------------------------------------
def epitope_summary(df):
    # Filter to rows that have an epitope
    epi_df = df[df["Epitope.peptide"] != ""]
    rows = []
    for epitope, grp in epi_df.groupby("Epitope.peptide", sort=True):
        primary_mhc = grp.loc[grp["MHC"] != "", "MHC"].mode()
        primary_mhc = primary_mhc.iloc[0] if not primary_mhc.empty else ""
        mhc_class = grp.loc[grp["mhc_class"] != "unknown", "mhc_class"].mode()
        mhc_class = mhc_class.iloc[0] if not mhc_class.empty else "unknown"

        antigen_protein = grp.loc[grp["Antigen.protein"] != "", "Antigen.protein"].mode()
        antigen_protein = antigen_protein.iloc[0] if not antigen_protein.empty else ""
        pathology = grp.loc[grp["Pathology"] != "", "Pathology"].mode()
        pathology = pathology.iloc[0] if not pathology.empty else ""
        category = grp.loc[grp["Category"] != "", "Category"].mode()
        category = category.iloc[0] if not category.empty else ""

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["has_beta"] & ~grp["has_alpha"]]

        rows.append(
            {
                "epitope": epitope,
                "antigen_protein": antigen_protein,
                "pathology": pathology,
                "category": category,
                "primary_mhc": primary_mhc,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "paired_records": len(paired),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
                "unique_beta_only_tcrs": beta_only["CDR3.beta.aa"].nunique()
                if len(beta_only)
                else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. pMHC summary
# ---------------------------------------------------------------------------
def pmhc_summary(df):
    # Filter to rows with both epitope and MHC
    pmhc_df = df[(df["Epitope.peptide"] != "") & (df["MHC"] != "")]
    rows = []
    for (epitope, mhc), grp in pmhc_df.groupby(
        ["Epitope.peptide", "MHC"], sort=True
    ):
        mhc_class = grp["mhc_class"].mode()
        mhc_class = mhc_class.iloc[0] if not mhc_class.empty else "unknown"

        antigen_protein = grp.loc[grp["Antigen.protein"] != "", "Antigen.protein"].mode()
        antigen_protein = antigen_protein.iloc[0] if not antigen_protein.empty else ""
        pathology = grp.loc[grp["Pathology"] != "", "Pathology"].mode()
        pathology = pathology.iloc[0] if not pathology.empty else ""

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["has_beta"] & ~grp["has_alpha"]]

        rows.append(
            {
                "epitope": epitope,
                "mhc": mhc,
                "mhc_class": mhc_class,
                "antigen_protein": antigen_protein,
                "pathology": pathology,
                "total_records": len(grp),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
                "unique_beta_only_tcrs": beta_only["CDR3.beta.aa"].nunique()
                if len(beta_only)
                else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Study summary
# ---------------------------------------------------------------------------
def study_summary(df):
    # Clean PubMed.ID
    df_studies = df.copy()
    df_studies["PubMed.ID"] = df_studies["PubMed.ID"].replace("NA", "")

    rows = []
    for pubmed_id, grp in df_studies.groupby("PubMed.ID", sort=True):
        if pubmed_id == "":
            pubmed_id = "(empty)"
        paired = grp[grp["is_paired"]]

        method_col = "Antigen.identification.method"
        method_id = grp.loc[grp[method_col].replace("NA", "") != "", method_col].mode()
        method_id = method_id.iloc[0] if not method_id.empty else ""

        categories = grp.loc[grp["Category"] != "", "Category"].dropna().unique()
        species_list = grp.loc[grp["Species"] != "", "Species"].dropna().unique()

        rows.append(
            {
                "pubmed_id": pubmed_id,
                "total_records": len(grp),
                "paired_records": len(paired),
                "unique_epitopes": grp.loc[
                    grp["Epitope.peptide"] != "", "Epitope.peptide"
                ].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
                "categories": ";".join(sorted(categories)),
                "method_identification": method_id,
                "species": ";".join(sorted(species_list)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. MHC allele summary
# ---------------------------------------------------------------------------
def mhc_allele_summary(df):
    mhc_df = df[df["MHC"] != ""]
    rows = []
    for (mhc, mhc_class), grp in mhc_df.groupby(["MHC", "mhc_class"], sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "mhc": mhc,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "unique_epitopes": grp.loc[
                    grp["Epitope.peptide"] != "", "Epitope.peptide"
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
    # Host species
    species_df = df[df["Species"] != ""]
    for species, grp in species_df.groupby("Species", sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "category": "host_species",
                "species": species,
                "total_records": len(grp),
                "unique_epitopes": grp.loc[
                    grp["Epitope.peptide"] != "", "Epitope.peptide"
                ].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            }
        )
    # Pathology (organism/disease)
    path_df = df[df["Pathology"] != ""]
    for pathology, grp in path_df.groupby("Pathology", sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "category": "pathology",
                "species": pathology,
                "total_records": len(grp),
                "unique_epitopes": grp.loc[
                    grp["Epitope.peptide"] != "", "Epitope.peptide"
                ].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Data completeness
# ---------------------------------------------------------------------------
def data_completeness(df):
    has_epitope = df["Epitope.peptide"] != ""
    has_mhc = df["MHC"] != ""

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
                "unique_beta_cdr3": sub.loc[sub["has_beta"], "CDR3.beta.aa"].nunique(),
                "unique_epitopes": sub.loc[
                    sub["Epitope.peptide"] != "", "Epitope.peptide"
                ].nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SUMMARY.txt generation
# ---------------------------------------------------------------------------
def write_summary_txt(df, output_dir):
    """Write a human-readable SUMMARY.txt after all TSVs are exported."""
    from datetime import date

    # Read all exported TSVs
    ov = pd.read_csv(os.path.join(output_dir, "overview.tsv"), sep="\t")
    cat_ov = pd.read_csv(os.path.join(output_dir, "category_overview.tsv"), sep="\t")
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
    studies = pd.read_csv(os.path.join(output_dir, "study_summary.tsv"), sep="\t")

    r = ov.iloc[0]
    total = int(r["total_records"])
    lines = []

    # --- Header ---
    lines.append("=" * 80)
    lines.append("                    McPAS-TCR Analysis Summary")
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {INPUT_FILE}")
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
    lines.append(f"Unique beta CDR3s:      {r['unique_beta_cdr3']:>10,}")
    lines.append(f"Unique alpha CDR3s:     {r['unique_alpha_cdr3']:>10,}")
    lines.append(
        f"Paired records (a+b):   {paired:>10,}  ({paired_pct:.1f}% of total)"
    )
    lines.append(f"Unique paired TCRs:     {r['unique_paired_tcrs']:>10,}")
    lines.append(f"Unique epitopes:        {r['unique_epitopes']:>10,}")
    lines.append(f"Unique MHC alleles:     {r['unique_mhc']:>10,}")
    lines.append(f"Unique pMHC combos:     {r['unique_pmhc']:>10,}")
    lines.append(f"Unique TCR-pMHC pairs:  {r['unique_tcr_pmhc_paired']:>10,}")
    lines.append("")

    # Category distribution sub-block
    lines.append("Category distribution:")
    for _, row in cat_ov.iterrows():
        cat = row["category"]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cat:<20s} {cnt:>7,}  ({pct:>5.1f}%)")
    lines.append("")

    # Records with full annotation sub-block
    lines.append(
        f"Records with full annotation (epitope + MHC): "
        f"{has_epi_mhc:,}  ({has_epi_mhc_pct:.1f}% of total)"
    )
    lines.append("")

    # --- 2. MHC CLASS BREAKDOWN ---
    lines.append("2. MHC CLASS BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    # Aggregate mhc_alleles by class
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

    # Commentary
    mhci_row = mhc_class_agg[mhc_class_agg["mhc_class"] == "MHCI"]
    if not mhci_row.empty and mhc_total_records > 0:
        mhci_pct = 100.0 * int(mhci_row["records"].iloc[0]) / mhc_total_records
        lines.append(
            f"Among records with MHC annotation, MHC class I accounts for "
            f"{mhci_pct:.1f}% of records."
        )
    else:
        lines.append(
            "Note: McPAS uses heterogeneous MHC naming; many alleles fall into "
            "'unknown' class."
        )
    lines.append("")

    # --- 3. CATEGORY BREAKDOWN ---
    lines.append("3. CATEGORY BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Category':<20s} {'Records':>9s} {'%':>7s} {'Paired TCRs':>13s} "
        f"{'Epitopes':>10s}"
    )
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {row['category']:<20s} {cnt:>9,} {pct:>6.1f}% "
            f"{int(row['unique_paired_tcrs']):>13,} {int(row['unique_epitopes']):>10,}"
        )
    lines.append("")

    # Identify dominant category
    dominant = cat_ov.loc[cat_ov["total_records"].idxmax()]
    dom_pct = 100.0 * int(dominant["total_records"]) / total if total else 0
    lines.append(
        f"The '{dominant['category']}' category dominates the database with "
        f"{dom_pct:.1f}% of all records."
    )
    lines.append("")

    # --- 4. TOP EPITOPES ---
    lines.append("4. TOP EPITOPES")
    lines.append("-" * 80)
    lines.append("")
    lines.append("By total records:")
    lines.append("")
    lines.append(
        f"  {'Epitope':<22s} {'Pathology':<28s} {'MHC':<18s} "
        f"{'Records':>8s} {'Paired TCRs':>13s}"
    )

    top_epi = epi.nlargest(10, "total_records")
    for _, row in top_epi.iterrows():
        epi_name = str(row["epitope"])[:20]
        pathology = str(row["pathology"])[:26]
        mhc = str(row["primary_mhc"])[:16]
        lines.append(
            f"  {epi_name:<22s} {pathology:<28s} {mhc:<18s} "
            f"{int(row['total_records']):>8,} {int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

    lines.append("By unique paired TCRs:")
    lines.append("")
    lines.append(
        f"  {'Epitope':<22s} {'Pathology':<28s} {'Paired TCRs':>13s}"
    )

    top_paired = epi.nlargest(10, "unique_paired_tcrs")
    for _, row in top_paired.iterrows():
        epi_name = str(row["epitope"])[:20]
        pathology = str(row["pathology"])[:26]
        lines.append(
            f"  {epi_name:<22s} {pathology:<28s} "
            f"{int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

    # GILGFVFTL note
    gilg = epi[epi["epitope"] == "GILGFVFTL"]
    if not gilg.empty:
        g = gilg.iloc[0]
        lines.append(
            f"Note: GILGFVFTL (influenza M1 / HLA-A*02:01) has "
            f"{int(g['unique_paired_tcrs']):,} unique paired TCRs -- "
            f"a strong benchmark target."
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

    top_mhc = mhc_alleles.nlargest(10, "total_records")
    for _, row in top_mhc.iterrows():
        allele = str(row["mhc"])[:23]
        lines.append(
            f"  {allele:<25s} {row['mhc_class']:<10s} "
            f"{int(row['total_records']):>9,} "
            f"{int(row['unique_epitopes']):>10,} "
            f"{int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

    # Commentary on dominant allele
    if len(top_mhc) > 0:
        top_allele = top_mhc.iloc[0]
        top_allele_pct = (
            100.0 * int(top_allele["total_records"]) / mhc_total_records
            if mhc_total_records > 0
            else 0
        )
        lines.append(
            f"{top_allele['mhc']} is the most represented allele with "
            f"{top_allele_pct:.1f}% of MHC-annotated records."
        )
    lines.append("")

    # --- 6. PATHOLOGY & SPECIES ---
    lines.append("6. PATHOLOGY & SPECIES")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Top pathologies by epitope diversity:")
    lines.append("")

    pathologies = species[species["category"] == "pathology"].copy()
    top_path = pathologies.nlargest(10, "unique_epitopes")
    lines.append(
        f"  {'Pathology':<40s} {'Epitopes':>10s} {'Records':>9s} {'Paired TCRs':>13s}"
    )
    for _, row in top_path.iterrows():
        path_name = str(row["species"])[:38]
        lines.append(
            f"  {path_name:<40s} {int(row['unique_epitopes']):>10,} "
            f"{int(row['total_records']):>9,} {int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

    # Host species breakdown
    host_species = species[species["category"] == "host_species"].copy()
    host_species = host_species.sort_values("total_records", ascending=False)
    host_total = int(host_species["total_records"].sum())
    lines.append("Host species breakdown:")
    for _, row in host_species.iterrows():
        sp = str(row["species"])
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / host_total if host_total else 0
        epi_cnt = int(row["unique_epitopes"])
        paired_cnt = int(row["unique_paired_tcrs"])
        lines.append(
            f"  {sp + ':':<18s} {cnt:>7,} records  ({pct:>5.1f}%)  "
            f"{epi_cnt:>5,} epitopes  {paired_cnt:>6,} paired TCRs"
        )
    lines.append("")

    # Commentary
    if len(host_species) >= 2:
        top_host = host_species.iloc[0]
        top_host_pct = (
            100.0 * int(top_host["total_records"]) / host_total
            if host_total
            else 0
        )
        lines.append(
            f"{top_host['species']} accounts for {top_host_pct:.1f}% of records "
            f"with species annotation."
        )
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
    studies_large = int((studies["total_records"] > 1000).sum())

    lines.append(f"Total studies (PubMed IDs): {total_studies}")
    lines.append(
        f"Studies with paired TCR data: {studies_with_paired} "
        f"({studies_with_paired_pct:.0f}%)"
    )
    lines.append(f"Studies with >1,000 records: {studies_large}")
    lines.append("")

    lines.append("Largest contributors:")
    top_studies = studies.nlargest(5, "total_records")
    for _, row in top_studies.iterrows():
        pubmed = str(row["pubmed_id"])
        label = f"PMID:{pubmed}" if pubmed != "(empty)" else "(no PubMed ID)"
        lines.append(
            f"  {label:<25s} {int(row['total_records']):>7,} records  "
            f"({int(row['paired_records']):>5,} paired, "
            f"{int(row['unique_epitopes']):>3,} epitopes)"
        )
    lines.append("")

    # Commentary on top-study concentration
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
        f"Records with epitope + MHC ({has_epi_mhc:,}) are the most usable for "
        f"TCR-pMHC modeling."
    )
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

    # a) Usable annotated data
    has_epi_mhc_paired = int(
        completeness.loc[
            completeness["category"] == "has_epitope_and_mhc", "paired_tcrs"
        ].iloc[0]
    )
    has_epi_mhc_epitopes = int(
        completeness.loc[
            completeness["category"] == "has_epitope_and_mhc", "unique_epitopes"
        ].iloc[0]
    )
    lines.append(
        f"a) Usable annotated data (epitope + MHC): {has_epi_mhc:,} records with "
        f"{has_epi_mhc_paired:,} unique paired\n"
        f"   TCRs across {has_epi_mhc_epitopes:,} epitopes. "
        f"This is the high-confidence training set."
    )
    lines.append("")

    # b) Paired TCR availability
    unique_paired = int(r["unique_paired_tcrs"])
    lines.append(
        f"b) Paired TCR availability: {unique_paired:,} unique paired TCRs total "
        f"({paired_pct:.1f}% of records\n"
        f"   have both alpha+beta chains). Including records without epitope/MHC\n"
        f"   annotation expands paired data for unsupervised pre-training."
    )
    lines.append("")

    # c) Category bias
    lines.append(
        f"c) Category bias: '{dominant['category']}' accounts for {dom_pct:.1f}% "
        f"of records. Models trained\n"
        f"   on this data will perform best on pathogen-specific TCRs but may\n"
        f"   underperform on cancer neoantigens or autoimmune targets."
    )
    lines.append("")

    # d) HLA bias
    if len(top_mhc) > 0:
        top_a = top_mhc.iloc[0]
        top_a_pct = (
            100.0 * int(top_a["total_records"]) / mhc_total_records
            if mhc_total_records > 0
            else 0
        )
        lines.append(
            f"d) HLA bias: {top_a['mhc']} represents {top_a_pct:.1f}% of "
            f"MHC-annotated records.\n"
            f"   Models will have strong performance on this allele but may "
            f"underperform\n"
            f"   on rare alleles."
        )
    lines.append("")

    # e) Epitope imbalance
    top10_epi = epi.nlargest(10, "total_records")
    top10_epi_records = int(top10_epi["total_records"].sum())
    epi_with_records = epi[epi["total_records"] > 0]
    all_epi_records = int(epi_with_records["total_records"].sum())
    top10_epi_pct = (
        100.0 * top10_epi_records / all_epi_records if all_epi_records else 0
    )
    lines.append(
        f"e) Epitope imbalance: The top 10 epitopes account for "
        f"{top10_epi_pct:.0f}% of epitope-annotated\n"
        f"   records. GILGFVFTL and NLVPMVATV are the best-represented targets,\n"
        f"   making them ideal benchmarks but also sources of training bias."
    )
    lines.append("")

    # f) Species coverage
    if len(host_species) >= 2:
        for _, row in host_species.iterrows():
            sp = str(row["species"])
            if sp.lower() == "human":
                human_pct = (
                    100.0 * int(row["total_records"]) / host_total
                    if host_total
                    else 0
                )
            elif sp.lower() == "mouse":
                mouse_pct = (
                    100.0 * int(row["total_records"]) / host_total
                    if host_total
                    else 0
                )
        lines.append(
            f"f) Species coverage: {human_pct:.0f}% human data, "
            f"{mouse_pct:.0f}% mouse. Human-centric models\n"
            f"   will have limited cross-species generalization."
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
    input_path = os.path.join(project_root, INPUT_FILE)
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {input_path} ...")
    df = load_data(input_path)
    print(f"  Loaded {len(df):,} records, {df.columns.size} columns")
    print(f"  Category distribution: {df['Category'].value_counts().to_dict()}")
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

    # Write human-readable summary
    print("Writing SUMMARY.txt ...")
    write_summary_txt(df, output_dir)

    # --- Verification ---
    print("\n--- Verification ---")

    # Check category_overview sums match overview
    ov = pd.read_csv(os.path.join(output_dir, "overview.tsv"), sep="\t")
    cat_ov = pd.read_csv(os.path.join(output_dir, "category_overview.tsv"), sep="\t")
    cat_sum = cat_ov["total_records"].sum()
    all_total = ov.iloc[0]["total_records"]
    print(
        f"Category sum: {cat_sum}, Overview total: {all_total}, "
        f"match: {cat_sum == all_total}"
    )

    # Spot-check GILGFVFTL
    epi = pd.read_csv(os.path.join(output_dir, "epitope_summary.tsv"), sep="\t")
    gilg = epi[epi["epitope"] == "GILGFVFTL"]
    if not gilg.empty:
        g = gilg.iloc[0]
        print(
            f"GILGFVFTL: {g['total_records']} records, "
            f"{g['unique_paired_tcrs']} paired TCRs, "
            f"MHC={g['primary_mhc']}, class={g['mhc_class']}"
        )
    else:
        print("GILGFVFTL not found in McPAS epitope summary")

    # Data completeness verification
    comp = pd.read_csv(os.path.join(output_dir, "data_completeness.tsv"), sep="\t")
    comp_sum = comp["record_count"].sum()
    print(
        f"Completeness sum: {comp_sum}, Overview total: {all_total}, "
        f"match: {comp_sum == all_total}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
