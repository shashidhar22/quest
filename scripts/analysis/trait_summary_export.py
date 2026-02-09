#!/usr/bin/env python3
"""Export comprehensive TRAIT analysis tables."""

import os
import re
import zipfile

import numpy as np
import pandas as pd

INPUT_DIR = "data/databases/trait"
OUTPUT_DIR = "data/analysis/trait"

# MHC allele conversion: A0201 -> HLA-A*02:01
MHC_PATTERN = re.compile(r"^([A-Z]+)(\d{2})(\d{2})$")
# Handle NR(B0801) pattern
NR_PATTERN = re.compile(r"^NR\(([A-Z]+\d{4})\)$")


def convert_mhc_allele(raw):
    """Convert compact MHC allele name to standard notation."""
    nr_match = NR_PATTERN.match(raw)
    if nr_match:
        raw = nr_match.group(1)
    m = MHC_PATTERN.match(raw)
    if m:
        gene, group, protein = m.group(1), m.group(2), m.group(3)
        return f"HLA-{gene}*{group}:{protein}"
    return raw


def parse_filename(fname):
    """Parse TRAIT filename to extract MHC, epitope, protein, disease, binding.

    Pattern: A0201_GILGFVFTL_Flu-MP_Influenza_binder_pos.txt
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    if len(parts) < 5:
        return None

    mhc_raw = parts[0]
    epitope = parts[1]
    # Parts between epitope and "binder" are protein and disease
    try:
        binder_idx = parts.index("binder")
    except ValueError:
        return None

    middle = parts[2:binder_idx]
    protein = middle[0] if len(middle) >= 1 else ""
    disease = middle[1] if len(middle) >= 2 else ""
    binding_label = parts[binder_idx + 1] if binder_idx + 1 < len(parts) else ""

    mhc = convert_mhc_allele(mhc_raw)

    return {
        "mhc_raw": mhc_raw,
        "mhc": mhc,
        "epitope": epitope,
        "protein": protein,
        "disease": disease,
        "binding_label": binding_label,
    }


def read_trait_tsv(fileobj, file_info):
    """Read a single TRAIT TSV from a file-like object."""
    try:
        df = pd.read_csv(fileobj, sep="\t", dtype=str, na_filter=False)
    except Exception:
        return None

    if df.empty:
        return None

    # Attach parsed metadata
    for key, val in file_info.items():
        df[key] = val

    return df


def load_data(input_dir):
    """Load all TRAIT data from Omics.zip and epitope zips."""
    chunks = []

    # 1. Omics.zip
    omics_path = os.path.join(input_dir, "main", "Omics.zip")
    if os.path.exists(omics_path):
        print(f"  Reading {omics_path} ...")
        with zipfile.ZipFile(omics_path, "r") as zf:
            txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
            for name in txt_files:
                basename = os.path.basename(name)
                info = parse_filename(basename)
                if info is None:
                    continue
                with zf.open(name) as f:
                    chunk = read_trait_tsv(f, info)
                    if chunk is not None:
                        chunks.append(chunk)
        print(f"    {len(chunks)} files loaded from Omics.zip")

    # 2. Epitope zips (binding + non_binding)
    for subdir in ["binding", "non_binding"]:
        epi_dir = os.path.join(input_dir, "epitopes", subdir)
        if not os.path.isdir(epi_dir):
            continue
        for zname in sorted(os.listdir(epi_dir)):
            zpath = os.path.join(epi_dir, zname)
            if not zname.endswith(".zip"):
                continue
            if not zipfile.is_zipfile(zpath):
                print(f"    Skipping bad zip: {zpath}")
                continue
            with zipfile.ZipFile(zpath, "r") as zf:
                for name in zf.namelist():
                    if not name.endswith(".txt"):
                        continue
                    basename = os.path.basename(name)
                    info = parse_filename(basename)
                    if info is None:
                        info = parse_filename(os.path.splitext(zname)[0] + ".txt")
                    if info is None:
                        continue
                    with zf.open(name) as f:
                        chunk = read_trait_tsv(f, info)
                        if chunk is not None:
                            chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No TRAIT data loaded")

    df = pd.concat(chunks, ignore_index=True)

    # Derive standard columns
    df["CDR3a"] = df.get("CDR3a", pd.Series("", index=df.index)).replace("", pd.NA).fillna("")
    df["CDR3b"] = df.get("CDR3b", pd.Series("", index=df.index)).replace("", pd.NA).fillna("")

    # Handle 'None' string values
    for col in ["CDR3a", "CDR3b", "TRAV", "TRAJ", "TRBV", "TRBD", "TRBJ"]:
        if col in df.columns:
            df[col] = df[col].replace("None", "")

    df["has_alpha"] = df["CDR3a"] != ""
    df["has_beta"] = df["CDR3b"] != ""
    df["is_paired"] = df["has_alpha"] & df["has_beta"]

    df["paired_tcr"] = np.where(
        df["is_paired"],
        df["CDR3a"] + "|" + df["CDR3b"],
        pd.NA,
    )

    df["has_epitope"] = True  # All records have epitope from filename
    df["has_mhc"] = True  # All records have MHC from filename

    # MHC class derivation
    mhc_upper = df["mhc"].str.upper()
    df["mhc_class"] = np.where(
        mhc_upper.str.contains(r"HLA-[ABC]", regex=True, na=False),
        "MHCI",
        np.where(
            mhc_upper.str.contains(r"HLA-D", regex=True, na=False),
            "MHCII",
            "unknown",
        ),
    )

    # pMHC key
    df["pmhc"] = df["epitope"] + "|" + df["mhc"]

    # TCR-pMHC paired key
    df["tcr_pmhc_paired"] = np.where(
        df["is_paired"],
        df["paired_tcr"] + "|" + df["epitope"],
        pd.NA,
    )

    # Category = disease from filename
    df["category"] = df["disease"].replace("", "(unknown)")

    # Study = Donor column
    df["study"] = df.get("Donor", pd.Series("", index=df.index)).replace("", "(unknown)")

    return df


# ---------------------------------------------------------------------------
# 1. Overview (single-row dataset summary)
# ---------------------------------------------------------------------------
def overview(df):
    rows = [
        {
            "total_records": len(df),
            "unique_beta_cdr3": df.loc[df["has_beta"], "CDR3b"].nunique(),
            "unique_alpha_cdr3": df.loc[df["has_alpha"], "CDR3a"].nunique(),
            "paired_records": int(df["is_paired"].sum()),
            "unique_paired_tcrs": df["paired_tcr"].dropna().nunique(),
            "unique_epitopes": df["epitope"].nunique(),
            "unique_mhc": df["mhc"].nunique(),
            "unique_pmhc": df["pmhc"].nunique(),
            "unique_tcr_pmhc_paired": df["tcr_pmhc_paired"].dropna().nunique(),
        }
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Category overview (one row per disease category)
# ---------------------------------------------------------------------------
def category_overview(df):
    rows = []
    for category, grp in df.groupby("category", sort=True):
        rows.append(
            {
                "category": category,
                "total_records": len(grp),
                "unique_beta_cdr3": grp.loc[grp["has_beta"], "CDR3b"].nunique(),
                "unique_alpha_cdr3": grp.loc[grp["has_alpha"], "CDR3a"].nunique(),
                "paired_records": int(grp["is_paired"].sum()),
                "unique_paired_tcrs": grp["paired_tcr"].dropna().nunique(),
                "unique_epitopes": grp["epitope"].nunique(),
                "unique_mhc": grp.loc[grp["mhc"] != "", "mhc"].nunique(),
                "unique_pmhc": grp["pmhc"].nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Epitope summary
# ---------------------------------------------------------------------------
def epitope_summary(df):
    rows = []
    for epitope, grp in df.groupby("epitope", sort=True):
        primary_mhc = grp["mhc"].mode()
        primary_mhc = primary_mhc.iloc[0] if not primary_mhc.empty else ""
        mhc_class = grp.loc[grp["mhc_class"] != "unknown", "mhc_class"].mode()
        mhc_class = mhc_class.iloc[0] if not mhc_class.empty else "unknown"

        protein = grp.loc[grp["protein"] != "", "protein"].mode()
        protein = protein.iloc[0] if not protein.empty else ""
        disease = grp.loc[grp["disease"] != "", "disease"].mode()
        disease = disease.iloc[0] if not disease.empty else ""

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["has_beta"] & ~grp["has_alpha"]]

        rows.append(
            {
                "epitope": epitope,
                "antigen_protein": protein,
                "disease": disease,
                "primary_mhc": primary_mhc,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "paired_records": len(paired),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
                "unique_beta_only_tcrs": beta_only["CDR3b"].nunique()
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
    for (epitope, mhc), grp in df.groupby(["epitope", "mhc"], sort=True):
        mhc_class = grp["mhc_class"].mode()
        mhc_class = mhc_class.iloc[0] if not mhc_class.empty else "unknown"

        protein = grp.loc[grp["protein"] != "", "protein"].mode()
        protein = protein.iloc[0] if not protein.empty else ""
        disease = grp.loc[grp["disease"] != "", "disease"].mode()
        disease = disease.iloc[0] if not disease.empty else ""

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["has_beta"] & ~grp["has_alpha"]]

        rows.append(
            {
                "epitope": epitope,
                "mhc": mhc,
                "mhc_class": mhc_class,
                "antigen_protein": protein,
                "disease": disease,
                "total_records": len(grp),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
                "unique_beta_only_tcrs": beta_only["CDR3b"].nunique()
                if len(beta_only)
                else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Study summary (by Donor)
# ---------------------------------------------------------------------------
def study_summary(df):
    rows = []
    for donor, grp in df.groupby("study", sort=True):
        paired = grp[grp["is_paired"]]
        epitopes = grp["epitope"].unique()
        diseases = grp.loc[grp["disease"] != "", "disease"].dropna().unique()

        rows.append(
            {
                "donor": donor,
                "total_records": len(grp),
                "paired_records": len(paired),
                "unique_epitopes": grp["epitope"].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
                "categories": ";".join(sorted(diseases)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. MHC allele summary
# ---------------------------------------------------------------------------
def mhc_allele_summary(df):
    rows = []
    for (mhc, mhc_class), grp in df.groupby(["mhc", "mhc_class"], sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "mhc": mhc,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "unique_epitopes": grp["epitope"].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Species breakdown
# ---------------------------------------------------------------------------
def species_breakdown(df):
    """TRAIT data is all human; provide binding label breakdown instead."""
    rows = [
        {
            "category": "host_species",
            "species": "Human",
            "total_records": len(df),
            "unique_epitopes": df["epitope"].nunique(),
            "unique_paired_tcrs": df["paired_tcr"].dropna().nunique(),
        }
    ]
    # Binding label breakdown
    if "binding_label" in df.columns:
        for label, grp in df.groupby("binding_label", sort=True):
            paired = grp[grp["is_paired"]]
            rows.append(
                {
                    "category": "binding_label",
                    "species": label,
                    "total_records": len(grp),
                    "unique_epitopes": grp["epitope"].nunique(),
                    "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
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
                "unique_beta_cdr3": sub.loc[sub["has_beta"], "CDR3b"].nunique(),
                "unique_epitopes": sub["epitope"].nunique() if len(sub) > 0 else 0,
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
    lines.append("                    TRAIT Analysis Summary")
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {INPUT_DIR}")
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

    lines.append("Category distribution:")
    for _, row in cat_ov.iterrows():
        cat = row["category"]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cat:<20s} {cnt:>7,}  ({pct:>5.1f}%)")
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
            f"Among records with MHC annotation, MHC class I accounts for "
            f"{mhci_pct:.1f}% of records."
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
        f"  {'Epitope':<22s} {'Disease':<28s} {'MHC':<18s} "
        f"{'Records':>8s} {'Paired TCRs':>13s}"
    )

    top_epi = epi.nlargest(10, "total_records")
    for _, row in top_epi.iterrows():
        epi_name = str(row["epitope"])[:20]
        disease = str(row["disease"])[:26]
        mhc = str(row["primary_mhc"])[:16]
        lines.append(
            f"  {epi_name:<22s} {disease:<28s} {mhc:<18s} "
            f"{int(row['total_records']):>8,} {int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

    lines.append("By unique paired TCRs:")
    lines.append("")
    lines.append(
        f"  {'Epitope':<22s} {'Disease':<28s} {'Paired TCRs':>13s}"
    )

    top_paired = epi.nlargest(10, "unique_paired_tcrs")
    for _, row in top_paired.iterrows():
        epi_name = str(row["epitope"])[:20]
        disease = str(row["disease"])[:26]
        lines.append(
            f"  {epi_name:<22s} {disease:<28s} "
            f"{int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

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

    host_species = species[species["category"] == "host_species"]
    lines.append("Host species: All records are from Human TCR data.")
    lines.append("")

    binding_labels = species[species["category"] == "binding_label"]
    if not binding_labels.empty:
        lines.append("Binding label breakdown:")
        bl_total = int(binding_labels["total_records"].sum())
        for _, row in binding_labels.sort_values(
            "total_records", ascending=False
        ).iterrows():
            lbl = str(row["species"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / bl_total if bl_total else 0
            lines.append(f"  {lbl:<20s} {cnt:>9,}  ({pct:>5.1f}%)")
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

    lines.append(f"Total donors: {total_studies}")
    lines.append(
        f"Donors with paired TCR data: {studies_with_paired} "
        f"({studies_with_paired_pct:.0f}%)"
    )
    lines.append("")

    lines.append("Largest contributors:")
    top_studies = studies.nlargest(5, "total_records")
    for _, row in top_studies.iterrows():
        donor = str(row["donor"])
        lines.append(
            f"  {donor:<25s} {int(row['total_records']):>7,} records  "
            f"({int(row['paired_records']):>5,} paired, "
            f"{int(row['unique_epitopes']):>3,} epitopes)"
        )
    lines.append("")

    top5_records = int(top_studies["total_records"].sum())
    top5_pct = 100.0 * top5_records / total if total else 0
    lines.append(
        f"The top 5 donors account for ~{top5_pct:.0f}% of all records."
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
        f"All {total:,} records have both epitope and MHC annotation "
        f"(derived from TRAIT filenames)."
    )
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

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
        f"All TRAIT records are fully annotated."
    )
    lines.append("")

    unique_paired = int(r["unique_paired_tcrs"])
    lines.append(
        f"b) Paired TCR availability: {unique_paired:,} unique paired TCRs total "
        f"({paired_pct:.1f}% of records\n"
        f"   have both alpha+beta chains). TRAIT provides rich paired TCR data\n"
        f"   with full V/D/J gene segment annotation."
    )
    lines.append("")

    lines.append(
        f"c) Category bias: '{dominant['category']}' accounts for {dom_pct:.1f}% "
        f"of records. Models trained\n"
        f"   on this data may underperform on underrepresented categories."
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
            f"records. Models will have stronger performance on\n"
            f"   well-represented alleles."
        )
    lines.append("")

    top10_epi = epi.nlargest(10, "total_records")
    top10_epi_records = int(top10_epi["total_records"].sum())
    all_epi_records = int(epi["total_records"].sum())
    top10_epi_pct = (
        100.0 * top10_epi_records / all_epi_records if all_epi_records else 0
    )
    lines.append(
        f"e) Epitope imbalance: The top 10 epitopes account for "
        f"{top10_epi_pct:.0f}% of all records.\n"
        f"   TRAIT includes both binders and non-binders, making it valuable\n"
        f"   for training binding classifiers."
    )
    lines.append("")

    lines.append(
        f"f) Species coverage: All data is from human TCR repertoires. No\n"
        f"   cross-species data is available."
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
    input_dir = os.path.join(project_root, INPUT_DIR)
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading TRAIT data from {input_dir} ...")
    df = load_data(input_dir)
    print(f"  Loaded {len(df):,} records, {df.columns.size} columns")
    print(f"  Category distribution: {df['category'].value_counts().to_dict()}")
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
    cat_ov = pd.read_csv(os.path.join(output_dir, "category_overview.tsv"), sep="\t")
    cat_sum = cat_ov["total_records"].sum()
    all_total = ov.iloc[0]["total_records"]
    print(
        f"Category sum: {cat_sum}, Overview total: {all_total}, "
        f"match: {cat_sum == all_total}"
    )

    epi = pd.read_csv(os.path.join(output_dir, "epitope_summary.tsv"), sep="\t")
    gilg = epi[epi["epitope"] == "GILGFVFTL"]
    if not gilg.empty:
        g = gilg.iloc[0]
        print(
            f"GILGFVFTL: {g['total_records']} records, "
            f"{g['unique_paired_tcrs']} paired TCRs, "
            f"MHC={g['primary_mhc']}, class={g['mhc_class']}"
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
