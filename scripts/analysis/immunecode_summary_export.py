#!/usr/bin/env python3
"""Export comprehensive immuneCODE analysis tables."""

import os
import subprocess

import numpy as np
import pandas as pd

MIRA_DIR = "data/databases/immuneCODE/immunecode_covid/ImmuneCODE-MIRA-Release002.1"
REVIEW_DIR = "data/databases/immuneCODE/immunecode_covid/ImmuneCODE-Review-002"
OUTPUT_DIR = "data/analysis/immunecode"


def parse_tcr_bioidentity(bio_id):
    """Parse 'CASSAQGTGDRGYTF+TCRBV27-01+TCRBJ01-02' into components."""
    parts = str(bio_id).split("+")
    cdr3 = parts[0] if len(parts) >= 1 else ""
    v_gene = parts[1] if len(parts) >= 2 else ""
    j_gene = parts[2] if len(parts) >= 3 else ""
    return cdr3, v_gene, j_gene


def load_subject_metadata(mira_dir):
    """Load subject metadata with HLA typing."""
    path = os.path.join(mira_dir, "subject-metadata.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    meta = pd.read_csv(path, dtype=str, na_filter=False, encoding="latin-1")
    # Replace N/A strings
    meta = meta.replace("N/A", "")
    return meta


def load_data(mira_dir):
    """Load MIRA release data (peptide CI, CII, minigene)."""
    chunks = []

    # 1. peptide-detail-ci.csv (Class I)
    ci_path = os.path.join(mira_dir, "peptide-detail-ci.csv")
    if os.path.exists(ci_path):
        print(f"  Reading {ci_path} ...")
        ci = pd.read_csv(ci_path, dtype=str, na_filter=False)
        ci["mhc_class"] = "MHCI"
        ci["source_file"] = "peptide-detail-ci"
        # CI has "ORF Coverage" and "Amino Acids" (comma-separated)
        if "ORF Coverage" not in ci.columns and "ORF" in ci.columns:
            ci["ORF Coverage"] = ci["ORF"]
        if "Amino Acids" not in ci.columns and "Amino Acid" in ci.columns:
            ci["Amino Acids"] = ci["Amino Acid"]
        chunks.append(ci)
        print(f"    {len(ci):,} rows")

    # 2. peptide-detail-cii.csv (Class II)
    cii_path = os.path.join(mira_dir, "peptide-detail-cii.csv")
    if os.path.exists(cii_path):
        print(f"  Reading {cii_path} ...")
        cii = pd.read_csv(cii_path, dtype=str, na_filter=False)
        cii["mhc_class"] = "MHCII"
        cii["source_file"] = "peptide-detail-cii"
        # CII has "ORF" and "Amino Acid" (single)
        if "ORF Coverage" not in cii.columns:
            cii["ORF Coverage"] = cii.get("ORF", "")
        if "Amino Acids" not in cii.columns:
            cii["Amino Acids"] = cii.get("Amino Acid", "")
        chunks.append(cii)
        print(f"    {len(cii):,} rows")

    # 3. minigene-detail.csv
    mini_path = os.path.join(mira_dir, "minigene-detail.csv")
    if os.path.exists(mini_path):
        print(f"  Reading {mini_path} ...")
        mini = pd.read_csv(mini_path, dtype=str, na_filter=False)
        mini["mhc_class"] = "unknown"
        mini["source_file"] = "minigene-detail"
        if "ORF Coverage" not in mini.columns:
            mini["ORF Coverage"] = mini.get("ORF", "")
        if "Amino Acids" not in mini.columns:
            mini["Amino Acids"] = mini.get("Amino Acid", "")
        chunks.append(mini)
        print(f"    {len(mini):,} rows")

    if not chunks:
        raise RuntimeError("No immuneCODE MIRA data loaded")

    df = pd.concat(chunks, ignore_index=True)

    # Parse TCR BioIdentity
    parsed = df["TCR BioIdentity"].apply(parse_tcr_bioidentity)
    df["cdr3_beta"] = parsed.apply(lambda x: x[0])
    df["v_call_beta"] = parsed.apply(lambda x: x[1])
    df["j_call_beta"] = parsed.apply(lambda x: x[2])

    # Beta-only data
    df["has_alpha"] = False
    df["has_beta"] = df["cdr3_beta"] != ""
    df["is_paired"] = False

    df["paired_tcr"] = pd.NA

    # Epitope from Amino Acids column
    df["has_epitope"] = df["Amino Acids"] != ""
    df["has_mhc"] = True  # All subjects are HLA-typed

    # Category = ORF Coverage
    df["category"] = df["ORF Coverage"].replace("", "(unknown)")

    # Study = Experiment
    df["study"] = df.get("Experiment", pd.Series("", index=df.index))

    return df


def count_review_records(review_dir):
    """Count total records in Review-002 directory via batch wc -l."""
    if not os.path.isdir(review_dir):
        return 0, 0

    tsv_files = [f for f in os.listdir(review_dir) if f.endswith(".tsv")]
    if not tsv_files:
        return 0, 0

    # Use find + xargs for a single batched wc -l (much faster than per-file)
    try:
        result = subprocess.run(
            f'find "{review_dir}" -name "*.tsv" -print0 | xargs -0 wc -l',
            capture_output=True, text=True, timeout=300, shell=True,
        )
        if result.returncode == 0:
            # Last line of wc -l output is "TOTAL total"
            lines = result.stdout.strip().split("\n")
            total_line = lines[-1].strip()
            total_lines = int(total_line.split()[0])
            # Subtract one header line per file
            total_lines -= len(tsv_files)
            return len(tsv_files), max(total_lines, 0)
    except (subprocess.TimeoutExpired, ValueError):
        pass

    return len(tsv_files), 0


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
def overview(df):
    rows = [
        {
            "total_records": len(df),
            "unique_beta_cdr3": df.loc[df["has_beta"], "cdr3_beta"].nunique(),
            "unique_alpha_cdr3": 0,
            "paired_records": 0,
            "unique_paired_tcrs": 0,
            "unique_epitopes": _count_unique_epitopes(df),
            "unique_mhc": 0,
            "unique_pmhc": 0,
            "unique_tcr_pmhc_paired": 0,
        }
    ]
    return pd.DataFrame(rows)


def _count_unique_epitopes(df):
    """Count unique epitopes across comma-separated Amino Acids fields."""
    all_peptides = set()
    for val in df.loc[df["has_epitope"], "Amino Acids"]:
        for pep in str(val).split(","):
            pep = pep.strip()
            if pep:
                all_peptides.add(pep)
    return len(all_peptides)


# ---------------------------------------------------------------------------
# 2. Category overview (by ORF)
# ---------------------------------------------------------------------------
def category_overview(df):
    rows = []
    for category, grp in df.groupby("category", sort=True):
        rows.append(
            {
                "category": category,
                "total_records": len(grp),
                "unique_beta_cdr3": grp.loc[
                    grp["has_beta"], "cdr3_beta"
                ].nunique(),
                "unique_alpha_cdr3": 0,
                "paired_records": 0,
                "unique_paired_tcrs": 0,
                "unique_epitopes": _count_unique_epitopes(grp),
                "unique_mhc": 0,
                "unique_pmhc": 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Epitope summary (explode comma-separated Amino Acids)
# ---------------------------------------------------------------------------
def epitope_summary(df):
    # Vectorized explode of comma-separated peptides
    epi_df = df[df["has_epitope"]][["Amino Acids", "cdr3_beta", "mhc_class", "category"]].copy()
    if epi_df.empty:
        return pd.DataFrame(
            columns=["epitope", "category", "mhc_class", "total_records", "unique_beta_cdr3"]
        )

    epi_df["epitope"] = epi_df["Amino Acids"].str.split(",")
    epi_df = epi_df.explode("epitope")
    epi_df["epitope"] = epi_df["epitope"].str.strip()
    epi_df = epi_df[epi_df["epitope"] != ""]
    rows = []
    for epitope, grp in epi_df.groupby("epitope", sort=True):
        mhc_class = grp["mhc_class"].mode()
        mhc_class = mhc_class.iloc[0] if not mhc_class.empty else "unknown"
        category = grp["category"].mode()
        category = category.iloc[0] if not category.empty else ""

        rows.append(
            {
                "epitope": epitope,
                "category": category,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "unique_beta_cdr3": grp["cdr3_beta"].nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. pMHC summary (partial — MHC is per-subject not per-record)
# ---------------------------------------------------------------------------
def pmhc_summary(df):
    return pd.DataFrame(
        columns=[
            "epitope",
            "mhc",
            "mhc_class",
            "total_records",
            "unique_beta_cdr3",
            "note",
        ]
    )


# ---------------------------------------------------------------------------
# 5. Study summary (by Experiment)
# ---------------------------------------------------------------------------
def study_summary(df):
    rows = []
    for experiment, grp in df.groupby("study", sort=True):
        categories = grp.loc[grp["category"] != "(unknown)", "category"].unique()
        rows.append(
            {
                "experiment": experiment,
                "total_records": len(grp),
                "unique_beta_cdr3": grp.loc[
                    grp["has_beta"], "cdr3_beta"
                ].nunique(),
                "unique_epitopes": _count_unique_epitopes(grp),
                "mhc_classes": ";".join(
                    sorted(grp["mhc_class"].unique())
                ),
                "categories": ";".join(sorted(categories)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. MHC allele summary (from subject metadata)
# ---------------------------------------------------------------------------
def mhc_allele_summary_from_metadata(mira_dir):
    """Build MHC allele summary from subject-metadata.csv."""
    meta = load_subject_metadata(mira_dir)
    if meta.empty:
        return pd.DataFrame(
            columns=["mhc", "mhc_class", "subject_count", "note"]
        )

    allele_counts = {}
    hla_cols_class_i = [
        c for c in meta.columns if c.startswith("HLA-A") or c.startswith("HLA-B") or c.startswith("HLA-C")
    ]
    hla_cols_class_ii = [
        c
        for c in meta.columns
        if any(c.startswith(p) for p in ["DPA1", "DPB1", "DQA1", "DQB1", "DRB1", "DRB3", "DRB4", "DRB5"])
    ]

    for _, row in meta.iterrows():
        for col in hla_cols_class_i:
            allele = str(row[col]).strip()
            if allele and allele != "N/A":
                key = (allele, "MHCI")
                allele_counts[key] = allele_counts.get(key, 0) + 1
        for col in hla_cols_class_ii:
            allele = str(row[col]).strip()
            if allele and allele != "N/A":
                key = (allele, "MHCII")
                allele_counts[key] = allele_counts.get(key, 0) + 1

    rows = []
    for (allele, mhc_class), count in sorted(allele_counts.items()):
        rows.append(
            {
                "mhc": allele,
                "mhc_class": mhc_class,
                "subject_count": count,
                "note": "from subject HLA typing, not per-TCR annotation",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Species breakdown
# ---------------------------------------------------------------------------
def species_breakdown(df):
    rows = [
        {
            "category": "host_species",
            "species": "Human",
            "total_records": len(df),
            "unique_epitopes": _count_unique_epitopes(df),
            "unique_beta_cdr3": df.loc[df["has_beta"], "cdr3_beta"].nunique(),
        },
        {
            "category": "antigen_species",
            "species": "SARS-CoV-2",
            "total_records": len(df),
            "unique_epitopes": _count_unique_epitopes(df),
            "unique_beta_cdr3": df.loc[df["has_beta"], "cdr3_beta"].nunique(),
        },
    ]
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
                "paired_tcrs": 0,
                "unique_beta_cdr3": sub.loc[
                    sub["has_beta"], "cdr3_beta"
                ].nunique(),
                "unique_epitopes": _count_unique_epitopes(sub) if len(sub) > 0 else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SUMMARY.txt generation
# ---------------------------------------------------------------------------
def write_summary_txt(df, output_dir, mira_dir, review_dir):
    """Write a human-readable SUMMARY.txt."""
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
    lines.append("                    immuneCODE Analysis Summary")
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {MIRA_DIR}")
    lines.append("=" * 80)
    lines.append("")

    # --- 1. DATASET OVERVIEW ---
    # Count Review-002 records
    print("  Counting Review-002 records (this may take a moment) ...")
    review_files, review_records = count_review_records(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            review_dir,
        )
        if not os.path.isabs(review_dir)
        else review_dir
    )

    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append("MIRA Release 002.1 (TCR-antigen mapping):")
    lines.append(f"  Total MIRA records:     {total:>10,}")
    lines.append(f"  Unique beta CDR3s:      {r['unique_beta_cdr3']:>10,}")
    lines.append(f"  Unique epitopes:        {r['unique_epitopes']:>10,}")
    lines.append(f"  Note: Beta-chain only data (no alpha chains in MIRA release)")
    lines.append("")
    if review_records > 0:
        lines.append("Review-002 (raw repertoire data):")
        lines.append(f"  Subject files:          {review_files:>10,}")
        lines.append(f"  Total TCR records:      {review_records:>10,}")
        lines.append(
            f"  Note: Raw TCRB repertoire data, not loaded for detailed analysis"
        )
        lines.append("")

    lines.append("MHC class distribution (MIRA):")
    for _, row in cat_ov.iterrows():
        cat = row["category"]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cat:<40s} {cnt:>7,}  ({pct:>5.1f}%)")
    lines.append("")

    # Source file breakdown
    for src, grp in df.groupby("source_file"):
        lines.append(f"  {src}: {len(grp):,} records")
    lines.append("")

    # --- 2. MHC CLASS BREAKDOWN ---
    lines.append("2. MHC CLASS BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    mhc_class_counts = df["mhc_class"].value_counts()
    for cls, cnt in mhc_class_counts.items():
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cls:<12s} {cnt:>10,}  ({pct:.1f}%)")
    lines.append("")

    lines.append(
        "Note: MHC class is determined by the experiment type (peptide-CI = MHCI,\n"
        "peptide-CII = MHCII, minigene = unknown)."
    )
    lines.append("")

    # Subject HLA allele summary
    if not mhc_alleles.empty and "subject_count" in mhc_alleles.columns:
        lines.append("Subject HLA allele distribution (from subject-metadata.csv):")
        top_alleles = mhc_alleles.nlargest(10, "subject_count")
        for _, row in top_alleles.iterrows():
            lines.append(
                f"  {row['mhc']:<25s} {row['mhc_class']:<8s} "
                f"{int(row['subject_count']):>3} subjects"
            )
        lines.append("")
        lines.append(
            "Note: MHC allele stats reflect the subject population, not direct\n"
            "TCR-MHC associations."
        )
    lines.append("")

    # --- 3. CATEGORY BREAKDOWN ---
    lines.append("3. CATEGORY BREAKDOWN (by ORF)")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'ORF Coverage':<40s} {'Records':>9s} {'%':>7s} {'Epitopes':>10s}"
    )
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['category'])[:38]:<40s} {cnt:>9,} {pct:>6.1f}% "
            f"{int(row['unique_epitopes']):>10,}"
        )
    lines.append("")

    # --- 4. TOP EPITOPES ---
    lines.append("4. TOP EPITOPES")
    lines.append("-" * 80)
    lines.append("")

    if not epi.empty:
        lines.append("By total records (TCR-epitope associations):")
        lines.append("")
        lines.append(
            f"  {'Epitope':<22s} {'ORF':<28s} {'Class':<8s} "
            f"{'Records':>8s} {'Beta CDR3s':>12s}"
        )

        top_epi = epi.nlargest(10, "total_records")
        for _, row in top_epi.iterrows():
            epi_name = str(row["epitope"])[:20]
            cat = str(row.get("category", ""))[:26]
            cls = str(row.get("mhc_class", ""))
            lines.append(
                f"  {epi_name:<22s} {cat:<28s} {cls:<8s} "
                f"{int(row['total_records']):>8,} {int(row['unique_beta_cdr3']):>12,}"
            )
        lines.append("")

        lines.append("By unique beta CDR3s:")
        lines.append("")
        top_beta = epi.nlargest(10, "unique_beta_cdr3")
        lines.append(
            f"  {'Epitope':<22s} {'ORF':<28s} {'Beta CDR3s':>12s}"
        )
        for _, row in top_beta.iterrows():
            epi_name = str(row["epitope"])[:20]
            cat = str(row.get("category", ""))[:26]
            lines.append(
                f"  {epi_name:<22s} {cat:<28s} "
                f"{int(row['unique_beta_cdr3']):>12,}"
            )
    else:
        lines.append("No epitope data available.")
    lines.append("")

    # --- 5. TOP MHC ALLELES ---
    lines.append("5. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")
    if not mhc_alleles.empty and "subject_count" in mhc_alleles.columns:
        lines.append(
            f"  {'Allele':<25s} {'Class':<10s} {'Subjects':>10s}"
        )
        top_mhc = mhc_alleles.nlargest(10, "subject_count")
        for _, row in top_mhc.iterrows():
            allele = str(row["mhc"])[:23]
            lines.append(
                f"  {allele:<25s} {row['mhc_class']:<10s} "
                f"{int(row['subject_count']):>10,}"
            )
        lines.append("")
        lines.append(
            "Note: MHC allele data is from subject HLA typing, not per-TCR annotation.\n"
            "Allele counts reflect the number of subjects carrying each allele."
        )
    else:
        lines.append(
            "Not available in this database. immuneCODE MHC data is per-subject,\n"
            "not per-TCR record."
        )
    lines.append("")

    # --- 6. PATHOLOGY & SPECIES ---
    lines.append("6. PATHOLOGY & SPECIES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "All immuneCODE MIRA data is from human subjects targeting SARS-CoV-2 antigens."
    )
    lines.append("Host species: Human")
    lines.append("Antigen species: SARS-CoV-2")
    lines.append("")

    # --- 7. STUDY LANDSCAPE ---
    lines.append("7. STUDY LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")

    total_experiments = len(studies)
    lines.append(f"Total experiments: {total_experiments}")
    lines.append("")

    lines.append("Largest experiments:")
    top_studies = studies.nlargest(5, "total_records")
    for _, row in top_studies.iterrows():
        exp = str(row["experiment"])
        lines.append(
            f"  {exp:<25s} {int(row['total_records']):>7,} records  "
            f"({int(row['unique_beta_cdr3']):>5,} unique CDR3b, "
            f"{int(row['unique_epitopes']):>3,} epitopes)"
        )
    lines.append("")

    top5_records = int(top_studies["total_records"].sum())
    top5_pct = 100.0 * top5_records / total if total else 0
    lines.append(
        f"The top 5 experiments account for ~{top5_pct:.0f}% of all records."
    )
    lines.append("")

    # --- 8. DATA COMPLETENESS ---
    lines.append("8. DATA COMPLETENESS")
    lines.append("-" * 80)
    lines.append("")

    criteria = {
        "has_epitope_and_mhc": "Has epitope + subjects are HLA-typed.",
        "has_epitope_only": "Has epitope but no subject HLA typing.",
        "has_mhc_only": "Has subject HLA typing but no epitope.",
        "no_epitope_no_mhc": "Missing both epitope and HLA typing.",
    }

    lines.append(
        f"  {'Category':<25s} {'Records':>9s} {'%':>7s} "
        f"{'Beta CDR3s':>12s} {'Epitopes':>10s}"
    )
    for _, row in completeness.iterrows():
        cat = row["category"]
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {cat:<25s} {cnt:>9,} {pct:>6.1f}% "
            f"{int(row['unique_beta_cdr3']):>12,} {int(row['unique_epitopes']):>10,}"
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
        f"a) Usable annotated data: {total:,} MIRA records with TCR-antigen mappings.\n"
        f"   {int(r['unique_beta_cdr3']):,} unique beta CDR3s across "
        f"{int(r['unique_epitopes']):,} epitopes.\n"
        f"   All data has peptide annotation; MHC is available at subject level."
    )
    lines.append("")

    lines.append(
        f"b) Paired TCR availability: No paired alpha+beta data in MIRA release.\n"
        f"   All records are beta-chain only. Review-002 raw data is also\n"
        f"   TCRB-only repertoire data."
    )
    lines.append("")

    lines.append(
        f"c) Antigen focus: All data targets SARS-CoV-2 antigens. Models trained\n"
        f"   on this data will be COVID-19 specific and may not generalize\n"
        f"   to other pathogens."
    )
    lines.append("")

    lines.append(
        f"d) HLA bias: Subject HLA typing is available but not directly linked\n"
        f"   to individual TCR-peptide records. Population-level HLA\n"
        f"   distribution can inform MHC-aware models."
    )
    lines.append("")

    lines.append(
        f"e) Epitope imbalance: Class I peptide data ({len(df[df['mhc_class'] == 'MHCI']):,} records)\n"
        f"   far exceeds Class II ({len(df[df['mhc_class'] == 'MHCII']):,} records).\n"
        f"   Multiple peptides per record (comma-separated) add complexity."
    )
    lines.append("")

    lines.append(
        f"f) Species coverage: All human, all SARS-CoV-2. No cross-species data."
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
    mira_dir = os.path.join(project_root, MIRA_DIR)
    review_dir = os.path.join(project_root, REVIEW_DIR)
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading immuneCODE MIRA data from {mira_dir} ...")
    df = load_data(mira_dir)
    print(f"  Loaded {len(df):,} records")
    print(f"  MHC class distribution: {df['mhc_class'].value_counts().to_dict()}")
    print(f"  Source file distribution: {df['source_file'].value_counts().to_dict()}")
    print()

    # Build MHC allele summary from subject metadata
    print("Building MHC allele summary from subject metadata ...")
    mhc_df = mhc_allele_summary_from_metadata(mira_dir)

    exports = [
        ("overview.tsv", lambda d: overview(d)),
        ("category_overview.tsv", lambda d: category_overview(d)),
        ("epitope_summary.tsv", lambda d: epitope_summary(d)),
        ("pmhc_summary.tsv", lambda d: pmhc_summary(d)),
        ("study_summary.tsv", lambda d: study_summary(d)),
        ("mhc_allele_summary.tsv", lambda d: mhc_df),
        ("species_breakdown.tsv", lambda d: species_breakdown(d)),
        ("data_completeness.tsv", lambda d: data_completeness(d)),
    ]

    for filename, func in exports:
        print(f"Computing {filename} ...")
        result = func(df)
        path = os.path.join(output_dir, filename)
        result.to_csv(path, sep="\t", index=False)
        print(f"  -> {path}  ({len(result)} rows, {result.columns.size} cols)")

    print("Writing SUMMARY.txt ...")
    write_summary_txt(df, output_dir, mira_dir, review_dir)

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

    epi = pd.read_csv(os.path.join(output_dir, "epitope_summary.tsv"), sep="\t")
    if not epi.empty:
        top = epi.nlargest(3, "total_records")
        print("Top 3 epitopes:")
        for _, row in top.iterrows():
            print(f"  {row['epitope']}: {row['total_records']} records")

    print("\nDone.")


if __name__ == "__main__":
    main()
