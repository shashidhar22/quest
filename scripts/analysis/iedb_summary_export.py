#!/usr/bin/env python3
"""Export comprehensive IEDB TCR-specificity analysis tables.

Reads the existing IEDB SQLite database (built by iedb_sqlite_summary.py)
and produces SUMMARY.txt + TSV files in the same format as the VDJdb and
McPAS summary exports, enabling cross-database comparison.

Usage:
    python scripts/analysis/iedb_summary_export.py
"""

import os
import re
import sqlite3

import pandas as pd

DB_PATH = "data/analysis/iedb/iedb.sqlite"
OUTPUT_DIR = "data/analysis/iedb"

# SQL fragments for coalescing curated/calculated CDR3 values
CDR3A = "COALESCE(NULLIF(chain_1_cdr3_curated, ''), NULLIF(chain_1_cdr3_calculated, ''))"
CDR3B = "COALESCE(NULLIF(chain_2_cdr3_curated, ''), NULLIF(chain_2_cdr3_calculated, ''))"
VGA = "COALESCE(NULLIF(chain_1_curated_v_gene, ''), NULLIF(chain_1_calculated_v_gene, ''))"
VGB = "COALESCE(NULLIF(chain_2_curated_v_gene, ''), NULLIF(chain_2_calculated_v_gene, ''))"
JGA = "COALESCE(NULLIF(chain_1_curated_j_gene, ''), NULLIF(chain_1_calculated_j_gene, ''))"
JGB = "COALESCE(NULLIF(chain_2_curated_j_gene, ''), NULLIF(chain_2_calculated_j_gene, ''))"

# Map NCBITaxon IRIs to species names
NCBI_TAXON_MAP = {
    "http://purl.obolibrary.org/obo/NCBITaxon_9606": "Human",
    "http://purl.obolibrary.org/obo/NCBITaxon_10090": "Mouse",
}


def classify_mhc(allele):
    """Heuristic MHC class assignment from allele name."""
    a = allele.upper()
    # Literal class annotations from IEDB
    if a == "HLA CLASS I":
        return "Class I"
    if a == "HLA CLASS II":
        return "Class II"
    # Class I: HLA-A, HLA-B, HLA-C, HLA-E, HLA-F, HLA-G
    if re.match(r"HLA-[ABCEFG]", a):
        return "Class I"
    # Mouse class I: H-2K, H-2D, H-2L (with or without hyphen after H)
    if re.match(r"H-?2-?[KDL]", a):
        return "Class I"
    # Class II: HLA-D* (DR, DP, DQ, etc.)
    if re.match(r"HLA-D", a):
        return "Class II"
    # Mouse class II: H-2I
    if re.match(r"H-?2-?I", a):
        return "Class II"
    return "Unknown"


def map_species(iri):
    """Map organism IRI to species name."""
    return NCBI_TAXON_MAP.get(iri, "Other")


# ---------------------------------------------------------------------------
# Helper: scalar query
# ---------------------------------------------------------------------------
def _scalar(conn, sql):
    return conn.execute(sql).fetchone()[0]


# ---------------------------------------------------------------------------
# 1. Overview (single-row dataset summary)
# ---------------------------------------------------------------------------
def overview(conn):
    total = _scalar(conn, "SELECT COUNT(*) FROM receptor")
    unique_beta = _scalar(conn, f"SELECT COUNT(DISTINCT {CDR3B}) FROM receptor WHERE {CDR3B} IS NOT NULL")
    unique_alpha = _scalar(conn, f"SELECT COUNT(DISTINCT {CDR3A}) FROM receptor WHERE {CDR3A} IS NOT NULL")
    paired_records = _scalar(conn, f"""
        SELECT COUNT(*) FROM receptor
        WHERE {CDR3A} IS NOT NULL AND {CDR3B} IS NOT NULL
    """)
    unique_paired = _scalar(conn, f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT {CDR3A}, {CDR3B} FROM receptor
            WHERE {CDR3A} IS NOT NULL AND {CDR3B} IS NOT NULL
        )
    """)
    unique_epitopes = _scalar(conn, "SELECT COUNT(DISTINCT epitope_name) FROM receptor WHERE epitope_name != ''")
    unique_mhc = _scalar(conn, "SELECT COUNT(DISTINCT mhc_allele) FROM receptor_mhc_alleles")

    unique_pmhc = _scalar(conn, """
        SELECT COUNT(*) FROM (
            SELECT DISTINCT r.epitope_name, rma.mhc_allele
            FROM receptor r
            JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
            WHERE r.epitope_name != ''
        )
    """)

    unique_tcr_pmhc_paired = _scalar(conn, f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT {CDR3A}, {CDR3B}, r.epitope_name, rma.mhc_allele
            FROM receptor r
            JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
            WHERE {CDR3A} IS NOT NULL AND {CDR3B} IS NOT NULL
              AND r.epitope_name != ''
        )
    """)

    # Tcell assay counts
    tcell_total = _scalar(conn, "SELECT COUNT(*) FROM tcell")
    tcell_positive = _scalar(conn, """
        SELECT COUNT(*) FROM tcell
        WHERE assay_qualitative_measurement IN ('Positive', 'Positive-Low', 'Positive-High', 'Positive-Intermediate')
    """)
    tcell_unique_pmids = _scalar(conn, "SELECT COUNT(DISTINCT reference_iedb_iri) FROM tcell WHERE reference_iedb_iri != ''")

    rows = [{
        "total_records": total,
        "unique_beta_cdr3": unique_beta,
        "unique_alpha_cdr3": unique_alpha,
        "paired_records": paired_records,
        "unique_paired_tcrs": unique_paired,
        "unique_epitopes": unique_epitopes,
        "unique_mhc_alleles": unique_mhc,
        "unique_pmhc": unique_pmhc,
        "unique_tcr_pmhc_paired": unique_tcr_pmhc_paired,
        "tcell_total_assays": tcell_total,
        "tcell_positive_assays": tcell_positive,
        "tcell_unique_pmids": tcell_unique_pmids,
    }]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Epitope summary
# ---------------------------------------------------------------------------
def epitope_summary(conn):
    # Get epitope stats from receptor table
    df = pd.read_sql_query(f"""
        SELECT
            r.epitope_name AS epitope,
            r.epitope_source_organism AS antigen_source_organism,
            {CDR3A} AS cdr3a,
            {CDR3B} AS cdr3b
        FROM receptor r
        WHERE r.epitope_name != ''
    """, conn)

    df["is_paired"] = df["cdr3a"].notna() & df["cdr3b"].notna()
    df["paired_tcr"] = df.apply(
        lambda row: f"{row['cdr3a']}|{row['cdr3b']}" if row["is_paired"] else None, axis=1
    )

    # Get primary MHC per epitope from junction table
    mhc_df = pd.read_sql_query("""
        SELECT r.epitope_name AS epitope, rma.mhc_allele
        FROM receptor r
        JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
        WHERE r.epitope_name != ''
    """, conn)
    primary_mhc_map = mhc_df.groupby("epitope")["mhc_allele"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else ""
    ).to_dict()

    # Get tcell positive assay counts per epitope
    tcell_pos = pd.read_sql_query("""
        SELECT epitope_name AS epitope, COUNT(*) AS tcell_positive_assays
        FROM tcell
        WHERE epitope_name != ''
          AND assay_qualitative_measurement IN ('Positive', 'Positive-Low', 'Positive-High', 'Positive-Intermediate')
        GROUP BY epitope_name
    """, conn)
    tcell_pos_map = dict(zip(tcell_pos["epitope"], tcell_pos["tcell_positive_assays"]))

    rows = []
    for epitope, grp in df.groupby("epitope", sort=True):
        organism = grp.loc[grp["antigen_source_organism"].notna() & (grp["antigen_source_organism"] != ""),
                           "antigen_source_organism"]
        organism = organism.mode().iloc[0] if not organism.mode().empty else ""

        primary_mhc = primary_mhc_map.get(epitope, "")
        mhc_class = classify_mhc(primary_mhc) if primary_mhc else "Unknown"

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["cdr3b"].notna() & grp["cdr3a"].isna()]

        rows.append({
            "epitope": epitope,
            "antigen_source_organism": organism,
            "primary_mhc": primary_mhc,
            "mhc_class": mhc_class,
            "total_records": len(grp),
            "paired_records": len(paired),
            "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            "unique_beta_only_tcrs": beta_only["cdr3b"].nunique() if len(beta_only) else 0,
            "tcell_positive_assays": tcell_pos_map.get(epitope, 0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. pMHC summary
# ---------------------------------------------------------------------------
def pmhc_summary(conn):
    df = pd.read_sql_query(f"""
        SELECT
            r.epitope_name AS epitope,
            rma.mhc_allele,
            r.epitope_source_organism AS antigen_source_organism,
            {CDR3A} AS cdr3a,
            {CDR3B} AS cdr3b
        FROM receptor r
        JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
        WHERE r.epitope_name != ''
    """, conn)

    df["is_paired"] = df["cdr3a"].notna() & df["cdr3b"].notna()
    df["paired_tcr"] = df.apply(
        lambda row: f"{row['cdr3a']}|{row['cdr3b']}" if row["is_paired"] else None, axis=1
    )

    rows = []
    for (epitope, mhc_allele), grp in df.groupby(["epitope", "mhc_allele"], sort=True):
        mhc_class = classify_mhc(mhc_allele)

        organism = grp.loc[grp["antigen_source_organism"].notna() & (grp["antigen_source_organism"] != ""),
                           "antigen_source_organism"]
        organism = organism.mode().iloc[0] if not organism.mode().empty else ""

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["cdr3b"].notna() & grp["cdr3a"].isna()]

        rows.append({
            "epitope": epitope,
            "mhc_allele": mhc_allele,
            "mhc_class": mhc_class,
            "antigen_source_organism": organism,
            "total_records": len(grp),
            "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            "unique_beta_only_tcrs": beta_only["cdr3b"].nunique() if len(beta_only) else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. MHC allele summary
# ---------------------------------------------------------------------------
def mhc_allele_summary(conn):
    df = pd.read_sql_query(f"""
        SELECT
            rma.mhc_allele,
            r.epitope_name,
            {CDR3A} AS cdr3a,
            {CDR3B} AS cdr3b
        FROM receptor r
        JOIN receptor_mhc_alleles rma ON r.rowid = rma.receptor_rowid
    """, conn)

    df["is_paired"] = df["cdr3a"].notna() & df["cdr3b"].notna()
    df["paired_tcr"] = df.apply(
        lambda row: f"{row['cdr3a']}|{row['cdr3b']}" if row["is_paired"] else None, axis=1
    )

    rows = []
    for mhc_allele, grp in df.groupby("mhc_allele", sort=True):
        mhc_class = classify_mhc(mhc_allele)
        paired = grp[grp["is_paired"]]
        epi = grp.loc[grp["epitope_name"].notna() & (grp["epitope_name"] != ""), "epitope_name"]

        rows.append({
            "mhc_allele": mhc_allele,
            "mhc_class": mhc_class,
            "total_records": len(grp),
            "unique_epitopes": epi.nunique(),
            "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Species breakdown
# ---------------------------------------------------------------------------
def species_breakdown(conn):
    rows = []

    # Host species from organism IRIs (prefer chain_2, fall back to chain_1)
    host_df = pd.read_sql_query(f"""
        SELECT
            COALESCE(NULLIF(chain_2_organism_iri, ''), NULLIF(chain_1_organism_iri, '')) AS organism_iri,
            epitope_name,
            {CDR3A} AS cdr3a,
            {CDR3B} AS cdr3b
        FROM receptor
    """, conn)

    host_df["species"] = host_df["organism_iri"].apply(
        lambda x: map_species(x) if pd.notna(x) else "Unknown"
    )
    host_df["is_paired"] = host_df["cdr3a"].notna() & host_df["cdr3b"].notna()
    host_df["paired_tcr"] = host_df.apply(
        lambda row: f"{row['cdr3a']}|{row['cdr3b']}" if row["is_paired"] else None, axis=1
    )

    for species, grp in host_df.groupby("species", sort=True):
        paired = grp[grp["is_paired"]]
        epi = grp.loc[grp["epitope_name"].notna() & (grp["epitope_name"] != ""), "epitope_name"]
        rows.append({
            "category": "host_species",
            "species": species,
            "total_records": len(grp),
            "unique_epitopes": epi.nunique(),
            "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
        })

    # Antigen species from epitope_source_organism
    antigen_df = pd.read_sql_query(f"""
        SELECT
            epitope_source_organism AS species,
            epitope_name,
            {CDR3A} AS cdr3a,
            {CDR3B} AS cdr3b
        FROM receptor
        WHERE epitope_source_organism != ''
    """, conn)

    antigen_df["is_paired"] = antigen_df["cdr3a"].notna() & antigen_df["cdr3b"].notna()
    antigen_df["paired_tcr"] = antigen_df.apply(
        lambda row: f"{row['cdr3a']}|{row['cdr3b']}" if row["is_paired"] else None, axis=1
    )

    for species, grp in antigen_df.groupby("species", sort=True):
        paired = grp[grp["is_paired"]]
        epi = grp.loc[grp["epitope_name"].notna() & (grp["epitope_name"] != ""), "epitope_name"]
        rows.append({
            "category": "antigen_species",
            "species": species,
            "total_records": len(grp),
            "unique_epitopes": epi.nunique(),
            "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Study summary
# ---------------------------------------------------------------------------
def study_summary(conn):
    df = pd.read_sql_query(f"""
        SELECT
            reference_iedb_iri AS reference,
            epitope_name,
            {CDR3A} AS cdr3a,
            {CDR3B} AS cdr3b
        FROM receptor
        WHERE reference_iedb_iri != ''
    """, conn)

    df["is_paired"] = df["cdr3a"].notna() & df["cdr3b"].notna()
    df["paired_tcr"] = df.apply(
        lambda row: f"{row['cdr3a']}|{row['cdr3b']}" if row["is_paired"] else None, axis=1
    )

    rows = []
    for reference, grp in df.groupby("reference", sort=True):
        paired = grp[grp["is_paired"]]
        epi = grp.loc[grp["epitope_name"].notna() & (grp["epitope_name"] != ""), "epitope_name"]

        rows.append({
            "reference": reference,
            "total_records": len(grp),
            "paired_records": len(paired),
            "unique_epitopes": epi.nunique(),
            "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Data completeness
# ---------------------------------------------------------------------------
def data_completeness(conn):
    df = pd.read_sql_query(f"""
        SELECT
            epitope_name,
            {CDR3A} AS cdr3a,
            {CDR3B} AS cdr3b
        FROM receptor
    """, conn)

    # Check which rows have MHC allele via junction table
    mhc_rowids = pd.read_sql_query(
        "SELECT DISTINCT receptor_rowid FROM receptor_mhc_alleles",
        conn,
    )["receptor_rowid"].values

    # We need rowid to join; re-query with rowid
    df_full = pd.read_sql_query(f"""
        SELECT
            rowid,
            epitope_name,
            {CDR3A} AS cdr3a,
            {CDR3B} AS cdr3b
        FROM receptor
    """, conn)

    df_full["has_mhc"] = df_full["rowid"].isin(mhc_rowids)
    df_full["has_epitope"] = df_full["epitope_name"].notna() & (df_full["epitope_name"] != "")
    df_full["has_alpha"] = df_full["cdr3a"].notna()
    df_full["has_beta"] = df_full["cdr3b"].notna()
    df_full["is_paired"] = df_full["has_alpha"] & df_full["has_beta"]
    df_full["paired_tcr"] = df_full.apply(
        lambda row: f"{row['cdr3a']}|{row['cdr3b']}" if row["is_paired"] else None, axis=1
    )

    rows = []

    # Chain pairing dimension
    chain_categories = {
        "paired_both_chains": df_full["has_alpha"] & df_full["has_beta"],
        "beta_only": ~df_full["has_alpha"] & df_full["has_beta"],
        "alpha_only": df_full["has_alpha"] & ~df_full["has_beta"],
        "no_cdr3": ~df_full["has_alpha"] & ~df_full["has_beta"],
    }
    for cat, mask in chain_categories.items():
        sub = df_full[mask]
        rows.append({
            "dimension": "chain_pairing",
            "category": cat,
            "record_count": len(sub),
            "unique_paired_tcrs": sub["paired_tcr"].dropna().nunique(),
            "unique_beta_cdr3": sub.loc[sub["has_beta"], "cdr3b"].nunique() if sub["has_beta"].any() else 0,
            "unique_epitopes": sub.loc[sub["has_epitope"], "epitope_name"].nunique() if sub["has_epitope"].any() else 0,
        })

    # Annotation dimension
    ann_categories = {
        "has_epitope_and_mhc": df_full["has_epitope"] & df_full["has_mhc"],
        "has_epitope_only": df_full["has_epitope"] & ~df_full["has_mhc"],
        "has_mhc_only": ~df_full["has_epitope"] & df_full["has_mhc"],
        "no_epitope_no_mhc": ~df_full["has_epitope"] & ~df_full["has_mhc"],
    }
    for cat, mask in ann_categories.items():
        sub = df_full[mask]
        rows.append({
            "dimension": "annotation",
            "category": cat,
            "record_count": len(sub),
            "unique_paired_tcrs": sub["paired_tcr"].dropna().nunique(),
            "unique_beta_cdr3": sub.loc[sub["has_beta"], "cdr3b"].nunique() if sub["has_beta"].any() else 0,
            "unique_epitopes": sub.loc[sub["has_epitope"], "epitope_name"].nunique() if sub["has_epitope"].any() else 0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SUMMARY.txt generation
# ---------------------------------------------------------------------------
def write_summary_txt(conn, output_dir):
    """Write a human-readable SUMMARY.txt after all TSVs are exported."""
    from datetime import date

    ov = pd.read_csv(os.path.join(output_dir, "overview.tsv"), sep="\t")
    epi = pd.read_csv(os.path.join(output_dir, "epitope_summary.tsv"), sep="\t")
    mhc_alleles = pd.read_csv(os.path.join(output_dir, "mhc_allele_summary.tsv"), sep="\t")
    species = pd.read_csv(os.path.join(output_dir, "species_breakdown.tsv"), sep="\t")
    studies = pd.read_csv(os.path.join(output_dir, "study_summary.tsv"), sep="\t")
    completeness = pd.read_csv(os.path.join(output_dir, "data_completeness.tsv"), sep="\t")

    r = ov.iloc[0]
    total = int(r["total_records"])
    lines = []

    # --- Header ---
    lines.append("=" * 80)
    lines.append("                    IEDB TCR-Specificity Analysis Summary")
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {DB_PATH}")
    lines.append("=" * 80)
    lines.append("")

    # --- 1. DATASET OVERVIEW ---
    paired = int(r["paired_records"])
    paired_pct = 100.0 * paired / total if total else 0

    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Total receptor records:  {total:>10,}")
    lines.append(f"Unique beta CDR3s:      {int(r['unique_beta_cdr3']):>10,}")
    lines.append(f"Unique alpha CDR3s:     {int(r['unique_alpha_cdr3']):>10,}")
    lines.append(
        f"Paired records (a+b):   {paired:>10,}  ({paired_pct:.1f}% of total)"
    )
    lines.append(f"Unique paired TCRs:     {int(r['unique_paired_tcrs']):>10,}")
    lines.append(f"Unique epitopes:        {int(r['unique_epitopes']):>10,}")
    lines.append(f"Unique MHC alleles:     {int(r['unique_mhc_alleles']):>10,}")
    lines.append(f"Unique pMHC combos:     {int(r['unique_pmhc']):>10,}")
    lines.append(f"Unique TCR-pMHC pairs:  {int(r['unique_tcr_pmhc_paired']):>10,}")
    lines.append("")

    # Receptor type breakdown
    type_counts = conn.execute(
        "SELECT receptor_type, COUNT(*) FROM receptor GROUP BY receptor_type ORDER BY COUNT(*) DESC"
    ).fetchall()
    lines.append("Receptor type breakdown:")
    for rtype, cnt in type_counts:
        rtype_label = rtype if rtype else "(empty)"
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {rtype_label:<20s} {cnt:>9,}  ({pct:>5.1f}%)")
    lines.append("")

    # Tcell assay counts
    lines.append(f"T-cell assays (tcell table): {int(r['tcell_total_assays']):,} total")
    lines.append(f"  Positive assays: {int(r['tcell_positive_assays']):,}")
    lines.append(f"  Unique references: {int(r['tcell_unique_pmids']):,}")
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

    class1_row = mhc_class_agg[mhc_class_agg["mhc_class"] == "Class I"]
    if not class1_row.empty and mhc_total_records > 0:
        class1_pct = 100.0 * int(class1_row["records"].iloc[0]) / mhc_total_records
        lines.append(
            f"Among records with MHC annotation, Class I accounts for "
            f"{class1_pct:.1f}% of records."
        )
    lines.append(
        'Note: "HLA class I" generic entries (no specific allele) account for '
        "a significant share of Class I records."
    )
    lines.append("")

    # --- 3. TOP EPITOPES ---
    lines.append("3. TOP EPITOPES")
    lines.append("-" * 80)
    lines.append("")
    lines.append("By total records:")
    lines.append("")
    lines.append(
        f"  {'Epitope':<22s} {'Antigen Source':<30s} {'MHC':<18s} "
        f"{'Records':>8s} {'Paired TCRs':>13s}"
    )

    top_epi = epi.nlargest(10, "total_records")
    for _, row in top_epi.iterrows():
        epi_name = str(row["epitope"])[:20]
        organism = str(row["antigen_source_organism"])[:28]
        mhc = str(row["primary_mhc"])[:16]
        lines.append(
            f"  {epi_name:<22s} {organism:<30s} {mhc:<18s} "
            f"{int(row['total_records']):>8,} {int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

    lines.append("By unique paired TCRs:")
    lines.append("")
    lines.append(
        f"  {'Epitope':<22s} {'Antigen Source':<30s} {'Paired TCRs':>13s}"
    )
    top_paired = epi.nlargest(10, "unique_paired_tcrs")
    for _, row in top_paired.iterrows():
        epi_name = str(row["epitope"])[:20]
        organism = str(row["antigen_source_organism"])[:28]
        lines.append(
            f"  {epi_name:<22s} {organism:<30s} "
            f"{int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

    # GILGFVFTL benchmark note
    gilg = epi[epi["epitope"] == "GILGFVFTL"]
    if not gilg.empty:
        g = gilg.iloc[0]
        lines.append(
            f"Note: GILGFVFTL (influenza M1 / HLA-A*02:01) has "
            f"{int(g['total_records']):,} records and "
            f"{int(g['unique_paired_tcrs']):,} unique paired TCRs -- "
            f"a strong benchmark target."
        )
        lines.append("")

    # --- 4. TOP MHC ALLELES ---
    lines.append("4. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Allele':<30s} {'Class':<12s} {'Records':>9s} "
        f"{'Epitopes':>10s} {'Paired TCRs':>13s}"
    )

    top_mhc = mhc_alleles.nlargest(10, "total_records")
    for _, row in top_mhc.iterrows():
        allele = str(row["mhc_allele"])[:28]
        lines.append(
            f"  {allele:<30s} {row['mhc_class']:<12s} "
            f"{int(row['total_records']):>9,} "
            f"{int(row['unique_epitopes']):>10,} "
            f"{int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

    if len(top_mhc) > 0:
        top_allele = top_mhc.iloc[0]
        top_pct = (
            100.0 * int(top_allele["total_records"]) / mhc_total_records
            if mhc_total_records > 0
            else 0
        )
        lines.append(
            f"{top_allele['mhc_allele']} is the most represented allele with "
            f"{top_pct:.1f}% of MHC-annotated records."
        )
        lines.append(
            'Note: "HLA class I" is a generic annotation (no specific allele) '
            "and appears prominently in the data."
        )
    lines.append("")

    # --- 5. ANTIGEN SPECIES ---
    lines.append("5. ANTIGEN SPECIES")
    lines.append("-" * 80)
    lines.append("")

    antigen = species[species["category"] == "antigen_species"].copy()
    top_antigen = antigen.nlargest(10, "unique_epitopes")

    lines.append("Top antigen sources by epitope diversity:")
    lines.append("")
    lines.append(
        f"  {'Antigen Species':<45s} {'Epitopes':>10s} {'Records':>9s} {'Paired TCRs':>13s}"
    )
    for _, row in top_antigen.iterrows():
        sp = str(row["species"])[:43]
        lines.append(
            f"  {sp:<45s} {int(row['unique_epitopes']):>10,} "
            f"{int(row['total_records']):>9,} {int(row['unique_paired_tcrs']):>13,}"
        )
    lines.append("")

    # Host species breakdown
    host = species[species["category"] == "host_species"].copy()
    host = host.sort_values("total_records", ascending=False)
    host_total = int(host["total_records"].sum())

    lines.append("Host species breakdown:")
    for _, row in host.iterrows():
        sp = str(row["species"])
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / host_total if host_total else 0
        lines.append(
            f"  {sp + ':':<18s} {cnt:>9,} records  ({pct:>5.1f}%)  "
            f"{int(row['unique_epitopes']):>5,} epitopes  "
            f"{int(row['unique_paired_tcrs']):>6,} paired TCRs"
        )
    lines.append("")

    # --- 6. STUDY LANDSCAPE ---
    lines.append("6. STUDY LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")

    total_studies = len(studies)
    studies_with_paired = int((studies["paired_records"] > 0).sum())
    studies_with_paired_pct = (
        100.0 * studies_with_paired / total_studies if total_studies else 0
    )
    studies_large = int((studies["total_records"] > 1000).sum())

    lines.append(f"Total references: {total_studies}")
    lines.append(
        f"References with paired TCR data: {studies_with_paired} "
        f"({studies_with_paired_pct:.0f}%)"
    )
    lines.append(f"References with >1,000 records: {studies_large}")
    lines.append("")

    lines.append("Largest contributors:")
    top_studies = studies.nlargest(5, "total_records")
    for _, row in top_studies.iterrows():
        ref = str(row["reference"])
        lines.append(
            f"  {ref:<55s} {int(row['total_records']):>7,} records  "
            f"({int(row['paired_records']):>5,} paired, "
            f"{int(row['unique_epitopes']):>3,} epitopes)"
        )
    lines.append("")

    top5_records = int(top_studies["total_records"].sum())
    top5_pct = 100.0 * top5_records / total if total else 0
    lines.append(
        f"The top 5 references account for ~{top5_pct:.0f}% of all records."
    )
    lines.append("")

    # --- 7. DATA COMPLETENESS ---
    lines.append("7. DATA COMPLETENESS")
    lines.append("-" * 80)
    lines.append("")

    # Chain pairing
    chain_comp = completeness[completeness["dimension"] == "chain_pairing"]
    lines.append("Chain pairing:")
    lines.append(
        f"  {'Category':<25s} {'Records':>9s} {'%':>7s} "
        f"{'Paired TCRs':>13s} {'Beta CDR3':>11s}"
    )
    for _, row in chain_comp.iterrows():
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {row['category']:<25s} {cnt:>9,} {pct:>6.1f}% "
            f"{int(row['unique_paired_tcrs']):>13,} {int(row['unique_beta_cdr3']):>11,}"
        )
    lines.append("")

    # Annotation completeness
    ann_comp = completeness[completeness["dimension"] == "annotation"]
    lines.append("Annotation completeness:")
    lines.append(
        f"  {'Category':<25s} {'Records':>9s} {'%':>7s} "
        f"{'Paired TCRs':>13s} {'Epitopes':>10s}"
    )
    for _, row in ann_comp.iterrows():
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {row['category']:<25s} {cnt:>9,} {pct:>6.1f}% "
            f"{int(row['unique_paired_tcrs']):>13,} {int(row['unique_epitopes']):>10,}"
        )
    lines.append("")

    # V/J gene annotation rates
    vga_cnt = _scalar(conn, f"SELECT COUNT(*) FROM receptor WHERE {VGA} IS NOT NULL")
    vgb_cnt = _scalar(conn, f"SELECT COUNT(*) FROM receptor WHERE {VGB} IS NOT NULL")
    jga_cnt = _scalar(conn, f"SELECT COUNT(*) FROM receptor WHERE {JGA} IS NOT NULL")
    jgb_cnt = _scalar(conn, f"SELECT COUNT(*) FROM receptor WHERE {JGB} IS NOT NULL")

    lines.append("V/J gene annotation rates:")
    lines.append(
        f"  Alpha V gene: {vga_cnt:>9,} / {total:,} ({100.0*vga_cnt/total:.1f}%)"
    )
    lines.append(
        f"  Alpha J gene: {jga_cnt:>9,} / {total:,} ({100.0*jga_cnt/total:.1f}%)"
    )
    lines.append(
        f"  Beta V gene:  {vgb_cnt:>9,} / {total:,} ({100.0*vgb_cnt/total:.1f}%)"
    )
    lines.append(
        f"  Beta J gene:  {jgb_cnt:>9,} / {total:,} ({100.0*jgb_cnt/total:.1f}%)"
    )
    lines.append("")

    # --- 8. TCELL ASSAY PERSPECTIVE ---
    lines.append("8. TCELL ASSAY PERSPECTIVE")
    lines.append("-" * 80)
    lines.append("")

    tcell_total = int(r["tcell_total_assays"])
    tcell_pos = int(r["tcell_positive_assays"])
    tcell_neg = tcell_total - tcell_pos

    lines.append(f"Total T-cell assays: {tcell_total:,}")
    lines.append(
        f"  Positive: {tcell_pos:,} ({100.0*tcell_pos/tcell_total:.1f}%)"
    )
    lines.append(
        f"  Negative: {tcell_neg:,} ({100.0*tcell_neg/tcell_total:.1f}%)"
    )
    lines.append("")

    # Top assay methods
    methods = conn.execute("""
        SELECT assay_method, COUNT(*) AS cnt
        FROM tcell WHERE assay_method != ''
        GROUP BY assay_method ORDER BY cnt DESC LIMIT 5
    """).fetchall()
    lines.append("Top assay methods:")
    for method, cnt in methods:
        lines.append(f"  {method:<30s} {cnt:>9,}")
    lines.append("")

    # Positive-assay epitope count
    tcell_pos_epi = _scalar(conn, """
        SELECT COUNT(DISTINCT epitope_name) FROM tcell
        WHERE epitope_name != ''
          AND assay_qualitative_measurement IN ('Positive', 'Positive-Low', 'Positive-High', 'Positive-Intermediate')
    """)
    lines.append(f"Unique epitopes with positive T-cell assays: {tcell_pos_epi:,}")
    lines.append("")
    lines.append(
        "The tcell table captures a broader landscape of T-cell reactivity than the "
        "receptor table,\nwhich focuses specifically on TCR sequence data. The receptor "
        "table is the primary source\nfor TCR-pMHC modeling."
    )
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

    # a) Usable annotated data
    ann_epi_mhc = ann_comp[ann_comp["category"] == "has_epitope_and_mhc"]
    if not ann_epi_mhc.empty:
        epi_mhc_records = int(ann_epi_mhc.iloc[0]["record_count"])
        epi_mhc_paired = int(ann_epi_mhc.iloc[0]["unique_paired_tcrs"])
        epi_mhc_epitopes = int(ann_epi_mhc.iloc[0]["unique_epitopes"])
    else:
        epi_mhc_records = epi_mhc_paired = epi_mhc_epitopes = 0

    lines.append(
        f"a) Usable annotated data (epitope + MHC): {epi_mhc_records:,} records with "
        f"{epi_mhc_paired:,} unique paired\n"
        f"   TCRs across {epi_mhc_epitopes:,} epitopes. "
        f"This is the high-confidence training set."
    )
    lines.append("")

    # b) Paired TCR availability
    lines.append(
        f"b) Paired TCR availability: {int(r['unique_paired_tcrs']):,} unique paired TCRs "
        f"({paired_pct:.1f}% of records\n"
        f"   have both alpha+beta chains). Compared to ~62% in VDJdb and ~31% in McPAS,\n"
        f"   IEDB is heavily beta-chain focused."
    )
    lines.append("")

    # c) Antigen bias
    top3_antigen = antigen.nlargest(3, "total_records")
    top3_names = ", ".join(str(s)[:30] for s in top3_antigen["species"].values)
    lines.append(
        f"c) Antigen bias: The top antigen sources are {top3_names}.\n"
        f"   SARS-CoV-2, EBV, and CMV-related epitopes dominate the database."
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
        # Find HLA-A*02:01 specifically
        a0201 = mhc_alleles[mhc_alleles["mhc_allele"] == "HLA-A*02:01"]
        if not a0201.empty:
            a0201_pct = 100.0 * int(a0201.iloc[0]["total_records"]) / mhc_total_records
            lines.append(
                f"d) HLA bias: HLA-A*02:01 represents {a0201_pct:.1f}% of "
                f"MHC-annotated records.\n"
                f"   Models will have strong performance on this allele but may underperform\n"
                f"   on rare alleles."
            )
        else:
            lines.append(
                f"d) HLA bias: {top_a['mhc_allele']} represents {top_a_pct:.1f}% of "
                f"MHC-annotated records."
            )
    lines.append("")

    # e) Epitope imbalance
    top5_epi = epi.nlargest(5, "total_records")
    top5_epi_records = int(top5_epi["total_records"].sum())
    epi_with_records = epi[epi["total_records"] > 0]
    all_epi_records = int(epi_with_records["total_records"].sum())
    top5_epi_pct = 100.0 * top5_epi_records / all_epi_records if all_epi_records else 0
    lines.append(
        f"e) Epitope imbalance: The top 5 epitopes account for "
        f"~{top5_epi_pct:.0f}% of epitope-annotated\n"
        f"   records. GILGFVFTL and NLVPMVATV are ideal benchmarks but also\n"
        f"   sources of training bias."
    )
    lines.append("")

    # f) Species coverage
    human_row = host[host["species"] == "Human"]
    mouse_row = host[host["species"] == "Mouse"]
    if not human_row.empty and host_total > 0:
        human_pct = 100.0 * int(human_row.iloc[0]["total_records"]) / host_total
        mouse_pct = (
            100.0 * int(mouse_row.iloc[0]["total_records"]) / host_total
            if not mouse_row.empty
            else 0
        )
        lines.append(
            f"f) Species coverage: ~{human_pct:.0f}% human data, "
            f"~{mouse_pct:.1f}% mouse. Human-centric models\n"
            f"   will have limited cross-species generalization."
        )
    lines.append("")

    # g) V/J gene gap
    lines.append(
        f"g) V/J gene gap: Alpha V genes only {100.0*vga_cnt/total:.0f}% annotated; "
        f"beta V genes {100.0*vgb_cnt/total:.0f}%.\n"
        f"   Alpha chain gene annotation is sparse, limiting alpha-chain modeling."
    )
    lines.append("")

    # --- Footer ---
    lines.append("=" * 80)
    lines.append(f"Output files in {OUTPUT_DIR}/:")

    tsv_files = [
        "overview.tsv",
        "epitope_summary.tsv",
        "pmhc_summary.tsv",
        "mhc_allele_summary.tsv",
        "species_breakdown.tsv",
        "study_summary.tsv",
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
    db_path = os.path.join(project_root, DB_PATH)
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(db_path):
        print(f"ERROR: SQLite DB not found at {db_path}")
        print("Run iedb_sqlite_summary.py first to build the database.")
        return

    conn = sqlite3.connect(db_path)
    total = _scalar(conn, "SELECT COUNT(*) FROM receptor")
    print(f"Connected to {db_path}")
    print(f"  Receptor table: {total:,} rows")
    print()

    exports = [
        ("overview.tsv", overview),
        ("epitope_summary.tsv", epitope_summary),
        ("pmhc_summary.tsv", pmhc_summary),
        ("mhc_allele_summary.tsv", mhc_allele_summary),
        ("species_breakdown.tsv", species_breakdown),
        ("study_summary.tsv", study_summary),
        ("data_completeness.tsv", data_completeness),
    ]

    for filename, func in exports:
        print(f"Computing {filename} ...")
        result = func(conn)
        path = os.path.join(output_dir, filename)
        result.to_csv(path, sep="\t", index=False)
        print(f"  -> {path}  ({len(result)} rows, {result.columns.size} cols)")

    # Write human-readable summary
    print("Writing SUMMARY.txt ...")
    write_summary_txt(conn, output_dir)

    conn.close()

    # --- Verification ---
    print("\n--- Verification ---")

    # Check data_completeness chain_pairing rows sum to total
    comp = pd.read_csv(os.path.join(output_dir, "data_completeness.tsv"), sep="\t")
    chain_sum = int(comp[comp["dimension"] == "chain_pairing"]["record_count"].sum())
    ann_sum = int(comp[comp["dimension"] == "annotation"]["record_count"].sum())
    print(
        f"Chain pairing sum: {chain_sum}, total: {total}, "
        f"match: {chain_sum == total}"
    )
    print(
        f"Annotation sum: {ann_sum}, total: {total}, "
        f"match: {ann_sum == total}"
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
        print("GILGFVFTL not found in IEDB epitope summary")

    # Check SUMMARY.txt has all sections
    summary_path = os.path.join(output_dir, "SUMMARY.txt")
    with open(summary_path) as f:
        summary_text = f.read()
    for section in range(1, 10):
        marker = f"{section}. "
        if marker in summary_text:
            print(f"  Section {section}: present")
        else:
            print(f"  Section {section}: MISSING")

    print("\nDone.")


if __name__ == "__main__":
    main()
