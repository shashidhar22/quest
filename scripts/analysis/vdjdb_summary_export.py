#!/usr/bin/env python3
"""Export comprehensive VDJdb analysis tables."""

import os
import pandas as pd
import numpy as np

INPUT_FILE = "data/databases/vdjdb/vdjdb_full_filtered.txt"
OUTPUT_DIR = "data/analysis/vdjdb"


def load_data(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["vdjdb.score"] = pd.to_numeric(df["vdjdb.score"], errors="coerce").astype(
        "Int64"
    )
    # Derived columns
    df["has_alpha"] = df["cdr3.alpha"].notna() & (df["cdr3.alpha"] != "")
    df["has_beta"] = df["cdr3.beta"].notna() & (df["cdr3.beta"] != "")
    df["is_paired"] = df["has_alpha"] & df["has_beta"]
    # Composite keys
    df["paired_tcr"] = np.where(
        df["is_paired"],
        df["cdr3.alpha"] + "|" + df["cdr3.beta"],
        pd.NA,
    )
    df["pmhc"] = df["antigen.epitope"] + "|" + df["mhc.a"].fillna("")
    df["tcr_pmhc_paired"] = np.where(
        df["is_paired"],
        df["paired_tcr"] + "|" + df["antigen.epitope"],
        pd.NA,
    )
    return df


# ---------------------------------------------------------------------------
# 1. Per-score overview
# ---------------------------------------------------------------------------
def per_score_overview(df):
    rows = []
    for label, mask in score_filters(df):
        sub = df[mask]
        rows.append(
            {
                "score": label,
                "total_records": len(sub),
                "unique_beta_cdr3": sub.loc[sub["has_beta"], "cdr3.beta"].nunique(),
                "unique_alpha_cdr3": sub.loc[sub["has_alpha"], "cdr3.alpha"].nunique(),
                "paired_records": sub["is_paired"].sum(),
                "unique_paired_tcrs": sub["paired_tcr"].dropna().nunique(),
                "unique_epitopes": sub["antigen.epitope"].nunique(),
                "unique_mhc_a": sub["mhc.a"].nunique(),
                "unique_pmhc": sub["pmhc"].nunique(),
                "unique_tcr_pmhc_paired": sub["tcr_pmhc_paired"].dropna().nunique(),
            }
        )
    return pd.DataFrame(rows)


def score_filters(df):
    """Yield (label, boolean mask) for each score bucket."""
    for s in [0, 1, 2, 3]:
        yield str(s), df["vdjdb.score"] == s
    yield ">=1", df["vdjdb.score"] >= 1
    yield "all", pd.Series(True, index=df.index)


# ---------------------------------------------------------------------------
# 2. Epitope summary
# ---------------------------------------------------------------------------
def epitope_summary(df):
    rows = []
    for epitope, grp in df.groupby("antigen.epitope", sort=True):
        primary_mhc = grp["mhc.a"].mode().iloc[0] if not grp["mhc.a"].mode().empty else ""
        mhc_class = grp["mhc.class"].mode().iloc[0] if not grp["mhc.class"].mode().empty else ""

        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["has_beta"] & ~grp["has_alpha"]]

        row = {
            "epitope": epitope,
            "antigen_gene": grp["antigen.gene"].mode().iloc[0] if not grp["antigen.gene"].mode().empty else "",
            "antigen_species": grp["antigen.species"].mode().iloc[0] if not grp["antigen.species"].mode().empty else "",
            "primary_mhc_a": primary_mhc,
            "mhc_class": mhc_class,
            "total_records": len(grp),
            "paired_records": len(paired),
            "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            "unique_beta_only_tcrs": beta_only["cdr3.beta"].nunique() if len(beta_only) else 0,
            "records_score0": (grp["vdjdb.score"] == 0).sum(),
            "records_score1": (grp["vdjdb.score"] == 1).sum(),
            "records_score2": (grp["vdjdb.score"] == 2).sum(),
            "records_score3": (grp["vdjdb.score"] == 3).sum(),
            "paired_tcrs_score0": paired.loc[paired["vdjdb.score"] == 0, "paired_tcr"].dropna().nunique(),
            "paired_tcrs_score_gte1": paired.loc[paired["vdjdb.score"] >= 1, "paired_tcr"].dropna().nunique(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. pMHC summary
# ---------------------------------------------------------------------------
def pmhc_summary(df):
    group_cols = ["antigen.epitope", "mhc.a", "mhc.b", "mhc.class"]
    rows = []
    for keys, grp in df.groupby(group_cols, sort=True):
        epitope, mhc_a, mhc_b, mhc_class = keys
        paired = grp[grp["is_paired"]]
        beta_only = grp[grp["has_beta"] & ~grp["has_alpha"]]

        score_dist = {}
        for s in [0, 1, 2, 3]:
            score_dist[f"records_score{s}"] = (grp["vdjdb.score"] == s).sum()

        row = {
            "epitope": epitope,
            "mhc_a": mhc_a,
            "mhc_b": mhc_b,
            "mhc_class": mhc_class,
            "antigen_gene": grp["antigen.gene"].mode().iloc[0] if not grp["antigen.gene"].mode().empty else "",
            "antigen_species": grp["antigen.species"].mode().iloc[0] if not grp["antigen.species"].mode().empty else "",
            "total_records": len(grp),
            "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            "unique_beta_only_tcrs": beta_only["cdr3.beta"].nunique() if len(beta_only) else 0,
            **score_dist,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Study summary
# ---------------------------------------------------------------------------
def study_summary(df):
    rows = []
    for ref_id, grp in df.groupby("reference.id", sort=True):
        paired = grp[grp["is_paired"]]
        method_id = grp["method.identification"].mode().iloc[0] if not grp["method.identification"].mode().empty else ""
        sc_yes = (grp["method.singlecell"].str.lower() == "yes").sum() if grp["method.singlecell"].notna().any() else 0
        sc_no = len(grp) - sc_yes
        species_list = grp["species"].dropna().unique()

        row = {
            "reference_id": ref_id,
            "total_records": len(grp),
            "paired_records": len(paired),
            "unique_epitopes": grp["antigen.epitope"].nunique(),
            "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            "records_score0": (grp["vdjdb.score"] == 0).sum(),
            "records_score1": (grp["vdjdb.score"] == 1).sum(),
            "records_score2": (grp["vdjdb.score"] == 2).sum(),
            "records_score3": (grp["vdjdb.score"] == 3).sum(),
            "method_identification": method_id,
            "method_singlecell_yes": sc_yes,
            "method_singlecell_no": sc_no,
            "species": ";".join(sorted(species_list)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. MHC allele summary
# ---------------------------------------------------------------------------
def mhc_allele_summary(df):
    rows = []
    for (mhc_a, mhc_class), grp in df.groupby(["mhc.a", "mhc.class"], sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "mhc_a": mhc_a,
                "mhc_class": mhc_class,
                "total_records": len(grp),
                "records_score_gte1": (grp["vdjdb.score"] >= 1).sum(),
                "unique_epitopes": grp["antigen.epitope"].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Species breakdown
# ---------------------------------------------------------------------------
def species_breakdown(df):
    rows = []
    # Host species
    for species, grp in df.groupby("species", sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "category": "host_species",
                "species": species,
                "total_records": len(grp),
                "unique_epitopes": grp["antigen.epitope"].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            }
        )
    # Antigen species
    for species, grp in df.groupby("antigen.species", sort=True):
        paired = grp[grp["is_paired"]]
        rows.append(
            {
                "category": "antigen_species",
                "species": species,
                "total_records": len(grp),
                "unique_epitopes": grp["antigen.epitope"].nunique(),
                "unique_paired_tcrs": paired["paired_tcr"].dropna().nunique(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Score-0 salvageability
# ---------------------------------------------------------------------------
def score0_salvageability(df):
    s0 = df[df["vdjdb.score"] == 0].copy()

    # Criteria for salvageability
    # Likely salvageable: has single-cell method OR has both alpha+beta (paired)
    has_sc = s0["method.singlecell"].str.lower() == "yes"
    is_paired = s0["is_paired"]

    # Possibly salvageable: has a known reference AND has beta CDR3
    has_ref = s0["reference.id"].notna() & (s0["reference.id"] != "")
    has_beta = s0["has_beta"]

    likely = has_sc | is_paired
    possibly = ~likely & has_ref & has_beta
    low_quality = ~likely & ~possibly

    rows = [
        {
            "category": "likely_salvageable",
            "criteria": "single-cell method OR paired alpha+beta",
            "record_count": likely.sum(),
            "unique_paired_tcrs": s0.loc[likely & is_paired, "paired_tcr"]
            .dropna()
            .nunique(),
            "unique_beta_cdr3": s0.loc[likely & has_beta, "cdr3.beta"].nunique(),
            "unique_epitopes": s0.loc[likely, "antigen.epitope"].nunique(),
        },
        {
            "category": "possibly_salvageable",
            "criteria": "has reference AND beta CDR3 but not single-cell/paired",
            "record_count": possibly.sum(),
            "unique_paired_tcrs": s0.loc[possibly & is_paired, "paired_tcr"]
            .dropna()
            .nunique(),
            "unique_beta_cdr3": s0.loc[possibly & has_beta, "cdr3.beta"].nunique(),
            "unique_epitopes": s0.loc[possibly, "antigen.epitope"].nunique(),
        },
        {
            "category": "low_quality",
            "criteria": "no reference or no beta CDR3",
            "record_count": low_quality.sum(),
            "unique_paired_tcrs": s0.loc[low_quality & is_paired, "paired_tcr"]
            .dropna()
            .nunique(),
            "unique_beta_cdr3": s0.loc[low_quality & has_beta, "cdr3.beta"].nunique(),
            "unique_epitopes": s0.loc[low_quality, "antigen.epitope"].nunique(),
        },
    ]
    return pd.DataFrame(rows)


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
    print(f"  Score distribution: {df['vdjdb.score'].value_counts().sort_index().to_dict()}")
    print(f"  Paired records: {df['is_paired'].sum():,}")
    print()

    exports = [
        ("per_score_overview.tsv", per_score_overview),
        ("epitope_summary.tsv", epitope_summary),
        ("pmhc_summary.tsv", pmhc_summary),
        ("study_summary.tsv", study_summary),
        ("mhc_allele_summary.tsv", mhc_allele_summary),
        ("species_breakdown.tsv", species_breakdown),
        ("score0_salvageability.tsv", score0_salvageability),
    ]

    for filename, func in exports:
        print(f"Computing {filename} ...")
        result = func(df)
        path = os.path.join(output_dir, filename)
        result.to_csv(path, sep="\t", index=False)
        print(f"  -> {path}  ({len(result)} rows, {result.columns.size} cols)")

    # Verification spot-checks
    print("\n--- Verification ---")
    epi = pd.read_csv(os.path.join(output_dir, "epitope_summary.tsv"), sep="\t")
    gilg = epi[epi["epitope"] == "GILGFVFTL"]
    if not gilg.empty:
        print(f"GILGFVFTL paired_tcrs_score_gte1: {gilg.iloc[0]['paired_tcrs_score_gte1']}")

    overview = pd.read_csv(os.path.join(output_dir, "per_score_overview.tsv"), sep="\t")
    print(f"Total records (all): {overview[overview['score'] == 'all'].iloc[0]['total_records']}")
    print(f"Score>=1 records: {overview[overview['score'] == '>=1'].iloc[0]['total_records']}")

    # Check sum consistency
    score_rows = overview[overview["score"].isin(["0", "1", "2", "3"])]
    sum_records = score_rows["total_records"].sum()
    all_records = overview[overview["score"] == "all"].iloc[0]["total_records"]
    print(f"Sum of score 0+1+2+3: {sum_records}, all: {all_records}, match: {sum_records == all_records}")

    print("\nDone.")


if __name__ == "__main__":
    main()
