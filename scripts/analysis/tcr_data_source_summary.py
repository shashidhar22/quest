#!/usr/bin/env python3
"""
TCR Raw Data Source Summary

Analyzes raw TCR data to report:
- Data source counts (databases vs public studies)
- Study breakdown by repository (GEO, SRA, Zenodo, other)
- Database statistics (epitope-MHC pairs, TCR-pMHC records)
- Unique TCR chain counts (paired, TRA, TRB)
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def convert_to_native(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Study naming patterns
STUDY_PATTERNS = {
    "GEO": re.compile(r"^GSE\d+|^GSM\d+", re.IGNORECASE),
    "SRA": re.compile(r"^PRJNA\d+|^PRJEB\d+", re.IGNORECASE),
    "Zenodo": re.compile(r"^ZEN\d+", re.IGNORECASE),
}

# Known database names (case-insensitive matching)
DATABASE_NAMES = {
    "VdjDB",
    "IEDB",
    "McPASDB",
    "TCRdb",
    "CEDAR",
    "ireceptor",
    "GLIPHDB",
    "WarrenDB",
    "IMGTHLA",
    "netMHCpan",
}


def classify_study_source(name: str) -> str:
    """Classify a study by its naming pattern."""
    for source_type, pattern in STUDY_PATTERNS.items():
        if pattern.match(name):
            return source_type
    return "Other"


def scan_study_sources(raw_data_path: Path) -> dict:
    """
    Walk directory structure and identify study types by naming pattern.

    Returns dict with:
    - databases: list of database names
    - studies_by_category: dict mapping category -> list of study names
    - studies_by_source: dict mapping source type (GEO, SRA, Zenodo, Other) -> list
    """
    results = {
        "databases": [],
        "studies_by_category": defaultdict(list),
        "studies_by_source": defaultdict(list),
        "total_studies": 0,
    }

    # Scan databases directory
    db_path = raw_data_path / "databases"
    if db_path.exists():
        for db_dir in db_path.iterdir():
            if db_dir.is_dir() and not db_dir.name.startswith("."):
                results["databases"].append(db_dir.name)

    # Scan study directories
    study_categories = [
        "acute_illness",
        "autoimmune_studies",
        "cancer_studies",
        "chronic_illness",
        "healthy_studies",
        "multiple_classifications",
        "unclassified_data",
        "viral_studies",
        "adaptive_bulk",
    ]

    for category in study_categories:
        cat_path = raw_data_path / category
        if not cat_path.exists():
            continue

        # Skip adaptive_bulk in normal loop - handled specially below
        if category == "adaptive_bulk":
            continue

        for study_dir in cat_path.iterdir():
            if study_dir.is_dir() and not study_dir.name.startswith("."):
                study_name = study_dir.name
                results["studies_by_category"][category].append(study_name)

                source_type = classify_study_source(study_name)
                results["studies_by_source"][source_type].append(study_name)
                results["total_studies"] += 1

    # Special handling for adaptive_bulk: count each .tsv file as a sample
    adaptive_path = raw_data_path / "adaptive_bulk" / "bulk_data" / "tcr"
    if adaptive_path.exists():
        for subdir in ["bulk_survey_tra", "bulk_survey_trb"]:
            survey_path = adaptive_path / subdir
            if survey_path.exists():
                for sample_file in survey_path.glob("*.tsv"):
                    sample_name = sample_file.stem
                    results["studies_by_category"]["adaptive_bulk"].append(sample_name)
                    source_type = classify_study_source(sample_name)
                    results["studies_by_source"][source_type].append(sample_name)
                    results["total_studies"] += 1

    return results


def analyze_vdjdb(db_path: Path) -> dict:
    """
    Parse vdjdb.txt (TSV) and extract statistics.

    Returns:
    - unique_epitopes: count of unique epitope sequences
    - unique_mhc_alleles: count of unique MHC alleles
    - unique_epitope_mhc_pairs: count of unique epitope-MHC combinations
    - total_records: total TCR-pMHC records
    - unique_tcr_pmhc: deduplicated TCR-pMHC records
    - paired_tcrs: count of paired TCRs (using complex.id)
    - unique_tra: count of unique TRA CDR3 sequences
    - unique_trb: count of unique TRB CDR3 sequences
    """
    vdjdb_file = db_path / "VdjDB" / "vdjdb.txt"
    if not vdjdb_file.exists():
        return None

    print("  Reading VdjDB...")
    df = pd.read_csv(vdjdb_file, sep="\t", low_memory=False)

    results = {
        "total_records": len(df),
        "unique_epitopes": df["antigen.epitope"].nunique(),
        "unique_mhc_alleles": 0,
        "unique_epitope_mhc_pairs": 0,
        "unique_tcr_pmhc": 0,
        "paired_tcrs": 0,
        "unique_tra": 0,
        "unique_trb": 0,
    }

    # Unique MHC alleles (combine mhc.a and mhc.b)
    mhc_alleles = set()
    if "mhc.a" in df.columns:
        mhc_alleles.update(df["mhc.a"].dropna().unique())
    if "mhc.b" in df.columns:
        mhc_alleles.update(df["mhc.b"].dropna().unique())
    results["unique_mhc_alleles"] = len(mhc_alleles)

    # Unique epitope-MHC pairs
    df["epitope_mhc"] = (
        df["antigen.epitope"].fillna("") + "|" +
        df["mhc.a"].fillna("") + "|" +
        df["mhc.b"].fillna("")
    )
    results["unique_epitope_mhc_pairs"] = df["epitope_mhc"].nunique()

    # Unique TCR-pMHC (cdr3 + epitope + mhc)
    df["tcr_pmhc"] = (
        df["cdr3"].fillna("") + "|" +
        df["antigen.epitope"].fillna("") + "|" +
        df["mhc.a"].fillna("") + "|" +
        df["mhc.b"].fillna("")
    )
    results["unique_tcr_pmhc"] = df["tcr_pmhc"].nunique()

    # Count paired TCRs using complex.id
    if "complex.id" in df.columns:
        # A paired TCR has the same complex.id for both TRA and TRB
        complex_groups = df.groupby("complex.id")
        paired_count = 0
        for _, group in complex_groups:
            genes = group["gene"].unique()
            if "TRA" in genes and "TRB" in genes:
                paired_count += 1
        results["paired_tcrs"] = paired_count

    # Unique TRA and TRB
    if "gene" in df.columns:
        tra_mask = df["gene"] == "TRA"
        trb_mask = df["gene"] == "TRB"
        results["unique_tra"] = df.loc[tra_mask, "cdr3"].nunique()
        results["unique_trb"] = df.loc[trb_mask, "cdr3"].nunique()

    return results


def analyze_iedb(db_path: Path) -> dict:
    """
    Parse IEDB receptor results table and extract statistics.
    """
    iedb_dir = db_path / "IEDB"
    if not iedb_dir.exists():
        return None

    # Find the receptor results file
    receptor_files = list(iedb_dir.glob("iedb_receptor_results_table*.tsv"))
    if not receptor_files:
        return None

    print("  Reading IEDB...")
    df = pd.read_csv(receptor_files[0], sep="\t", low_memory=False)

    results = {
        "total_records": len(df),
        "unique_epitopes": 0,
        "unique_mhc_alleles": 0,
        "unique_epitope_mhc_pairs": 0,
        "unique_tcr_pmhc": 0,
        "paired_tcrs": 0,
        "unique_tra": 0,
        "unique_trb": 0,
    }

    # Column names with quotes need special handling
    epitope_col = "Epitope - Name"
    mhc_col = "Assay - MHC Allele Names"
    chain1_type = "Chain 1 - Type"
    chain1_cdr3 = "Chain 1 - CDR3 Calculated"
    chain2_type = "Chain 2 - Type"
    chain2_cdr3 = "Chain 2 - CDR3 Calculated"

    if epitope_col in df.columns:
        results["unique_epitopes"] = df[epitope_col].nunique()

    if mhc_col in df.columns:
        # MHC alleles may be comma-separated
        mhc_alleles = set()
        for val in df[mhc_col].dropna().unique():
            if isinstance(val, str):
                for allele in val.split(","):
                    mhc_alleles.add(allele.strip())
        results["unique_mhc_alleles"] = len(mhc_alleles)

    # Unique epitope-MHC pairs
    if epitope_col in df.columns and mhc_col in df.columns:
        df["epitope_mhc"] = df[epitope_col].fillna("") + "|" + df[mhc_col].fillna("")
        results["unique_epitope_mhc_pairs"] = df["epitope_mhc"].nunique()

    # Count paired TCRs (both Chain 1 and Chain 2 CDR3 present)
    if chain1_cdr3 in df.columns and chain2_cdr3 in df.columns:
        paired_mask = df[chain1_cdr3].notna() & df[chain2_cdr3].notna()
        results["paired_tcrs"] = paired_mask.sum()

        # Collect unique alpha/beta sequences
        tra_seqs = set()
        trb_seqs = set()

        for _, row in df.iterrows():
            chain1_t = row.get(chain1_type, "")
            chain2_t = row.get(chain2_type, "")
            cdr3_1 = row.get(chain1_cdr3)
            cdr3_2 = row.get(chain2_cdr3)

            if pd.notna(cdr3_1):
                if chain1_t == "alpha":
                    tra_seqs.add(cdr3_1)
                elif chain1_t == "beta":
                    trb_seqs.add(cdr3_1)

            if pd.notna(cdr3_2):
                if chain2_t == "alpha":
                    tra_seqs.add(cdr3_2)
                elif chain2_t == "beta":
                    trb_seqs.add(cdr3_2)

        results["unique_tra"] = len(tra_seqs)
        results["unique_trb"] = len(trb_seqs)

    # Unique TCR-pMHC
    if chain1_cdr3 in df.columns and epitope_col in df.columns:
        df["tcr_pmhc"] = (
            df[chain1_cdr3].fillna("") + "|" +
            df.get(chain2_cdr3, pd.Series([""] * len(df))).fillna("") + "|" +
            df[epitope_col].fillna("") + "|" +
            df.get(mhc_col, pd.Series([""] * len(df))).fillna("")
        )
        results["unique_tcr_pmhc"] = df["tcr_pmhc"].nunique()

    return results


def analyze_mcpas(db_path: Path) -> dict:
    """
    Parse McPAS-TCR.csv and extract statistics.
    """
    mcpas_file = db_path / "McPASDB" / "McPAS-TCR.csv"
    if not mcpas_file.exists():
        return None

    print("  Reading McPAS-TCR...")
    df = pd.read_csv(mcpas_file, low_memory=False)

    results = {
        "total_records": len(df),
        "unique_epitopes": 0,
        "unique_mhc_alleles": 0,
        "unique_epitope_mhc_pairs": 0,
        "unique_tcr_pmhc": 0,
        "paired_tcrs": 0,
        "unique_tra": 0,
        "unique_trb": 0,
    }

    epitope_col = "Epitope.peptide"
    mhc_col = "MHC"
    alpha_col = "CDR3.alpha.aa"
    beta_col = "CDR3.beta.aa"

    if epitope_col in df.columns:
        results["unique_epitopes"] = df[epitope_col].nunique()

    if mhc_col in df.columns:
        results["unique_mhc_alleles"] = df[mhc_col].nunique()

    # Unique epitope-MHC pairs
    if epitope_col in df.columns and mhc_col in df.columns:
        df["epitope_mhc"] = df[epitope_col].fillna("") + "|" + df[mhc_col].fillna("")
        results["unique_epitope_mhc_pairs"] = df["epitope_mhc"].nunique()

    # Paired TCRs (both alpha and beta present)
    if alpha_col in df.columns and beta_col in df.columns:
        paired_mask = df[alpha_col].notna() & df[beta_col].notna()
        results["paired_tcrs"] = paired_mask.sum()

        # Unique alpha and beta
        results["unique_tra"] = df[alpha_col].nunique()
        results["unique_trb"] = df[beta_col].nunique()

    # Unique TCR-pMHC
    df["tcr_pmhc"] = (
        df.get(alpha_col, pd.Series([""] * len(df))).fillna("") + "|" +
        df.get(beta_col, pd.Series([""] * len(df))).fillna("") + "|" +
        df.get(epitope_col, pd.Series([""] * len(df))).fillna("") + "|" +
        df.get(mhc_col, pd.Series([""] * len(df))).fillna("")
    )
    results["unique_tcr_pmhc"] = df["tcr_pmhc"].nunique()

    return results


def analyze_cedar(db_path: Path) -> dict:
    """
    Parse CEDAR receptor results table (similar format to IEDB).
    """
    cedar_dir = db_path / "CEDAR"
    if not cedar_dir.exists():
        return None

    receptor_files = list(cedar_dir.glob("cedar_receptor_results_table*.tsv"))
    if not receptor_files:
        return None

    print("  Reading CEDAR...")
    df = pd.read_csv(receptor_files[0], sep="\t", low_memory=False)

    results = {
        "total_records": len(df),
        "unique_epitopes": 0,
        "unique_mhc_alleles": 0,
        "unique_epitope_mhc_pairs": 0,
        "unique_tcr_pmhc": 0,
        "paired_tcrs": 0,
        "unique_tra": 0,
        "unique_trb": 0,
    }

    epitope_col = "Epitope - Name"
    chain1_type = "Chain 1 - Type"
    chain1_cdr3 = "Chain 1 - CDR3 Calculated"
    chain2_type = "Chain 2 - Type"
    chain2_cdr3 = "Chain 2 - CDR3 Calculated"

    if epitope_col in df.columns:
        results["unique_epitopes"] = df[epitope_col].nunique()

    # Count paired TCRs (both Chain 1 and Chain 2 CDR3 present)
    if chain1_cdr3 in df.columns and chain2_cdr3 in df.columns:
        paired_mask = df[chain1_cdr3].notna() & df[chain2_cdr3].notna()
        results["paired_tcrs"] = paired_mask.sum()

        # Collect unique alpha/beta sequences
        tra_seqs = set()
        trb_seqs = set()

        for _, row in df.iterrows():
            chain1_t = row.get(chain1_type, "")
            chain2_t = row.get(chain2_type, "")
            cdr3_1 = row.get(chain1_cdr3)
            cdr3_2 = row.get(chain2_cdr3)

            if pd.notna(cdr3_1):
                if chain1_t == "alpha":
                    tra_seqs.add(cdr3_1)
                elif chain1_t == "beta":
                    trb_seqs.add(cdr3_1)

            if pd.notna(cdr3_2):
                if chain2_t == "alpha":
                    tra_seqs.add(cdr3_2)
                elif chain2_t == "beta":
                    trb_seqs.add(cdr3_2)

        results["unique_tra"] = len(tra_seqs)
        results["unique_trb"] = len(trb_seqs)

    return results


def analyze_tcrdb(db_path: Path) -> dict:
    """
    Aggregate across all PRJNA/PRJEB*.tsv files in TCRdb.
    These contain bulk TRB data (mostly unpaired).
    """
    tcrdb_dir = db_path / "TCRdb"
    if not tcrdb_dir.exists():
        return None

    tsv_files = list(tcrdb_dir.glob("*.tsv"))
    if not tsv_files:
        return None

    print(f"  Reading TCRdb ({len(tsv_files)} files)...")
    unique_trb = set()
    total_records = 0

    for tsv_file in tqdm(tsv_files, desc="  TCRdb files", leave=False):
        try:
            df = pd.read_csv(tsv_file, sep="\t", low_memory=False)
            total_records += len(df)
            if "AASeq" in df.columns:
                unique_trb.update(df["AASeq"].dropna().unique())
        except Exception as e:
            print(f"    Warning: Could not read {tsv_file.name}: {e}")

    return {
        "total_records": total_records,
        "unique_epitopes": 0,
        "unique_mhc_alleles": 0,
        "unique_epitope_mhc_pairs": 0,
        "unique_tcr_pmhc": 0,
        "paired_tcrs": 0,
        "unique_tra": 0,
        "unique_trb": len(unique_trb),
    }


def analyze_ireceptor(db_path: Path, sample_size: int | None = None) -> dict:
    """
    Parse iReceptor TSV files (AIRR format).
    Note: These files can be very large, so we may sample.
    """
    ireceptor_dir = db_path / "ireceptor"
    if not ireceptor_dir.exists():
        return None

    # Find TSV files (excluding metadata files)
    tsv_files = [f for f in ireceptor_dir.glob("*.tsv") if "metadata" not in f.name]
    if not tsv_files:
        return None

    print(f"  Reading iReceptor ({len(tsv_files)} files)...")
    print("    Note: Large files - this may take a while...")

    unique_tra = set()
    unique_trb = set()
    total_records = 0

    for tsv_file in tqdm(tsv_files, desc="  iReceptor files", leave=False):
        try:
            # For very large files, read in chunks
            chunk_iter = pd.read_csv(
                tsv_file,
                sep="\t",
                low_memory=False,
                chunksize=500000,
                usecols=lambda col: col in ["junction_aa", "locus"],
            )

            for chunk in chunk_iter:
                total_records += len(chunk)

                if "locus" in chunk.columns and "junction_aa" in chunk.columns:
                    tra_mask = chunk["locus"].str.contains("TRA", na=False)
                    trb_mask = chunk["locus"].str.contains("TRB", na=False)

                    unique_tra.update(chunk.loc[tra_mask, "junction_aa"].dropna().unique())
                    unique_trb.update(chunk.loc[trb_mask, "junction_aa"].dropna().unique())
                elif "junction_aa" in chunk.columns:
                    # If no locus column, assume all are TRB
                    unique_trb.update(chunk["junction_aa"].dropna().unique())

        except Exception as e:
            print(f"    Warning: Could not read {tsv_file.name}: {e}")

    return {
        "total_records": total_records,
        "unique_epitopes": 0,
        "unique_mhc_alleles": 0,
        "unique_epitope_mhc_pairs": 0,
        "unique_tcr_pmhc": 0,
        "paired_tcrs": 0,
        "unique_tra": len(unique_tra),
        "unique_trb": len(unique_trb),
    }


def analyze_warrendb(db_path: Path) -> dict:
    """
    Parse WarrenDB public TCR files.
    """
    warren_dir = db_path / "WarrenDB"
    if not warren_dir.exists():
        return None

    csv_files = list(warren_dir.glob("*.csv"))
    if not csv_files:
        return None

    print(f"  Reading WarrenDB ({len(csv_files)} files)...")
    unique_trb = set()
    total_records = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            total_records += len(df)
            # Try common column names for TCR sequences
            for col in ["CDR3", "cdr3", "sequence", "TRB"]:
                if col in df.columns:
                    unique_trb.update(df[col].dropna().unique())
                    break
        except Exception as e:
            print(f"    Warning: Could not read {csv_file.name}: {e}")

    return {
        "total_records": total_records,
        "unique_epitopes": 0,
        "unique_mhc_alleles": 0,
        "unique_epitope_mhc_pairs": 0,
        "unique_tcr_pmhc": 0,
        "paired_tcrs": 0,
        "unique_tra": 0,
        "unique_trb": len(unique_trb),
    }


def count_study_tcr_sequences(study_path: Path) -> dict:
    """
    Count TCR sequences from a study's tcr/ folder.

    Looks for:
    - tcr/contigs/*.csv (10x format)
    - tcr/*.tsv or tcr/*.csv files

    Avoids double-counting by only processing one data type.
    """
    tcr_dir = study_path / "tcr"
    if not tcr_dir.exists():
        return None

    unique_tra = set()
    unique_trb = set()
    paired_barcodes = set()  # Track barcodes with both TRA and TRB

    # Priority: contigs folder (10x format)
    contigs_dir = tcr_dir / "contigs"
    if contigs_dir.exists():
        csv_files = list(contigs_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)

                # 10x format has chain, cdr3, and barcode columns
                if "chain" in df.columns and "cdr3" in df.columns:
                    # Filter for productive sequences only if column exists
                    if "productive" in df.columns:
                        df = df[df["productive"] == True]  # noqa: E712

                    tra_mask = df["chain"] == "TRA"
                    trb_mask = df["chain"] == "TRB"

                    unique_tra.update(df.loc[tra_mask, "cdr3"].dropna().unique())
                    unique_trb.update(df.loc[trb_mask, "cdr3"].dropna().unique())

                    # Count paired (barcodes with both TRA and TRB)
                    if "barcode" in df.columns:
                        tra_barcodes = set(df.loc[tra_mask, "barcode"].dropna())
                        trb_barcodes = set(df.loc[trb_mask, "barcode"].dropna())
                        paired_barcodes.update(tra_barcodes & trb_barcodes)
            except Exception:
                pass

        return {
            "paired_tcrs": len(paired_barcodes),
            "unique_tra": len(unique_tra),
            "unique_trb": len(unique_trb),
        }

    # Fallback: look for TSV/CSV files directly in tcr/
    data_files = list(tcr_dir.glob("*.csv")) + list(tcr_dir.glob("*.tsv"))
    for data_file in data_files:
        try:
            sep = "\t" if data_file.suffix == ".tsv" else ","
            df = pd.read_csv(data_file, sep=sep, low_memory=False)

            # Try to identify chain and CDR3 columns
            chain_col = None
            cdr3_col = None

            for col in df.columns:
                col_lower = col.lower()
                if "chain" in col_lower:
                    chain_col = col
                if "cdr3" in col_lower and "aa" in col_lower:
                    cdr3_col = col
                elif "cdr3" in col_lower and cdr3_col is None:
                    cdr3_col = col

            if chain_col and cdr3_col:
                tra_mask = df[chain_col].str.contains("TRA", na=False, case=False)
                trb_mask = df[chain_col].str.contains("TRB", na=False, case=False)

                unique_tra.update(df.loc[tra_mask, cdr3_col].dropna().unique())
                unique_trb.update(df.loc[trb_mask, cdr3_col].dropna().unique())
            elif cdr3_col:
                # If no chain column, add to TRB (assumption for bulk data)
                unique_trb.update(df[cdr3_col].dropna().unique())
        except Exception:
            pass

    if unique_tra or unique_trb:
        return {
            "paired_tcrs": 0,  # Can't determine from non-10x data
            "unique_tra": len(unique_tra),
            "unique_trb": len(unique_trb),
        }

    return None


def analyze_all_studies(raw_data_path: Path, study_sources: dict) -> dict:
    """
    Analyze TCR sequences from all study folders.
    """
    print("\nAnalyzing study TCR sequences...")

    results = {
        "total_paired": 0,
        "total_tra": 0,
        "total_trb": 0,
        "all_tra_seqs": set(),
        "all_trb_seqs": set(),
        "studies_with_tcr": 0,
    }

    study_categories = [
        "acute_illness",
        "autoimmune_studies",
        "cancer_studies",
        "chronic_illness",
        "healthy_studies",
        "multiple_classifications",
        "unclassified_data",
        "viral_studies",
    ]

    all_studies = []
    for category in study_categories:
        cat_path = raw_data_path / category
        if cat_path.exists():
            for study_dir in cat_path.iterdir():
                if study_dir.is_dir() and not study_dir.name.startswith("."):
                    all_studies.append(study_dir)

    for study_dir in tqdm(all_studies, desc="  Processing studies"):
        study_result = count_study_tcr_sequences(study_dir)
        if study_result:
            results["studies_with_tcr"] += 1
            results["total_paired"] += study_result["paired_tcrs"]
            results["total_tra"] += study_result["unique_tra"]
            results["total_trb"] += study_result["unique_trb"]

    # Special handling for adaptive_bulk: scan sample files for TCR sequences
    adaptive_path = raw_data_path / "adaptive_bulk" / "bulk_data" / "tcr"
    if adaptive_path.exists():
        for subdir, chain_type in [("bulk_survey_tra", "TRA"), ("bulk_survey_trb", "TRB")]:
            survey_path = adaptive_path / subdir
            if survey_path.exists():
                sample_files = list(survey_path.glob("*.tsv"))
                for sample_file in tqdm(sample_files, desc=f"  Adaptive {chain_type}"):
                    try:
                        df = pd.read_csv(
                            sample_file, sep="\t", usecols=["amino_acid"], low_memory=False
                        )
                        seqs = df["amino_acid"].dropna().unique()
                        if chain_type == "TRA":
                            results["total_tra"] += len(seqs)
                        else:
                            results["total_trb"] += len(seqs)
                        results["studies_with_tcr"] += 1
                    except Exception:
                        pass

    return results


def generate_report(
    source_scan: dict,
    db_results: dict[str, dict],
    study_results: dict,
    output_path: Path,
) -> None:
    """Generate markdown and JSON reports."""

    # Calculate totals
    total_db_paired = sum(r.get("paired_tcrs", 0) for r in db_results.values() if r)
    total_db_tra = sum(r.get("unique_tra", 0) for r in db_results.values() if r)
    total_db_trb = sum(r.get("unique_trb", 0) for r in db_results.values() if r)

    # Build markdown report
    lines = [
        "=" * 80,
        "TCR RAW DATA SOURCE SUMMARY",
        "=" * 80,
        "",
        "### DATA SOURCES ###",
        "",
        f"Databases:                {len(source_scan['databases']):>6}",
        f"Public Studies:           {source_scan['total_studies']:>6}",
        f"  - GEO (GSE/GSM):        {len(source_scan['studies_by_source']['GEO']):>6}",
        f"  - SRA (PRJNA/PRJEB):    {len(source_scan['studies_by_source']['SRA']):>6}",
        f"  - Zenodo (ZEN):         {len(source_scan['studies_by_source']['Zenodo']):>6}",
        f"  - Other/Named:          {len(source_scan['studies_by_source']['Other']):>6}",
        "",
        "### DATABASE STATISTICS ###",
        "",
        f"{'Database':<15} {'Epitopes':>10} {'MHC Alleles':>12} {'Epitope-MHC':>12} {'TCR-pMHC':>12}",
        "-" * 65,
    ]

    for db_name, stats in db_results.items():
        if stats and stats.get("unique_epitopes", 0) > 0:
            lines.append(
                f"{db_name:<15} {stats['unique_epitopes']:>10,} "
                f"{stats['unique_mhc_alleles']:>12,} "
                f"{stats['unique_epitope_mhc_pairs']:>12,} "
                f"{stats['unique_tcr_pmhc']:>12,}"
            )

    lines.extend([
        "",
        "### TCR SEQUENCE COUNTS ###",
        "",
        f"{'Source':<20} {'Paired TCRs':>15} {'Unique TRA':>15} {'Unique TRB':>15}",
        "-" * 70,
    ])

    for db_name, stats in db_results.items():
        if stats and (stats.get("unique_tra", 0) > 0 or stats.get("unique_trb", 0) > 0):
            paired_str = f"{stats['paired_tcrs']:>15,}" if stats["paired_tcrs"] > 0 else f"{'-':>15}"
            tra_str = f"{stats['unique_tra']:>15,}" if stats["unique_tra"] > 0 else f"{'-':>15}"
            trb_str = f"{stats['unique_trb']:>15,}" if stats["unique_trb"] > 0 else f"{'-':>15}"
            lines.append(f"{db_name:<20} {paired_str} {tra_str} {trb_str}")

    lines.extend([
        "-" * 70,
        f"{'All Databases':<20} {total_db_paired:>15,} {total_db_tra:>15,} {total_db_trb:>15,}",
    ])

    if study_results:
        study_paired_str = f"{study_results['total_paired']:>15,}" if study_results["total_paired"] > 0 else f"{'-':>15}"
        lines.append(
            f"{'All Studies':<20} {study_paired_str} "
            f"{study_results['total_tra']:>15,} {study_results['total_trb']:>15,}"
        )
        lines.append(
            f"  Studies with TCR data: {study_results['studies_with_tcr']}"
        )

    lines.extend([
        "",
        "=" * 80,
        "",
        "### STUDIES BY CATEGORY ###",
        "",
    ])

    for category, studies in sorted(source_scan["studies_by_category"].items()):
        lines.append(f"{category}: {len(studies)} studies")

    lines.extend([
        "",
        "### DATABASE NAMES ###",
        "",
        ", ".join(sorted(source_scan["databases"])),
        "",
    ])

    # Write markdown report
    output_path.mkdir(parents=True, exist_ok=True)
    md_file = output_path / "tcr_source_summary.md"
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"\nMarkdown report written to: {md_file}")

    # Build JSON report
    json_data = {
        "data_sources": {
            "databases": source_scan["databases"],
            "total_databases": len(source_scan["databases"]),
            "total_studies": source_scan["total_studies"],
            "studies_by_repository": {
                "GEO": len(source_scan["studies_by_source"]["GEO"]),
                "SRA": len(source_scan["studies_by_source"]["SRA"]),
                "Zenodo": len(source_scan["studies_by_source"]["Zenodo"]),
                "Other": len(source_scan["studies_by_source"]["Other"]),
            },
            "studies_by_category": {
                cat: len(studies)
                for cat, studies in source_scan["studies_by_category"].items()
            },
        },
        "database_statistics": {
            db_name: stats for db_name, stats in db_results.items() if stats
        },
        "database_totals": {
            "total_paired_tcrs": total_db_paired,
            "total_unique_tra": total_db_tra,
            "total_unique_trb": total_db_trb,
        },
    }

    if study_results:
        json_data["study_statistics"] = {
            "studies_with_tcr_data": study_results["studies_with_tcr"],
            "total_paired_tcrs": study_results["total_paired"],
            "total_unique_tra": study_results["total_tra"],
            "total_unique_trb": study_results["total_trb"],
        }

    json_file = output_path / "tcr_source_summary.json"
    with open(json_file, "w") as f:
        json.dump(convert_to_native(json_data), f, indent=2)

    print(f"JSON report written to: {json_file}")

    # Print summary to console
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TCR raw data sources and generate summary statistics."
    )
    parser.add_argument(
        "--raw_data_path",
        type=Path,
        required=True,
        help="Path to the raw_data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/analysis"),
        help="Output directory for reports (default: data/analysis)",
    )
    parser.add_argument(
        "--skip_ireceptor",
        action="store_true",
        help="Skip iReceptor analysis (very large files)",
    )
    parser.add_argument(
        "--skip_studies",
        action="store_true",
        help="Skip individual study analysis",
    )
    parser.add_argument(
        "--databases_only",
        action="store_true",
        help="Only analyze databases, not studies",
    )

    args = parser.parse_args()

    if not args.raw_data_path.exists():
        print(f"Error: Raw data path does not exist: {args.raw_data_path}")
        return 1

    print("=" * 60)
    print("TCR Raw Data Source Analysis")
    print("=" * 60)

    # Step 1: Scan sources
    print("\nStep 1: Scanning data sources...")
    source_scan = scan_study_sources(args.raw_data_path)
    print(f"  Found {len(source_scan['databases'])} databases")
    print(f"  Found {source_scan['total_studies']} studies")

    # Step 2: Analyze databases
    print("\nStep 2: Analyzing databases...")
    db_path = args.raw_data_path / "databases"
    db_results = {}

    db_results["VdjDB"] = analyze_vdjdb(db_path)
    db_results["IEDB"] = analyze_iedb(db_path)
    db_results["McPAS-TCR"] = analyze_mcpas(db_path)
    db_results["CEDAR"] = analyze_cedar(db_path)
    db_results["TCRdb"] = analyze_tcrdb(db_path)
    db_results["WarrenDB"] = analyze_warrendb(db_path)

    if not args.skip_ireceptor:
        db_results["iReceptor"] = analyze_ireceptor(db_path)
    else:
        print("  Skipping iReceptor (--skip_ireceptor flag)")
        db_results["iReceptor"] = None

    # Step 3: Analyze studies (if not skipped)
    study_results = None
    if not args.skip_studies and not args.databases_only:
        print("\nStep 3: Analyzing study TCR data...")
        study_results = analyze_all_studies(args.raw_data_path, source_scan)
    else:
        print("\nStep 3: Skipping study analysis")

    # Step 4: Generate report
    print("\nStep 4: Generating reports...")
    generate_report(source_scan, db_results, study_results, args.output_dir)

    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
