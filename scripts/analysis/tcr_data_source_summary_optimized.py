#!/usr/bin/env python3
"""
TCR Raw Data Source Summary - Optimized for p4d.24xlarge

Optimizations for 96 vCPUs, 1.5TB RAM:
1. Parallel database analyzers using ThreadPoolExecutor
2. Parallel file processing within analyzers using ProcessPoolExecutor
3. Vectorized operations instead of row-by-row iteration
4. Larger chunk sizes for chunked reading (leverage RAM)
5. Polars for faster DataFrame operations where beneficial
6. Parallel study processing
7. PyArrow for faster parquet reading

Usage:
    python tcr_data_source_summary_optimized.py \
        --raw_data_path /path/to/raw_data \
        --workers 80 \
        --foundation_path /path/to/foundation_permutations
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Try to import polars for faster operations
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("Warning: Polars not installed. Using pandas (slower).")


# Configuration for p4d.24xlarge
DEFAULT_WORKERS = 80  # Leave some CPUs for system
CHUNK_SIZE = 5_000_000  # Large chunks for 1.5TB RAM
PARQUET_BATCH_SIZE = 100  # Process parquet files in batches


def load_checkpoint(checkpoint_file: Path) -> dict | None:
    """Load checkpoint if it exists."""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_checkpoint(checkpoint_file: Path, data: dict) -> None:
    """Save checkpoint to file."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(convert_to_native(data), f, indent=2)
    print(f"  Checkpoint saved: {checkpoint_file}")


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


MOLECULES = {'tra', 'trb', 'peptide', 'mhc_one', 'mhc_two'}


def normalize_permutation_key(key: str) -> str:
    """
    Normalize permutation key by sorting molecule components alphabetically.

    Handles multi-part molecule names like mhc_one and mhc_two.

    Examples:
        'tra_trb' -> 'tra+trb'
        'trb_tra' -> 'tra+trb'
        'mhc_one_mhc_two' -> 'mhc_one+mhc_two'
        'trb_mhc_one_tra' -> 'mhc_one+tra+trb'
    """
    if not key or key == 'empty':
        return key

    # Extract molecules by matching known names
    components = []
    remaining = key
    while remaining:
        matched = False
        for mol in sorted(MOLECULES, key=len, reverse=True):  # Try longer names first
            if remaining.startswith(mol):
                components.append(mol)
                remaining = remaining[len(mol):]
                if remaining.startswith('_'):
                    remaining = remaining[1:]
                matched = True
                break
        if not matched:
            break  # Unknown component, stop parsing

    return '+'.join(sorted(components))


# --- Optimized Database Analyzers ---


def analyze_vdjdb(db_path: Path) -> dict | None:
    """Optimized VDJdb analysis with vectorized operations."""
    vdjdb_file = db_path / "VdjDB" / "vdjdb.txt"
    if not vdjdb_file.exists():
        return None

    print("  Reading VDJdb...")

    if HAS_POLARS:
        df = pl.read_csv(vdjdb_file, separator="\t", infer_schema_length=0)

        # Filter: HomoSapiens and score != 0
        df = df.filter(
            (pl.col("species") == "HomoSapiens") & (pl.col("vdjdb.score") != "0")
        )

        # Build TCR-pMHC key
        df = df.with_columns(
            (pl.col("cdr3") + "|" + pl.col("antigen.epitope") + "|" +
             pl.col("mhc.a") + "|" + pl.col("mhc.b")).alias("tcr_pmhc")
        )

        # Split by MHC class
        class_i = df.filter(pl.col("mhc.class") == "MHCI")
        class_ii = df.filter(pl.col("mhc.class") == "MHCII")

        # Paired TCRs via complex.id
        paired_df = df.filter(pl.col("complex.id") != "0")
        if len(paired_df) > 0:
            paired_ids = (
                paired_df.group_by("complex.id")
                .agg(pl.col("gene").unique())
                .filter(
                    pl.col("gene").list.contains("TRA") &
                    pl.col("gene").list.contains("TRB")
                )
            )
            paired_count = len(paired_ids)
        else:
            paired_count = 0

        return {
            "total_records": len(df),
            "class_i": {
                "total": len(class_i),
                "unique_tcr_pmhc": class_i["tcr_pmhc"].n_unique(),
            },
            "class_ii": {
                "total": len(class_ii),
                "unique_tcr_pmhc": class_ii["tcr_pmhc"].n_unique(),
            },
            "unique_tra": df.filter(pl.col("gene") == "TRA")["cdr3"].n_unique(),
            "unique_trb": df.filter(pl.col("gene") == "TRB")["cdr3"].n_unique(),
            "paired_tcrs": paired_count,
        }
    else:
        # Pandas fallback with optimizations
        df = pd.read_csv(vdjdb_file, sep="\t", dtype=str, na_filter=False)
        df = df[(df["species"] == "HomoSapiens") & (df["vdjdb.score"] != "0")]

        df["tcr_pmhc"] = (
            df["cdr3"] + "|" + df["antigen.epitope"] + "|" + df["mhc.a"] + "|" + df["mhc.b"]
        )

        class_i = df[df["mhc.class"] == "MHCI"]
        class_ii = df[df["mhc.class"] == "MHCII"]

        paired_df = df[df["complex.id"] != "0"]
        if len(paired_df) > 0:
            paired_ids = (
                paired_df.groupby("complex.id")["gene"]
                .apply(lambda g: set(g.unique()) >= {"TRA", "TRB"})
            )
            paired_count = int(paired_ids.sum())
        else:
            paired_count = 0

        return {
            "total_records": len(df),
            "class_i": {
                "total": len(class_i),
                "unique_tcr_pmhc": int(class_i["tcr_pmhc"].nunique()),
            },
            "class_ii": {
                "total": len(class_ii),
                "unique_tcr_pmhc": int(class_ii["tcr_pmhc"].nunique()),
            },
            "unique_tra": int(df[df["gene"] == "TRA"]["cdr3"].nunique()),
            "unique_trb": int(df[df["gene"] == "TRB"]["cdr3"].nunique()),
            "paired_tcrs": paired_count,
        }


def analyze_mcpas(db_path: Path) -> dict | None:
    """Optimized McPAS-TCR analysis with vectorized MHC classification."""
    mcpas_file = db_path / "McPASDB" / "McPAS-TCR.csv"
    if not mcpas_file.exists():
        return None

    print("  Reading McPAS-TCR...")

    if HAS_POLARS:
        df = pl.read_csv(mcpas_file, infer_schema_length=0)

        alpha_col = "CDR3.alpha.aa"
        beta_col = "CDR3.beta.aa"
        epitope_col = "Epitope.peptide"
        mhc_col = "MHC"

        # Vectorized MHC classification
        df = df.with_columns(
            pl.when(pl.col(mhc_col).str.to_uppercase().str.contains(r"HLA-[ABC]"))
            .then(pl.lit("MHCI"))
            .when(pl.col(mhc_col).str.to_uppercase().str.contains(r"HLA-D"))
            .then(pl.lit("MHCII"))
            .otherwise(pl.lit("unknown"))
            .alias("mhc_class")
        )

        # Build TCR-pMHC key
        df = df.with_columns(
            (pl.col(alpha_col) + "|" + pl.col(beta_col) + "|" +
             pl.col(epitope_col) + "|" + pl.col(mhc_col)).alias("tcr_pmhc")
        )

        class_i = df.filter(pl.col("mhc_class") == "MHCI")
        class_ii = df.filter(pl.col("mhc_class") == "MHCII")

        # Paired detection
        has_alpha = (pl.col(alpha_col) != "") & (pl.col(alpha_col) != "NA")
        has_beta = (pl.col(beta_col) != "") & (pl.col(beta_col) != "NA")
        paired_count = df.filter(has_alpha & has_beta).height

        # Unique chains
        tra_seqs = df.filter(has_alpha)[alpha_col].n_unique()
        trb_seqs = df.filter(has_beta)[beta_col].n_unique()

        return {
            "total_records": len(df),
            "class_i": {
                "total": len(class_i),
                "unique_tcr_pmhc": class_i["tcr_pmhc"].n_unique(),
            },
            "class_ii": {
                "total": len(class_ii),
                "unique_tcr_pmhc": class_ii["tcr_pmhc"].n_unique(),
            },
            "unique_tra": tra_seqs,
            "unique_trb": trb_seqs,
            "paired_tcrs": paired_count,
        }
    else:
        # Pandas with vectorized operations
        df = pd.read_csv(mcpas_file, dtype=str, na_filter=False)

        alpha_col = "CDR3.alpha.aa"
        beta_col = "CDR3.beta.aa"
        epitope_col = "Epitope.peptide"
        mhc_col = "MHC"

        df["tcr_pmhc"] = (
            df[alpha_col] + "|" + df[beta_col] + "|" + df[epitope_col] + "|" + df[mhc_col]
        )

        # Vectorized MHC classification
        mhc_upper = df[mhc_col].str.upper()
        df["mhc_class"] = np.where(
            mhc_upper.str.contains(r"HLA-[ABC]", regex=True, na=False),
            "MHCI",
            np.where(
                mhc_upper.str.contains(r"HLA-D", regex=True, na=False),
                "MHCII",
                "unknown"
            )
        )

        class_i = df[df["mhc_class"] == "MHCI"]
        class_ii = df[df["mhc_class"] == "MHCII"]

        has_alpha = (df[alpha_col] != "") & (df[alpha_col] != "NA")
        has_beta = (df[beta_col] != "") & (df[beta_col] != "NA")
        paired_count = int((has_alpha & has_beta).sum())

        tra_seqs = df.loc[has_alpha, alpha_col]
        trb_seqs = df.loc[has_beta, beta_col]

        return {
            "total_records": len(df),
            "class_i": {
                "total": len(class_i),
                "unique_tcr_pmhc": int(class_i["tcr_pmhc"].nunique()),
            },
            "class_ii": {
                "total": len(class_ii),
                "unique_tcr_pmhc": int(class_ii["tcr_pmhc"].nunique()),
            },
            "unique_tra": int(tra_seqs.nunique()),
            "unique_trb": int(trb_seqs.nunique()),
            "paired_tcrs": paired_count,
        }


def _analyze_iedb_cedar_tcell_mhcligand_optimized(
    tcell_file: Path | None,
    mhcligand_file: Path | None,
    has_qualitative_in_mhcligand: bool = True,
) -> dict:
    """Optimized IEDB/CEDAR analysis with vectorized operations."""
    result = {
        "tcell_positive": 0,
        "tcell_total": 0,
        "mhcligand_total": 0,
        "unique_peptide_mhc": 0,
    }

    peptide_mhc_pairs = set()

    if HAS_POLARS:
        # T-cell assay
        if tcell_file and tcell_file.exists():
            tcell_df = pl.read_csv(tcell_file, separator="\t", infer_schema_length=0)
            result["tcell_total"] = len(tcell_df)

            qual_col = "Assay - Qualitative Measurement"
            if qual_col in tcell_df.columns:
                pos_df = tcell_df.filter(pl.col(qual_col).str.starts_with("Positive"))
                result["tcell_positive"] = len(pos_df)
            else:
                pos_df = tcell_df

            epitope_col = "Epitope - Name"
            mhc_col = "MHC Restriction - Name"
            if epitope_col in pos_df.columns and mhc_col in pos_df.columns:
                pairs = pos_df.select([epitope_col, mhc_col]).filter(
                    (pl.col(epitope_col) != "") & (pl.col(mhc_col) != "")
                )
                for row in pairs.iter_rows():
                    peptide_mhc_pairs.add(f"{row[0]}|{row[1]}")

        # MHC ligand assay
        if mhcligand_file and mhcligand_file.exists():
            mhc_df = pl.read_csv(mhcligand_file, separator="\t", infer_schema_length=0)
            result["mhcligand_total"] = len(mhc_df)

            qual_col = "Assay - Qualitative Measurement"
            if has_qualitative_in_mhcligand and qual_col in mhc_df.columns:
                pos_mhc_df = mhc_df.filter(pl.col(qual_col).str.starts_with("Positive"))
            else:
                pos_mhc_df = mhc_df

            epitope_col = "Epitope - Name"
            mhc_col = "MHC Restriction - Name"
            if epitope_col in pos_mhc_df.columns and mhc_col in pos_mhc_df.columns:
                pairs = pos_mhc_df.select([epitope_col, mhc_col]).filter(
                    (pl.col(epitope_col) != "") & (pl.col(mhc_col) != "")
                )
                for row in pairs.iter_rows():
                    peptide_mhc_pairs.add(f"{row[0]}|{row[1]}")
    else:
        # Pandas fallback
        if tcell_file and tcell_file.exists():
            tcell_df = pd.read_csv(tcell_file, sep="\t", dtype=str, na_filter=False)
            result["tcell_total"] = len(tcell_df)

            qual_col = "Assay - Qualitative Measurement"
            if qual_col in tcell_df.columns:
                positive_mask = tcell_df[qual_col].str.startswith("Positive")
                result["tcell_positive"] = int(positive_mask.sum())
                pos_df = tcell_df[positive_mask]
            else:
                pos_df = tcell_df

            epitope_col = "Epitope - Name"
            mhc_col = "MHC Restriction - Name"
            if epitope_col in pos_df.columns and mhc_col in pos_df.columns:
                valid = pos_df[(pos_df[epitope_col] != "") & (pos_df[mhc_col] != "")]
                peptide_mhc_pairs.update(
                    valid[epitope_col] + "|" + valid[mhc_col]
                )

        if mhcligand_file and mhcligand_file.exists():
            mhc_df = pd.read_csv(mhcligand_file, sep="\t", dtype=str, na_filter=False)
            result["mhcligand_total"] = len(mhc_df)

            qual_col = "Assay - Qualitative Measurement"
            if has_qualitative_in_mhcligand and qual_col in mhc_df.columns:
                positive_mask = mhc_df[qual_col].str.startswith("Positive")
                pos_mhc_df = mhc_df[positive_mask]
            else:
                pos_mhc_df = mhc_df

            epitope_col = "Epitope - Name"
            mhc_col = "MHC Restriction - Name"
            if epitope_col in pos_mhc_df.columns and mhc_col in pos_mhc_df.columns:
                valid = pos_mhc_df[(pos_mhc_df[epitope_col] != "") & (pos_mhc_df[mhc_col] != "")]
                peptide_mhc_pairs.update(
                    valid[epitope_col] + "|" + valid[mhc_col]
                )

    result["unique_peptide_mhc"] = len(peptide_mhc_pairs)
    return result


def _analyze_receptor_table_optimized(receptor_file: Path, has_mhc_col: bool = True) -> dict:
    """Optimized receptor table analysis with vectorized operations."""
    result = {
        "total_records": 0,
        "unique_tra": 0,
        "unique_trb": 0,
        "paired_tcrs": 0,
    }

    if not receptor_file.exists():
        return result

    chain1_type_col = "Chain 1 - Type"
    chain1_cdr3_col = "Chain 1 - CDR3 Calculated"
    chain2_type_col = "Chain 2 - Type"
    chain2_cdr3_col = "Chain 2 - CDR3 Calculated"

    if HAS_POLARS:
        df = pl.read_csv(receptor_file, separator="\t", infer_schema_length=0)
        result["total_records"] = len(df)

        # Vectorized chain extraction
        tra_c1 = df.filter(
            (pl.col(chain1_type_col) == "alpha") & (pl.col(chain1_cdr3_col) != "")
        )[chain1_cdr3_col]
        tra_c2 = df.filter(
            (pl.col(chain2_type_col) == "alpha") & (pl.col(chain2_cdr3_col) != "")
        )[chain2_cdr3_col]

        trb_c1 = df.filter(
            (pl.col(chain1_type_col) == "beta") & (pl.col(chain1_cdr3_col) != "")
        )[chain1_cdr3_col]
        trb_c2 = df.filter(
            (pl.col(chain2_type_col) == "beta") & (pl.col(chain2_cdr3_col) != "")
        )[chain2_cdr3_col]

        tra_seqs = set(tra_c1.to_list()) | set(tra_c2.to_list())
        trb_seqs = set(trb_c1.to_list()) | set(trb_c2.to_list())

        # Paired count
        paired_count = df.filter(
            (pl.col(chain1_cdr3_col) != "") & (pl.col(chain2_cdr3_col) != "")
        ).height

        result["unique_tra"] = len(tra_seqs)
        result["unique_trb"] = len(trb_seqs)
        result["paired_tcrs"] = paired_count
    else:
        df = pd.read_csv(receptor_file, sep="\t", dtype=str, na_filter=False)
        result["total_records"] = len(df)

        # Vectorized operations
        tra_c1_mask = (df[chain1_type_col] == "alpha") & (df[chain1_cdr3_col] != "")
        tra_c2_mask = (df[chain2_type_col] == "alpha") & (df[chain2_cdr3_col] != "")
        trb_c1_mask = (df[chain1_type_col] == "beta") & (df[chain1_cdr3_col] != "")
        trb_c2_mask = (df[chain2_type_col] == "beta") & (df[chain2_cdr3_col] != "")

        tra_seqs = set(df.loc[tra_c1_mask, chain1_cdr3_col]) | set(df.loc[tra_c2_mask, chain2_cdr3_col])
        trb_seqs = set(df.loc[trb_c1_mask, chain1_cdr3_col]) | set(df.loc[trb_c2_mask, chain2_cdr3_col])

        paired_count = ((df[chain1_cdr3_col] != "") & (df[chain2_cdr3_col] != "")).sum()

        result["unique_tra"] = len(tra_seqs)
        result["unique_trb"] = len(trb_seqs)
        result["paired_tcrs"] = int(paired_count)

    return result


def analyze_iedb(db_path: Path) -> dict | None:
    """Optimized IEDB analysis."""
    iedb_dir = db_path / "IEDB"
    if not iedb_dir.exists():
        return None

    print("  Reading IEDB...")

    tcell_files = list(iedb_dir.glob("iedb_tcell_assay_results_table*.tsv"))
    mhcligand_files = list(iedb_dir.glob("iedb_mhcligand_assay_results_table*.tsv"))
    receptor_files = list(iedb_dir.glob("iedb_receptor_results_table*.tsv"))

    tcell_file = tcell_files[0] if tcell_files else None
    mhcligand_file = mhcligand_files[0] if mhcligand_files else None
    receptor_file = receptor_files[0] if receptor_files else None

    pmhc_results = _analyze_iedb_cedar_tcell_mhcligand_optimized(
        tcell_file, mhcligand_file, has_qualitative_in_mhcligand=True
    )

    receptor_results = {"total_records": 0, "unique_tra": 0, "unique_trb": 0, "paired_tcrs": 0}
    if receptor_file:
        receptor_results = _analyze_receptor_table_optimized(receptor_file, has_mhc_col=True)

    return {
        "tcell_positive": pmhc_results["tcell_positive"],
        "tcell_total": pmhc_results["tcell_total"],
        "mhcligand_total": pmhc_results["mhcligand_total"],
        "unique_peptide_mhc": pmhc_results["unique_peptide_mhc"],
        "receptor_total": receptor_results["total_records"],
        "unique_tra": receptor_results["unique_tra"],
        "unique_trb": receptor_results["unique_trb"],
        "paired_tcrs": receptor_results["paired_tcrs"],
    }


def analyze_cedar(db_path: Path) -> dict | None:
    """Optimized CEDAR analysis."""
    cedar_dir = db_path / "CEDAR"
    if not cedar_dir.exists():
        return None

    print("  Reading CEDAR...")

    tcell_files = list(cedar_dir.glob("cedar_tcell_assay_results_table*.tsv"))
    mhcligand_files = list(cedar_dir.glob("cedar_mhc_ligand_assay_results_table*.tsv"))
    receptor_files = list(cedar_dir.glob("cedar_receptor_results_table*.tsv"))

    tcell_file = tcell_files[0] if tcell_files else None
    mhcligand_file = mhcligand_files[0] if mhcligand_files else None
    receptor_file = receptor_files[0] if receptor_files else None

    pmhc_results = _analyze_iedb_cedar_tcell_mhcligand_optimized(
        tcell_file, mhcligand_file, has_qualitative_in_mhcligand=False
    )

    receptor_results = {"total_records": 0, "unique_tra": 0, "unique_trb": 0, "paired_tcrs": 0}
    if receptor_file:
        receptor_results = _analyze_receptor_table_optimized(receptor_file, has_mhc_col=False)

    return {
        "tcell_positive": pmhc_results["tcell_positive"],
        "tcell_total": pmhc_results["tcell_total"],
        "mhcligand_total": pmhc_results["mhcligand_total"],
        "unique_peptide_mhc": pmhc_results["unique_peptide_mhc"],
        "receptor_total": receptor_results["total_records"],
        "unique_tra": receptor_results["unique_tra"],
        "unique_trb": receptor_results["unique_trb"],
        "paired_tcrs": receptor_results["paired_tcrs"],
    }


def analyze_imgt(db_path: Path) -> dict | None:
    """Optimized IMGT/HLA analysis."""
    imgt_dir = db_path / "IMGTHLA" / "fasta"
    if not imgt_dir.exists():
        return None

    print("  Reading IMGT/HLA...")

    prot_files = list(imgt_dir.glob("*_prot.fasta"))
    if not prot_files:
        return None

    total_entries = 0
    alleles_full = set()
    alleles_4digit = set()
    alleles_2digit = set()

    for fasta_file in prot_files:
        with open(fasta_file) as f:
            for line in f:
                if line.startswith(">"):
                    total_entries += 1
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        allele_name = parts[1]
                        alleles_full.add(allele_name)

                        fields = allele_name.split(":")
                        if len(fields) >= 2:
                            alleles_4digit.add(":".join(fields[:2]))
                        if len(fields) >= 1:
                            alleles_2digit.add(fields[0])

    return {
        "total_entries": total_entries,
        "unique_alleles_full": len(alleles_full),
        "unique_alleles_4digit": len(alleles_4digit),
        "unique_alleles_2digit": len(alleles_2digit),
    }


def _process_tcrdb_file(tsv_file_str: str) -> tuple[int, int, list]:
    """Process a single TCRdb file. Returns (record_count, total_trb, unique_sequences as list)."""
    tsv_file = Path(tsv_file_str)
    try:
        df = pd.read_csv(tsv_file, sep="\t", usecols=["AASeq"], dtype=str, na_filter=False)
        seqs = [seq for seq in df["AASeq"] if seq]
        return len(df), len(seqs), seqs
    except Exception as e:
        print(f"    Error processing {tsv_file}: {e}")
        return 0, 0, []


def analyze_tcrdb(db_path: Path, workers: int = DEFAULT_WORKERS) -> dict | None:
    """Optimized TCRdb analysis with parallel file processing."""
    tcrdb_dir = db_path / "TCRdb"
    if not tcrdb_dir.exists():
        return None

    tsv_files = list(tcrdb_dir.glob("*.tsv"))
    if not tsv_files:
        return None

    print(f"  Reading TCRdb ({len(tsv_files)} files) with {workers} workers...")

    unique_trb = set()
    total_records = 0
    total_trb = 0

    # Convert to strings for pickling
    tsv_file_strs = [str(f) for f in tsv_files]

    with ProcessPoolExecutor(max_workers=min(workers, len(tsv_files))) as executor:
        results = list(tqdm(
            executor.map(_process_tcrdb_file, tsv_file_strs),
            total=len(tsv_file_strs),
            desc="  TCRdb files",
            leave=False
        ))

    for count, trb_count, seqs in results:
        total_records += count
        total_trb += trb_count
        unique_trb.update(seqs)

    return {
        "total_records": total_records,
        "total_tra": 0,
        "total_trb": total_trb,
        "unique_tra": 0,
        "unique_trb": len(unique_trb),
        "paired_tcrs": 0,
    }


def _process_ireceptor_file(tsv_file_str: str) -> tuple[int, int, int, int, list, list]:
    """Process a single iReceptor file.

    Returns (record_count, total_tra, total_trb, paired_cells,
             unique_tra as list, unique_trb as list).
    """
    tsv_file = Path(tsv_file_str)
    unique_tra = set()
    unique_trb = set()
    total_records = 0
    total_tra = 0
    total_trb = 0
    cell_chains: dict[str, set] = {}

    try:
        print(f"    Processing {tsv_file.name}...")
        chunk_iter = pd.read_csv(
            tsv_file,
            sep="\t",
            dtype=str,
            na_filter=False,
            chunksize=CHUNK_SIZE,
            usecols=lambda col: col in ["junction_aa", "locus", "cell_id"],
        )

        for chunk in chunk_iter:
            total_records += len(chunk)

            if "locus" in chunk.columns and "junction_aa" in chunk.columns:
                valid = chunk["junction_aa"] != ""
                tra_mask = valid & chunk["locus"].str.contains("TRA", na=False)
                trb_mask = valid & chunk["locus"].str.contains("TRB", na=False)

                tra_seqs = chunk.loc[tra_mask, "junction_aa"]
                trb_seqs = chunk.loc[trb_mask, "junction_aa"]

                total_tra += len(tra_seqs)
                total_trb += len(trb_seqs)
                unique_tra.update(tra_seqs)
                unique_trb.update(trb_seqs)

                # Track cell_id for pairing
                if "cell_id" in chunk.columns:
                    chain_mask = tra_mask | trb_mask
                    for _, row in chunk.loc[chain_mask, ["cell_id", "locus"]].iterrows():
                        cell_id = row["cell_id"]
                        if cell_id:
                            chain_type = "TRA" if "TRA" in row["locus"] else "TRB"
                            if cell_id not in cell_chains:
                                cell_chains[cell_id] = set()
                            cell_chains[cell_id].add(chain_type)
            elif "junction_aa" in chunk.columns:
                valid_seqs = [seq for seq in chunk["junction_aa"] if seq]
                total_trb += len(valid_seqs)
                unique_trb.update(valid_seqs)

        print(f"    Finished {tsv_file.name}: {total_records:,} records")
    except Exception as e:
        print(f"    Error processing {tsv_file.name}: {e}")

    paired_cells = sum(
        1 for chains in cell_chains.values() if "TRA" in chains and "TRB" in chains
    )

    return total_records, total_tra, total_trb, paired_cells, list(unique_tra), list(unique_trb)


def analyze_ireceptor(db_path: Path, workers: int = DEFAULT_WORKERS) -> dict | None:
    """Optimized iReceptor analysis with parallel file processing."""
    ireceptor_dir = db_path / "ireceptor"
    if not ireceptor_dir.exists():
        return None

    tsv_files = [f for f in ireceptor_dir.glob("*.tsv") if "metadata" not in f.name.lower()]
    if not tsv_files:
        return None

    print(f"  Reading iReceptor ({len(tsv_files)} files) with {workers} workers...")

    unique_tra = set()
    unique_trb = set()
    total_records = 0
    total_tra = 0
    total_trb = 0
    total_paired = 0

    # Convert to strings for pickling
    tsv_file_strs = [str(f) for f in tsv_files]

    with ProcessPoolExecutor(max_workers=min(workers, len(tsv_files))) as executor:
        results = list(tqdm(
            executor.map(_process_ireceptor_file, tsv_file_strs),
            total=len(tsv_file_strs),
            desc="  iReceptor files",
            leave=False
        ))

    for count, file_tra, file_trb, file_paired, tra, trb in results:
        total_records += count
        total_tra += file_tra
        total_trb += file_trb
        total_paired += file_paired
        unique_tra.update(tra)
        unique_trb.update(trb)

    return {
        "total_records": total_records,
        "total_tra": total_tra,
        "total_trb": total_trb,
        "unique_tra": len(unique_tra),
        "unique_trb": len(unique_trb),
        "paired_tcrs": total_paired,
    }


def _process_warren_file(csv_file_str: str) -> tuple[int, list]:
    """Process a single WarrenDB file. Returns (record_count, sequences as list)."""
    csv_file = Path(csv_file_str)
    try:
        df = pd.read_csv(csv_file, dtype=str, na_filter=False)
        seqs = []
        if "aminoAcid.beta" in df.columns:
            seqs = [seq for seq in df["aminoAcid.beta"] if seq]
        return len(df), seqs
    except Exception as e:
        print(f"    Error processing {csv_file}: {e}")
        return 0, []


def analyze_warrendb(db_path: Path, workers: int = DEFAULT_WORKERS) -> dict | None:
    """Optimized WarrenDB analysis with parallel file processing."""
    warren_dir = db_path / "WarrenDB"
    if not warren_dir.exists():
        return None

    csv_files = list(warren_dir.glob("*.csv"))
    if not csv_files:
        return None

    print(f"  Reading WarrenDB ({len(csv_files)} files) with {workers} workers...")

    unique_trb = set()
    total_records = 0

    # Convert to strings for pickling
    csv_file_strs = [str(f) for f in csv_files]

    with ProcessPoolExecutor(max_workers=min(workers, len(csv_files))) as executor:
        results = list(tqdm(
            executor.map(_process_warren_file, csv_file_strs),
            total=len(csv_file_strs),
            desc="  WarrenDB files",
            leave=False
        ))

    for count, seqs in results:
        total_records += count
        unique_trb.update(seqs)

    return {
        "total_records": total_records,
        "unique_trb": len(unique_trb),
    }


# --- Optimized Study Analyzers ---


def _parse_contigs_optimized(contigs_dir: Path) -> dict:
    """Optimized contig parsing with vectorized operations."""
    unique_tra = set()
    unique_trb = set()
    total_tra = 0
    total_trb = 0
    paired_barcodes = set()

    csv_files = list(contigs_dir.glob("*.csv"))

    for csv_file in csv_files:
        try:
            if HAS_POLARS:
                df = pl.read_csv(csv_file, infer_schema_length=0)

                if "chain" not in df.columns or "cdr3" not in df.columns:
                    continue

                if "productive" in df.columns:
                    df = df.filter(pl.col("productive").str.to_lowercase() == "true")

                tra_df = df.filter((pl.col("chain") == "TRA") & (pl.col("cdr3") != ""))
                trb_df = df.filter((pl.col("chain") == "TRB") & (pl.col("cdr3") != ""))

                total_tra += len(tra_df)
                total_trb += len(trb_df)
                unique_tra.update(tra_df["cdr3"].to_list())
                unique_trb.update(trb_df["cdr3"].to_list())

                if "barcode" in df.columns:
                    tra_barcodes = set(tra_df["barcode"].to_list())
                    trb_barcodes = set(trb_df["barcode"].to_list())
                    paired_barcodes.update(tra_barcodes & trb_barcodes)
            else:
                df = pd.read_csv(csv_file, dtype=str, na_filter=False)

                if "chain" not in df.columns or "cdr3" not in df.columns:
                    continue

                if "productive" in df.columns:
                    df = df[df["productive"].str.lower() == "true"]

                tra_mask = df["chain"] == "TRA"
                trb_mask = df["chain"] == "TRB"

                tra_valid = df.loc[tra_mask & (df["cdr3"] != ""), "cdr3"]
                trb_valid = df.loc[trb_mask & (df["cdr3"] != ""), "cdr3"]

                total_tra += len(tra_valid)
                total_trb += len(trb_valid)
                unique_tra.update(tra_valid)
                unique_trb.update(trb_valid)

                if "barcode" in df.columns:
                    tra_barcodes = set(df.loc[tra_mask & (df["cdr3"] != ""), "barcode"])
                    trb_barcodes = set(df.loc[trb_mask & (df["cdr3"] != ""), "barcode"])
                    paired_barcodes.update(tra_barcodes & trb_barcodes)
        except Exception:
            pass

    return {
        "data_type": "contigs",
        "total_tra": total_tra,
        "total_trb": total_trb,
        "unique_tra": len(unique_tra),
        "unique_trb": len(unique_trb),
        "paired": len(paired_barcodes),
    }


def _parse_clonotypes_optimized(clonotypes_dir: Path) -> dict:
    """Optimized clonotype parsing."""
    unique_tra = set()
    unique_trb = set()
    total_tra = 0
    total_trb = 0
    paired_count = 0

    csv_files = list(clonotypes_dir.glob("*.csv"))

    for csv_file in csv_files:
        try:
            if HAS_POLARS:
                df = pl.read_csv(csv_file, infer_schema_length=0)

                if "cdr3s_aa" not in df.columns:
                    continue

                freq_col = "frequency" if "frequency" in df.columns else None

                for row in df.iter_rows(named=True):
                    cdr3s_str = row["cdr3s_aa"]
                    if not cdr3s_str:
                        continue

                    freq = int(row[freq_col]) if freq_col and row.get(freq_col) else 1
                    chains = cdr3s_str.split(";")
                    has_tra = False
                    has_trb = False

                    for chain_entry in chains:
                        if ":" not in chain_entry:
                            continue
                        chain_type, cdr3 = chain_entry.split(":", 1)
                        chain_type = chain_type.strip()
                        cdr3 = cdr3.strip()

                        if chain_type == "TRA" and cdr3:
                            unique_tra.add(cdr3)
                            total_tra += freq
                            has_tra = True
                        elif chain_type == "TRB" and cdr3:
                            unique_trb.add(cdr3)
                            total_trb += freq
                            has_trb = True

                    if has_tra and has_trb:
                        paired_count += freq
            else:
                df = pd.read_csv(csv_file, dtype=str, na_filter=False)

                if "cdr3s_aa" not in df.columns:
                    continue

                freq_col = "frequency" if "frequency" in df.columns else None

                for _, row in df.iterrows():
                    cdr3s_str = row["cdr3s_aa"]
                    if not cdr3s_str:
                        continue

                    freq = int(row[freq_col]) if freq_col and row[freq_col] else 1
                    chains = cdr3s_str.split(";")
                    has_tra = False
                    has_trb = False

                    for chain_entry in chains:
                        if ":" not in chain_entry:
                            continue
                        chain_type, cdr3 = chain_entry.split(":", 1)
                        chain_type = chain_type.strip()
                        cdr3 = cdr3.strip()

                        if chain_type == "TRA" and cdr3:
                            unique_tra.add(cdr3)
                            total_tra += freq
                            has_tra = True
                        elif chain_type == "TRB" and cdr3:
                            unique_trb.add(cdr3)
                            total_trb += freq
                            has_trb = True

                    if has_tra and has_trb:
                        paired_count += freq
        except Exception:
            pass

    return {
        "data_type": "clonotypes",
        "total_tra": total_tra,
        "total_trb": total_trb,
        "unique_tra": len(unique_tra),
        "unique_trb": len(unique_trb),
        "paired": paired_count,
    }


def _parse_airr_optimized(airr_dir: Path) -> dict:
    """Optimized AIRR parsing with vectorized operations."""
    unique_tra = set()
    unique_trb = set()
    total_tra = 0
    total_trb = 0
    cell_chains: dict[str, set] = {}

    tsv_files = list(airr_dir.glob("*.tsv"))

    for tsv_file in tsv_files:
        try:
            if HAS_POLARS:
                df = pl.read_csv(tsv_file, separator="\t", infer_schema_length=0)

                if "junction_aa" not in df.columns:
                    continue

                # Vectorized chain type detection
                df = df.with_columns([
                    pl.when(
                        pl.col("locus").str.contains("TRA") |
                        pl.col("v_call").str.contains("TRAV")
                    ).then(pl.lit("TRA"))
                    .when(
                        pl.col("locus").str.contains("TRB") |
                        pl.col("v_call").str.contains("TRBV")
                    ).then(pl.lit("TRB"))
                    .otherwise(pl.lit(None))
                    .alias("chain_type")
                ])

                valid = df.filter(
                    (pl.col("junction_aa") != "") & (pl.col("chain_type").is_not_null())
                )

                tra_df = valid.filter(pl.col("chain_type") == "TRA")
                trb_df = valid.filter(pl.col("chain_type") == "TRB")

                total_tra += len(tra_df)
                total_trb += len(trb_df)
                unique_tra.update(tra_df["junction_aa"].to_list())
                unique_trb.update(trb_df["junction_aa"].to_list())

                # Track cell_id for pairing
                if "cell_id" in valid.columns:
                    for row in valid.select(["cell_id", "chain_type"]).iter_rows():
                        cell_id, chain_type = row
                        if cell_id:
                            if cell_id not in cell_chains:
                                cell_chains[cell_id] = set()
                            cell_chains[cell_id].add(chain_type)
            else:
                df = pd.read_csv(tsv_file, sep="\t", dtype=str, na_filter=False)

                if "junction_aa" not in df.columns:
                    continue

                # Vectorized chain type detection
                locus = df.get("locus", pd.Series([""] * len(df)))
                v_call = df.get("v_call", pd.Series([""] * len(df)))

                is_tra = locus.str.contains("TRA", na=False) | v_call.str.contains("TRAV", na=False)
                is_trb = locus.str.contains("TRB", na=False) | v_call.str.contains("TRBV", na=False)

                valid_cdr3 = df["junction_aa"] != ""

                tra_valid = df.loc[valid_cdr3 & is_tra, "junction_aa"]
                trb_valid = df.loc[valid_cdr3 & is_trb, "junction_aa"]

                total_tra += len(tra_valid)
                total_trb += len(trb_valid)
                unique_tra.update(tra_valid)
                unique_trb.update(trb_valid)

                # Track cell_id for pairing
                if "cell_id" in df.columns:
                    for _, row in df[valid_cdr3 & (is_tra | is_trb)].iterrows():
                        cell_id = row.get("cell_id", "")
                        if cell_id:
                            chain_type = "TRA" if row.get("locus", "").find("TRA") >= 0 or row.get("v_call", "").find("TRAV") >= 0 else "TRB"
                            if cell_id not in cell_chains:
                                cell_chains[cell_id] = set()
                            cell_chains[cell_id].add(chain_type)
        except Exception:
            pass

    paired_count = sum(
        1 for chains in cell_chains.values() if "TRA" in chains and "TRB" in chains
    )

    return {
        "data_type": "airr",
        "total_tra": total_tra,
        "total_trb": total_trb,
        "unique_tra": len(unique_tra),
        "unique_trb": len(unique_trb),
        "paired": paired_count,
    }


def _parse_bulk_survey_optimized(tcr_dir: Path) -> dict:
    """Optimized bulk survey parsing."""
    unique_tra = set()
    unique_trb = set()
    total_tra = 0
    total_trb = 0

    for subdir, chain in [("bulk_survey_tra", "TRA"), ("bulk_survey_trb", "TRB")]:
        survey_path = tcr_dir / subdir
        if not survey_path.exists():
            continue

        tsv_files = list(survey_path.glob("*.tsv"))
        for tsv_file in tsv_files:
            try:
                if HAS_POLARS:
                    df = pl.read_csv(tsv_file, separator="\t", columns=["aminoAcid"], infer_schema_length=0)
                    valid = df.filter(pl.col("aminoAcid") != "")["aminoAcid"]
                    if chain == "TRA":
                        total_tra += len(valid)
                        unique_tra.update(valid.to_list())
                    else:
                        total_trb += len(valid)
                        unique_trb.update(valid.to_list())
                else:
                    df = pd.read_csv(tsv_file, sep="\t", dtype=str, na_filter=False, usecols=["aminoAcid"])
                    valid = df["aminoAcid"][df["aminoAcid"] != ""]
                    if chain == "TRA":
                        total_tra += len(valid)
                        unique_tra.update(valid)
                    else:
                        total_trb += len(valid)
                        unique_trb.update(valid)
            except Exception:
                pass

    return {
        "data_type": "bulk_survey",
        "total_tra": total_tra,
        "total_trb": total_trb,
        "unique_tra": len(unique_tra),
        "unique_trb": len(unique_trb),
        "paired": 0,
    }


def analyze_study_paired(study_path: Path) -> dict | None:
    """Analyze a single study's TCR data."""
    tcr_dir = study_path / "tcr"
    if not tcr_dir.exists():
        return None

    if (tcr_dir / "contigs").exists() and any((tcr_dir / "contigs").glob("*.csv")):
        return _parse_contigs_optimized(tcr_dir / "contigs")
    elif (tcr_dir / "clonotypes").exists() and any((tcr_dir / "clonotypes").glob("*.csv")):
        return _parse_clonotypes_optimized(tcr_dir / "clonotypes")
    elif (tcr_dir / "airr").exists() and any((tcr_dir / "airr").glob("*.tsv")):
        return _parse_airr_optimized(tcr_dir / "airr")

    has_tra = (tcr_dir / "bulk_survey_tra").exists()
    has_trb = (tcr_dir / "bulk_survey_trb").exists()
    if has_tra or has_trb:
        return _parse_bulk_survey_optimized(tcr_dir)

    return None


def _process_study(args: tuple) -> tuple[str, str, dict | None]:
    """Process a single study (for parallel execution)."""
    category, study_dir_str = args
    study_dir = Path(study_dir_str)
    result = analyze_study_paired(study_dir)
    # Return study name for progress tracking
    return category, study_dir.name, result


def analyze_studies(raw_data_path: Path, workers: int = DEFAULT_WORKERS) -> dict:
    """Optimized study analysis with parallel processing."""
    print("\n  Analyzing studies...")

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

    results = {
        "total_studies_with_tcr": 0,
        "paired_studies": 0,
        "unpaired_studies": 0,
        "total_tra": 0,
        "total_trb": 0,
        "total_paired": 0,
        "by_category": {},
    }

    all_studies = []
    for category in study_categories:
        cat_path = raw_data_path / category
        if cat_path.exists():
            for study_dir in cat_path.iterdir():
                if study_dir.is_dir() and not study_dir.name.startswith("."):
                    all_studies.append((category, str(study_dir)))  # Convert to string for pickling

    print(f"  Processing {len(all_studies)} studies with {workers} workers...")

    from concurrent.futures import as_completed

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_study, s) for s in all_studies]

        for future in tqdm(as_completed(futures), total=len(futures), desc="  Studies", leave=True):
            category, study_name, study_result = future.result()

            if study_result is None:
                continue

            results["total_studies_with_tcr"] += 1

            if study_result["data_type"] in ("contigs", "clonotypes", "airr"):
                results["paired_studies"] += 1
            else:
                results["unpaired_studies"] += 1

            results["total_tra"] += study_result["total_tra"]
            results["total_trb"] += study_result["total_trb"]
            results["total_paired"] += study_result["paired"]

            if category not in results["by_category"]:
                results["by_category"][category] = {"studies": 0, "paired": 0, "unpaired": 0}
            results["by_category"][category]["studies"] += 1
            if study_result["data_type"] in ("contigs", "clonotypes", "airr"):
                results["by_category"][category]["paired"] += 1
            else:
                results["by_category"][category]["unpaired"] += 1

    return results


def _process_adaptive_file(args: tuple) -> tuple[str, int, list]:
    """Process a single adaptive bulk file. Returns (chain, count, sequences as list).

    Handles multiple column name formats for amino acid sequences:
    - amino_acid, aminoAcid, aaCDR3, CDR3 amino acid sequence, CDR3(aa), cdr3_b_aa
    """
    tsv_file_str, chain = args
    tsv_file = Path(tsv_file_str)

    # Possible column names for amino acid sequences
    AA_COLUMNS = [
        "amino_acid", "aminoAcid", "aaCDR3",
        "CDR3 amino acid sequence", "CDR3(aa)",
        "cdr3_b_aa", "cdr3_a_aa", "junction_aa"
    ]

    try:
        # First, read just the header to find which column exists
        df_header = pd.read_csv(tsv_file, sep="\t", nrows=0)
        if len(df_header.columns) <= 1:
            # Try comma delimiter
            df_header = pd.read_csv(tsv_file, nrows=0)

        if len(df_header.columns) == 0:
            return chain, 0, []  # Empty file

        # Find which AA column exists
        aa_col = None
        for col in AA_COLUMNS:
            if col in df_header.columns:
                aa_col = col
                break

        if aa_col is None:
            # No recognized AA column
            return chain, 0, []

        # Detect delimiter
        sep = "\t" if tsv_file.suffix == ".tsv" else ","

        # Read the file with the found column
        df = pd.read_csv(tsv_file, sep=sep, dtype=str, na_filter=False, usecols=[aa_col])
        valid = df[aa_col][(df[aa_col] != "") & (df[aa_col].notna())]
        return chain, len(valid), list(valid)
    except pd.errors.EmptyDataError:
        return chain, 0, []  # Empty file
    except Exception as e:
        # Only print for unexpected errors
        if "No columns to parse" not in str(e):
            print(f"    Warning: {tsv_file.name}: {e}")
        return chain, 0, []


def analyze_adaptive_bulk(raw_data_path: Path, workers: int = DEFAULT_WORKERS) -> dict | None:
    """Optimized adaptive bulk analysis with parallel processing."""
    adaptive_path = raw_data_path / "adaptive_bulk" / "bulk_data" / "tcr"
    if not adaptive_path.exists():
        return None

    print("  Reading adaptive bulk survey...")

    unique_tra = set()
    unique_trb = set()
    total_tra = 0
    total_trb = 0

    all_files = []
    for subdir, chain in [("bulk_survey_tra", "TRA"), ("bulk_survey_trb", "TRB")]:
        survey_path = adaptive_path / subdir
        if survey_path.exists():
            for f in survey_path.glob("*.tsv"):
                all_files.append((str(f), chain))  # Convert to string for pickling

    if not all_files:
        return None

    with ProcessPoolExecutor(max_workers=min(workers, len(all_files))) as executor:
        results = list(tqdm(
            executor.map(_process_adaptive_file, all_files),
            total=len(all_files),
            desc="  Adaptive files",
            leave=False
        ))

    for chain, count, seqs in results:
        if chain == "TRA":
            total_tra += count
            unique_tra.update(seqs)
        else:
            total_trb += count
            unique_trb.update(seqs)

    return {
        "total_tra": total_tra,
        "total_trb": total_trb,
        "unique_tra": len(unique_tra),
        "unique_trb": len(unique_trb),
        "tra_samples": len(list((adaptive_path / "bulk_survey_tra").glob("*.tsv")))
            if (adaptive_path / "bulk_survey_tra").exists() else 0,
        "trb_samples": len(list((adaptive_path / "bulk_survey_trb").glob("*.tsv")))
            if (adaptive_path / "bulk_survey_trb").exists() else 0,
    }


# --- Optimized Foundation Permutations ---


def _process_parquet_batch(parquet_file_strs: list[str]) -> tuple[int, dict]:
    """Process a batch of parquet files. Returns (records, counts dict)."""
    total_records = 0
    permutation_counts: dict[str, int] = {}

    for pq_file_str in parquet_file_strs:
        pq_file = Path(pq_file_str)
        try:
            table = pq.read_table(pq_file, columns=["permutation_key"])
            df = table.to_pandas()
            total_records += len(df)
            for key, count in df["permutation_key"].value_counts().items():
                permutation_counts[key] = permutation_counts.get(key, 0) + count
        except Exception as e:
            print(f"    Error processing {pq_file}: {e}")

    return total_records, permutation_counts


def analyze_foundation_permutations(foundation_path: Path, workers: int = DEFAULT_WORKERS) -> dict | None:
    """Optimized foundation permutations analysis with parallel batch processing."""
    if not foundation_path.exists():
        return None

    print("  Reading foundation_permutations...")

    parquet_files = list(foundation_path.glob("batch_*.parquet"))
    if not parquet_files:
        return None

    print(f"  Processing {len(parquet_files)} parquet files with {workers} workers...")

    # Convert to strings and split into batches for parallel processing
    parquet_file_strs = [str(f) for f in parquet_files]
    batch_size = max(1, len(parquet_file_strs) // workers)
    batches = [parquet_file_strs[i:i + batch_size] for i in range(0, len(parquet_file_strs), batch_size)]

    total_records = 0
    permutation_counts: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=min(workers, len(batches))) as executor:
        results = list(tqdm(
            executor.map(_process_parquet_batch, batches),
            total=len(batches),
            desc="  Parquet batches",
            leave=False
        ))

    for batch_records, batch_counts in results:
        total_records += batch_records
        for key, count in batch_counts.items():
            permutation_counts[key] = permutation_counts.get(key, 0) + count

    # Normalize permutation keys to combinations (sort fields alphabetically)
    # Pick one permutation's count per combination (max), not sum
    combination_counts: dict[str, int] = {}
    for key, count in permutation_counts.items():
        normalized = normalize_permutation_key(key)
        if normalized not in combination_counts or count > combination_counts[normalized]:
            combination_counts[normalized] = count

    return {
        "total_records": total_records,
        "num_parquet_files": len(parquet_files),
        "num_combinations": len(combination_counts),
        "by_combination": dict(sorted(combination_counts.items(), key=lambda x: -x[1])),
    }


# --- Combined Specificity ---


def analyze_combined_specificity(db_path: Path) -> dict | None:
    """
    Compute unique TCR-peptide-MHC interactions across VDJdb and McPAS-TCR.
    Returns counts for:
      - Unique TRA + epitope + MHC
      - Unique TRB + epitope + MHC
      - Unique TRA + TRB + epitope + MHC (paired)
    """
    tra_epitope_mhc = set()
    trb_epitope_mhc = set()
    paired_epitope_mhc = set()

    # --- VDJdb ---
    vdjdb_file = db_path / "VdjDB" / "vdjdb.txt"
    if vdjdb_file.exists():
        print("  Reading VDJdb for combined specificity...")
        df = pd.read_csv(vdjdb_file, sep="\t", dtype=str, na_filter=False, low_memory=False)
        df = df[(df["species"] == "HomoSapiens") & (df["vdjdb.score"] != "0")]

        grouped = df.groupby("complex.id")
        for cid, group in grouped:
            epitope = group["antigen.epitope"].iloc[0]
            mhc = group["mhc.a"].iloc[0]
            if not epitope or not mhc:
                continue

            genes = {}
            for _, row in group.iterrows():
                gene = row["gene"]
                cdr3 = row["cdr3"]
                if cdr3:
                    genes.setdefault(gene, []).append(cdr3)

            if cid == "0":
                for cdr3 in genes.get("TRA", []):
                    tra_epitope_mhc.add(f"{cdr3}|{epitope}|{mhc}")
                for cdr3 in genes.get("TRB", []):
                    trb_epitope_mhc.add(f"{cdr3}|{epitope}|{mhc}")
            else:
                tra_list = genes.get("TRA", [])
                trb_list = genes.get("TRB", [])
                for cdr3 in tra_list:
                    tra_epitope_mhc.add(f"{cdr3}|{epitope}|{mhc}")
                for cdr3 in trb_list:
                    trb_epitope_mhc.add(f"{cdr3}|{epitope}|{mhc}")
                if tra_list and trb_list:
                    for tra_cdr3 in tra_list:
                        for trb_cdr3 in trb_list:
                            paired_epitope_mhc.add(
                                f"{tra_cdr3}|{trb_cdr3}|{epitope}|{mhc}"
                            )

    # --- McPAS-TCR ---
    mcpas_file = db_path / "McPASDB" / "McPAS-TCR.csv"
    if mcpas_file.exists():
        print("  Reading McPAS-TCR for combined specificity...")
        df = pd.read_csv(mcpas_file, dtype=str, na_filter=False, low_memory=False)

        for _, row in df.iterrows():
            tra = row.get("CDR3.alpha.aa", "")
            trb = row.get("CDR3.beta.aa", "")
            epitope = row.get("Epitope.peptide", "")
            mhc = row.get("MHC", "")

            if not epitope or not mhc or epitope == "NA" or mhc == "NA":
                continue

            has_tra = tra and tra != "NA"
            has_trb = trb and trb != "NA"

            if has_tra:
                tra_epitope_mhc.add(f"{tra}|{epitope}|{mhc}")
            if has_trb:
                trb_epitope_mhc.add(f"{trb}|{epitope}|{mhc}")
            if has_tra and has_trb:
                paired_epitope_mhc.add(f"{tra}|{trb}|{epitope}|{mhc}")

    if not tra_epitope_mhc and not trb_epitope_mhc:
        return None

    return {
        "unique_tra_epitope_mhc": len(tra_epitope_mhc),
        "unique_trb_epitope_mhc": len(trb_epitope_mhc),
        "unique_paired_epitope_mhc": len(paired_epitope_mhc),
    }


# --- Report Generation ---


def generate_report(db_results: dict, study_results: dict | None,
                    adaptive_results: dict | None, output_path: Path,
                    foundation_results: dict | None = None,
                    combined_specificity: dict | None = None) -> None:
    """Generate markdown and JSON reports."""
    output_path.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 80,
        "TCR RAW DATA SOURCE SUMMARY",
        "=" * 80,
        "",
    ]

    # --- Specificity Databases ---
    lines.extend([
        "### SPECIFICITY DATABASES (TCR-peptide-MHC) ###",
        "",
    ])

    vdjdb = db_results.get("VDJdb")
    if vdjdb:
        lines.extend([
            "VDJdb:",
            f"  Total records (after filtering):       {vdjdb['total_records']:>10,}",
            f"  MHC Class I:",
            f"    Total TCR-pMHC records:              {vdjdb['class_i']['total']:>10,}",
            f"    Unique TCR-pMHC:                     {vdjdb['class_i']['unique_tcr_pmhc']:>10,}",
            f"  MHC Class II:",
            f"    Total TCR-pMHC records:              {vdjdb['class_ii']['total']:>10,}",
            f"    Unique TCR-pMHC:                     {vdjdb['class_ii']['unique_tcr_pmhc']:>10,}",
            f"  Paired TCRs (TRA+TRB):                 {vdjdb['paired_tcrs']:>10,}",
            f"  Unique TRA:                            {vdjdb['unique_tra']:>10,}",
            f"  Unique TRB:                            {vdjdb['unique_trb']:>10,}",
            "",
        ])

    mcpas = db_results.get("McPAS-TCR")
    if mcpas:
        lines.extend([
            "McPAS-TCR:",
            f"  Total records:                         {mcpas['total_records']:>10,}",
            f"  MHC Class I:",
            f"    Total TCR-pMHC records:              {mcpas['class_i']['total']:>10,}",
            f"    Unique TCR-pMHC:                     {mcpas['class_i']['unique_tcr_pmhc']:>10,}",
            f"  MHC Class II:",
            f"    Total TCR-pMHC records:              {mcpas['class_ii']['total']:>10,}",
            f"    Unique TCR-pMHC:                     {mcpas['class_ii']['unique_tcr_pmhc']:>10,}",
            f"  Paired TCRs (TRA+TRB):                 {mcpas['paired_tcrs']:>10,}",
            f"  Unique TRA:                            {mcpas['unique_tra']:>10,}",
            f"  Unique TRB:                            {mcpas['unique_trb']:>10,}",
            "",
        ])

    # --- Combined Specificity ---
    if combined_specificity:
        lines.extend([
            "### COMBINED SPECIFICITY (VDJdb + McPAS-TCR) ###",
            "",
            f"  Unique TRA + epitope + MHC:            {combined_specificity['unique_tra_epitope_mhc']:>10,}",
            f"  Unique TRB + epitope + MHC:            {combined_specificity['unique_trb_epitope_mhc']:>10,}",
            f"  Unique TRA+TRB + epitope + MHC:        {combined_specificity['unique_paired_epitope_mhc']:>10,}",
            "",
        ])

    lines.extend([
        "### PEPTIDE-MHC DATABASES ###",
        "",
    ])

    for db_name in ["IEDB", "CEDAR"]:
        db = db_results.get(db_name)
        if db:
            lines.extend([
                f"{db_name}:",
                f"  T-cell assay (positive):               {db['tcell_positive']:>10,}",
                f"  T-cell assay (total):                  {db['tcell_total']:>10,}",
                f"  MHC ligand assay:                      {db['mhcligand_total']:>10,}",
                f"  Unique peptide-MHC pairs:              {db['unique_peptide_mhc']:>10,}",
                f"  Receptor table:",
                f"    Total records:                       {db['receptor_total']:>10,}",
                f"    Unique TRA:                          {db['unique_tra']:>10,}",
                f"    Unique TRB:                          {db['unique_trb']:>10,}",
                f"    Paired (both chains):                {db['paired_tcrs']:>10,}",
                "",
            ])

    lines.extend([
        "### HLA ALLELE DATABASE ###",
        "",
    ])

    imgt = db_results.get("IMGT/HLA")
    if imgt:
        lines.extend([
            "IMGT/HLA:",
            f"  Total allele entries:                  {imgt['total_entries']:>10,}",
            f"  Unique alleles (full):                 {imgt['unique_alleles_full']:>10,}",
            f"  Unique alleles (4-digit):              {imgt['unique_alleles_4digit']:>10,}",
            f"  Unique alleles (2-digit):              {imgt['unique_alleles_2digit']:>10,}",
            "",
        ])

    lines.extend([
        "### TCR SEQUENCE DATABASES ###",
        "",
    ])

    for db_name in ["TCRdb", "iReceptor", "WarrenDB"]:
        db = db_results.get(db_name)
        if db:
            total = db.get("total_records", 0)
            total_tra = db.get("total_tra", 0)
            total_trb = db.get("total_trb", 0)
            unique_tra = db.get("unique_tra", 0)
            unique_trb = db.get("unique_trb", 0)
            paired = db.get("paired_tcrs", 0)

            lines.extend([
                f"{db_name}:",
                f"  Total records:                         {total:>10,}",
                f"  Total TRA:                             {total_tra:>10,}" if total_tra > 0
                    else f"  Total TRA:                                    -",
                f"  Total TRB:                             {total_trb:>10,}" if total_trb > 0
                    else f"  Total TRB:                                    -",
                f"  Unique TRA:                            {unique_tra:>10,}" if unique_tra > 0
                    else f"  Unique TRA:                                   -",
                f"  Unique TRB:                            {unique_trb:>10,}" if unique_trb > 0
                    else f"  Unique TRB:                                   -",
                f"  Paired TCRs (TRA+TRB by cell):         {paired:>10,}" if paired > 0
                    else f"  Paired TCRs (TRA+TRB by cell):                -",
                "",
            ])

    lines.extend(["", ""])

    lines.extend([
        "### PUBLIC STUDIES ###",
        "",
    ])

    if study_results:
        lines.extend([
            f"Total studies with TCR data:             {study_results['total_studies_with_tcr']:>10,}",
            f"  Paired studies (contigs/clonotypes/airr): {study_results['paired_studies']:>7,}",
            f"  Unpaired studies (bulk survey only):    {study_results['unpaired_studies']:>10,}",
            "",
            f"  Total TRA records:                     {study_results['total_tra']:>10,}",
            f"  Total TRB records:                     {study_results['total_trb']:>10,}",
            f"  Total paired TRA-TRB:                  {study_results['total_paired']:>10,}",
            "",
        ])

        if study_results.get("by_category"):
            lines.append("  By category:")
            for cat, counts in sorted(study_results["by_category"].items()):
                lines.append(
                    f"    {cat:<30} {counts['studies']:>3} studies "
                    f"({counts['paired']} paired, {counts['unpaired']} unpaired)"
                )
            lines.append("")

    if adaptive_results:
        lines.extend([
            "### ADAPTIVE BULK SURVEY ###",
            "",
            f"  TRA samples:                           {adaptive_results['tra_samples']:>10,}",
            f"  TRB samples:                           {adaptive_results['trb_samples']:>10,}",
            f"  Total TRA records:                     {adaptive_results['total_tra']:>10,}",
            f"  Total TRB records:                     {adaptive_results['total_trb']:>10,}",
            f"  Unique TRA:                            {adaptive_results['unique_tra']:>10,}",
            f"  Unique TRB:                            {adaptive_results['unique_trb']:>10,}",
            "",
        ])

    if foundation_results:
        lines.extend([
            "### FOUNDATION COMBINATIONS ###",
            "",
            f"  Parquet files:                         {foundation_results['num_parquet_files']:>10,}",
            f"  Total records:                         {foundation_results['total_records']:>10,}",
            f"  Unique combinations:                   {foundation_results['num_combinations']:>10,}",
            "",
        ])
        if foundation_results.get("by_combination"):
            lines.append("  Records by field combination:")
            for key, count in foundation_results["by_combination"].items():
                lines.append(f"    {key:<30} {count:>15,}")
            lines.append("")

    lines.extend([
        "=" * 80,
        "",
    ])

    md_file = output_path / "tcr_source_summary.md"
    with open(md_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown report written to: {md_file}")

    # Write foundation summary if available
    if foundation_results:
        foundation_summary_file = output_path / "foundation_combinations_summary.txt"
        with open(foundation_summary_file, "w") as f:
            f.write("FOUNDATION COMBINATIONS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total parquet files:     {foundation_results['num_parquet_files']:>15,}\n")
            f.write(f"Total records:           {foundation_results['total_records']:>15,}\n")
            f.write(f"Unique combinations:     {foundation_results['num_combinations']:>15,}\n")
            f.write("\n" + "-" * 50 + "\n")

            if foundation_results.get("by_combination"):
                sorted_combinations = sorted(
                    foundation_results["by_combination"].items(),
                    key=lambda x: -x[1]
                )

                f.write("COMBINATIONS (sorted by record count):\n")
                f.write("-" * 50 + "\n")
                for key, count in sorted_combinations:
                    f.write(f"  {key:<30} {count:>15,}\n")
                f.write("\n")
                f.write("Note: Keys are normalized alphabetically (e.g., tra_trb and\n")
                f.write("      trb_tra are both reported as 'tra+trb').\n")

            f.write("\n" + "=" * 50 + "\n")
        print(f"Foundation summary written to: {foundation_summary_file}")

    json_data = {
        "databases": convert_to_native(db_results),
        "combined_specificity": convert_to_native(combined_specificity) if combined_specificity else None,
        "studies": convert_to_native(study_results) if study_results else None,
        "adaptive_bulk": convert_to_native(adaptive_results) if adaptive_results else None,
        "foundation_combinations": convert_to_native(foundation_results) if foundation_results else None,
    }

    json_file = output_path / "tcr_source_summary.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON report written to: {json_file}")

    print("\n" + "\n".join(lines))


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TCR raw data sources (optimized for p4d.24xlarge)."
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
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
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
        "--foundation_path",
        type=Path,
        default=None,
        help="Path to foundation_permutations directory (default: None)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if available",
    )

    args = parser.parse_args()

    if not args.raw_data_path.exists():
        print(f"Error: Raw data path does not exist: {args.raw_data_path}")
        return 1

    print("=" * 60)
    print("TCR Raw Data Source Analysis (Optimized)")
    print(f"Workers: {args.workers}")
    print(f"Polars available: {HAS_POLARS}")
    print(f"Resume mode: {args.resume}")
    print("=" * 60)

    # Checkpoint files
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / ".checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    db_checkpoint = checkpoint_dir / "step1_databases.json"
    studies_checkpoint = checkpoint_dir / "step2_studies.json"
    adaptive_checkpoint = checkpoint_dir / "step2_adaptive.json"
    foundation_checkpoint = checkpoint_dir / "step3_foundation.json"

    db_path = args.raw_data_path / "databases"

    # --- Step 1: Database analysis ---
    db_results = None
    if args.resume:
        db_results = load_checkpoint(db_checkpoint)
        if db_results:
            print("\nStep 1: Loaded databases from checkpoint")

    if db_results is None:
        print("\nStep 1: Analyzing databases...")
        db_results = {}

        db_results["VDJdb"] = analyze_vdjdb(db_path)
        db_results["McPAS-TCR"] = analyze_mcpas(db_path)
        db_results["IEDB"] = analyze_iedb(db_path)
        db_results["CEDAR"] = analyze_cedar(db_path)
        db_results["IMGT/HLA"] = analyze_imgt(db_path)
        db_results["TCRdb"] = analyze_tcrdb(db_path, args.workers)
        db_results["WarrenDB"] = analyze_warrendb(db_path, args.workers)

        if not args.skip_ireceptor:
            db_results["iReceptor"] = analyze_ireceptor(db_path, args.workers)
        else:
            print("  Skipping iReceptor (--skip_ireceptor flag)")
            db_results["iReceptor"] = None

        save_checkpoint(db_checkpoint, db_results)

    # --- Step 2: Study analysis ---
    study_results = None
    adaptive_results = None

    if not args.skip_studies:
        # Check for study checkpoint
        if args.resume:
            study_results = load_checkpoint(studies_checkpoint)
            adaptive_results = load_checkpoint(adaptive_checkpoint)
            if study_results and adaptive_results:
                print("\nStep 2: Loaded studies from checkpoint")

        if study_results is None:
            print("\nStep 2: Analyzing studies...")
            study_results = analyze_studies(args.raw_data_path, args.workers)
            save_checkpoint(studies_checkpoint, study_results)

        if adaptive_results is None:
            print("  Analyzing adaptive bulk...")
            adaptive_results = analyze_adaptive_bulk(args.raw_data_path, args.workers)
            if adaptive_results:
                save_checkpoint(adaptive_checkpoint, adaptive_results)
    else:
        print("\nStep 2: Skipping study analysis")

    # --- Step 3: Combined specificity analysis ---
    combined_specificity_checkpoint = checkpoint_dir / "step3_combined_specificity.json"
    combined_specificity = None
    if args.resume:
        combined_specificity = load_checkpoint(combined_specificity_checkpoint)
        if combined_specificity:
            print("\nStep 3: Loaded combined specificity from checkpoint")

    if combined_specificity is None:
        print("\nStep 3: Analyzing combined specificity (VDJdb + McPAS-TCR)...")
        combined_specificity = analyze_combined_specificity(db_path)
        if combined_specificity:
            save_checkpoint(combined_specificity_checkpoint, combined_specificity)

    # --- Step 4: Foundation combinations analysis ---
    foundation_results = None
    if args.foundation_path:
        if args.resume:
            foundation_results = load_checkpoint(foundation_checkpoint)
            if foundation_results:
                print("\nStep 4: Loaded foundation combinations from checkpoint")

        if foundation_results is None:
            print("\nStep 4: Analyzing foundation combinations...")
            foundation_results = analyze_foundation_permutations(args.foundation_path, args.workers)
            if foundation_results:
                save_checkpoint(foundation_checkpoint, foundation_results)
    else:
        print("\nStep 4: Skipping foundation combinations (no path provided)")

    # --- Generate report ---
    print("\nStep 5: Generating reports...")
    generate_report(db_results, study_results, adaptive_results, args.output_dir,
                    foundation_results, combined_specificity)

    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
