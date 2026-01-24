#!/usr/bin/env python3
"""
TCR Raw Data Source Summary

Analyzes raw TCR data to report:
- Database statistics: VDJdb, McPAS-TCR, IEDB, CEDAR, IMGT/HLA, TCRdb, iReceptor, WarrenDB
- Study breakdown: paired vs unpaired, TCR counts per category
- Adaptive bulk survey counts
"""

import argparse
import json
import re
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


# --- Database analyzers ---


def analyze_vdjdb(db_path: Path) -> dict | None:
    """
    Parse vdjdb.txt: filter HomoSapiens + score != 0, split by mhc.class,
    count paired via complex.id.
    """
    vdjdb_file = db_path / "VdjDB" / "vdjdb.txt"
    if not vdjdb_file.exists():
        return None

    print("  Reading VDJdb...")
    df = pd.read_csv(vdjdb_file, sep="\t", dtype=str, na_filter=False, low_memory=False)

    # Filter: HomoSapiens and score != 0
    df = df[(df["species"] == "HomoSapiens") & (df["vdjdb.score"] != "0")]

    # Build TCR-pMHC key
    df["tcr_pmhc"] = (
        df["cdr3"] + "|" + df["antigen.epitope"] + "|" + df["mhc.a"] + "|" + df["mhc.b"]
    )

    # Split by MHC class
    class_i = df[df["mhc.class"] == "MHCI"]
    class_ii = df[df["mhc.class"] == "MHCII"]

    # Paired TCRs via complex.id (non-zero, has both TRA and TRB)
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
    """
    Parse McPAS-TCR.csv. Infer MHC class from MHC column:
    HLA-A/B/C = class I, HLA-D* = class II.
    """
    mcpas_file = db_path / "McPASDB" / "McPAS-TCR.csv"
    if not mcpas_file.exists():
        return None

    print("  Reading McPAS-TCR...")
    df = pd.read_csv(mcpas_file, dtype=str, na_filter=False, low_memory=False)

    alpha_col = "CDR3.alpha.aa"
    beta_col = "CDR3.beta.aa"
    epitope_col = "Epitope.peptide"
    mhc_col = "MHC"

    # Build TCR-pMHC key
    df["tcr_pmhc"] = (
        df[alpha_col] + "|" + df[beta_col] + "|" + df[epitope_col] + "|" + df[mhc_col]
    )

    # Infer MHC class
    def classify_mhc(mhc_val: str) -> str:
        if not mhc_val or mhc_val == "NA":
            return "unknown"
        mhc_upper = mhc_val.upper()
        if re.search(r"HLA-[ABC]", mhc_upper):
            return "MHCI"
        elif re.search(r"HLA-D", mhc_upper):
            return "MHCII"
        return "unknown"

    df["mhc_class"] = df[mhc_col].apply(classify_mhc)

    class_i = df[df["mhc_class"] == "MHCI"]
    class_ii = df[df["mhc_class"] == "MHCII"]

    # Paired: both alpha and beta present (non-empty, non-NA)
    has_alpha = (df[alpha_col] != "") & (df[alpha_col] != "NA")
    has_beta = (df[beta_col] != "") & (df[beta_col] != "NA")
    paired_count = int((has_alpha & has_beta).sum())

    # Unique chains (exclude empty/NA)
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


def _analyze_iedb_cedar_tcell_mhcligand(
    tcell_file: Path | None,
    mhcligand_file: Path | None,
    has_qualitative_in_mhcligand: bool = True,
) -> dict:
    """
    Shared logic for IEDB/CEDAR tcell + mhc_ligand assay files.
    Returns peptide-MHC association counts.
    """
    result = {
        "tcell_positive": 0,
        "tcell_total": 0,
        "mhcligand_total": 0,
        "unique_peptide_mhc": 0,
    }

    peptide_mhc_pairs = set()

    # T-cell assay
    if tcell_file and tcell_file.exists():
        tcell_df = pd.read_csv(tcell_file, sep="\t", dtype=str, na_filter=False, low_memory=False)
        result["tcell_total"] = len(tcell_df)

        # Filter positive results
        qual_col = "Assay - Qualitative Measurement"
        if qual_col in tcell_df.columns:
            positive_mask = tcell_df[qual_col].str.startswith("Positive")
            result["tcell_positive"] = int(positive_mask.sum())
            pos_df = tcell_df[positive_mask]
        else:
            pos_df = tcell_df

        # Collect peptide-MHC pairs from positive results
        epitope_col = "Epitope - Name"
        mhc_col = "MHC Restriction - Name"
        if epitope_col in pos_df.columns and mhc_col in pos_df.columns:
            for _, row in pos_df[[epitope_col, mhc_col]].iterrows():
                ep = row[epitope_col]
                mhc = row[mhc_col]
                if ep and mhc:
                    peptide_mhc_pairs.add(f"{ep}|{mhc}")

    # MHC ligand assay
    if mhcligand_file and mhcligand_file.exists():
        mhc_df = pd.read_csv(mhcligand_file, sep="\t", dtype=str, na_filter=False, low_memory=False)
        result["mhcligand_total"] = len(mhc_df)

        # Filter positive if column exists
        qual_col = "Assay - Qualitative Measurement"
        if has_qualitative_in_mhcligand and qual_col in mhc_df.columns:
            positive_mask = mhc_df[qual_col].str.startswith("Positive")
            pos_mhc_df = mhc_df[positive_mask]
        else:
            pos_mhc_df = mhc_df

        epitope_col = "Epitope - Name"
        mhc_col = "MHC Restriction - Name"
        if epitope_col in pos_mhc_df.columns and mhc_col in pos_mhc_df.columns:
            for _, row in pos_mhc_df[[epitope_col, mhc_col]].iterrows():
                ep = row[epitope_col]
                mhc = row[mhc_col]
                if ep and mhc:
                    peptide_mhc_pairs.add(f"{ep}|{mhc}")

    result["unique_peptide_mhc"] = len(peptide_mhc_pairs)
    return result


def _analyze_receptor_table(receptor_file: Path, has_mhc_col: bool = True) -> dict:
    """
    Shared logic for IEDB/CEDAR receptor results tables.
    IEDB has 'Assay - MHC Allele Names'; CEDAR does not.
    """
    result = {
        "total_records": 0,
        "unique_tra": 0,
        "unique_trb": 0,
        "paired_tcrs": 0,
    }

    if not receptor_file.exists():
        return result

    df = pd.read_csv(receptor_file, sep="\t", dtype=str, na_filter=False, low_memory=False)
    result["total_records"] = len(df)

    chain1_type_col = "Chain 1 - Type"
    chain1_cdr3_col = "Chain 1 - CDR3 Calculated"
    chain2_type_col = "Chain 2 - Type"
    chain2_cdr3_col = "Chain 2 - CDR3 Calculated"

    tra_seqs = set()
    trb_seqs = set()
    paired_count = 0

    for _, row in df.iterrows():
        c1_type = row.get(chain1_type_col, "")
        c1_cdr3 = row.get(chain1_cdr3_col, "")
        c2_type = row.get(chain2_type_col, "")
        c2_cdr3 = row.get(chain2_cdr3_col, "")

        has_c1 = c1_cdr3 != ""
        has_c2 = c2_cdr3 != ""

        if has_c1:
            if c1_type == "alpha":
                tra_seqs.add(c1_cdr3)
            elif c1_type == "beta":
                trb_seqs.add(c1_cdr3)

        if has_c2:
            if c2_type == "alpha":
                tra_seqs.add(c2_cdr3)
            elif c2_type == "beta":
                trb_seqs.add(c2_cdr3)

        if has_c1 and has_c2:
            paired_count += 1

    result["unique_tra"] = len(tra_seqs)
    result["unique_trb"] = len(trb_seqs)
    result["paired_tcrs"] = paired_count

    return result


def analyze_iedb(db_path: Path) -> dict | None:
    """
    Parse IEDB: tcell assay, mhc_ligand assay, and receptor results.
    """
    iedb_dir = db_path / "IEDB"
    if not iedb_dir.exists():
        return None

    print("  Reading IEDB...")

    # Find files
    tcell_files = list(iedb_dir.glob("iedb_tcell_assay_results_table*.tsv"))
    mhcligand_files = list(iedb_dir.glob("iedb_mhcligand_assay_results_table*.tsv"))
    receptor_files = list(iedb_dir.glob("iedb_receptor_results_table*.tsv"))

    tcell_file = tcell_files[0] if tcell_files else None
    mhcligand_file = mhcligand_files[0] if mhcligand_files else None
    receptor_file = receptor_files[0] if receptor_files else None

    # Peptide-MHC from tcell + mhc_ligand
    pmhc_results = _analyze_iedb_cedar_tcell_mhcligand(
        tcell_file, mhcligand_file, has_qualitative_in_mhcligand=True
    )

    # TCR from receptor
    receptor_results = {"total_records": 0, "unique_tra": 0, "unique_trb": 0, "paired_tcrs": 0}
    if receptor_file:
        receptor_results = _analyze_receptor_table(receptor_file, has_mhc_col=True)

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
    """
    Parse CEDAR: tcell assay, mhc_ligand assay, and receptor results.
    Key difference from IEDB: receptor table has no 'Assay - MHC Allele Names';
    mhc_ligand has no 'Assay - Qualitative Measurement'.
    """
    cedar_dir = db_path / "CEDAR"
    if not cedar_dir.exists():
        return None

    print("  Reading CEDAR...")

    # Find files
    tcell_files = list(cedar_dir.glob("cedar_tcell_assay_results_table*.tsv"))
    mhcligand_files = list(cedar_dir.glob("cedar_mhc_ligand_assay_results_table*.tsv"))
    receptor_files = list(cedar_dir.glob("cedar_receptor_results_table*.tsv"))

    tcell_file = tcell_files[0] if tcell_files else None
    mhcligand_file = mhcligand_files[0] if mhcligand_files else None
    receptor_file = receptor_files[0] if receptor_files else None

    # Peptide-MHC from tcell + mhc_ligand
    # CEDAR mhc_ligand does NOT have 'Assay - Qualitative Measurement'
    pmhc_results = _analyze_iedb_cedar_tcell_mhcligand(
        tcell_file, mhcligand_file, has_qualitative_in_mhcligand=False
    )

    # TCR from receptor (no MHC allele column in CEDAR)
    receptor_results = {"total_records": 0, "unique_tra": 0, "unique_trb": 0, "paired_tcrs": 0}
    if receptor_file:
        receptor_results = _analyze_receptor_table(receptor_file, has_mhc_col=False)

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
    """
    Parse IMGT/HLA protein FASTA files.
    Headers: >HLA:HLA00001 A*01:01:01:01 365 bp
    Report total alleles, unique at 4-digit and 2-digit resolution.
    """
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
                    # Parse header: >HLA:HLA00001 A*01:01:01:01 365 bp
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        allele_name = parts[1]
                        alleles_full.add(allele_name)

                        # Extract resolution levels
                        # e.g., A*01:01:01:01 -> 4-digit: A*01:01, 2-digit: A*01
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


def analyze_tcrdb(db_path: Path) -> dict | None:
    """
    Aggregate TRB CDR3 sequences from TCRdb TSV files.
    Column: AASeq
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
            df = pd.read_csv(tsv_file, sep="\t", usecols=["AASeq"], dtype=str,
                             na_filter=False, low_memory=False)
            total_records += len(df)
            unique_trb.update(seq for seq in df["AASeq"] if seq)
        except Exception as e:
            print(f"    Warning: Could not read {tsv_file.name}: {e}")

    return {
        "total_records": total_records,
        "unique_trb": len(unique_trb),
    }


def analyze_ireceptor(db_path: Path) -> dict | None:
    """
    Parse iReceptor TSV files (AIRR format) using chunked reading.
    Columns: junction_aa, locus
    """
    ireceptor_dir = db_path / "ireceptor"
    if not ireceptor_dir.exists():
        return None

    tsv_files = [f for f in ireceptor_dir.glob("*.tsv") if "metadata" not in f.name.lower()]
    if not tsv_files:
        return None

    print(f"  Reading iReceptor ({len(tsv_files)} files)...")
    print("    Note: Large files - using chunked reading...")

    unique_tra = set()
    unique_trb = set()
    total_records = 0

    for tsv_file in tqdm(tsv_files, desc="  iReceptor files", leave=False):
        try:
            chunk_iter = pd.read_csv(
                tsv_file,
                sep="\t",
                dtype=str,
                na_filter=False,
                low_memory=False,
                chunksize=500000,
                usecols=lambda col: col in ["junction_aa", "locus"],
            )

            for chunk in chunk_iter:
                total_records += len(chunk)

                if "locus" in chunk.columns and "junction_aa" in chunk.columns:
                    tra_mask = chunk["locus"].str.contains("TRA", na=False)
                    trb_mask = chunk["locus"].str.contains("TRB", na=False)

                    unique_tra.update(
                        seq for seq in chunk.loc[tra_mask, "junction_aa"] if seq
                    )
                    unique_trb.update(
                        seq for seq in chunk.loc[trb_mask, "junction_aa"] if seq
                    )
                elif "junction_aa" in chunk.columns:
                    unique_trb.update(seq for seq in chunk["junction_aa"] if seq)

        except Exception as e:
            print(f"    Warning: Could not read {tsv_file.name}: {e}")

    return {
        "total_records": total_records,
        "unique_tra": len(unique_tra),
        "unique_trb": len(unique_trb),
    }


def analyze_warrendb(db_path: Path) -> dict | None:
    """
    Parse WarrenDB CSV files. Column: aminoAcid.beta (TRB CDR3).
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
            df = pd.read_csv(csv_file, dtype=str, na_filter=False, low_memory=False)
            total_records += len(df)
            if "aminoAcid.beta" in df.columns:
                unique_trb.update(seq for seq in df["aminoAcid.beta"] if seq)
        except Exception as e:
            print(f"    Warning: Could not read {csv_file.name}: {e}")

    return {
        "total_records": total_records,
        "unique_trb": len(unique_trb),
    }


# --- Study analyzers ---


def _parse_contigs(contigs_dir: Path) -> dict:
    """
    Parse 10x filtered_contig_annotations CSVs.
    Paired = barcodes with both TRA and TRB productive chains.
    """
    unique_tra = set()
    unique_trb = set()
    total_tra = 0
    total_trb = 0
    paired_barcodes = set()

    csv_files = list(contigs_dir.glob("*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, dtype=str, na_filter=False, low_memory=False)

            if "chain" not in df.columns or "cdr3" not in df.columns:
                continue

            # Filter productive
            if "productive" in df.columns:
                df = df[df["productive"].str.lower() == "true"]

            tra_mask = df["chain"] == "TRA"
            trb_mask = df["chain"] == "TRB"

            tra_cdr3s = df.loc[tra_mask, "cdr3"]
            trb_cdr3s = df.loc[trb_mask, "cdr3"]

            tra_valid = tra_cdr3s[tra_cdr3s != ""]
            trb_valid = trb_cdr3s[trb_cdr3s != ""]

            total_tra += len(tra_valid)
            total_trb += len(trb_valid)
            unique_tra.update(tra_valid)
            unique_trb.update(trb_valid)

            # Paired via barcode
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


def _parse_clonotypes(clonotypes_dir: Path) -> dict:
    """
    Parse 10x clonotype CSVs. cdr3s_aa format: TRB:CASSXX;TRA:CAVXX
    """
    unique_tra = set()
    unique_trb = set()
    total_tra = 0
    total_trb = 0
    paired_count = 0

    csv_files = list(clonotypes_dir.glob("*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, dtype=str, na_filter=False, low_memory=False)

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


def _parse_airr(airr_dir: Path) -> dict:
    """
    Parse AIRR rearrangement TSVs. Paired via cell_id.
    Locus inferred from v_call (TRAV/TRBV).
    """
    unique_tra = set()
    unique_trb = set()
    total_tra = 0
    total_trb = 0
    cell_chains: dict[str, set] = {}  # cell_id -> set of chain types

    tsv_files = list(airr_dir.glob("*.tsv"))
    for tsv_file in tsv_files:
        try:
            df = pd.read_csv(tsv_file, sep="\t", dtype=str, na_filter=False, low_memory=False)

            if "junction_aa" not in df.columns:
                continue

            # Determine locus from v_call or locus column
            for _, row in df.iterrows():
                cdr3 = row.get("junction_aa", "")
                if not cdr3:
                    continue

                # Determine chain type
                locus = row.get("locus", "")
                v_call = row.get("v_call", "")
                chain_type = None

                if "TRA" in locus or "TRAV" in v_call:
                    chain_type = "TRA"
                elif "TRB" in locus or "TRBV" in v_call:
                    chain_type = "TRB"

                if chain_type == "TRA":
                    unique_tra.add(cdr3)
                    total_tra += 1
                elif chain_type == "TRB":
                    unique_trb.add(cdr3)
                    total_trb += 1

                # Track cell_id for pairing
                cell_id = row.get("cell_id", "")
                if cell_id and chain_type:
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


def _parse_bulk_survey(tcr_dir: Path) -> dict:
    """
    Parse Adaptive bulk survey TSVs (bulk_survey_tra / bulk_survey_trb).
    Column: aminoAcid (for study-level bulk).
    """
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
                df = pd.read_csv(tsv_file, sep="\t", dtype=str, na_filter=False,
                                 low_memory=False, usecols=["aminoAcid"])
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
    """
    Analyze a single study's TCR data using priority:
    contigs > clonotypes > airr > bulk_survey
    """
    tcr_dir = study_path / "tcr"
    if not tcr_dir.exists():
        return None

    # Priority: contigs > clonotypes > airr
    if (tcr_dir / "contigs").exists() and any((tcr_dir / "contigs").glob("*.csv")):
        return _parse_contigs(tcr_dir / "contigs")
    elif (tcr_dir / "clonotypes").exists() and any((tcr_dir / "clonotypes").glob("*.csv")):
        return _parse_clonotypes(tcr_dir / "clonotypes")
    elif (tcr_dir / "airr").exists() and any((tcr_dir / "airr").glob("*.tsv")):
        return _parse_airr(tcr_dir / "airr")

    # Unpaired: bulk survey
    has_tra = (tcr_dir / "bulk_survey_tra").exists()
    has_trb = (tcr_dir / "bulk_survey_trb").exists()
    if has_tra or has_trb:
        return _parse_bulk_survey(tcr_dir)

    return None


def analyze_studies(raw_data_path: Path) -> dict:
    """
    Analyze TCR data across all study categories.
    """
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
                    all_studies.append((category, study_dir))

    for category, study_dir in tqdm(all_studies, desc="  Processing studies"):
        study_result = analyze_study_paired(study_dir)
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

        # Category tracking
        if category not in results["by_category"]:
            results["by_category"][category] = {
                "studies": 0, "paired": 0, "unpaired": 0,
            }
        results["by_category"][category]["studies"] += 1
        if study_result["data_type"] in ("contigs", "clonotypes", "airr"):
            results["by_category"][category]["paired"] += 1
        else:
            results["by_category"][category]["unpaired"] += 1

    return results


def analyze_adaptive_bulk(raw_data_path: Path) -> dict | None:
    """
    Analyze adaptive bulk survey at data/raw_data/adaptive_bulk/bulk_data/tcr/.
    Column: amino_acid
    """
    adaptive_path = raw_data_path / "adaptive_bulk" / "bulk_data" / "tcr"
    if not adaptive_path.exists():
        return None

    print("  Reading adaptive bulk survey...")

    unique_tra = set()
    unique_trb = set()
    total_tra = 0
    total_trb = 0

    for subdir, chain in [("bulk_survey_tra", "TRA"), ("bulk_survey_trb", "TRB")]:
        survey_path = adaptive_path / subdir
        if not survey_path.exists():
            continue

        tsv_files = list(survey_path.glob("*.tsv"))
        for tsv_file in tqdm(tsv_files, desc=f"  Adaptive {chain}", leave=False):
            try:
                df = pd.read_csv(tsv_file, sep="\t", dtype=str, na_filter=False,
                                 low_memory=False, usecols=["amino_acid"])
                valid = df["amino_acid"][df["amino_acid"] != ""]
                if chain == "TRA":
                    total_tra += len(valid)
                    unique_tra.update(valid)
                else:
                    total_trb += len(valid)
                    unique_trb.update(valid)
            except Exception:
                pass

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


# --- Report ---


def generate_report(db_results: dict, study_results: dict | None,
                    adaptive_results: dict | None, output_path: Path) -> None:
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

    # VDJdb
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

    # McPAS-TCR
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

    # --- Peptide-MHC Databases ---
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

    # --- HLA Allele Database ---
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

    # --- TCR Sequence Databases ---
    lines.extend([
        "### TCR SEQUENCE DATABASES ###",
        "",
        f"{'Database':<15} {'Total records':>15} {'Unique TRA':>12} {'Unique TRB':>12}",
        "-" * 60,
    ])

    for db_name in ["TCRdb", "iReceptor", "WarrenDB"]:
        db = db_results.get(db_name)
        if db:
            total = db.get("total_records", 0)
            tra = db.get("unique_tra", 0)
            trb = db.get("unique_trb", 0)
            tra_str = f"{tra:>12,}" if tra > 0 else f"{'-':>12}"
            trb_str = f"{trb:>12,}" if trb > 0 else f"{'-':>12}"
            lines.append(f"{db_name:<15} {total:>15,} {tra_str} {trb_str}")

    lines.extend(["", ""])

    # --- Public Studies ---
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

    # --- Adaptive Bulk ---
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

    lines.extend([
        "=" * 80,
        "",
    ])

    # Write markdown report
    md_file = output_path / "tcr_source_summary.md"
    with open(md_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown report written to: {md_file}")

    # Build JSON report
    json_data = {
        "databases": convert_to_native(db_results),
        "studies": convert_to_native(study_results) if study_results else None,
        "adaptive_bulk": convert_to_native(adaptive_results) if adaptive_results else None,
    }

    json_file = output_path / "tcr_source_summary.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON report written to: {json_file}")

    # Print summary to console
    print("\n" + "\n".join(lines))


# --- Main ---


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

    args = parser.parse_args()

    if not args.raw_data_path.exists():
        print(f"Error: Raw data path does not exist: {args.raw_data_path}")
        return 1

    print("=" * 60)
    print("TCR Raw Data Source Analysis")
    print("=" * 60)

    db_path = args.raw_data_path / "databases"
    db_results = {}

    # --- Database analysis ---
    print("\nStep 1: Analyzing databases...")

    db_results["VDJdb"] = analyze_vdjdb(db_path)
    db_results["McPAS-TCR"] = analyze_mcpas(db_path)
    db_results["IEDB"] = analyze_iedb(db_path)
    db_results["CEDAR"] = analyze_cedar(db_path)
    db_results["IMGT/HLA"] = analyze_imgt(db_path)
    db_results["TCRdb"] = analyze_tcrdb(db_path)
    db_results["WarrenDB"] = analyze_warrendb(db_path)

    if not args.skip_ireceptor:
        db_results["iReceptor"] = analyze_ireceptor(db_path)
    else:
        print("  Skipping iReceptor (--skip_ireceptor flag)")
        db_results["iReceptor"] = None

    # --- Study analysis ---
    study_results = None
    adaptive_results = None

    if not args.skip_studies:
        print("\nStep 2: Analyzing studies...")
        study_results = analyze_studies(args.raw_data_path)
        adaptive_results = analyze_adaptive_bulk(args.raw_data_path)
    else:
        print("\nStep 2: Skipping study analysis")

    # --- Generate report ---
    print("\nStep 3: Generating reports...")
    generate_report(db_results, study_results, adaptive_results, args.output_dir)

    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
