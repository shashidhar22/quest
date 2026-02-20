#!/usr/bin/env python3
"""Export comprehensive studies analysis tables.

Processes ~72 heterogeneous VDJ datasets from data/studies/.
Only VDJ/TCR sequencing data is included; gene expression files are skipped.

Supported formats:
  - 10X Genomics: all_contig_annotations.csv, clonotypes.csv, consensus
  - immunoSEQ: bulk_survey .tsv/.align.txt files
  - MiTCR: .clones_TRA.tsv / .clones_TRB.tsv
  - AIRR: airr_rearrangement.tsv
  - Generic TCR TSV: count/freq/cdr3aa/v/j columns
  - tcrdist3-ready: bulk_deep_trb .tcrdist3.ready.tsv
"""

import gc
import os
import re
from collections import Counter, defaultdict

import pandas as pd

DATA_DIR = "data/studies"
OUTPUT_DIR = "data/analysis/studies"

# File patterns to skip (gene expression, spatial, etc.)
SKIP_PATTERNS = [
    "count.txt", "TPM", "matrix", "genes.tsv", "barcodes.tsv",
    "features.tsv", ".h5", ".mtx", ".loom", "spatial",
    "metrics_summary", "web_summary",
]

# File extensions to consider
VDJ_EXTENSIONS = {".csv", ".tsv", ".txt"}


def classify_study_source(study_name):
    """Classify study by data source."""
    if study_name.startswith("GSE") or study_name.startswith("GSM"):
        return "GEO"
    if study_name.startswith("ZEN"):
        return "Zenodo"
    return "Custom"


def should_skip_file(fpath):
    """Check if file should be skipped (non-VDJ data)."""
    basename = os.path.basename(fpath)
    for pattern in SKIP_PATTERNS:
        if pattern in basename:
            return True
    return False


def detect_format(columns):
    """Detect file format from column names.

    Returns: (format_name, column_mapping) or (None, None)
    """
    cols_lower = {c.lower(): c for c in columns}

    # 10X Genomics contig annotations
    if "barcode" in cols_lower and "chain" in cols_lower and "cdr3" in cols_lower:
        mapping = {
            "cdr3": cols_lower.get("cdr3"),
            "chain": cols_lower.get("chain"),
            "v_gene": cols_lower.get("v_gene"),
            "j_gene": cols_lower.get("j_gene"),
            "productive": cols_lower.get("productive"),
        }
        return "10X_contig", mapping

    # 10X AIRR format
    if "junction_aa" in cols_lower and "v_call" in cols_lower:
        # Detect chain from locus column or v_call prefix
        mapping = {
            "cdr3": cols_lower.get("junction_aa"),
            "v_gene": cols_lower.get("v_call"),
            "j_gene": cols_lower.get("j_call"),
            "productive": cols_lower.get("productive"),
        }
        return "AIRR", mapping

    # 10X clonotypes (cdr3s_aa contains chain-prefixed sequences)
    if "clonotype_id" in cols_lower and "cdr3s_aa" in cols_lower:
        mapping = {"cdr3s_aa": cols_lower.get("cdr3s_aa")}
        return "10X_clonotype", mapping

    # 10X consensus annotations
    if "consensus_id" in cols_lower and "cdr3" in cols_lower and "chain" in cols_lower:
        mapping = {
            "cdr3": cols_lower.get("cdr3"),
            "chain": cols_lower.get("chain"),
            "v_gene": cols_lower.get("v_gene"),
            "j_gene": cols_lower.get("j_gene"),
            "productive": cols_lower.get("productive"),
        }
        return "10X_consensus", mapping

    # MiTCR clones (aaSeqCDR3)
    if "aaseqcdr3" in cols_lower:
        mapping = {
            "cdr3": cols_lower.get("aaseqcdr3"),
            "v_gene": cols_lower.get("allvhitswithscore",
                       cols_lower.get("bestvhit")),
            "j_gene": cols_lower.get("alljhitswithscore",
                       cols_lower.get("bestjhit")),
        }
        return "MiTCR", mapping

    # tcrdist3-ready format
    if "cdr3_b_aa" in cols_lower or "cdr3_a_aa" in cols_lower:
        mapping = {
            "cdr3_beta": cols_lower.get("cdr3_b_aa"),
            "cdr3_alpha": cols_lower.get("cdr3_a_aa"),
            "v_beta": cols_lower.get("v_b_gene"),
            "v_alpha": cols_lower.get("v_a_gene"),
            "j_beta": cols_lower.get("j_b_gene"),
            "j_alpha": cols_lower.get("j_a_gene"),
        }
        return "tcrdist3", mapping

    # immunoSEQ / generic bulk TCR (cdr3aa or amino_acid)
    if "cdr3aa" in cols_lower or "amino_acid" in cols_lower:
        mapping = {
            "cdr3": cols_lower.get("cdr3aa", cols_lower.get("amino_acid")),
            "v_gene": cols_lower.get("v", cols_lower.get("v_gene",
                       cols_lower.get("v_resolved"))),
            "j_gene": cols_lower.get("j", cols_lower.get("j_gene",
                       cols_lower.get("j_resolved"))),
        }
        return "immunoSEQ", mapping

    # Generic with rearrangement/cdr3 columns
    for cdr3_col in ["cdr3", "cdr3_amino_acid", "junction_aa", "cdr3_aa"]:
        if cdr3_col in cols_lower:
            mapping = {
                "cdr3": cols_lower.get(cdr3_col),
                "v_gene": next(
                    (cols_lower[k] for k in cols_lower
                     if "v_gene" in k or "v_call" in k or k == "v"),
                    None,
                ),
                "j_gene": next(
                    (cols_lower[k] for k in cols_lower
                     if "j_gene" in k or "j_call" in k or k == "j"),
                    None,
                ),
            }
            return "generic_TCR", mapping

    return None, None


def infer_chain_from_path(fpath):
    """Infer chain type from file path."""
    path_lower = fpath.lower()
    if "tra" in os.path.basename(path_lower) or "alpha" in path_lower:
        return "TRA"
    if "trb" in os.path.basename(path_lower) or "beta" in path_lower:
        return "TRB"
    if "bulk_survey_tra" in path_lower:
        return "TRA"
    if "bulk_survey_trb" in path_lower or "bulk_deep_trb" in path_lower:
        return "TRB"
    return "unknown"


def infer_data_type_from_path(fpath):
    """Infer single-cell vs bulk from file path."""
    path_lower = fpath.lower()
    if any(x in path_lower for x in ["contigs", "contig", "barcode",
                                       "clonotype", "consensus", "airr"]):
        return "single-cell"
    if any(x in path_lower for x in ["bulk", "align"]):
        return "bulk"
    return "unknown"


def find_vdj_files(data_dir):
    """Find all VDJ/TCR files in the studies directory.

    Returns list of (study_name, fpath) tuples.
    """
    files = []
    for study_name in sorted(os.listdir(data_dir)):
        study_dir = os.path.join(data_dir, study_name)
        if not os.path.isdir(study_dir):
            continue

        for root, dirs, filenames in os.walk(study_dir):
            # Only look in tcr/ or vdj/ subdirectories
            rel_path = os.path.relpath(root, study_dir).lower()
            if not any(x in rel_path for x in ["tcr", "vdj"]):
                continue

            # Skip gex, spatial subdirectories
            if any(x in rel_path for x in ["gex", "spatial", "count"]):
                continue

            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VDJ_EXTENSIONS:
                    continue

                fpath = os.path.join(root, fname)
                if should_skip_file(fpath):
                    continue

                # Skip FASTA files by content hint
                if fname.endswith(".fa") or "fasta" in fname.lower():
                    continue

                files.append((study_name, fpath))

    return files


def process_file(fpath, study_name):
    """Process a single VDJ file and return stats dict or None."""
    try:
        # Detect separator and read header
        with open(fpath) as f:
            first_line = f.readline()

        # Skip files that start with XML/non-tabular content
        if first_line.startswith("<?xml") or first_line.startswith("#"):
            # MiTCR XML header - skip header lines and try next
            with open(fpath) as f:
                for line in f:
                    if not line.startswith(("#", "<", "?")):
                        first_line = line
                        break

        # Detect separator
        if "\t" in first_line:
            sep = "\t"
        else:
            sep = ","

        # Read small sample to detect format
        try:
            sample = pd.read_csv(
                fpath, sep=sep, nrows=5, dtype=str, na_filter=False,
                comment="#", on_bad_lines="skip",
            )
        except Exception:
            return None

        if len(sample.columns) < 2:
            return None

        fmt, mapping = detect_format(sample.columns)
        if fmt is None:
            return None

        # Now read full file
        try:
            df = pd.read_csv(
                fpath, sep=sep, dtype=str, na_filter=False,
                comment="#", on_bad_lines="skip",
            )
        except Exception:
            return None

        if len(df) == 0:
            return None

        n_records = len(df)
        chain_from_path = infer_chain_from_path(fpath)
        data_type = infer_data_type_from_path(fpath)

        stats = {
            "study": study_name,
            "file": os.path.basename(fpath),
            "format": fmt,
            "records": n_records,
            "data_type": data_type,
            "chain_counts": Counter(),
            "unique_cdr3_beta": set(),
            "unique_cdr3_alpha": set(),
            "unique_v_genes": set(),
            "unique_j_genes": set(),
            "has_cdr3": 0,
            "has_v_gene": 0,
            "has_j_gene": 0,
            "productive_count": 0,
        }

        # Extract chain-specific CDR3s
        if fmt == "10X_clonotype":
            # Parse cdr3s_aa column: "TRA:CAVS...;TRB:CASS..."
            cdr3s_col = mapping.get("cdr3s_aa")
            if cdr3s_col and cdr3s_col in df.columns:
                for val in df[cdr3s_col]:
                    if not val:
                        continue
                    for part in str(val).split(";"):
                        part = part.strip()
                        if part.startswith("TRA:"):
                            stats["unique_cdr3_alpha"].add(part[4:])
                            stats["chain_counts"]["TRA"] += 1
                            stats["has_cdr3"] += 1
                        elif part.startswith("TRB:"):
                            stats["unique_cdr3_beta"].add(part[4:])
                            stats["chain_counts"]["TRB"] += 1
                            stats["has_cdr3"] += 1
            return stats

        if fmt == "tcrdist3":
            # Separate alpha/beta columns
            if mapping.get("cdr3_beta") and mapping["cdr3_beta"] in df.columns:
                valid = df[mapping["cdr3_beta"]] != ""
                stats["unique_cdr3_beta"] = set(
                    df.loc[valid, mapping["cdr3_beta"]]
                )
                stats["has_cdr3"] += int(valid.sum())
                stats["chain_counts"]["TRB"] += int(valid.sum())
            if mapping.get("cdr3_alpha") and mapping["cdr3_alpha"] in df.columns:
                valid = df[mapping["cdr3_alpha"]] != ""
                stats["unique_cdr3_alpha"] = set(
                    df.loc[valid, mapping["cdr3_alpha"]]
                )
                stats["has_cdr3"] += int(valid.sum())
                stats["chain_counts"]["TRA"] += int(valid.sum())
            for vk in ["v_beta", "v_alpha"]:
                if mapping.get(vk) and mapping[vk] in df.columns:
                    valid = df[mapping[vk]] != ""
                    stats["unique_v_genes"].update(
                        df.loc[valid, mapping[vk]]
                    )
                    stats["has_v_gene"] += int(valid.sum())
            for jk in ["j_beta", "j_alpha"]:
                if mapping.get(jk) and mapping[jk] in df.columns:
                    valid = df[mapping[jk]] != ""
                    stats["unique_j_genes"].update(
                        df.loc[valid, mapping[jk]]
                    )
                    stats["has_j_gene"] += int(valid.sum())
            return stats

        # Standard single-CDR3 formats
        cdr3_col = mapping.get("cdr3")
        if cdr3_col and cdr3_col in df.columns:
            valid_cdr3 = (df[cdr3_col] != "") & (df[cdr3_col] != "None")
            stats["has_cdr3"] = int(valid_cdr3.sum())
            cdr3_values = set(df.loc[valid_cdr3, cdr3_col])

            # Determine chain per-row or per-file
            chain_col = mapping.get("chain")
            if chain_col and chain_col in df.columns:
                for chain_val, grp in df[valid_cdr3].groupby(chain_col):
                    chain_str = str(chain_val).upper().strip()
                    stats["chain_counts"][chain_str] += len(grp)
                    cdr3_set = set(grp[cdr3_col])
                    if chain_str == "TRA":
                        stats["unique_cdr3_alpha"].update(cdr3_set)
                    elif chain_str == "TRB":
                        stats["unique_cdr3_beta"].update(cdr3_set)
            else:
                # Use path-based chain inference
                if chain_from_path == "TRA":
                    stats["unique_cdr3_alpha"] = cdr3_values
                    stats["chain_counts"]["TRA"] = int(valid_cdr3.sum())
                elif chain_from_path == "TRB":
                    stats["unique_cdr3_beta"] = cdr3_values
                    stats["chain_counts"]["TRB"] = int(valid_cdr3.sum())
                else:
                    stats["unique_cdr3_beta"] = cdr3_values
                    stats["chain_counts"]["unknown"] = int(valid_cdr3.sum())

        # V gene
        v_col = mapping.get("v_gene")
        if v_col and v_col in df.columns:
            valid_v = (df[v_col] != "") & (df[v_col] != "None") & (
                df[v_col] != "unresolved"
            )
            stats["has_v_gene"] = int(valid_v.sum())
            stats["unique_v_genes"] = set(df.loc[valid_v, v_col])

        # J gene
        j_col = mapping.get("j_gene")
        if j_col and j_col in df.columns:
            valid_j = (df[j_col] != "") & (df[j_col] != "None") & (
                df[j_col] != "unresolved"
            )
            stats["has_j_gene"] = int(valid_j.sum())
            stats["unique_j_genes"] = set(df.loc[valid_j, j_col])

        # Productive
        prod_col = mapping.get("productive")
        if prod_col and prod_col in df.columns:
            stats["productive_count"] = int(
                df[prod_col].str.lower().isin(["true", "t", "in"]).sum()
            )

        return stats

    except Exception as e:
        return None


def aggregate_data(data_dir):
    """Find and process all VDJ files, accumulating statistics."""
    print("  Finding VDJ files ...")
    vdj_files = find_vdj_files(data_dir)
    print(f"  Found {len(vdj_files)} VDJ files")

    # --- Accumulators ---
    total_records = 0
    total_files = 0
    total_studies = set()
    skipped = 0

    global_cdr3_beta = set()
    global_cdr3_alpha = set()
    global_v_genes = set()
    global_j_genes = set()

    # Per-study
    study_records = Counter()
    study_files = Counter()
    study_cdr3_beta = defaultdict(set)
    study_cdr3_alpha = defaultdict(set)
    study_v_genes = defaultdict(set)
    study_j_genes = defaultdict(set)
    study_formats = defaultdict(set)
    study_data_types = defaultdict(set)
    study_has_cdr3 = Counter()
    study_has_v_gene = Counter()
    study_has_j_gene = Counter()
    study_productive = Counter()
    study_source = {}

    # Chain distribution
    chain_records = Counter()

    # Format distribution
    format_records = Counter()
    format_files = Counter()

    # Data type distribution
    data_type_records = Counter()

    # Completeness accumulators
    total_has_cdr3 = 0
    total_has_v_gene = 0
    total_has_j_gene = 0
    total_productive = 0

    for i, (study_name, fpath) in enumerate(vdj_files):
        result = process_file(fpath, study_name)
        if result is None:
            skipped += 1
            continue

        n = result["records"]
        fmt = result["format"]

        total_records += n
        total_files += 1
        total_studies.add(study_name)

        global_cdr3_beta.update(result["unique_cdr3_beta"])
        global_cdr3_alpha.update(result["unique_cdr3_alpha"])
        global_v_genes.update(result["unique_v_genes"])
        global_j_genes.update(result["unique_j_genes"])

        study_records[study_name] += n
        study_files[study_name] += 1
        study_cdr3_beta[study_name].update(result["unique_cdr3_beta"])
        study_cdr3_alpha[study_name].update(result["unique_cdr3_alpha"])
        study_v_genes[study_name].update(result["unique_v_genes"])
        study_j_genes[study_name].update(result["unique_j_genes"])
        study_formats[study_name].add(fmt)
        study_data_types[study_name].add(result["data_type"])
        study_has_cdr3[study_name] += result["has_cdr3"]
        study_has_v_gene[study_name] += result["has_v_gene"]
        study_has_j_gene[study_name] += result["has_j_gene"]
        study_productive[study_name] += result["productive_count"]
        study_source[study_name] = classify_study_source(study_name)

        for chain, cnt in result["chain_counts"].items():
            chain_records[chain] += cnt

        format_records[fmt] += n
        format_files[fmt] += 1
        data_type_records[result["data_type"]] += n

        total_has_cdr3 += result["has_cdr3"]
        total_has_v_gene += result["has_v_gene"]
        total_has_j_gene += result["has_j_gene"]
        total_productive += result["productive_count"]

        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{len(vdj_files)} files ...")
            gc.collect()

    print(
        f"  Processed {total_files} files ({skipped} skipped)"
        f" across {len(total_studies)} studies"
    )

    # --- Per-source category aggregation ---
    source_records = Counter()
    source_studies = defaultdict(set)
    for study, src in study_source.items():
        source_records[src] += study_records[study]
        source_studies[src].add(study)

    return {
        "total_records": total_records,
        "total_files": total_files,
        "total_studies": len(total_studies),
        "skipped": skipped,
        "global_cdr3_beta": global_cdr3_beta,
        "global_cdr3_alpha": global_cdr3_alpha,
        "global_v_genes": global_v_genes,
        "global_j_genes": global_j_genes,
        "study_records": study_records,
        "study_files": study_files,
        "study_cdr3_beta": study_cdr3_beta,
        "study_cdr3_alpha": study_cdr3_alpha,
        "study_v_genes": study_v_genes,
        "study_j_genes": study_j_genes,
        "study_formats": study_formats,
        "study_data_types": study_data_types,
        "study_has_cdr3": study_has_cdr3,
        "study_has_v_gene": study_has_v_gene,
        "study_has_j_gene": study_has_j_gene,
        "study_productive": study_productive,
        "study_source": study_source,
        "chain_records": chain_records,
        "format_records": format_records,
        "format_files": format_files,
        "data_type_records": data_type_records,
        "source_records": source_records,
        "source_studies": source_studies,
        "total_has_cdr3": total_has_cdr3,
        "total_has_v_gene": total_has_v_gene,
        "total_has_j_gene": total_has_j_gene,
        "total_productive": total_productive,
    }


# ---------------------------------------------------------------------------
# Build TSV DataFrames
# ---------------------------------------------------------------------------
def overview(stats):
    return pd.DataFrame(
        [
            {
                "total_records": stats["total_records"],
                "total_files": stats["total_files"],
                "total_studies": stats["total_studies"],
                "unique_cdr3_beta": len(stats["global_cdr3_beta"]),
                "unique_cdr3_alpha": len(stats["global_cdr3_alpha"]),
                "unique_v_genes": len(stats["global_v_genes"]),
                "unique_j_genes": len(stats["global_j_genes"]),
                "unique_epitopes": 0,
                "unique_mhc": 0,
            }
        ]
    )


def category_overview(stats):
    """Category = study source (GEO, Zenodo, Custom)."""
    rows = []
    for src in sorted(stats["source_records"]):
        studies_in_src = stats["source_studies"][src]
        cdr3_beta = set()
        cdr3_alpha = set()
        for s in studies_in_src:
            cdr3_beta.update(stats["study_cdr3_beta"].get(s, set()))
            cdr3_alpha.update(stats["study_cdr3_alpha"].get(s, set()))
        rows.append(
            {
                "category": src,
                "total_records": stats["source_records"][src],
                "num_studies": len(studies_in_src),
                "unique_cdr3_beta": len(cdr3_beta),
                "unique_cdr3_alpha": len(cdr3_alpha),
            }
        )
    return pd.DataFrame(rows)


def epitope_summary():
    """No epitope data in VDJ studies."""
    return pd.DataFrame(
        columns=[
            "epitope", "source_organism", "primary_mhc", "mhc_class",
            "total_records", "paired_records", "unique_paired_tcrs",
        ]
    )


def pmhc_summary():
    """No pMHC data in VDJ studies."""
    return pd.DataFrame(
        columns=[
            "epitope", "mhc", "mhc_class", "total_records",
            "unique_paired_tcrs",
        ]
    )


def study_summary(stats):
    """Per-study breakdown (main value of this summary)."""
    rows = []
    for study in sorted(stats["study_records"]):
        rows.append(
            {
                "study": study,
                "source": stats["study_source"].get(study, "unknown"),
                "total_records": stats["study_records"][study],
                "num_files": stats["study_files"][study],
                "unique_cdr3_beta": len(
                    stats["study_cdr3_beta"].get(study, set())
                ),
                "unique_cdr3_alpha": len(
                    stats["study_cdr3_alpha"].get(study, set())
                ),
                "unique_v_genes": len(
                    stats["study_v_genes"].get(study, set())
                ),
                "unique_j_genes": len(
                    stats["study_j_genes"].get(study, set())
                ),
                "formats": ";".join(
                    sorted(stats["study_formats"].get(study, set()))
                ),
                "data_types": ";".join(
                    sorted(stats["study_data_types"].get(study, set()))
                ),
                "has_cdr3_count": stats["study_has_cdr3"][study],
                "has_v_gene_count": stats["study_has_v_gene"][study],
                "productive_count": stats["study_productive"][study],
            }
        )
    return pd.DataFrame(rows)


def mhc_allele_summary():
    """No MHC data in VDJ studies."""
    return pd.DataFrame(
        columns=[
            "mhc", "mhc_class", "total_records", "unique_epitopes",
            "unique_paired_tcrs",
        ]
    )


def species_breakdown(stats):
    """Chain type, file format, data type distribution."""
    rows = []

    # Chain type distribution
    for chain in sorted(stats["chain_records"]):
        rows.append(
            {
                "category": "chain_type",
                "label": chain,
                "total_records": stats["chain_records"][chain],
            }
        )

    # File format distribution
    for fmt in sorted(stats["format_records"]):
        rows.append(
            {
                "category": "file_format",
                "label": fmt,
                "total_records": stats["format_records"][fmt],
            }
        )

    # Data type (single-cell vs bulk)
    for dt in sorted(stats["data_type_records"]):
        rows.append(
            {
                "category": "data_type",
                "label": dt,
                "total_records": stats["data_type_records"][dt],
            }
        )

    # Source distribution
    for src in sorted(stats["source_records"]):
        rows.append(
            {
                "category": "study_source",
                "label": src,
                "total_records": stats["source_records"][src],
            }
        )

    return pd.DataFrame(rows)


def data_completeness(stats):
    total = stats["total_records"]
    return pd.DataFrame(
        [
            {
                "category": "has_cdr3",
                "record_count": stats["total_has_cdr3"],
                "unique_cdr3_beta": len(stats["global_cdr3_beta"]),
                "unique_v_genes": len(stats["global_v_genes"]),
            },
            {
                "category": "has_v_gene",
                "record_count": stats["total_has_v_gene"],
                "unique_cdr3_beta": 0,
                "unique_v_genes": len(stats["global_v_genes"]),
            },
            {
                "category": "has_j_gene",
                "record_count": stats["total_has_j_gene"],
                "unique_cdr3_beta": 0,
                "unique_v_genes": 0,
            },
            {
                "category": "productive",
                "record_count": stats["total_productive"],
                "unique_cdr3_beta": 0,
                "unique_v_genes": 0,
            },
            {
                "category": "total",
                "record_count": total,
                "unique_cdr3_beta": len(stats["global_cdr3_beta"]),
                "unique_v_genes": len(stats["global_v_genes"]),
            },
        ]
    )


# ---------------------------------------------------------------------------
# SUMMARY.txt generation
# ---------------------------------------------------------------------------
def write_summary_txt(stats, output_dir):
    from datetime import date

    ov = pd.read_csv(os.path.join(output_dir, "overview.tsv"), sep="\t")
    cat_ov = pd.read_csv(
        os.path.join(output_dir, "category_overview.tsv"), sep="\t"
    )
    completeness = pd.read_csv(
        os.path.join(output_dir, "data_completeness.tsv"), sep="\t"
    )
    species = pd.read_csv(
        os.path.join(output_dir, "species_breakdown.tsv"), sep="\t"
    )
    studies = pd.read_csv(
        os.path.join(output_dir, "study_summary.tsv"), sep="\t"
    )

    r = ov.iloc[0]
    total = int(r["total_records"])
    n_files = int(r["total_files"])
    n_studies = int(r["total_studies"])
    lines = []

    # --- Header ---
    lines.append("=" * 80)
    lines.append("                    Studies VDJ Analysis Summary")
    lines.append(
        "       Heterogeneous VDJ Datasets from GEO, Zenodo, and Custom Sources"
    )
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {DATA_DIR}")
    lines.append("=" * 80)
    lines.append("")

    # --- 1. DATASET OVERVIEW ---
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Total records:          {total:>12,}")
    lines.append(f"Total files:            {n_files:>12,}")
    lines.append(f"Total studies:          {n_studies:>12,}")
    lines.append(
        f"Unique CDR3 beta:       {int(r['unique_cdr3_beta']):>12,}"
    )
    lines.append(
        f"Unique CDR3 alpha:      {int(r['unique_cdr3_alpha']):>12,}"
    )
    lines.append(
        f"Unique V genes:         {int(r['unique_v_genes']):>12,}"
    )
    lines.append(
        f"Unique J genes:         {int(r['unique_j_genes']):>12,}"
    )
    lines.append(
        f"Unique epitopes:        {'N/A':>12s}  (no epitope annotation)"
    )
    lines.append(
        f"Unique MHC alleles:     {'N/A':>12s}  (no MHC annotation)"
    )
    lines.append("")

    lines.append("Source distribution:")
    for _, row in cat_ov.sort_values(
        "total_records", ascending=False
    ).iterrows():
        cat = row["category"]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        n_st = int(row["num_studies"])
        lines.append(
            f"  {cat:<12s} {cnt:>12,}  ({pct:>5.1f}%)"
            f"  [{n_st:,} studies]"
        )
    lines.append("")

    # --- 2. MHC CLASS BREAKDOWN ---
    lines.append("2. MHC CLASS BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available. VDJ studies do not include MHC annotations."
    )
    lines.append("")

    # --- 3. CATEGORY BREAKDOWN ---
    lines.append("3. CATEGORY BREAKDOWN (BY SOURCE)")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Source':<12s} {'Records':>12s} {'%':>7s}"
        f" {'Studies':>9s} {'CDR3 beta':>12s} {'CDR3 alpha':>12s}"
    )
    for _, row in cat_ov.sort_values(
        "total_records", ascending=False
    ).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['category']):<12s} {cnt:>12,} {pct:>6.1f}%"
            f" {int(row['num_studies']):>9,}"
            f" {int(row['unique_cdr3_beta']):>12,}"
            f" {int(row['unique_cdr3_alpha']):>12,}"
        )
    lines.append("")

    # --- 4. TOP EPITOPES ---
    lines.append("4. TOP EPITOPES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available. VDJ studies are bulk/single-cell repertoire data\n"
        "without epitope annotations."
    )
    lines.append("")

    # --- 5. TOP MHC ALLELES ---
    lines.append("5. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available. VDJ studies do not include MHC annotations."
    )
    lines.append("")

    # --- 6. CHAIN TYPE & FORMAT BREAKDOWN ---
    lines.append("6. CHAIN TYPE, FORMAT & DATA TYPE BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    chain_rows = species[species["category"] == "chain_type"].copy()
    chain_rows = chain_rows.sort_values("total_records", ascending=False)
    chain_total = int(chain_rows["total_records"].sum())
    if len(chain_rows) > 0:
        lines.append("Chain type distribution:")
        for _, row in chain_rows.iterrows():
            ch = str(row["label"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / chain_total if chain_total else 0
            lines.append(
                f"  {ch:<12s} {cnt:>12,} records  ({pct:>5.1f}%)"
            )
        lines.append("")

    fmt_rows = species[species["category"] == "file_format"].copy()
    fmt_rows = fmt_rows.sort_values("total_records", ascending=False)
    if len(fmt_rows) > 0:
        lines.append("File format distribution:")
        for _, row in fmt_rows.iterrows():
            fmt = str(row["label"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total if total else 0
            lines.append(
                f"  {fmt:<18s} {cnt:>12,} records  ({pct:>5.1f}%)"
            )
        lines.append("")

    dt_rows = species[species["category"] == "data_type"].copy()
    dt_rows = dt_rows.sort_values("total_records", ascending=False)
    if len(dt_rows) > 0:
        lines.append("Data type distribution:")
        for _, row in dt_rows.iterrows():
            dt = str(row["label"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total if total else 0
            lines.append(
                f"  {dt:<18s} {cnt:>12,} records  ({pct:>5.1f}%)"
            )
        lines.append("")

    # --- 7. STUDY LANDSCAPE ---
    lines.append("7. STUDY LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")

    lines.append(f"Total studies: {n_studies}")
    lines.append(f"Total files: {n_files}")
    lines.append("")

    lines.append("Largest studies by record count:")
    for _, row in studies.nlargest(15, "total_records").iterrows():
        study = str(row["study"])[:20]
        lines.append(
            f"  {study:<22s} {int(row['total_records']):>12,} records"
            f"  [{int(row['num_files']):>4,} files,"
            f" {str(row['formats'])[:25]}]"
        )
    lines.append("")

    top5_records = int(
        studies.nlargest(5, "total_records")["total_records"].sum()
    )
    top5_pct = 100.0 * top5_records / total if total else 0
    lines.append(
        f"Top 5 studies account for ~{top5_pct:.0f}% of all records."
    )
    lines.append("")

    # --- 8. DATA COMPLETENESS ---
    lines.append("8. DATA COMPLETENESS")
    lines.append("-" * 80)
    lines.append("")

    criteria = {
        "has_cdr3": "Has CDR3 amino acid sequence.",
        "has_v_gene": "Has V gene annotation.",
        "has_j_gene": "Has J gene annotation.",
        "productive": "Productive rearrangement.",
        "total": "All records in the dataset.",
    }

    lines.append(
        f"  {'Category':<18s} {'Records':>12s} {'%':>7s}"
    )
    for _, row in completeness.iterrows():
        cat = row["category"]
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {cat:<18s} {cnt:>12,} {pct:>6.1f}%"
        )
    lines.append("")

    lines.append("Criteria:")
    for cat, desc in criteria.items():
        lines.append(f"  - {cat:<18s} {desc}")
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

    lines.append(
        f"a) Data scale: {total:,} records across {n_files:,} files"
        f" from {n_studies} studies.\n"
        f"   {int(r['unique_cdr3_beta']):,} unique CDR3 beta and"
        f" {int(r['unique_cdr3_alpha']):,} unique CDR3 alpha sequences."
    )
    lines.append("")

    lines.append(
        f"b) V/J gene annotation: {int(r['unique_v_genes']):,} unique"
        f" V genes and {int(r['unique_j_genes']):,} unique J genes.\n"
        f"   Gene usage analysis and conditional generation supported."
    )
    lines.append("")

    lines.append(
        "c) No epitope/MHC: VDJ studies are repertoire data without\n"
        "   epitope or MHC annotations. Useful for TCR language modeling,\n"
        "   repertoire analysis, and pre-training."
    )
    lines.append("")

    lines.append(
        "d) Format heterogeneity: Multiple formats (10X, immunoSEQ,\n"
        "   MiTCR, AIRR, tcrdist3) reflecting real-world data diversity.\n"
        "   Standardization is needed before unified analysis."
    )
    lines.append("")

    lines.append(
        f"e) Source diversity: {n_studies} studies from GEO, Zenodo,\n"
        f"   and custom cohorts. Broad tissue/disease coverage."
    )
    lines.append("")

    lines.append(
        "f) Mixed single-cell and bulk: Both single-cell (10X) and\n"
        "   bulk (immunoSEQ) data are represented. Single-cell data\n"
        "   provides alpha-beta chain pairing."
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

    print(f"Streaming VDJ studies data from {data_dir} ...")
    stats = aggregate_data(data_dir)
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  Unique CDR3 beta: {len(stats['global_cdr3_beta']):,}")
    print(f"  Unique CDR3 alpha: {len(stats['global_cdr3_alpha']):,}")
    print(f"  Unique V genes: {len(stats['global_v_genes']):,}")
    print()

    exports = [
        ("overview.tsv", lambda s: overview(s)),
        ("category_overview.tsv", lambda s: category_overview(s)),
        ("epitope_summary.tsv", lambda s: epitope_summary()),
        ("pmhc_summary.tsv", lambda s: pmhc_summary()),
        ("study_summary.tsv", lambda s: study_summary(s)),
        ("mhc_allele_summary.tsv", lambda s: mhc_allele_summary()),
        ("species_breakdown.tsv", lambda s: species_breakdown(s)),
        ("data_completeness.tsv", lambda s: data_completeness(s)),
    ]

    for filename, func in exports:
        print(f"Computing {filename} ...")
        result = func(stats)
        path = os.path.join(output_dir, filename)
        result.to_csv(path, sep="\t", index=False)
        print(
            f"  -> {path}  ({len(result)} rows, {result.columns.size} cols)"
        )

    print("Writing SUMMARY.txt ...")
    write_summary_txt(stats, output_dir)

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
    comp_total_row = comp[comp["category"] == "total"]["record_count"].iloc[0]
    print(
        f"Completeness total: {comp_total_row}, Overview total:"
        f" {all_total}, match: {comp_total_row == all_total}"
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
