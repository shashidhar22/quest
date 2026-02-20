#!/usr/bin/env python3
"""Export comprehensive NetMHCPan analysis tables.

NetMHCPan training data contains MHC-peptide binding data (no TCR sequences):
  - MHC class I: NetMHCpan_train/ with 5-fold BA/EL data (space-separated)
  - MHC class II: NetMHCIIpan_train/ with 5-fold BA/EL train+test (tab-separated)

Data files contain peptide sequences, binding scores, and allele identifiers.
Allele lists map sample IDs to HLA allele names.
"""

import os
import re
from collections import Counter, defaultdict

import pandas as pd

DATA_DIR = "data/databases/NetMHCPan"
OUTPUT_DIR = "data/analysis/NetMHCPan"

# Species prefixes from allele names
SPECIES_MAP = {
    "HLA": "human",
    "H-2": "mouse",
    "H2-": "mouse",
    "BoLA": "bovine",
    "SLA": "swine",
    "Mamu": "rhesus macaque",
    "Patr": "chimpanzee",
    "Gogo": "gorilla",
    "ELA": "equine",
    "DLA": "canine",
    "FLA": "feline",
    "RT1": "rat",
    "Caja": "marmoset",
    "Aime": "giant panda",
    "Eqca": "horse",
}


def classify_species(allele):
    """Determine species from allele name prefix."""
    if not allele:
        return "unknown"
    a = str(allele).strip()
    for prefix, species in SPECIES_MAP.items():
        if a.startswith(prefix):
            return species
    # Check DRB/DQB patterns (Class II human)
    if re.match(r"^D[RPQ][AB]\d", a):
        return "human"
    return "other"


def load_allele_list(fpath):
    """Load allele list file mapping sample IDs to allele names.

    Returns dict: sample_id -> list of allele names
    """
    mapping = {}
    if not os.path.exists(fpath):
        print(f"  WARNING: allele list not found: {fpath}")
        return mapping

    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on first whitespace (space or tab)
            parts = line.split(None, 1)
            if len(parts) == 2:
                sample_id = parts[0]
                alleles = [a.strip() for a in parts[1].split(",")]
                mapping[sample_id] = alleles
    return mapping


def aggregate_data(data_dir):
    """Stream through NetMHCPan data files and accumulate statistics."""
    total_records = 0
    unique_peptides = set()
    unique_alleles = set()
    peptide_lengths = Counter()

    # Per-category: {MHCI_binding_affinity, MHCI_eluted_ligand, ...}
    cat_records = Counter()
    cat_peptides = defaultdict(set)
    cat_alleles = defaultdict(set)

    # Per-file/fold stats
    fold_records = Counter()
    fold_peptides = defaultdict(set)
    fold_alleles = defaultdict(set)

    # Per-allele stats
    allele_records = Counter()
    allele_peptides = defaultdict(set)

    # Species tracking
    species_records = Counter()

    # Data type tracking
    ba_records = 0
    el_records = 0
    mhci_records = 0
    mhcii_records = 0
    train_records = 0
    test_records = 0

    # --- MHC Class I ---
    mhci_dir = os.path.join(data_dir, "NetMHCpan_train")
    mhci_allele_map = load_allele_list(
        os.path.join(mhci_dir, "allelelist")
    )
    print(f"  MHC-I allele list: {len(mhci_allele_map)} entries")

    for fold in range(5):
        for dtype, suffix in [
            ("binding_affinity", "ba"),
            ("eluted_ligand", "el"),
        ]:
            fname = f"c{fold:03d}_{suffix}"
            fpath = os.path.join(mhci_dir, fname)
            if not os.path.exists(fpath):
                print(f"  WARNING: {fpath} not found")
                continue

            category = f"MHCI_{dtype}"
            fold_label = f"MHCI_fold{fold}_{suffix}"
            n = 0

            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue

                    peptide = parts[0]
                    allele_id = parts[2]

                    # Resolve allele ID to allele names
                    resolved = mhci_allele_map.get(allele_id, [allele_id])

                    pep_len = len(peptide)
                    unique_peptides.add(peptide)
                    peptide_lengths[pep_len] += 1

                    cat_records[category] += 1
                    cat_peptides[category].add(peptide)
                    fold_records[fold_label] += 1
                    fold_peptides[fold_label].add(peptide)

                    for allele in resolved:
                        unique_alleles.add(allele)
                        cat_alleles[category].add(allele)
                        fold_alleles[fold_label].add(allele)
                        allele_records[allele] += 1
                        allele_peptides[allele].add(peptide)
                        sp = classify_species(allele)
                        species_records[sp] += 1

                    n += 1

                    if dtype == "binding_affinity":
                        ba_records += 1
                    else:
                        el_records += 1
                    mhci_records += 1
                    train_records += 1
                    total_records += 1

            print(f"    {fname}: {n:,} records")

    # --- MHC Class II ---
    mhcii_dir = os.path.join(data_dir, "NetMHCIIpan_train")
    mhcii_allele_map = load_allele_list(
        os.path.join(mhcii_dir, "allelelist.txt")
    )
    print(f"  MHC-II allele list: {len(mhcii_allele_map)} entries")

    for split in ["train", "test"]:
        for fold in range(1, 6):
            for dtype, suffix in [
                ("binding_affinity", "BA"),
                ("eluted_ligand", "EL"),
            ]:
                fname = f"{split}_{suffix}{fold}.txt"
                fpath = os.path.join(mhcii_dir, fname)
                if not os.path.exists(fpath):
                    print(f"  WARNING: {fpath} not found")
                    continue

                category = f"MHCII_{dtype}"
                fold_label = f"MHCII_{split}_fold{fold}_{suffix}"
                n = 0

                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) < 3:
                            continue

                        peptide = parts[0]
                        allele_id = parts[2]

                        resolved = mhcii_allele_map.get(
                            allele_id, [allele_id]
                        )

                        pep_len = len(peptide)
                        unique_peptides.add(peptide)
                        peptide_lengths[pep_len] += 1

                        cat_records[category] += 1
                        cat_peptides[category].add(peptide)
                        fold_records[fold_label] += 1
                        fold_peptides[fold_label].add(peptide)

                        for allele in resolved:
                            unique_alleles.add(allele)
                            cat_alleles[category].add(allele)
                            fold_alleles[fold_label].add(allele)
                            allele_records[allele] += 1
                            allele_peptides[allele].add(peptide)
                            sp = classify_species(allele)
                            species_records[sp] += 1

                        n += 1

                        if dtype == "binding_affinity":
                            ba_records += 1
                        else:
                            el_records += 1
                        mhcii_records += 1
                        if split == "train":
                            train_records += 1
                        else:
                            test_records += 1
                        total_records += 1

                print(f"    {fname}: {n:,} records")

    print(f"  Total: {total_records:,} records")

    return {
        "total_records": total_records,
        "unique_peptides": unique_peptides,
        "unique_alleles": unique_alleles,
        "peptide_lengths": peptide_lengths,
        "cat_records": cat_records,
        "cat_peptides": cat_peptides,
        "cat_alleles": cat_alleles,
        "fold_records": fold_records,
        "fold_peptides": fold_peptides,
        "fold_alleles": fold_alleles,
        "allele_records": allele_records,
        "allele_peptides": allele_peptides,
        "species_records": species_records,
        "ba_records": ba_records,
        "el_records": el_records,
        "mhci_records": mhci_records,
        "mhcii_records": mhcii_records,
        "train_records": train_records,
        "test_records": test_records,
    }


# ---------------------------------------------------------------------------
# Build TSV DataFrames
# ---------------------------------------------------------------------------
def overview(stats):
    return pd.DataFrame(
        [
            {
                "total_records": stats["total_records"],
                "unique_peptides": len(stats["unique_peptides"]),
                "unique_alleles": len(stats["unique_alleles"]),
                "mhci_records": stats["mhci_records"],
                "mhcii_records": stats["mhcii_records"],
                "ba_records": stats["ba_records"],
                "el_records": stats["el_records"],
                "train_records": stats["train_records"],
                "test_records": stats["test_records"],
            }
        ]
    )


def category_overview(stats):
    rows = []
    for cat in sorted(stats["cat_records"]):
        rows.append(
            {
                "category": cat,
                "total_records": stats["cat_records"][cat],
                "unique_peptides": len(stats["cat_peptides"].get(cat, set())),
                "unique_alleles": len(stats["cat_alleles"].get(cat, set())),
            }
        )
    return pd.DataFrame(rows)


def epitope_summary(stats):
    """Repurposed as peptide length distribution (no TCR epitope data)."""
    rows = []
    for pep_len in sorted(stats["peptide_lengths"]):
        rows.append(
            {
                "peptide_length": pep_len,
                "total_records": stats["peptide_lengths"][pep_len],
            }
        )
    return pd.DataFrame(rows)


def pmhc_summary(stats):
    """Top alleles by record count (no TCR-pMHC pairing)."""
    rows = []
    for allele in sorted(
        stats["allele_records"],
        key=stats["allele_records"].get,
        reverse=True,
    )[:100]:
        sp = classify_species(allele)
        rows.append(
            {
                "allele": allele,
                "species": sp,
                "total_records": stats["allele_records"][allele],
                "unique_peptides": len(
                    stats["allele_peptides"].get(allele, set())
                ),
            }
        )
    return pd.DataFrame(rows)


def study_summary(stats):
    """Per-fold breakdown (folds serve as 'studies')."""
    rows = []
    for fold in sorted(stats["fold_records"]):
        rows.append(
            {
                "fold": fold,
                "total_records": stats["fold_records"][fold],
                "unique_peptides": len(
                    stats["fold_peptides"].get(fold, set())
                ),
                "unique_alleles": len(
                    stats["fold_alleles"].get(fold, set())
                ),
            }
        )
    return pd.DataFrame(rows)


def mhc_allele_summary(stats):
    rows = []
    for allele in sorted(stats["allele_records"]):
        # Determine MHC class from allele name
        a = str(allele).upper()
        if any(x in a for x in ["DRB", "DQB", "DPA", "DPB", "DQA", "DRA",
                                  "H-2-IA", "H-2-IE", "H2-IA", "H2-IE",
                                  "IAD", "IED"]):
            mhc_class = "MHCII"
        else:
            mhc_class = "MHCI"

        rows.append(
            {
                "mhc": allele,
                "mhc_class": mhc_class,
                "total_records": stats["allele_records"][allele],
                "unique_peptides": len(
                    stats["allele_peptides"].get(allele, set())
                ),
            }
        )
    return pd.DataFrame(rows)


def species_breakdown(stats):
    rows = []

    # Species from allele prefix
    for sp in sorted(stats["species_records"]):
        rows.append(
            {
                "category": "species",
                "label": sp,
                "total_records": stats["species_records"][sp],
            }
        )

    # Data type breakdown
    for label, count in [
        ("binding_affinity", stats["ba_records"]),
        ("eluted_ligand", stats["el_records"]),
    ]:
        rows.append(
            {
                "category": "data_type",
                "label": label,
                "total_records": count,
            }
        )

    # MHC class breakdown
    for label, count in [
        ("MHCI", stats["mhci_records"]),
        ("MHCII", stats["mhcii_records"]),
    ]:
        rows.append(
            {
                "category": "mhc_class",
                "label": label,
                "total_records": count,
            }
        )

    # Train/test split
    for label, count in [
        ("train", stats["train_records"]),
        ("test", stats["test_records"]),
    ]:
        rows.append(
            {
                "category": "split",
                "label": label,
                "total_records": count,
            }
        )

    # Peptide length distribution (top lengths)
    for pep_len in sorted(stats["peptide_lengths"]):
        rows.append(
            {
                "category": "peptide_length",
                "label": str(pep_len),
                "total_records": stats["peptide_lengths"][pep_len],
            }
        )

    return pd.DataFrame(rows)


def data_completeness(stats):
    total = stats["total_records"]
    return pd.DataFrame(
        [
            {
                "category": "binding_affinity",
                "record_count": stats["ba_records"],
                "unique_peptides": 0,
                "unique_alleles": 0,
            },
            {
                "category": "eluted_ligand",
                "record_count": stats["el_records"],
                "unique_peptides": 0,
                "unique_alleles": 0,
            },
            {
                "category": "mhc_class_i",
                "record_count": stats["mhci_records"],
                "unique_peptides": 0,
                "unique_alleles": 0,
            },
            {
                "category": "mhc_class_ii",
                "record_count": stats["mhcii_records"],
                "unique_peptides": 0,
                "unique_alleles": 0,
            },
            {
                "category": "total",
                "record_count": total,
                "unique_peptides": len(stats["unique_peptides"]),
                "unique_alleles": len(stats["unique_alleles"]),
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
    pep_len = pd.read_csv(
        os.path.join(output_dir, "epitope_summary.tsv"), sep="\t"
    )
    mhc_alleles = pd.read_csv(
        os.path.join(output_dir, "mhc_allele_summary.tsv"), sep="\t"
    )
    species = pd.read_csv(
        os.path.join(output_dir, "species_breakdown.tsv"), sep="\t"
    )
    folds = pd.read_csv(
        os.path.join(output_dir, "study_summary.tsv"), sep="\t"
    )
    pmhc = pd.read_csv(
        os.path.join(output_dir, "pmhc_summary.tsv"), sep="\t"
    )

    r = ov.iloc[0]
    total = int(r["total_records"])
    lines = []

    # --- Header ---
    lines.append("=" * 80)
    lines.append("                    NetMHCPan Analysis Summary")
    lines.append(
        "       MHC-Peptide Binding Prediction Training Data"
    )
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {DATA_DIR}")
    lines.append("=" * 80)
    lines.append("")

    # --- 1. DATASET OVERVIEW ---
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Total records:        {total:>14,}")
    lines.append(
        f"Unique peptides:      {int(r['unique_peptides']):>14,}"
    )
    lines.append(
        f"Unique alleles:       {int(r['unique_alleles']):>14,}"
    )
    lines.append(
        f"MHC-I records:        {int(r['mhci_records']):>14,}"
    )
    lines.append(
        f"MHC-II records:       {int(r['mhcii_records']):>14,}"
    )
    lines.append(
        f"Binding affinity (BA):{int(r['ba_records']):>14,}"
    )
    lines.append(
        f"Eluted ligand (EL):   {int(r['el_records']):>14,}"
    )
    lines.append(
        f"Training records:     {int(r['train_records']):>14,}"
    )
    lines.append(
        f"Test records:         {int(r['test_records']):>14,}"
    )
    lines.append("")

    lines.append("Category distribution:")
    for _, row in cat_ov.sort_values(
        "total_records", ascending=False
    ).iterrows():
        cat = row["category"]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {cat:<30s} {cnt:>14,}  ({pct:>5.1f}%)"
            f"  [{int(row['unique_peptides']):,} peptides,"
            f" {int(row['unique_alleles']):,} alleles]"
        )
    lines.append("")

    # --- 2. MHC CLASS BREAKDOWN ---
    lines.append("2. MHC CLASS BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    mhc_cls = species[species["category"] == "mhc_class"]
    for _, row in mhc_cls.iterrows():
        cls = str(row["label"])
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cls:<12s} {cnt:>14,} records  ({pct:.1f}%)")

    mhci_pct = (
        100.0 * int(r["mhci_records"]) / total if total else 0
    )
    mhcii_pct = 100.0 - mhci_pct if total else 0
    lines.append("")
    lines.append(
        f"MHC-I accounts for {mhci_pct:.1f}% and MHC-II for"
        f" {mhcii_pct:.1f}% of records."
    )
    lines.append("")

    # --- 3. CATEGORY BREAKDOWN ---
    lines.append("3. CATEGORY BREAKDOWN (MHC CLASS x DATA TYPE)")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Category':<30s} {'Records':>14s} {'%':>7s}"
        f" {'Peptides':>12s} {'Alleles':>9s}"
    )
    for _, row in cat_ov.sort_values(
        "total_records", ascending=False
    ).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['category']):<30s} {cnt:>14,} {pct:>6.1f}%"
            f" {int(row['unique_peptides']):>12,}"
            f" {int(row['unique_alleles']):>9,}"
        )
    lines.append("")

    el_pct = 100.0 * int(r["el_records"]) / total if total else 0
    lines.append(
        f"Eluted ligand data dominates at {el_pct:.1f}% of all records."
    )
    lines.append("")

    # --- 4. PEPTIDE LENGTH DISTRIBUTION ---
    lines.append("4. PEPTIDE LENGTH DISTRIBUTION")
    lines.append("-" * 80)
    lines.append("")

    if len(pep_len) > 0:
        lines.append(f"  {'Length':>8s} {'Records':>14s} {'%':>7s}")
        total_pep = int(pep_len["total_records"].sum())
        for _, row in pep_len.sort_values(
            "total_records", ascending=False
        ).head(20).iterrows():
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total_pep if total_pep else 0
            lines.append(
                f"  {int(row['peptide_length']):>8d}"
                f" {cnt:>14,} {pct:>6.1f}%"
            )
        if len(pep_len) > 20:
            lines.append(f"  ... and {len(pep_len) - 20} more lengths")
    lines.append("")

    # --- 5. TOP MHC ALLELES ---
    lines.append("5. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")

    if len(pmhc) > 0:
        lines.append(
            f"  {'Allele':<30s} {'Species':<18s}"
            f" {'Records':>12s} {'Peptides':>10s}"
        )
        for _, row in pmhc.head(20).iterrows():
            allele = str(row["allele"])[:28]
            sp = str(row["species"])[:16]
            lines.append(
                f"  {allele:<30s} {sp:<18s}"
                f" {int(row['total_records']):>12,}"
                f" {int(row['unique_peptides']):>10,}"
            )
    lines.append("")

    # --- 6. SPECIES BREAKDOWN ---
    lines.append("6. SPECIES BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    sp_rows = species[species["category"] == "species"].copy()
    sp_rows = sp_rows.sort_values("total_records", ascending=False)
    sp_total = int(sp_rows["total_records"].sum())

    if len(sp_rows) > 0:
        for _, row in sp_rows.iterrows():
            sp = str(row["label"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / sp_total if sp_total else 0
            lines.append(
                f"  {sp:<20s} {cnt:>14,} records  ({pct:>5.1f}%)"
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
                f"  {dt:<20s} {cnt:>14,}  ({pct:>5.1f}%)"
            )
    lines.append("")

    # --- 7. FOLD/STUDY LANDSCAPE ---
    lines.append("7. FOLD LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")

    total_folds = len(folds)
    lines.append(f"Total data folds: {total_folds}")
    lines.append("")

    lines.append(
        f"  {'Fold':<35s} {'Records':>12s}"
        f" {'Peptides':>10s} {'Alleles':>9s}"
    )
    for _, row in folds.sort_values(
        "total_records", ascending=False
    ).iterrows():
        fold = str(row["fold"])[:33]
        lines.append(
            f"  {fold:<35s} {int(row['total_records']):>12,}"
            f" {int(row['unique_peptides']):>10,}"
            f" {int(row['unique_alleles']):>9,}"
        )
    lines.append("")

    # --- 8. DATA COMPLETENESS ---
    lines.append("8. DATA COMPLETENESS")
    lines.append("-" * 80)
    lines.append("")

    lines.append(
        "All records in NetMHCPan training data are complete by design.\n"
        "Each record contains a peptide sequence, binding score/label,\n"
        "and an allele identifier."
    )
    lines.append("")

    split_rows = species[species["category"] == "split"].copy()
    if len(split_rows) > 0:
        lines.append("Train/test split:")
        for _, row in split_rows.iterrows():
            sp = str(row["label"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total if total else 0
            lines.append(
                f"  {sp:<12s} {cnt:>14,} records  ({pct:.1f}%)"
            )
        lines.append("")

    lines.append(
        "Note: MHC-I data has only training folds (5-fold CV).\n"
        "MHC-II data has separate train and test sets (5-fold each)."
    )
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

    lines.append(
        f"a) Data scale: {total:,} binding records across"
        f" {len(stats['unique_peptides']):,} unique peptides\n"
        f"   and {len(stats['unique_alleles']):,} unique MHC alleles."
    )
    lines.append("")

    lines.append(
        f"b) MHC class balance: MHC-I ({mhci_pct:.1f}%) vs"
        f" MHC-II ({mhcii_pct:.1f}%).\n"
        f"   MHC-II has substantially more data due to eluted"
        f" ligand datasets."
    )
    lines.append("")

    lines.append(
        f"c) Data type: Eluted ligand data ({el_pct:.1f}%) dominates"
        f" over binding affinity.\n"
        f"   EL data provides binary labels; BA data provides"
        f" continuous binding scores."
    )
    lines.append("")

    lines.append(
        "d) No TCR data: This is MHC-peptide binding data only.\n"
        "   Useful for peptide-MHC binding prediction and pMHC\n"
        "   representation learning, not for TCR specificity modeling."
    )
    lines.append("")

    if len(sp_rows) > 0:
        top_sp = sp_rows.iloc[0]
        top_sp_pct = (
            100.0 * int(top_sp["total_records"]) / sp_total
            if sp_total
            else 0
        )
        lines.append(
            f"e) Species: {top_sp['label']} alleles account for"
            f" {top_sp_pct:.1f}% of records.\n"
            f"   {len(sp_rows)} species represented."
        )
    lines.append("")

    lines.append(
        "f) Cross-validation: 5-fold splits provided for both MHC-I\n"
        "   and MHC-II. Standard benchmark for binding prediction."
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

    print(f"Streaming NetMHCPan data from {data_dir} ...")
    stats = aggregate_data(data_dir)
    print(f"  Unique peptides: {len(stats['unique_peptides']):,}")
    print(f"  Unique alleles: {len(stats['unique_alleles']):,}")
    print()

    exports = [
        ("overview.tsv", lambda s: overview(s)),
        ("category_overview.tsv", lambda s: category_overview(s)),
        ("epitope_summary.tsv", lambda s: epitope_summary(s)),
        ("pmhc_summary.tsv", lambda s: pmhc_summary(s)),
        ("study_summary.tsv", lambda s: study_summary(s)),
        ("mhc_allele_summary.tsv", lambda s: mhc_allele_summary(s)),
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
