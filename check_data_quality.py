#!/usr/bin/env python3
"""Data quality check for standardized_again parquet files."""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

BASE_DIR = Path("/home/ubuntu/quest/data/standardized_again")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")
HLA_PATTERN = re.compile(r"^HLA-[A-Z0-9]+\*\d+:\d+")
COLUMNS = ["tra", "trb", "peptide", "mhc_one", "mhc_two", "tra_full", "trb_full"]
SAMPLE_N = 50_000


def is_aa_sequence(s):
    """Check if string is a valid amino acid sequence."""
    if not isinstance(s, str) or len(s) == 0:
        return False
    return all(c in VALID_AA for c in s.upper())


def is_hla_allele(s):
    """Check if string looks like an HLA allele name."""
    if not isinstance(s, str) or len(s) == 0:
        return False
    return bool(HLA_PATTERN.match(s))


def analyze_database(db_path):
    """Analyze one database subdirectory."""
    parquet_files = sorted(glob(str(db_path / "part_*.parquet")))
    if not parquet_files:
        return None

    df = pd.read_parquet(parquet_files)
    results = {"total_rows": len(df)}

    for col in COLUMNS:
        if col not in df.columns:
            results[col] = {"present": False}
            continue

        non_null = df[col].dropna()
        n_non_null = len(non_null)
        results[col] = {"present": True, "non_null_count": n_non_null}

        if n_non_null == 0:
            results[col]["pct_valid_aa"] = None
            results[col]["median_len"] = None
            continue

        # Sample
        if n_non_null > SAMPLE_N:
            sample = non_null.sample(SAMPLE_N, random_state=42)
        else:
            sample = non_null

        sample_str = sample.astype(str)

        # Check AA validity
        aa_valid = sample_str.apply(is_aa_sequence)
        pct_aa = aa_valid.mean() * 100
        results[col]["pct_valid_aa"] = pct_aa

        # Median length
        lengths = sample_str.str.len()
        results[col]["median_len"] = float(lengths.median())

        # For mhc columns, also check HLA allele format
        if col in ("mhc_one", "mhc_two"):
            hla_valid = sample_str.apply(is_hla_allele)
            pct_hla = hla_valid.mean() * 100
            results[col]["pct_hla_allele"] = pct_hla
            # Check what format they are
            if pct_hla > 50:
                results[col]["format"] = "HLA allele names"
            elif pct_aa > 50:
                results[col]["format"] = "AA sequences"
            else:
                results[col]["format"] = "mixed/other"
            # Show some examples of non-AA, non-HLA values
            neither = sample_str[~aa_valid & ~hla_valid]
            if len(neither) > 0:
                results[col]["other_examples"] = neither.head(3).tolist()

        # For tra_full/trb_full, check length distribution
        if col in ("tra_full", "trb_full"):
            results[col]["pct_ge_100"] = float((lengths >= 100).mean() * 100)
            results[col]["len_q25"] = float(lengths.quantile(0.25))
            results[col]["len_q75"] = float(lengths.quantile(0.75))

    return results


def main():
    databases = sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])

    all_results = {}
    for db in databases:
        db_path = BASE_DIR / db
        print(f"Analyzing {db}...")
        res = analyze_database(db_path)
        if res is not None:
            all_results[db] = res

    # Print summary table
    print("\n" + "=" * 140)
    print("DATA QUALITY SUMMARY")
    print("=" * 140)

    # Table 1: Non-null counts and AA validity
    print(f"\n{'Database':<16} {'Rows':>8} | {'tra':>14} | {'trb':>14} | {'peptide':>14} | {'mhc_one':>14} | {'mhc_two':>14} | {'tra_full':>14} | {'trb_full':>14}")
    print(f"{'':16} {'':>8} | {'%AA  /  cnt':>14} | {'%AA  /  cnt':>14} | {'%AA  /  cnt':>14} | {'%AA  /  cnt':>14} | {'%AA  /  cnt':>14} | {'%AA  /  cnt':>14} | {'%AA  /  cnt':>14}")
    print("-" * 140)

    for db, res in sorted(all_results.items()):
        parts = []
        parts.append(f"{db:<16}")
        parts.append(f"{res['total_rows']:>8}")
        for col in COLUMNS:
            info = res.get(col, {})
            if not info.get("present", False):
                parts.append(f"{'N/A':>14}")
            elif info.get("non_null_count", 0) == 0:
                parts.append(f"{'0 vals':>14}")
            else:
                pct = info["pct_valid_aa"]
                cnt = info["non_null_count"]
                if cnt >= 1_000_000:
                    cnt_str = f"{cnt/1_000_000:.1f}M"
                elif cnt >= 1_000:
                    cnt_str = f"{cnt/1_000:.1f}K"
                else:
                    cnt_str = str(cnt)
                parts.append(f"{pct:5.1f}% {cnt_str:>7}")
        print(" | ".join(parts))

    # Table 2: MHC format details
    print(f"\n\n{'='*80}")
    print("MHC COLUMN FORMAT DETAILS")
    print(f"{'='*80}")
    print(f"{'Database':<16} {'mhc_one format':<25} {'%HLA':>6} {'%AA':>6} | {'mhc_two format':<25} {'%HLA':>6} {'%AA':>6}")
    print("-" * 80)

    for db, res in sorted(all_results.items()):
        m1 = res.get("mhc_one", {})
        m2 = res.get("mhc_two", {})

        m1_fmt = m1.get("format", "N/A") if m1.get("non_null_count", 0) > 0 else "no data"
        m1_hla = m1.get("pct_hla_allele", 0)
        m1_aa = m1.get("pct_valid_aa", 0)
        m2_fmt = m2.get("format", "N/A") if m2.get("non_null_count", 0) > 0 else "no data"
        m2_hla = m2.get("pct_hla_allele", 0)
        m2_aa = m2.get("pct_valid_aa", 0)

        print(f"{db:<16} {m1_fmt:<25} {m1_hla:5.1f}% {m1_aa:5.1f}% | {m2_fmt:<25} {m2_hla:5.1f}% {m2_aa:5.1f}%")

    # Table 3: Full-length vs CDR3 length comparison
    print(f"\n\n{'='*100}")
    print("FULL-LENGTH vs CDR3 LENGTH COMPARISON")
    print(f"{'='*100}")
    print(f"{'Database':<16} {'tra med':>8} {'tra_full med':>12} {'tra_full %>=100':>16} | {'trb med':>8} {'trb_full med':>12} {'trb_full %>=100':>16}")
    print("-" * 100)

    for db, res in sorted(all_results.items()):
        tra = res.get("tra", {})
        tra_f = res.get("tra_full", {})
        trb = res.get("trb", {})
        trb_f = res.get("trb_full", {})

        tra_med = f"{tra.get('median_len', 0):.0f}" if tra.get("median_len") else "-"
        tra_f_med = f"{tra_f.get('median_len', 0):.0f}" if tra_f.get("median_len") else "-"
        tra_f_pct = f"{tra_f.get('pct_ge_100', 0):.1f}%" if tra_f.get("pct_ge_100") is not None else "-"
        trb_med = f"{trb.get('median_len', 0):.0f}" if trb.get("median_len") else "-"
        trb_f_med = f"{trb_f.get('median_len', 0):.0f}" if trb_f.get("median_len") else "-"
        trb_f_pct = f"{trb_f.get('pct_ge_100', 0):.1f}%" if trb_f.get("pct_ge_100") is not None else "-"

        print(f"{db:<16} {tra_med:>8} {tra_f_med:>12} {tra_f_pct:>16} | {trb_med:>8} {trb_f_med:>12} {trb_f_pct:>16}")

    # Print any non-AA, non-HLA examples for mhc columns
    print(f"\n\n{'='*80}")
    print("MHC VALUES THAT ARE NEITHER AA SEQUENCES NOR HLA ALLELES (examples)")
    print(f"{'='*80}")
    for db, res in sorted(all_results.items()):
        for col in ("mhc_one", "mhc_two"):
            info = res.get(col, {})
            examples = info.get("other_examples", [])
            if examples:
                print(f"  {db}/{col}: {examples}")

    # Check for non-AA characters in sequence columns
    print(f"\n\n{'='*80}")
    print("NON-AA SEQUENCE ISSUES (columns with <100% valid AA)")
    print(f"{'='*80}")
    for db, res in sorted(all_results.items()):
        for col in ["tra", "trb", "peptide", "tra_full", "trb_full"]:
            info = res.get(col, {})
            pct = info.get("pct_valid_aa")
            if pct is not None and pct < 100.0 and info.get("non_null_count", 0) > 0:
                print(f"  {db}/{col}: {pct:.2f}% valid AA ({info['non_null_count']} non-null values)")


if __name__ == "__main__":
    main()
