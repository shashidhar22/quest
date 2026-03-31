#!/usr/bin/env python3
"""IEDB standardizer.

IEDB (Immune Epitope Database) — 226K rows, same structure as CEDAR.
Uses IEDB export format with multi-level headers.

Filters: receptor_type == TCR, assay_outcome == Positive.

Join strategy: receptor has Assay_IEDB IDs (comma-separated numeric IDs),
tcell has Assay ID_IEDB IRI (URLs like http://www.iedb.org/assay/29).
We extract numeric IDs from both and join on them.
"""

import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import (
    split_mhc_to_alpha_beta,
    standardize_dataframe,
)
from scripts.data_processing.standardize._base import BaseStandardizer
from scripts.data_processing.standardize.cedar import (
    _extract_cedar_id,
    _read_cedar_csv,
)


class IedbStandardizer(BaseStandardizer):
    name = "iedb"

    def get_column_map(self) -> dict:
        return {
            "tra": "tra",
            "trav_gene": "trav_gene",
            "trad_gene": "trad_gene",
            "traj_gene": "traj_gene",
            "trb": "trb",
            "trbv_gene": "trbv_gene",
            "trbd_gene": "trbd_gene",
            "trbj_gene": "trbj_gene",
            "peptide": "peptide",
            "mhc_one": "mhc_one",
            "mhc_two": "mhc_two",
            "binding": "binding",
        }

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        receptor_path = self.source_dir / "receptor" / "tcr_full_v3.csv"
        tcell_path = self.source_dir / "tcell" / "tcell_full_v3.csv"

        if not receptor_path.exists():
            return

        # Load receptor data (same format as CEDAR)
        receptor_df = _read_cedar_csv(receptor_path)

        # Filter to TCR only (exclude BCR)
        # Receptor_Type values are chain descriptions like "alphabeta",
        # "gammadelta", "beta" — not "TCR". Exclude known BCR types.
        type_col = None
        for col in receptor_df.columns:
            if "type" in col.lower() and "receptor" in col.lower():
                type_col = col
                break
        if type_col:
            bcr_types = {"ig", "immunoglobulin", "bcr"}
            receptor_df = receptor_df[
                ~receptor_df[type_col].str.strip().str.lower().isin(bcr_types)
            ].reset_index(drop=True)

        # Find CDR3 and gene columns
        merged = pd.DataFrame(index=receptor_df.index)
        col_mapping = {}

        for col in receptor_df.columns:
            cl = col.lower()
            if "chain 1" in cl and "cdr3" in cl and "curated" in cl and "start" not in cl and "end" not in cl:
                col_mapping["tra"] = col
            elif "chain 1" in cl and "v gene" in cl and "curated" in cl:
                col_mapping["trav_gene"] = col
            elif "chain 1" in cl and "d gene" in cl and "curated" in cl:
                col_mapping["trad_gene"] = col
            elif "chain 1" in cl and "j gene" in cl and "curated" in cl:
                col_mapping["traj_gene"] = col
            elif "chain 2" in cl and "cdr3" in cl and "curated" in cl and "start" not in cl and "end" not in cl:
                col_mapping["trb"] = col
            elif "chain 2" in cl and "v gene" in cl and "curated" in cl:
                col_mapping["trbv_gene"] = col
            elif "chain 2" in cl and "d gene" in cl and "curated" in cl:
                col_mapping["trbd_gene"] = col
            elif "chain 2" in cl and "j gene" in cl and "curated" in cl:
                col_mapping["trbj_gene"] = col

        # Fallback to calculated if curated not found
        for col in receptor_df.columns:
            cl = col.lower()
            if "tra" not in col_mapping and "chain 1" in cl and "cdr3" in cl and "calculated" in cl and "start" not in cl and "end" not in cl:
                col_mapping["tra"] = col
            if "trb" not in col_mapping and "chain 2" in cl and "cdr3" in cl and "calculated" in cl and "start" not in cl and "end" not in cl:
                col_mapping["trb"] = col

        for target_col, src_col in col_mapping.items():
            merged[target_col] = receptor_df[src_col]

        # Extract epitope (peptide) directly from receptor columns
        epitope_col_receptor = None
        for col in receptor_df.columns:
            cl = col.lower()
            if "epitope" in cl and "name" in cl:
                epitope_col_receptor = col
                break
        if epitope_col_receptor:
            merged["peptide"] = receptor_df[epitope_col_receptor]
        else:
            merged["peptide"] = ""

        # Extract MHC directly from receptor columns
        mhc_col_receptor = None
        for col in receptor_df.columns:
            cl = col.lower()
            if "mhc" in cl and "allele" in cl:
                mhc_col_receptor = col
                break
        if mhc_col_receptor:
            mhc_split = receptor_df[mhc_col_receptor].apply(split_mhc_to_alpha_beta)
            merged["mhc_one"] = mhc_split.apply(lambda x: x[0])
            merged["mhc_two"] = mhc_split.apply(lambda x: x[1])
        else:
            merged["mhc_one"] = ""
            merged["mhc_two"] = ""

        merged["binding"] = ""

        # Find the assay IDs column in receptor (comma-separated numeric IDs)
        receptor_assay_id_col = None
        for col in receptor_df.columns:
            cl = col.lower()
            if "assay" in cl and "id" in cl and "iri" not in cl:
                receptor_assay_id_col = col
                break

        # Join with tcell to enrich binding + fill peptide/MHC if missing
        if tcell_path.exists() and receptor_assay_id_col:
            merged["_assay_ids"] = receptor_df[receptor_assay_id_col]
            tcell_df = _read_cedar_csv(tcell_path)

            # Find columns in tcell
            epitope_col = mhc_col = tcell_assay_iri_col = outcome_col = None
            for col in tcell_df.columns:
                cl = col.lower()
                if "epitope" in cl and ("name" in cl or "description" in cl):
                    if epitope_col is None:
                        epitope_col = col
                if "mhc" in cl and ("restriction" in cl or "allele" in cl):
                    if mhc_col is None:
                        mhc_col = col
                if "assay" in cl and "iri" in cl:
                    if tcell_assay_iri_col is None:
                        tcell_assay_iri_col = col
                if "qualitative" in cl and "measure" in cl:
                    outcome_col = col

            # Filter to positive assays
            if outcome_col:
                tcell_df = tcell_df[
                    tcell_df[outcome_col].str.lower().str.contains("positive", na=False)
                ]

            # Filter tcell data to human organisms
            organism_col = None
            for col in tcell_df.columns:
                cl = col.lower()
                if "organism" in cl and "source" in cl and "iri" not in cl:
                    organism_col = col
                    break
            if organism_col is None:
                for col in tcell_df.columns:
                    if "organism" in col.lower() and "iri" not in col.lower():
                        organism_col = col
                        break
            if organism_col:
                tcell_df = tcell_df[
                    tcell_df[organism_col].str.strip().str.lower().str.contains(
                        "homo sapiens|human", na=False
                    )
                    | (tcell_df[organism_col].str.strip() == "")
                ].reset_index(drop=True)

            if tcell_assay_iri_col:
                # Vectorized join: extract assay IDs, explode, merge
                tcell_df["_assay_id"] = tcell_df[tcell_assay_iri_col].apply(
                    _extract_cedar_id
                )
                # Build tcell lookup table with binding + peptide + MHC
                tcell_keyed = tcell_df[["_assay_id"]].copy()
                tcell_keyed["_binding"] = (
                    tcell_df[outcome_col].str.strip() if outcome_col else ""
                )
                tcell_keyed["_epitope"] = (
                    tcell_df[epitope_col].str.strip() if epitope_col else ""
                )
                tcell_keyed["_mhc"] = (
                    tcell_df[mhc_col].str.strip() if mhc_col else ""
                )
                tcell_keyed = tcell_keyed[
                    tcell_keyed["_assay_id"] != ""
                ].drop_duplicates(subset=["_assay_id"], keep="first")

                # Explode receptor assay IDs and merge
                merged["_row_idx"] = range(len(merged))
                exploded = merged[["_row_idx", "_assay_ids"]].copy()
                exploded["_assay_id"] = exploded["_assay_ids"].str.split(",")
                exploded = exploded.explode("_assay_id", ignore_index=True)
                exploded["_assay_id"] = exploded["_assay_id"].str.strip()

                joined = exploded.merge(tcell_keyed, on="_assay_id", how="inner")
                # Keep first match per receptor row
                joined = joined.drop_duplicates(subset=["_row_idx"], keep="first")

                # Apply binding from tcell
                binding_map = joined.set_index("_row_idx")["_binding"]
                merged["binding"] = merged["_row_idx"].map(binding_map).fillna("")

                # Fill peptide/MHC from tcell where empty in receptor
                if epitope_col:
                    epitope_map = joined.set_index("_row_idx")["_epitope"]
                    tcell_peptide = merged["_row_idx"].map(epitope_map).fillna("")
                    empty_peptide = merged["peptide"].fillna("").str.strip() == ""
                    merged.loc[empty_peptide, "peptide"] = tcell_peptide[empty_peptide]
                if mhc_col:
                    mhc_map = joined.set_index("_row_idx")["_mhc"]
                    tcell_mhc = merged["_row_idx"].map(mhc_map).fillna("")
                    empty_mhc = merged["mhc_one"].fillna("").str.strip() == ""
                    for idx in empty_mhc[empty_mhc].index:
                        mhc_str = tcell_mhc.get(idx, "")
                        if mhc_str:
                            m1, m2 = split_mhc_to_alpha_beta(mhc_str)
                            merged.at[idx, "mhc_one"] = m1
                            merged.at[idx, "mhc_two"] = m2

                merged = merged.drop(columns=["_row_idx", "_assay_ids"], errors="ignore")
            else:
                merged = merged.drop(columns=["_assay_ids"], errors="ignore")
        # Ensure peptide/mhc columns exist
        for col in ("peptide", "mhc_one", "mhc_two", "binding"):
            if col not in merged.columns:
                merged[col] = ""

        column_map = self.get_column_map()
        result, dropped = standardize_dataframe(
            merged, column_map, source=self.name, stitch=self.stitch,
            hla_dir=self.hla_dir,
        )
        yield result, dropped


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize IEDB")
    parser.add_argument("--source-dir", default="data/databases/IEDB")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hla-dir", default="")
    args = parser.parse_args()

    standardizer = IedbStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
        hla_dir=args.hla_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
