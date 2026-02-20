#!/usr/bin/env python3
"""Export comprehensive ADC (AIRR Data Commons) analysis tables.

Memory-optimized: streams one TSV at a time, accumulates stats in
sets/counters. Never holds the full ~1.5 TB dataset in memory.
"""

import gc
import json
import os
from collections import Counter, defaultdict

import pandas as pd

DATA_DIR = "data/databases/adc"
OUTPUT_DIR = "data/analysis/adc"

# Only read these columns from the 160-column AIRR TSV
ADC_USECOLS = [
    "locus",
    "productive",
    "junction_aa",
    "v_call",
    "j_call",
    "cell_id",
    "repertoire_id",
]

# Repos that are metadata-only (no rearrangement data)
SKIP_REPOS = {"scireptor.dkfz.de"}


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------
def load_repertoire_metadata(data_dir):
    """Read all repertoires.json files, build per-repo metadata mappings.

    Returns:
        repo_metadata: dict[repo_name] -> {
            "rep_to_study": {repertoire_id -> study_title},
            "studies": {study_title -> {"study_id": ..., "species": ...}},
        }
    """
    repo_metadata = {}
    repos = sorted(
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d not in SKIP_REPOS
    )

    for repo in repos:
        rep_path = os.path.join(data_dir, repo, "repertoires.json")
        if not os.path.exists(rep_path):
            continue

        try:
            with open(rep_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        repertoires = data.get("Repertoire", [])
        rep_to_study = {}
        studies = {}

        for rep in repertoires:
            rep_id = rep.get("repertoire_id", "")
            study = rep.get("study", {})
            study_title = (study.get("study_title") or "").strip() or "(unknown)"
            study_id = (study.get("study_id") or "").strip()

            subject = rep.get("subject", {})
            species = subject.get("species", {})
            species_label = (species.get("label") or "").strip()

            rep_to_study[rep_id] = study_title

            if study_title not in studies:
                studies[study_title] = {
                    "study_id": study_id,
                    "species": set(),
                    "repertoire_ids": set(),
                }
            if species_label:
                studies[study_title]["species"].add(species_label)
            studies[study_title]["repertoire_ids"].add(rep_id)

        repo_metadata[repo] = {
            "rep_to_study": rep_to_study,
            "studies": studies,
        }

    return repo_metadata


# ---------------------------------------------------------------------------
# Streaming aggregation
# ---------------------------------------------------------------------------
def aggregate_data(data_dir, repo_metadata):
    """Stream through all TSV files and accumulate statistics.

    Processes one file at a time using chunked reading for large files.
    """
    repos = sorted(repo_metadata.keys())
    print(f"  Found {len(repos)} repositories with metadata")

    # --- Global accumulators ---
    total_records = 0
    productive_count = 0
    junction_aa_by_locus = defaultdict(set)  # locus -> set of junction_aa
    v_calls = set()
    j_calls = set()
    locus_counts = Counter()

    # --- Per-repo accumulators ---
    repo_records = Counter()
    repo_productive = Counter()
    repo_junction_by_locus = defaultdict(lambda: defaultdict(set))
    repo_paired = Counter()
    repo_paired_tcrs = defaultdict(set)

    # --- Per-study accumulators ---
    study_records = Counter()
    study_repos = defaultdict(set)
    study_species = defaultdict(set)
    study_repertoire_count = Counter()

    # --- Data completeness accumulators ---
    has_junction_aa_count = 0
    has_v_call_count = 0

    # --- Pairing accumulators (cell_id based) ---
    total_paired_records = 0
    paired_tcrs = set()

    file_count = 0
    skipped = 0

    for repo in repos:
        rearrangements_dir = os.path.join(data_dir, repo, "rearrangements")
        if not os.path.isdir(rearrangements_dir):
            continue

        tsv_files = sorted(
            f for f in os.listdir(rearrangements_dir) if f.endswith(".tsv")
        )

        meta = repo_metadata.get(repo, {})
        rep_to_study = meta.get("rep_to_study", {})
        study_info = meta.get("studies", {})

        # Pre-populate study species from metadata
        for study_title, sinfo in study_info.items():
            study_species[study_title].update(sinfo.get("species", set()))
            study_repertoire_count[study_title] = len(
                sinfo.get("repertoire_ids", set())
            )
            study_repos[study_title].add(repo)

        for tsv_file in tsv_files:
            fpath = os.path.join(rearrangements_dir, tsv_file)

            # Determine study from repertoire_id (filename without .tsv)
            rep_id = os.path.splitext(tsv_file)[0]
            study_title = rep_to_study.get(rep_id, "(unknown)")

            try:
                reader = pd.read_csv(
                    fpath,
                    sep="\t",
                    usecols=lambda c: c in ADC_USECOLS,
                    dtype=str,
                    na_filter=False,
                    chunksize=100_000,
                )
            except Exception:
                skipped += 1
                continue

            file_had_data = False
            # Track cell_id -> loci within this file for pairing
            cell_loci = defaultdict(set)

            for chunk in reader:
                if len(chunk) == 0:
                    continue
                file_had_data = True
                n = len(chunk)

                # Locus
                if "locus" in chunk.columns:
                    for locus, count in chunk["locus"].value_counts().items():
                        if locus:  # skip empty locus
                            locus_counts[locus] += count

                # Productive
                if "productive" in chunk.columns:
                    prod_mask = chunk["productive"].isin({"T", "true", "True"})
                    prod_n = int(prod_mask.sum())
                    productive_count += prod_n
                    repo_productive[repo] += prod_n

                # Junction AA
                if "junction_aa" in chunk.columns:
                    has_jaa = chunk["junction_aa"] != ""
                    has_junction_aa_count += int(has_jaa.sum())

                    if "locus" in chunk.columns:
                        for locus in chunk["locus"].unique():
                            if not locus:
                                continue
                            mask = (chunk["locus"] == locus) & has_jaa
                            seqs = set(chunk.loc[mask, "junction_aa"])
                            junction_aa_by_locus[locus].update(seqs)
                            repo_junction_by_locus[repo][locus].update(seqs)
                    else:
                        seqs = set(chunk.loc[has_jaa, "junction_aa"])
                        junction_aa_by_locus["unknown"].update(seqs)

                # V/J calls
                if "v_call" in chunk.columns:
                    has_vc = chunk["v_call"] != ""
                    has_v_call_count += int(has_vc.sum())
                    v_calls.update(set(chunk.loc[has_vc, "v_call"]))

                if "j_call" in chunk.columns:
                    has_jc = chunk["j_call"] != ""
                    j_calls.update(set(chunk.loc[has_jc, "j_call"]))

                # Cell-based pairing tracking
                if "cell_id" in chunk.columns and "locus" in chunk.columns:
                    has_cell = chunk["cell_id"] != ""
                    has_locus = chunk["locus"] != ""
                    pair_mask = has_cell & has_locus
                    if pair_mask.any():
                        for _, row in chunk.loc[
                            pair_mask, ["cell_id", "locus"]
                        ].iterrows():
                            cell_loci[row["cell_id"]].add(row["locus"])

                total_records += n
                repo_records[repo] += n
                study_records[study_title] += n

            # After processing all chunks for this file, check pairing
            paired_in_file = 0
            for cell_id, loci in cell_loci.items():
                if "TRA" in loci and "TRB" in loci:
                    paired_in_file += 1
                    paired_tcrs.add(f"{rep_id}|{cell_id}")
            total_paired_records += paired_in_file
            repo_paired[repo] += paired_in_file

            if not file_had_data:
                skipped += 1
                continue

            file_count += 1
            del cell_loci

            if file_count % 200 == 0:
                print(f"    Processed {file_count} files ...")
                gc.collect()

        print(f"    {repo}: {len(tsv_files)} files processed")

    print(f"  Processed {file_count} files ({skipped} skipped)")

    return {
        "total_records": total_records,
        "productive_count": productive_count,
        "junction_aa_by_locus": junction_aa_by_locus,
        "v_calls": v_calls,
        "j_calls": j_calls,
        "locus_counts": locus_counts,
        "repo_records": repo_records,
        "repo_productive": repo_productive,
        "repo_junction_by_locus": repo_junction_by_locus,
        "repo_paired": repo_paired,
        "repo_paired_tcrs": defaultdict(set),  # not tracked per-repo
        "study_records": study_records,
        "study_repos": study_repos,
        "study_species": study_species,
        "study_repertoire_count": study_repertoire_count,
        "has_junction_aa_count": has_junction_aa_count,
        "has_v_call_count": has_v_call_count,
        "total_paired_records": total_paired_records,
        "paired_tcrs": paired_tcrs,
        "file_count": file_count,
    }


# ---------------------------------------------------------------------------
# Build TSV DataFrames from aggregated stats
# ---------------------------------------------------------------------------
def overview(stats):
    # Sum unique junction_aa across TCR loci
    tra_unique = len(stats["junction_aa_by_locus"].get("TRA", set()))
    trb_unique = len(stats["junction_aa_by_locus"].get("TRB", set()))
    igh_unique = len(stats["junction_aa_by_locus"].get("IGH", set()))
    igk_unique = len(stats["junction_aa_by_locus"].get("IGK", set()))
    igl_unique = len(stats["junction_aa_by_locus"].get("IGL", set()))

    return pd.DataFrame(
        [
            {
                "total_records": stats["total_records"],
                "productive_records": stats["productive_count"],
                "unique_TRA_junction_aa": tra_unique,
                "unique_TRB_junction_aa": trb_unique,
                "unique_IGH_junction_aa": igh_unique,
                "unique_IGK_junction_aa": igk_unique,
                "unique_IGL_junction_aa": igl_unique,
                "paired_records_TRA_TRB": stats["total_paired_records"],
                "unique_paired_cells": len(stats["paired_tcrs"]),
                "unique_v_calls": len(stats["v_calls"]),
                "unique_j_calls": len(stats["j_calls"]),
                "unique_epitopes": 0,
                "unique_mhc": 0,
                "unique_pmhc": 0,
            }
        ]
    )


def category_overview(stats):
    """Per-repository breakdown (repository = category for ADC)."""
    rows = []
    for repo in sorted(stats["repo_records"]):
        locus_juncs = stats["repo_junction_by_locus"].get(repo, {})
        rows.append(
            {
                "category": repo,
                "total_records": stats["repo_records"][repo],
                "productive_records": stats["repo_productive"][repo],
                "unique_TRB_junction_aa": len(locus_juncs.get("TRB", set())),
                "unique_TRA_junction_aa": len(locus_juncs.get("TRA", set())),
                "unique_IGH_junction_aa": len(locus_juncs.get("IGH", set())),
                "paired_records_TRA_TRB": stats["repo_paired"][repo],
                "unique_epitopes": 0,
                "unique_mhc": 0,
                "unique_pmhc": 0,
            }
        )
    return pd.DataFrame(rows)


def epitope_summary():
    return pd.DataFrame(
        columns=[
            "epitope",
            "antigen_protein",
            "pathology",
            "category",
            "primary_mhc",
            "mhc_class",
            "total_records",
            "paired_records",
            "unique_paired_tcrs",
            "unique_beta_only_tcrs",
        ]
    )


def pmhc_summary():
    return pd.DataFrame(
        columns=[
            "epitope",
            "mhc",
            "mhc_class",
            "antigen_protein",
            "pathology",
            "total_records",
            "unique_paired_tcrs",
            "unique_beta_only_tcrs",
        ]
    )


def study_summary(stats):
    rows = []
    for study in sorted(stats["study_records"]):
        rows.append(
            {
                "study": study,
                "total_records": stats["study_records"][study],
                "num_repertoires": stats["study_repertoire_count"].get(study, 0),
                "repositories": ";".join(
                    sorted(stats["study_repos"].get(study, set()))
                ),
                "species": ";".join(
                    sorted(stats["study_species"].get(study, set()))
                ),
            }
        )
    return pd.DataFrame(rows)


def mhc_allele_summary():
    return pd.DataFrame(
        columns=[
            "mhc",
            "mhc_class",
            "total_records",
            "unique_epitopes",
            "unique_paired_tcrs",
        ]
    )


def species_breakdown(stats):
    """Locus type breakdown + productive/non-productive breakdown."""
    rows = []

    # Locus breakdown
    for locus in sorted(stats["locus_counts"]):
        cnt = stats["locus_counts"][locus]
        rows.append(
            {
                "category": "locus",
                "label": locus if locus else "(no locus)",
                "total_records": cnt,
                "unique_junction_aa": len(
                    stats["junction_aa_by_locus"].get(locus, set())
                ),
            }
        )

    # Productive breakdown
    prod = stats["productive_count"]
    nonprod = stats["total_records"] - prod
    rows.append(
        {
            "category": "productive",
            "label": "productive",
            "total_records": prod,
            "unique_junction_aa": 0,
        }
    )
    rows.append(
        {
            "category": "productive",
            "label": "non-productive",
            "total_records": nonprod,
            "unique_junction_aa": 0,
        }
    )

    return pd.DataFrame(rows)


def data_completeness(stats):
    total = stats["total_records"]
    has_jaa = stats["has_junction_aa_count"]
    has_vc = stats["has_v_call_count"]
    prod = stats["productive_count"]

    return pd.DataFrame(
        [
            {
                "category": "has_junction_aa",
                "record_count": has_jaa,
                "paired_tcrs": len(stats["paired_tcrs"]),
                "unique_beta_cdr3": len(
                    stats["junction_aa_by_locus"].get("TRB", set())
                ),
                "unique_epitopes": 0,
            },
            {
                "category": "has_v_call",
                "record_count": has_vc,
                "paired_tcrs": 0,
                "unique_beta_cdr3": 0,
                "unique_epitopes": 0,
            },
            {
                "category": "productive",
                "record_count": prod,
                "paired_tcrs": 0,
                "unique_beta_cdr3": 0,
                "unique_epitopes": 0,
            },
            {
                "category": "no_epitope_no_mhc",
                "record_count": total,
                "paired_tcrs": len(stats["paired_tcrs"]),
                "unique_beta_cdr3": len(
                    stats["junction_aa_by_locus"].get("TRB", set())
                ),
                "unique_epitopes": 0,
            },
        ]
    )


# ---------------------------------------------------------------------------
# SUMMARY.txt generation
# ---------------------------------------------------------------------------
def write_summary_txt(stats, output_dir):
    """Write a human-readable SUMMARY.txt after all TSVs are exported."""
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
    productive = int(r["productive_records"])
    prod_pct = 100.0 * productive / total if total else 0
    paired = int(r["paired_records_TRA_TRB"])
    paired_pct = 100.0 * paired / total if total else 0
    num_files = stats["file_count"]
    num_repos = len(stats["repo_records"])
    lines = []

    # --- Header ---
    lines.append("=" * 80)
    lines.append("                    ADC (AIRR Data Commons) Analysis Summary")
    lines.append(f"                    Generated: {date.today().isoformat()}")
    lines.append(f"          Source: {DATA_DIR}")
    lines.append("=" * 80)
    lines.append("")

    # --- 1. DATASET OVERVIEW ---
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Total records:          {total:>14,}")
    lines.append(
        f"Productive records:     {productive:>14,}  ({prod_pct:.1f}%)"
    )
    lines.append(f"Unique TRB junction_aa: {int(r['unique_TRB_junction_aa']):>14,}")
    lines.append(f"Unique TRA junction_aa: {int(r['unique_TRA_junction_aa']):>14,}")
    lines.append(f"Unique IGH junction_aa: {int(r['unique_IGH_junction_aa']):>14,}")
    lines.append(f"Unique IGK junction_aa: {int(r['unique_IGK_junction_aa']):>14,}")
    lines.append(f"Unique IGL junction_aa: {int(r['unique_IGL_junction_aa']):>14,}")
    lines.append(
        f"Paired cells (TRA+TRB): {paired:>14,}  ({paired_pct:.1f}% of total)"
    )
    lines.append(f"Unique V gene calls:    {int(r['unique_v_calls']):>14,}")
    lines.append(f"Unique J gene calls:    {int(r['unique_j_calls']):>14,}")
    lines.append(f"Unique epitopes:        {'N/A':>14s}  (no epitope annotation)")
    lines.append(f"Unique MHC alleles:     {'N/A':>14s}  (no MHC annotation)")
    lines.append(f"Source repositories:    {num_repos:>14,}")
    lines.append(f"Source files:           {num_files:>14,}")
    lines.append("")

    lines.append("Repository distribution:")
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cat = row["category"]
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cat:<40s} {cnt:>14,}  ({pct:>5.1f}%)")
    lines.append("")

    # --- 2. MHC CLASS BREAKDOWN ---
    lines.append("2. MHC CLASS BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available in this database. ADC bulk repertoire data does not\n"
        "include MHC annotations."
    )
    lines.append("")

    # --- 3. CATEGORY BREAKDOWN (REPOSITORY) ---
    lines.append("3. CATEGORY BREAKDOWN (BY REPOSITORY)")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"  {'Repository':<40s} {'Records':>14s} {'%':>7s} "
        f"{'Productive':>12s} {'TRB CDR3s':>12s}"
    )
    for _, row in cat_ov.sort_values("total_records", ascending=False).iterrows():
        cnt = int(row["total_records"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(
            f"  {str(row['category'])[:38]:<40s} {cnt:>14,} {pct:>6.1f}% "
            f"{int(row['productive_records']):>12,} "
            f"{int(row['unique_TRB_junction_aa']):>12,}"
        )
    lines.append("")

    dominant = cat_ov.loc[cat_ov["total_records"].idxmax()]
    dom_pct = 100.0 * int(dominant["total_records"]) / total if total else 0
    lines.append(
        f"The '{dominant['category']}' repository has the most records with "
        f"{dom_pct:.1f}% of the database."
    )
    lines.append("")

    # --- 4. TOP EPITOPES ---
    lines.append("4. TOP EPITOPES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available in this database. ADC bulk repertoire data does not\n"
        "include epitope annotations."
    )
    lines.append("")

    # --- 5. TOP MHC ALLELES ---
    lines.append("5. TOP MHC ALLELES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "Not available in this database. ADC bulk repertoire data does not\n"
        "include MHC annotations."
    )
    lines.append("")

    # --- 6. PATHOLOGY & SPECIES ---
    lines.append("6. LOCUS & SPECIES BREAKDOWN")
    lines.append("-" * 80)
    lines.append("")

    locus_rows = species[species["category"] == "locus"].copy()
    locus_rows = locus_rows.sort_values("total_records", ascending=False)
    locus_total = int(locus_rows["total_records"].sum())

    if len(locus_rows) > 0:
        lines.append("Locus breakdown:")
        for _, row in locus_rows.iterrows():
            lbl = str(row["label"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / locus_total if locus_total else 0
            unique = int(row["unique_junction_aa"])
            lines.append(
                f"  {lbl:<10s} {cnt:>14,} records  ({pct:>5.1f}%)  "
                f"{unique:>10,} unique junction_aa"
            )
        lines.append("")

    prod_rows = species[species["category"] == "productive"].copy()
    if len(prod_rows) > 0:
        lines.append("Productive breakdown:")
        for _, row in prod_rows.iterrows():
            lbl = str(row["label"])
            cnt = int(row["total_records"])
            pct = 100.0 * cnt / total if total else 0
            lines.append(f"  {lbl:<20s} {cnt:>14,}  ({pct:>5.1f}%)")
        lines.append("")

    # Species from study metadata
    all_species = set()
    for sp_set in stats["study_species"].values():
        all_species.update(sp_set)
    if all_species:
        lines.append(
            f"Species (from study metadata): {', '.join(sorted(all_species))}"
        )
        lines.append("")

    # --- 7. STUDY LANDSCAPE ---
    lines.append("7. STUDY LANDSCAPE")
    lines.append("-" * 80)
    lines.append("")

    total_studies = len(studies)

    lines.append(f"Total studies: {total_studies}")
    lines.append(f"Total repositories: {num_repos}")
    lines.append("")

    lines.append("Largest studies by record count:")
    top_studies = studies.nlargest(10, "total_records")
    for _, row in top_studies.iterrows():
        study_label = str(row["study"])[:60]
        lines.append(
            f"  {study_label:<62s} {int(row['total_records']):>14,} records"
        )
    lines.append("")

    top5_records = int(top_studies.head(5)["total_records"].sum())
    top5_pct = 100.0 * top5_records / total if total else 0
    lines.append(
        f"The top 5 studies account for ~{top5_pct:.0f}% of all records."
    )
    lines.append("")

    # --- 8. DATA COMPLETENESS ---
    lines.append("8. DATA COMPLETENESS")
    lines.append("-" * 80)
    lines.append("")

    criteria = {
        "has_junction_aa": "Record has a non-empty junction amino acid sequence.",
        "has_v_call": "Record has a V gene call annotation.",
        "productive": "Record is marked as productive rearrangement.",
        "no_epitope_no_mhc": "Missing both epitope and MHC (all ADC records).",
    }

    lines.append(
        f"  {'Category':<25s} {'Records':>14s} {'%':>7s}"
    )
    for _, row in completeness.iterrows():
        cat = row["category"]
        cnt = int(row["record_count"])
        pct = 100.0 * cnt / total if total else 0
        lines.append(f"  {cat:<25s} {cnt:>14,} {pct:>6.1f}%")
    lines.append("")

    lines.append("Criteria:")
    for cat, desc in criteria.items():
        lines.append(f"  - {cat:<25s} {desc}")
    lines.append("")

    lines.append(
        "All ADC records are TCR/BCR-only (no epitope or MHC annotation).\n"
        "This data is best suited for unsupervised pre-training, repertoire\n"
        "analysis, and language modeling."
    )
    lines.append("")

    # --- 9. KEY TAKEAWAYS FOR MODELING ---
    lines.append("9. KEY TAKEAWAYS FOR MODELING")
    lines.append("-" * 80)
    lines.append("")

    lines.append(
        f"a) Data scale: {total:,} records across {num_files:,} files from "
        f"{num_repos} repositories.\n"
        f"   {int(r['unique_TRB_junction_aa']):,} unique TRB junction_aa and "
        f"{int(r['unique_TRA_junction_aa']):,} unique TRA junction_aa.\n"
        f"   {int(r['unique_IGH_junction_aa']):,} unique IGH junction_aa."
    )
    lines.append("")

    lines.append(
        f"b) Locus diversity: Data includes both TCR (TRA/TRB) and BCR "
        f"(IGH/IGK/IGL) sequences.\n"
        f"   Mixed locus types enable cross-receptor analysis and multi-chain\n"
        f"   language modeling."
    )
    lines.append("")

    lines.append(
        f"c) Paired cell data: {paired:,} cells with paired TRA+TRB chains\n"
        f"   (from single-cell data with cell_id). Most data is bulk\n"
        f"   repertoire without pairing information."
    )
    lines.append("")

    lines.append(
        "d) No epitope/MHC: ADC does not include epitope or MHC annotations.\n"
        "   This data cannot be used for TCR-pMHC binding prediction\n"
        "   but is valuable for TCR/BCR language modeling and repertoire analysis."
    )
    lines.append("")

    lines.append(
        "e) V/J gene annotation: V and J gene calls are available for\n"
        "   most records, enabling gene usage analysis across repositories."
    )
    lines.append("")

    if all_species:
        lines.append(
            f"f) Species coverage: Data spans {len(all_species)} species: "
            f"{', '.join(sorted(all_species))}."
        )
    else:
        lines.append(
            "f) Species coverage: Species metadata available from repertoires.json."
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

    print(f"Loading repertoire metadata from {data_dir} ...")
    repo_metadata = load_repertoire_metadata(data_dir)
    total_repertoires = sum(
        len(m["rep_to_study"]) for m in repo_metadata.values()
    )
    total_studies = len(
        set(
            study
            for m in repo_metadata.values()
            for study in m["studies"].keys()
        )
    )
    print(f"  {len(repo_metadata)} repositories, {total_repertoires} repertoires, "
          f"{total_studies} studies")
    print()

    print(f"Streaming ADC data from {data_dir} ...")
    stats = aggregate_data(data_dir, repo_metadata)
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  Productive records: {stats['productive_count']:,}")
    for locus in sorted(stats["locus_counts"]):
        cnt = stats["locus_counts"][locus]
        unique = len(stats["junction_aa_by_locus"].get(locus, set()))
        print(f"  {locus}: {cnt:,} records, {unique:,} unique junction_aa")
    print(f"  Paired cells (TRA+TRB): {stats['total_paired_records']:,}")
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
        print(f"  -> {path}  ({len(result)} rows, {result.columns.size} cols)")

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
        f"Category sum: {cat_sum:,}, Overview total: {all_total:,}, "
        f"match: {cat_sum == all_total}"
    )

    comp = pd.read_csv(
        os.path.join(output_dir, "data_completeness.tsv"), sep="\t"
    )
    no_epi_row = comp[comp["category"] == "no_epitope_no_mhc"]
    if not no_epi_row.empty:
        comp_total = int(no_epi_row.iloc[0]["record_count"])
        print(
            f"Completeness (no_epitope_no_mhc): {comp_total:,}, "
            f"Overview total: {all_total:,}, match: {comp_total == all_total}"
        )

    summary_path = os.path.join(output_dir, "SUMMARY.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            content = f.read()
        section_count = content.count(". ")
        # Count numbered sections
        sections_found = sum(
            1 for i in range(1, 10) if f"\n{i}. " in content
        )
        print(f"SUMMARY.txt sections found: {sections_found}/9")

    print("\nDone.")


if __name__ == "__main__":
    main()
