#!/usr/bin/env python3
"""Phase 1 of the Stage 4 combined pipeline: recover cluster + split files
that the CPT dataset builder requires.

The original cluster outputs (/home/ubuntu/quest/data/molecule_clusters/)
were lost when the x2gd instance was decommissioned. Existing CPT datasets
(Run 08 + Run 06c) carry per-row cluster IDs and split membership, so we
derive the sequence -> cluster_id and cluster_id -> split mappings from
them.

Outputs match the schema the builder expects (see
``cpt_dataset_builder.py`` lines 657-668):

    /home/ubuntu/quest/data/molecule_clusters/clusters/
        trb_cdr3_clusters.parquet        (trb_cdr3, cluster_id)
        tra_cdr3_clusters.parquet        (tra_cdr3, cluster_id)
        peptide_clusters.parquet         (peptide,  cluster_id)
        mhc_one_pocket_clusters.parquet  (mhc_one_pocket, cluster_id)
        mhc_two_pocket_clusters.parquet  (mhc_two_pocket, cluster_id)

    /home/ubuntu/quest/data/molecule_clusters/split_assignments/
        trb_cdr3_split_assignment.parquet         (cluster_id, split)
        tra_cdr3_split_assignment.parquet         (cluster_id, split)
        peptide_split_assignment.parquet          (cluster_id, split)
        mhc_one_pocket_split_assignment.parquet   (cluster_id, split)
        mhc_two_pocket_split_assignment.parquet   (cluster_id, split)

Validation: re-route 1000 held-out rows from Run 01 through the derived
files and confirm >= 95% match the actual split.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk

OUTPUT_ROOT = Path("/home/ubuntu/quest/data/molecule_clusters")
CLUSTERS_DIR = OUTPUT_ROOT / "clusters"
SPLIT_DIR = OUTPUT_ROOT / "split_assignments"
PHASE_MARKER = Path("/home/ubuntu/quest/checkpoints/_phase1_done")
REPORT_PATH = OUTPUT_ROOT / "DERIVATION_REPORT.md"

SOURCES = [
    "/home/ubuntu/quest/data/cpt_datasets/run08_cdr3_pocket_proportional_100M",
    "/home/ubuntu/quest/data/cpt_datasets/run06c_cdr3_pocket_balanced_100M",
]
VALIDATION_DATASET = "/home/ubuntu/quest/data/cpt_datasets/run01_cdr3_pocket_proportional_10M"
VALIDATION_SAMPLES_PER_SPLIT = 333  # ~1000 total across train/val/test

ROUTING_ACCURACY_THRESHOLD = 0.95

# The CPT DatasetDict names its splits 'train' / 'validation' / 'test', but the
# builder's routing SQL (cpt_dataset_builder.py:740-742, 864) expects the split
# literals 'train' / 'val' / 'test'. We aggregate using the DatasetDict names
# (that is what we observe when scanning Run 08/06c) and translate only at write
# time so the persisted split_assignment parquet matches the builder's vocabulary.
BUILDER_SPLIT_NAME = {"train": "train", "validation": "val", "test": "test"}

# Builder uses `<eos>` (and the existing CPT input_text may have whitespace
# around it depending on tokenizer-fix mode). Strip whitespace on segments.
EOS = "<eos>"

# Molecule type -> (cluster file basename, split file basename, column name)
MOLECULE_SPEC = {
    "trb_cdr3":        ("trb_cdr3_clusters.parquet",        "trb_cdr3_split_assignment.parquet",        "trb_cdr3"),
    "tra_cdr3":        ("tra_cdr3_clusters.parquet",        "tra_cdr3_split_assignment.parquet",        "tra_cdr3"),
    "peptide":         ("peptide_clusters.parquet",         "peptide_split_assignment.parquet",         "peptide"),
    "mhc_one_pocket":  ("mhc_one_pocket_clusters.parquet",  "mhc_one_pocket_split_assignment.parquet",  "mhc_one_pocket"),
    "mhc_two_pocket":  ("mhc_two_pocket_clusters.parquet",  "mhc_two_pocket_split_assignment.parquet",  "mhc_two_pocket"),
}

# Map order_key component name -> (molecule_type, cluster_id_col_in_dataset)
COMPONENT_TO_MOLECULE = {
    "trb":     ("trb_cdr3",       "trb_cluster"),
    "tra":     ("tra_cdr3",       "tra_cluster"),
    "peptide": ("peptide",        "peptide_cluster"),
    "mhc_one": ("mhc_one_pocket", "mhc_one_cluster"),
    "mhc_two": ("mhc_two_pocket", "mhc_two_cluster"),
}


def _log(msg: str) -> None:
    print(f"[recover_clusters] {msg}", flush=True)


def parse_order_key(order_key: str) -> list[str]:
    """Split ``order_key`` honoring ``mhc_one`` / ``mhc_two`` as single tokens.

    Example: ``"tra_trb_peptide_mhc_one"`` -> ``['tra', 'trb', 'peptide', 'mhc_one']``.
    """
    parts = order_key.split("_")
    out: list[str] = []
    i = 0
    while i < len(parts):
        if parts[i] == "mhc" and i + 1 < len(parts) and parts[i + 1] in ("one", "two"):
            out.append(f"mhc_{parts[i + 1]}")
            i += 2
        else:
            out.append(parts[i])
            i += 1
    return out


def parse_input_text(input_text: str, order_key: str) -> dict[str, str] | None:
    """Return {component: sequence} dict, or None on shape mismatch."""
    # Normalize whitespace around <eos> (Stage 0 used " <eos> " padding for ESM-2 fixb).
    cleaned = input_text.replace(" " + EOS + " ", EOS).replace(EOS + " ", EOS).replace(" " + EOS, EOS)
    segments = cleaned.split(EOS)
    components = parse_order_key(order_key)
    if len(segments) != len(components):
        return None
    return {comp: seg.strip() for comp, seg in zip(components, segments)}


class AggregationState:
    """Per-molecule (sequence -> cluster_id) and (cluster_id -> set of splits)."""

    def __init__(self) -> None:
        self.seq_to_cluster: dict[str, dict[str, int]] = {m: {} for m in MOLECULE_SPEC}
        self.cluster_to_splits: dict[str, dict[int, set[str]]] = {
            m: defaultdict(set) for m in MOLECULE_SPEC
        }
        self.conflicts: dict[str, int] = {m: 0 for m in MOLECULE_SPEC}

    def update_row(self, sequences: dict[str, str], cluster_ids: dict[str, int], split: str) -> None:
        for comp, seq in sequences.items():
            if comp not in COMPONENT_TO_MOLECULE:
                continue
            molecule, cid_col = COMPONENT_TO_MOLECULE[comp]
            cid = cluster_ids.get(cid_col, -1)
            if cid is None or cid == -1 or not seq:
                continue
            prev = self.seq_to_cluster[molecule].get(seq)
            if prev is None:
                self.seq_to_cluster[molecule][seq] = cid
            elif prev != cid:
                self.conflicts[molecule] += 1
                # Keep first-seen cluster_id; cross-source disagreement is rare
                # and indicates the original clustering didn't fully dedupe.
            self.cluster_to_splits[molecule][cid].add(split)


def process_split(ds_split, split_name: str, state: AggregationState, batch_size: int = 20000) -> int:
    """Stream a dataset split through the aggregator."""
    cols = ["input_text", "order_key", "trb_cluster", "tra_cluster",
            "peptide_cluster", "mhc_one_cluster", "mhc_two_cluster"]
    seen = 0
    parse_fail = 0
    for batch in ds_split.select_columns(cols).iter(batch_size=batch_size):
        n = len(batch["input_text"])
        input_texts = batch["input_text"]
        order_keys = batch["order_key"]
        trb_c = batch["trb_cluster"]
        tra_c = batch["tra_cluster"]
        pep_c = batch["peptide_cluster"]
        m1_c = batch["mhc_one_cluster"]
        m2_c = batch["mhc_two_cluster"]
        for i in range(n):
            seqs = parse_input_text(input_texts[i], order_keys[i])
            if seqs is None:
                parse_fail += 1
                continue
            cluster_ids = {
                "trb_cluster":     trb_c[i],
                "tra_cluster":     tra_c[i],
                "peptide_cluster": pep_c[i],
                "mhc_one_cluster": m1_c[i],
                "mhc_two_cluster": m2_c[i],
            }
            state.update_row(seqs, cluster_ids, split_name)
            seen += 1
        if seen and seen % 5_000_000 < batch_size:
            _log(f"  {split_name}: {seen:,} rows processed (parse_fail={parse_fail})")
    _log(f"  {split_name}: total {seen:,} rows processed (parse_fail={parse_fail})")
    return seen


def derive_cluster_to_split(cluster_to_splits: dict[int, set[str]]) -> dict[int, str]:
    """Priority routing: test > val > train."""
    out: dict[int, str] = {}
    for cid, splits in cluster_to_splits.items():
        if splits == {"test"}:
            out[cid] = "test"
        elif "train" not in splits:
            out[cid] = "validation"
        else:
            out[cid] = "train"
    return out


def write_cluster_file(path: Path, sequences: dict[str, int], col_name: str) -> int:
    """Write (sequence, cluster_id) parquet. Returns rows."""
    if not sequences:
        _log(f"  WARNING: zero sequences for {path.name} — writing empty parquet")
    sorted_items = sorted(sequences.items())
    seq_arr = pa.array([s for s, _ in sorted_items], type=pa.large_string())
    cid_arr = pa.array([c for _, c in sorted_items], type=pa.int64())
    table = pa.table({col_name: seq_arr, "cluster_id": cid_arr})
    pq.write_table(table, str(path), compression="snappy")
    return len(sorted_items)


def write_split_file(path: Path, cluster_to_split: dict[int, str]) -> int:
    """Write (cluster_id, split) parquet using the builder's split vocabulary.

    cluster_to_split values use the DatasetDict names ('train'/'validation'/
    'test'); we translate 'validation' -> 'val' here so the builder's routing
    SQL matches (see BUILDER_SPLIT_NAME).
    """
    sorted_items = sorted(cluster_to_split.items())
    cid_arr = pa.array([c for c, _ in sorted_items], type=pa.int64())
    spl_arr = pa.array([BUILDER_SPLIT_NAME[s] for _, s in sorted_items], type=pa.string())
    table = pa.table({"cluster_id": cid_arr, "split": spl_arr})
    pq.write_table(table, str(path), compression="snappy")
    return len(sorted_items)


def validate_routing(state: AggregationState, cluster_to_split: dict[str, dict[int, str]]) -> tuple[float, dict]:
    """Re-route held-out Run 01 rows; return (accuracy, stats)."""
    _log(f"Loading validation dataset: {VALIDATION_DATASET}")
    val_ds = load_from_disk(VALIDATION_DATASET)

    correct = 0
    total = 0
    skipped_no_lookup = 0
    per_split = {"train": [0, 0], "validation": [0, 0], "test": [0, 0]}  # [correct, total]
    for split_name in ["train", "validation", "test"]:
        if split_name not in val_ds:
            continue
        n = min(VALIDATION_SAMPLES_PER_SPLIT, len(val_ds[split_name]))
        sample = val_ds[split_name].shuffle(seed=42).select(range(n))
        cols = ["input_text", "order_key", "trb_cluster", "tra_cluster",
                "peptide_cluster", "mhc_one_cluster", "mhc_two_cluster"]
        for row in sample.select_columns(cols):
            seqs = parse_input_text(row["input_text"], row["order_key"])
            if seqs is None:
                skipped_no_lookup += 1
                continue
            observed_splits: set[str] = set()
            for comp, seq in seqs.items():
                if comp not in COMPONENT_TO_MOLECULE:
                    continue
                molecule, cid_col = COMPONENT_TO_MOLECULE[comp]
                cid = row[cid_col]
                if cid is None or cid == -1 or not seq:
                    continue
                predicted = cluster_to_split[molecule].get(cid)
                if predicted is not None:
                    observed_splits.add(predicted)
            if not observed_splits:
                skipped_no_lookup += 1
                continue
            # Apply priority: test > val > train
            if "test" in observed_splits:
                predicted_split = "test"
            elif "validation" in observed_splits:
                predicted_split = "validation"
            else:
                predicted_split = "train"
            per_split[split_name][1] += 1
            total += 1
            if predicted_split == split_name:
                per_split[split_name][0] += 1
                correct += 1

    accuracy = correct / total if total else 0.0
    stats = {
        "total_checked": total,
        "correct": correct,
        "skipped_no_lookup": skipped_no_lookup,
        "per_split": {
            k: {"correct": v[0], "total": v[1], "accuracy": (v[0] / v[1] if v[1] else 0.0)}
            for k, v in per_split.items()
        },
    }
    return accuracy, stats


def main() -> None:
    start = time.time()
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    state = AggregationState()
    total_rows = 0
    for src in SOURCES:
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Source dataset missing: {src}")
        _log(f"Processing {src}")
        ds = load_from_disk(src)
        for split_name in ("train", "validation", "test"):
            if split_name in ds:
                total_rows += process_split(ds[split_name], split_name, state)
        _log(f"  conflicts so far: {state.conflicts}")

    # Per-molecule split derivation + file writes.
    per_molecule_stats: dict[str, dict] = {}
    cluster_to_split_lookup: dict[str, dict[int, str]] = {}
    for molecule, (cluster_fname, split_fname, col_name) in MOLECULE_SPEC.items():
        cluster_map = state.seq_to_cluster[molecule]
        c2s = derive_cluster_to_split(state.cluster_to_splits[molecule])
        cluster_to_split_lookup[molecule] = c2s

        cpath = CLUSTERS_DIR / cluster_fname
        spath = SPLIT_DIR / split_fname
        n_seq = write_cluster_file(cpath, cluster_map, col_name)
        n_clu = write_split_file(spath, c2s)

        split_counts = {"train": 0, "validation": 0, "test": 0}
        for split in c2s.values():
            split_counts[split] += 1
        per_molecule_stats[molecule] = {
            "unique_sequences": n_seq,
            "unique_clusters": n_clu,
            "cluster_split_counts": split_counts,
            "split_fractions": {
                k: (v / n_clu if n_clu else 0.0) for k, v in split_counts.items()
            },
            "conflicts_first_wins": state.conflicts[molecule],
            "cluster_file_bytes": cpath.stat().st_size,
            "split_file_bytes": spath.stat().st_size,
        }
        _log(
            f"{molecule}: seq={n_seq:,}  clusters={n_clu:,}  "
            f"splits={split_counts}  conflicts={state.conflicts[molecule]}"
        )

    # Validation: routing accuracy on held-out Run 01.
    accuracy, val_stats = validate_routing(state, cluster_to_split_lookup)
    _log(f"Routing validation accuracy on Run 01 sample: {accuracy:.1%}")
    for split, s in val_stats["per_split"].items():
        _log(f"  {split:10}  acc={s['accuracy']:.1%}  ({s['correct']}/{s['total']})")

    elapsed = time.time() - start
    status = "PASS" if accuracy >= ROUTING_ACCURACY_THRESHOLD else (
        f"FAIL: routing accuracy {accuracy:.1%} < {ROUTING_ACCURACY_THRESHOLD:.0%}"
    )

    # Write report
    with open(REPORT_PATH, "w") as f:
        f.write("# Cluster Derivation Report\n\n")
        f.write(f"- Wall time: {elapsed / 60:.1f} min\n")
        f.write(f"- Total source rows scanned: {total_rows:,}\n")
        f.write(f"- Routing validation accuracy on Run 01: {accuracy:.4f}\n")
        f.write(f"- Status: {status}\n\n")
        f.write("## Per-molecule stats\n\n")
        f.write("```json\n")
        f.write(json.dumps(per_molecule_stats, indent=2))
        f.write("\n```\n\n## Routing validation details\n\n```json\n")
        f.write(json.dumps(val_stats, indent=2))
        f.write("\n```\n")

    # Write phase marker
    PHASE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE_MARKER, "w") as f:
        f.write(f"STATUS={'PASS' if accuracy >= ROUTING_ACCURACY_THRESHOLD else 'FAIL'}\n")
        f.write(f"routing_accuracy={accuracy:.4f}\n")
        f.write(f"total_rows_scanned={total_rows}\n")
        f.write(f"elapsed_min={elapsed / 60:.1f}\n")

    if accuracy < ROUTING_ACCURACY_THRESHOLD:
        raise SystemExit(f"Phase 1 FAIL: routing accuracy {accuracy:.1%} below threshold")

    _log(f"Phase 1 PASS in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
