"""Benchmark isolation for the CPT corpus (corpus repair, step 6).

Before this, `cpt_dataset_builder` never consulted the benchmark: its manifest carried a
hard-coded `antigenic_specificity_exclusion: NOT_APPLIED`, and run05's train split held the exact
receptor-peptide association for 103 of the 1,589 `test_novel_pep` molecules and 118 of its 463
peptides.

Policy `heldout-peptides-v2` (decided 2026-11-09): every row that carries a held-out peptide, or
a peptide closer to one than `min_edit_distance`, is excluded from the corpus, in every projection
that has a peptide, peptide-only rows included. Excluding the peptide removes every held-out
receptor-peptide association with it, in all order variants, since an association cannot exist
without its peptide.

`min_edit_distance` is the smallest Levenshtein distance a corpus peptide may have to any
held-out peptide. 1 excludes exact matches only. 2, the value decided for the repaired corpus,
also excludes every one-edit neighbour (one substitution, insertion or deletion): exact-only
exclusion left a one-substitution neighbour of 308 of the 672 held-out peptides in the corpus,
which is not peptide novelty in any useful sense. Neighbours are direct only; clusters are not
followed outward. (v1, exact-only with no recorded distance, was never used in a build.)

What is and is not covered, deliberately:

  * A held-out peptide is one that occurs in a held-out part of the benchmark task. The parts are
    named explicitly by the caller and recorded with the list.
  * No excluded peptide, held-out or neighbour, may occur in the task's own train part. If one does, removing it from
    CPT would hide from CPT something the fine-tuning stage is allowed to see, so building the
    list refuses instead of quietly widening the exclusion. This is why the parts whose peptides
    are seen in training by design (novel-TCR, novel-allele) cannot be isolated by this policy:
    they need pair-level exclusion, which is not implemented.
  * Receptors and MHC chains may recur on their own. Only peptide-bearing rows are affected.
  * Matching is exact on the peptide's letters: everything that is not an ASCII letter is removed
    and the rest upper-cased, identically in Python (`normalize_peptide`) and SQL
    (`normalize_peptide_sql`). This is wider than the MHC-chain normalization on purpose: a
    held-out peptide written with a non-breaking or zero-width space must still be excluded.
    Peptide clusters are not expanded: the benchmark's clusters and the corpus's are different id
    spaces with no crosswalk.

The list is a plain TSV so that it can be read, diffed and hashed; the builder and the verifier
both record its sha256.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from quest.data.mhc_eligibility import as_text

NO_EXCLUSION = "none"
HELDOUT_PEPTIDES = "heldout-peptides-v2"
EXCLUSION_POLICIES = (NO_EXCLUSION, HELDOUT_PEPTIDES)

_POLICY_PREFIX = "#policy="
_HEADER = "peptide\tdistance\theldout_peptide\tparts\tmolecules"
# Every ASCII letter, not the 20 standard residues: source peptides can carry B, J, O, U, X, Z.
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ORDER_TOKEN = re.compile(r"mhc_one|mhc_two|tra|trb|peptide")
EOS = "<eos>"


def normalize_peptide(peptide) -> str:
    """Letters only, upper-case. Missing values (None, NaN) normalize to ''."""
    return re.sub(r"[^A-Za-z]", "", as_text(peptide)).upper()


def normalize_peptide_sql(expr: str) -> str:
    """SQL twin of `normalize_peptide`; a test pins their agreement. NULL in gives NULL out."""
    return f"upper(regexp_replace({expr}, '[^A-Za-z]', '', 'g'))"


def one_edit_neighbours(peptide: str) -> set[str]:
    """Every string at Levenshtein distance exactly 1 from `peptide`."""
    out = set()
    for i in range(len(peptide)):
        out.add(peptide[:i] + peptide[i + 1:])
        for c in _ALPHABET:
            out.add(peptide[:i] + c + peptide[i + 1:])
    for i in range(len(peptide) + 1):
        for c in _ALPHABET:
            out.add(peptide[:i] + c + peptide[i:])
    out.discard(peptide)
    out.discard("")
    return out


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_exclusion(splits_dir, task: str, heldout_parts,
                    min_edit_distance: int) -> tuple[list[dict], dict]:
    """Collect excluded peptides for `task` from `{task}_{part}.parquet` under `splits_dir`.

    Rows with distance 0 are the held-out peptides themselves; with `min_edit_distance=2`, rows
    with distance 1 are their one-edit neighbours. Molecules are counted once per `row_id`; the
    split files hold one row per ordering variant.
    """
    if min_edit_distance not in (1, 2):
        raise ValueError("min_edit_distance must be 1 (exact only) or 2 (also one-edit neighbours); "
                         "a wider radius cannot be enumerated and is not implemented")
    import pandas as pd

    splits_dir = Path(splits_dir)
    heldout_parts = list(heldout_parts)
    if not heldout_parts:
        raise ValueError("at least one held-out part is required")
    if "train" in heldout_parts:
        raise ValueError("'train' cannot be a held-out part")

    def read(part):
        path = splits_dir / f"{task}_{part}.parquet"
        frame = pd.read_parquet(path, columns=["row_id", "peptide"])
        frame = frame.assign(peptide=frame.peptide.map(normalize_peptide))
        # One line per (molecule, peptide). Deduplicating on row_id alone would keep one peptide
        # and silently drop the other if a row_id ever carried two.
        return path, frame[frame.peptide != ""].drop_duplicates(["row_id", "peptide"])

    train_path, train = read("train")
    train_peptides = set(train.peptide) - {""}
    inputs = {train_path.name: sha256_file(train_path)}
    per_peptide: dict[str, dict[str, int]] = {}
    for part in heldout_parts:
        path, frame = read(part)
        inputs[path.name] = sha256_file(path)
        for peptide, n in frame.peptide.value_counts().items():
            if peptide:
                per_peptide.setdefault(peptide, {})[part] = int(n)

    seen_in_train = sorted(set(per_peptide) & train_peptides)
    if seen_in_train:
        raise ValueError(
            f"{len(seen_in_train)} held-out peptide(s) also occur in {task} train "
            f"(e.g. {seen_in_train[:3]}); excluding them would hide training peptides from CPT. "
            "Parts whose peptides are seen in training need pair-level exclusion instead.")

    rows = [{"peptide": p, "distance": 0, "heldout_peptide": p,
             "parts": ",".join(sorted(parts)), "molecules": sum(parts.values())}
            for p, parts in sorted(per_peptide.items())]
    if min_edit_distance == 2:
        nearest: dict[str, str] = {}
        for p in sorted(per_peptide):
            for n in one_edit_neighbours(p):
                if n not in per_peptide:
                    nearest.setdefault(n, p)            # the first held-out peptide in sort order
        near_train = sorted(set(nearest) & train_peptides)
        if near_train:
            raise ValueError(
                f"{len(near_train)} {task} train peptide(s) are one edit from a held-out peptide "
                f"(e.g. {near_train[:3]}); excluding them would hide training peptides from CPT.")
        rows += [{"peptide": n, "distance": 1, "heldout_peptide": p,
                  "parts": ",".join(sorted(per_peptide[p])), "molecules": 0}
                 for n, p in sorted(nearest.items())]
    meta = {
        "policy": HELDOUT_PEPTIDES, "min_edit_distance": min_edit_distance, "task": task,
        "heldout_parts": heldout_parts, "n_heldout_peptides": len(per_peptide),
        "n_peptides": len(rows), "n_train_peptides": len(train_peptides),
        "excluded_peptides_also_in_train": 0, "input_sha256": inputs,
        "not_covered": ("peptide clusters (no crosswalk to the corpus's clusters); held-out parts not "
                        "named above; receptor-only and MHC-only recurrence, which is permitted"),
    }
    return rows, meta


def write_exclusion(rows, meta, out_path) -> str:
    """Write the TSV and a `.meta.json` beside it; return the TSV's sha256."""
    out_path = Path(out_path)
    with out_path.open("x") as fh:
        fh.write(f"{_POLICY_PREFIX}{meta['policy']};min_edit_distance={meta['min_edit_distance']}\n")
        fh.write(_HEADER + "\n")
        for r in rows:
            fh.write(f"{r['peptide']}\t{r['distance']}\t{r['heldout_peptide']}\t{r['parts']}\t{r['molecules']}\n")
    digest = sha256_file(out_path)
    with Path(str(out_path) + ".meta.json").open("x") as fh:
        json.dump({**meta, "list_sha256": digest}, fh, indent=2)
        fh.write("\n")
    return digest


def _read_list(path):
    path = Path(path)
    lines = path.read_text().splitlines()
    head = re.fullmatch(rf"{_POLICY_PREFIX}([a-z0-9-]+);min_edit_distance=([12])", lines[0]) if lines else None
    if head is None or len(lines) < 2 or lines[1] != _HEADER:
        raise ValueError(f"{path} is not an exclusion list: expected '{_POLICY_PREFIX}...' then {_HEADER!r}")
    return path, lines[2:], {"policy": head[1], "min_edit_distance": int(head[2])}


def exclusion_list_info(path) -> dict:
    """The policy and distance the list was built with, read from its own first line."""
    return _read_list(path)[2]


def load_excluded_peptides(path) -> tuple[frozenset[str], str]:
    """Read the TSV written by `write_exclusion`. Returns (excluded peptides, file sha256)."""
    path, body, _info = _read_list(path)
    peptides = set()
    for n, line in enumerate(body, start=3):
        fields = line.split("\t")
        # Strict on purpose: the file is machine-written, so any deviation means it was edited or
        # truncated, and a lenient reader would turn that into a silently different exclusion.
        if len(fields) != 5 or not re.fullmatch(r"[A-Z]+", fields[0]) or fields[1] not in ("0", "1"):
            raise ValueError(f"{path}:{n}: not an exclusion-list line: {line[:60]!r}")
        peptides.add(fields[0])
    if not peptides:
        raise ValueError(f"{path} holds no peptides")
    return frozenset(peptides), sha256_file(path)


def peptide_of_output_row(input_text: str, order_key: str) -> str | None:
    """The peptide component of a serialized CPT row, or None when the row has no peptide.

    `order_key` names the components in order ('trb_peptide_mhc_one'); `input_text` holds them
    joined by the literal EOS. Raises when the two disagree, because a silent mismatch would make
    every containment check built on this meaningless.
    """
    order = _ORDER_TOKEN.findall(order_key)
    if not order or "_".join(order) != order_key:
        raise ValueError(f"order_key {order_key!r} is not a '_'-joined list of known components")
    if "peptide" not in order:
        return None
    parts = input_text.split(EOS)
    if len(parts) != len(order):
        raise ValueError(f"order_key {order_key!r} names {len(order)} components, "
                         f"input_text has {len(parts)}")
    return normalize_peptide(parts[order.index("peptide")])
