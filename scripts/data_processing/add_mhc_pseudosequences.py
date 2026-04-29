"""Add MHC pseudosequence / contact-point columns to deduplicated parquet files.

Adds 6 columns:
  mhc_one_pocket, mhc_one_contact, mhc_one_pocket_contact  (from mhc_one, Class I)
  mhc_two_pocket, mhc_two_contact, mhc_two_pocket_contact  (from mhc_two, Class II)

Input : /home/ubuntu/quest/data/deduplicated_again/exploded_deduped
Output: /home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched
"""

from __future__ import annotations

import os
import sys
import time
import json
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


INPUT_DIR = Path("/home/ubuntu/quest/data/deduplicated_again/exploded_deduped")
OUTPUT_DIR = Path("/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched")
LOOKUP_CACHE = Path("/home/ubuntu/quest/data/deduplicated_again/mhc_pseudo_lookup.json")

MAX_WORKERS = 60  # Leave 4 cores for OS on 64-core box

# ---------- Position definitions (1-indexed mature protein numbering) --------

MHCI_POCKET_POSITIONS = [
    7, 9, 24, 25, 34, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76,
    77, 80, 81, 84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152,
    156, 158, 159, 160, 163, 167, 171,
]  # 37

MHCI_CONTACT_POSITIONS = [
    58, 62, 65, 66, 69, 72, 73, 75, 76, 79, 80,
    150, 151, 152, 154, 155, 158, 159, 162, 163, 166,
]  # 21

MHCI_POCKET_CONTACT_POSITIONS = sorted(
    set(MHCI_POCKET_POSITIONS) | set(MHCI_CONTACT_POSITIONS)
)  # 47

MHCII_BETA_POCKET_POSITIONS = [
    9, 11, 13, 26, 28, 30, 47, 57, 67, 70, 71, 74, 77, 78, 81, 85, 86, 89, 90
]  # 19

MHCII_BETA_CONTACT_POSITIONS = [64, 67, 70, 71, 73, 77, 78, 81]  # 8

MHCII_BETA_POCKET_CONTACT_POSITIONS = sorted(
    set(MHCII_BETA_POCKET_POSITIONS) | set(MHCII_BETA_CONTACT_POSITIONS)
)

MHCII_ALPHA_POCKET_POSITIONS = [9, 11, 22, 24, 31, 52, 53, 58, 59, 61, 65, 66, 68, 72, 73]  # 15
MHCII_ALPHA_CONTACT_POSITIONS = [57, 61, 62, 65, 68, 69, 72, 73, 76]  # 9
MHCII_ALPHA_POCKET_CONTACT_POSITIONS = sorted(
    set(MHCII_ALPHA_POCKET_POSITIONS) | set(MHCII_ALPHA_CONTACT_POSITIONS)
)

CLASS_II_CONCAT_THRESHOLD = 380  # Above this length we assume α+β concat

# --------- Offset detection ---------

# Known Class II β-chain mature N-terminal motifs
CLASS_II_BETA_START_MOTIFS = ["GDTRPR", "RDSPED", "RDSPE", "RATPEN", "RATPE"]
# Known Class II α-chain mature N-terminal motifs
CLASS_II_ALPHA_START_MOTIFS = ["IKEEHV", "EDIVAD", "IKADHV", "IKADHVS"]


def class_i_offset(seq: str) -> int:
    """Return 0-based index of mature position 1 in seq.

    Returns -1 if mature position 1 (G) is missing from the sequence (i.e., seq
    starts directly with SHSMRYF). In that case positions >= 2 are still
    recoverable via seq[pos - 2].
    """
    i = seq.find("SHSMRYF")
    if i >= 1:
        return i - 1
    if i == 0:
        return -1  # G stripped; position 1 absent
    i = seq.find("SMRYF")
    if i >= 3:
        return i - 3
    # Length-based fallback
    if len(seq) >= 300:
        return 24
    return 0


def class_ii_beta_offset(seq: str) -> int:
    """Return 0-based index of β-chain mature position 1."""
    for motif in CLASS_II_BETA_START_MOTIFS:
        idx = seq.find(motif)
        if idx >= 0:
            return idx
    # Try the conserved "FNGT" motif: appears at positions ~14-22 of mature
    # depending on isotype. Without more info we default to offset 0.
    return 0


def class_ii_alpha_offset(seq: str) -> int:
    for motif in CLASS_II_ALPHA_START_MOTIFS:
        idx = seq.find(motif)
        if idx >= 0:
            return idx
    return 0


def extract_positions(seq: str, offset: int, positions: list[int]) -> str:
    """Extract 1-indexed positions from mature start at `offset`.

    Returns empty string if the sequence is too short or any position is invalid.
    """
    if offset < -1 or offset >= len(seq):
        return ""
    out = []
    for p in positions:
        idx = offset + p - 1
        if idx < 0 or idx >= len(seq):
            return ""
        out.append(seq[idx])
    return "".join(out)


# --------- Class I lookup builder ---------

def build_class_i_entry(seq: str) -> tuple[str, str, str]:
    offset = class_i_offset(seq)
    pocket = extract_positions(seq, offset, MHCI_POCKET_POSITIONS)
    contact = extract_positions(seq, offset, MHCI_CONTACT_POSITIONS)
    pc_ = extract_positions(seq, offset, MHCI_POCKET_CONTACT_POSITIONS)
    return pocket, contact, pc_


def build_class_ii_entry(seq: str) -> tuple[str, str, str]:
    """Handle β-only or α+β concat. Extract β-chain pseudosequence; if the
    sequence is long enough to be an α+β concatenation we also prepend α-chain
    positions.
    """
    if len(seq) >= CLASS_II_CONCAT_THRESHOLD:
        # Try to find β-chain start in the second half
        half = len(seq) // 2
        beta_idx = -1
        for motif in CLASS_II_BETA_START_MOTIFS:
            idx = seq.find(motif, half - 50)
            if idx >= 0:
                beta_idx = idx
                break
        if beta_idx > 0:
            alpha_seq = seq[:beta_idx]
            beta_seq = seq[beta_idx:]
            alpha_off = class_ii_alpha_offset(alpha_seq)
            beta_off = class_ii_beta_offset(beta_seq)
            pocket = (
                extract_positions(alpha_seq, alpha_off, MHCII_ALPHA_POCKET_POSITIONS)
                + extract_positions(beta_seq, beta_off, MHCII_BETA_POCKET_POSITIONS)
            )
            contact = (
                extract_positions(alpha_seq, alpha_off, MHCII_ALPHA_CONTACT_POSITIONS)
                + extract_positions(beta_seq, beta_off, MHCII_BETA_CONTACT_POSITIONS)
            )
            pc_ = (
                extract_positions(alpha_seq, alpha_off, MHCII_ALPHA_POCKET_CONTACT_POSITIONS)
                + extract_positions(beta_seq, beta_off, MHCII_BETA_POCKET_CONTACT_POSITIONS)
            )
            # If any extraction failed (empty), fallback to β-only logic
            if pocket and contact and pc_:
                return pocket, contact, pc_

    # β-only path
    offset = class_ii_beta_offset(seq)
    pocket = extract_positions(seq, offset, MHCII_BETA_POCKET_POSITIONS)
    contact = extract_positions(seq, offset, MHCII_BETA_CONTACT_POSITIONS)
    pc_ = extract_positions(seq, offset, MHCII_BETA_POCKET_CONTACT_POSITIONS)
    return pocket, contact, pc_


# --------- File I/O helpers ---------

def find_parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


def _scan_file_unique(path: str) -> tuple[list[str], list[str]]:
    """Return unique non-empty mhc_one and mhc_two strings from one parquet file."""
    t = pq.read_table(path, columns=["mhc_one", "mhc_two"])
    def _uniq(col):
        arr = pc.unique(t[col])
        return [s for s in arr.to_pylist() if s]
    return _uniq("mhc_one"), _uniq("mhc_two")


def scan_all_unique(paths: list[Path]) -> tuple[set[str], set[str]]:
    m1: set[str] = set()
    m2: set[str] = set()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_scan_file_unique, str(p)) for p in paths]
        for i, fut in enumerate(as_completed(futures), 1):
            a, b = fut.result()
            m1.update(a)
            m2.update(b)
            if i % 50 == 0 or i == len(paths):
                dt = time.time() - t0
                print(
                    f"  scan {i}/{len(paths)} files ({dt:.1f}s) "
                    f"| uniq_mhc_one={len(m1)} uniq_mhc_two={len(m2)}",
                    flush=True,
                )
    return m1, m2


def build_lookups(
    m1_seqs: set[str], m2_seqs: set[str]
) -> tuple[dict, dict, dict]:
    """Return (class_i_lookup, class_ii_lookup, report)."""
    c1 = {s: build_class_i_entry(s) for s in m1_seqs}
    c2 = {s: build_class_ii_entry(s) for s in m2_seqs}

    report = {
        "unique_mhc_one": len(m1_seqs),
        "unique_mhc_two": len(m2_seqs),
        "mhc_one_short_or_invalid": sum(1 for v in c1.values() if not v[0]),
        "mhc_two_short_or_invalid": sum(1 for v in c2.values() if not v[0]),
    }
    return c1, c2, report


def validate_class_i(lookup: dict) -> dict:
    """Verify pos 45=M, pos 116=Y on sequences with the HLA-A*02:01 signature.

    Since we have no allele column, we instead scan for any sequence whose
    offset-detected mature protein has M at position 45 AND Y at position 116
    (the HLA-A*02:01 signature). A non-zero count means the offset logic is
    producing a canonical Bjorkman-numbered extraction for at least some
    sequences known to exhibit that signature.
    """
    a0201_signature_count = 0
    signature_ok_count = 0
    samples = []
    fail_examples = []
    for seq in lookup.keys():
        offset = class_i_offset(seq)
        if offset < 0 or offset + 115 >= len(seq):
            continue
        p45 = seq[offset + 44]
        p116 = seq[offset + 115]
        if p45 == "M" and p116 == "Y":
            a0201_signature_count += 1
            signature_ok_count += 1
            if len(samples) < 3:
                samples.append({
                    "len": len(seq),
                    "offset": offset,
                    "pos_45": p45,
                    "pos_116": p116,
                    "mature_start_40": seq[offset:offset + 40],
                    "pocket": lookup[seq][0],
                    "contact": lookup[seq][1],
                })
        else:
            # Collect a few counter-examples for diagnostics
            if len(fail_examples) < 3:
                fail_examples.append({
                    "len": len(seq),
                    "offset": offset,
                    "pos_45": p45,
                    "pos_116": p116,
                })
    return {
        "hla_a0201_signature_count": a0201_signature_count,
        "hla_a0201_samples": samples,
        "non_a0201_examples": fail_examples,
        "validated": signature_ok_count > 0,
    }


# --------- Per-file processing worker ---------

def _process_file(args: tuple[str, str, str]) -> tuple[str, int, int]:
    """Read one parquet file, add 6 columns via lookup, write to output dir."""
    in_path, out_path, lookup_path = args
    # Lazy load the lookup (each process loads once from a JSON cache)
    global _C1, _C2
    try:
        _C1
    except NameError:
        with open(lookup_path, "r") as f:
            data = json.load(f)
        _C1 = {k: tuple(v) for k, v in data["class_i"].items()}
        _C2 = {k: tuple(v) for k, v in data["class_ii"].items()}

    table = pq.read_table(in_path)
    n_rows = table.num_rows

    m1_col = table["mhc_one"].to_pylist()
    m2_col = table["mhc_two"].to_pylist()

    empty = ("", "", "")
    m1p, m1c, m1pc = [], [], []
    for s in m1_col:
        if not s:
            a, b, c = empty
        else:
            a, b, c = _C1.get(s, empty)
        m1p.append(a); m1c.append(b); m1pc.append(c)

    m2p, m2c, m2pc = [], [], []
    for s in m2_col:
        if not s:
            a, b, c = empty
        else:
            a, b, c = _C2.get(s, empty)
        m2p.append(a); m2c.append(b); m2pc.append(c)

    table = table.append_column("mhc_one_pocket", pa.array(m1p, type=pa.string()))
    table = table.append_column("mhc_one_contact", pa.array(m1c, type=pa.string()))
    table = table.append_column("mhc_one_pocket_contact", pa.array(m1pc, type=pa.string()))
    table = table.append_column("mhc_two_pocket", pa.array(m2p, type=pa.string()))
    table = table.append_column("mhc_two_contact", pa.array(m2c, type=pa.string()))
    table = table.append_column("mhc_two_pocket_contact", pa.array(m2pc, type=pa.string()))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
    )
    return in_path, n_rows, len(table.column_names)


# --------- Main ---------

def main() -> None:
    t_start = time.time()
    print(f"Input  : {INPUT_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = find_parquet_files(INPUT_DIR)
    print(f"Found {len(paths)} parquet files")

    # -------- Phase 1: scan --------
    print("\n== Phase 1: scanning for unique mhc_one / mhc_two sequences ==")
    m1_seqs, m2_seqs = scan_all_unique(paths)
    t_scan = time.time() - t_start
    print(f"  Unique mhc_one: {len(m1_seqs)}")
    print(f"  Unique mhc_two: {len(m2_seqs)}")
    print(f"  Scan time: {t_scan:.1f}s")

    # -------- Phase 2: build lookup --------
    print("\n== Phase 2: building lookup tables ==")
    c1, c2, report = build_lookups(m1_seqs, m2_seqs)

    # Determine dominant offset for reporting
    offset_counter = collections.Counter(class_i_offset(s) for s in m1_seqs)
    print(f"  Class I offset distribution (top 5): {offset_counter.most_common(5)}")

    # Class II length distribution
    len_counter = collections.Counter(len(s) for s in m2_seqs)
    print(f"  Class II length distribution (top 5): {len_counter.most_common(5)}")
    concat_count = sum(1 for s in m2_seqs if len(s) >= CLASS_II_CONCAT_THRESHOLD)
    print(f"  Class II sequences >= {CLASS_II_CONCAT_THRESHOLD} aa (α+β concat): {concat_count}")

    validation = validate_class_i(c1)
    print(f"  HLA-A*02:01 validation: {validation}")

    # Persist the lookup to disk so worker processes can load it
    LOOKUP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOOKUP_CACHE, "w") as f:
        json.dump({
            "class_i": {k: list(v) for k, v in c1.items()},
            "class_ii": {k: list(v) for k, v in c2.items()},
        }, f)
    print(f"  Lookup cache written: {LOOKUP_CACHE}")

    # Warnings
    warn = []
    if report["mhc_one_short_or_invalid"]:
        warn.append(f"{report['mhc_one_short_or_invalid']} mhc_one sequences produced empty pocket (too short or undetectable)")
    if report["mhc_two_short_or_invalid"]:
        warn.append(f"{report['mhc_two_short_or_invalid']} mhc_two sequences produced empty pocket (too short or undetectable)")
    for w in warn:
        print(f"  WARN: {w}")

    # -------- Phase 3: process files --------
    print("\n== Phase 3: processing files in parallel ==")
    work = []
    for p in paths:
        rel = p.relative_to(INPUT_DIR)
        out = OUTPUT_DIR / rel
        work.append((str(p), str(out), str(LOOKUP_CACHE)))

    t_proc0 = time.time()
    total_rows = 0
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_process_file, w) for w in work]
        for fut in as_completed(futures):
            _, n_rows, _ = fut.result()
            total_rows += n_rows
            done += 1
            if done % 50 == 0 or done == len(work):
                elapsed = time.time() - t_proc0
                eta = (elapsed / done) * (len(work) - done) if done else 0
                print(
                    f"  {done}/{len(work)} files | {total_rows:,} rows | "
                    f"{elapsed:.1f}s elapsed | ETA {eta:.1f}s",
                    flush=True,
                )

    t_total = time.time() - t_start
    print("\n== Summary ==")
    print(f"  Files processed        : {len(work)}")
    print(f"  Total rows             : {total_rows:,}")
    print(f"  Unique mhc_one         : {len(m1_seqs)}")
    print(f"  Unique mhc_two         : {len(m2_seqs)}")
    print(f"  Class I offset (mode)  : {offset_counter.most_common(1)[0][0]}")
    print(f"  HLA-A*02:01 validated  : {validation['validated']}")
    print(f"  HLA-A*02:01 matches    : {len(validation['hla_a0201_matches'])}")
    print(f"  Warnings               : {len(warn)}")
    print(f"  Scan time              : {t_scan:.1f}s")
    print(f"  Processing time        : {time.time() - t_proc0:.1f}s")
    print(f"  Total time             : {t_total:.1f}s")
    print(f"  Output directory       : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
