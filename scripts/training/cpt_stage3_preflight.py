#!/usr/bin/env python3
"""TCRBench v3 — Stage 3 preflight: pocket annotation sanity check.

Stage 3 sweeps the MHC representation axis (Run 04 = pocket+contact, Run 05 =
full MHC) holding TCR rep at CDR3. The primary decision metric is
``pocket_ppl`` on MHC-bearing rows in canonical eval v2. That metric is
computed by ``region_restricted_eval``, which relies on
``annotate_token_regions`` correctly labeling pocket tokens inside the
``pocket_contact`` and ``full`` MHC variants (the overlay logic in
``identify_segment_regions``).

This script samples 50 MHC-bearing canonical eval rows, projects each into
the three (TCR=cdr3) × (MHC ∈ {pocket, pocket_contact, full}) variants,
tokenizes with the ESM-C 300M tokenizer (the Stage 2/3 tokenizer), and
checks that pocket-labeled tokens are produced wherever the row has a
non-empty ``mhc_one_pocket`` or ``mhc_two_pocket`` column.

Pass criterion: ≤2 failures / 50 sample for each variant. Failure means the
substring-overlay annotation in ``identify_segment_regions`` (lines 254–290
of ``cpt_training_lib.py``) didn't find the pocket residues inside the
larger segment — usually because the source columns disagree on canonical
form (e.g. trailing whitespace, alternative ordering).

CPU-only; ~5 minutes.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_from_disk

from cpt_training_lib import (
    _collect_special_token_ids,
    _preprocess_for_eos_strategy,
    annotate_token_regions,
    get_tokenizer,
    project_canonical_row_to_run_input,
)


CANONICAL_EVAL = "/home/ubuntu/quest/data/cpt_canonical_eval_v2"
BASE_MODEL = "esmc_300m"
EOS_STRATEGY = "raw"           # matches Stage 2/3 training
SAMPLE_SIZE = 50
SAMPLE_SEED = 42
PASS_THRESHOLD = 2             # max allowed failures per variant
VARIANTS = [
    ("cdr3", "pocket"),
    ("cdr3", "pocket_contact"),
    ("cdr3", "full"),
]


def _has_pocket(row: dict) -> bool:
    """Row should be expected to produce pocket-labeled tokens.

    True iff at least one of mhc_one_pocket / mhc_two_pocket is a non-empty
    string. Note: when an MHC chain is present but only the full chain
    column is populated (no pocket annotation), the row is still expected
    to yield pocket-labeled tokens for the ``pocket_contact``/``full``
    variants — but only IF mhc_one_pocket is populated, because the
    overlay relies on the pocket-string lookup. If mhc_one_pocket is empty,
    a missing pocket label is not a failure.
    """
    p1 = row.get("mhc_one_pocket")
    p2 = row.get("mhc_two_pocket")
    return bool((p1 and p1.strip()) or (p2 and p2.strip()))


def main() -> int:
    print(f"[preflight] loading canonical eval v2 from {CANONICAL_EVAL}")
    ds = load_from_disk(CANONICAL_EVAL)
    # The dataset is a DatasetDict in Stage 2 builds; flatten to the test
    # split (the canonical eval is the test slice already).
    if hasattr(ds, "keys"):
        keys = list(ds.keys())
        # Prefer 'test' if present (canonical eval naming), else first key.
        split = "test" if "test" in keys else keys[0]
        print(f"[preflight] dataset is DatasetDict; using split={split}")
        ds = ds[split]

    print(f"[preflight] total rows: {len(ds):,}")
    print(f"[preflight] columns: {sorted(ds.column_names)}")

    mhc_rows = ds.filter(lambda x: x.get("mhc_class") != "none")
    print(f"[preflight] MHC-bearing rows (mhc_class != 'none'): {len(mhc_rows):,}")
    if len(mhc_rows) < SAMPLE_SIZE:
        print(f"[preflight] ERROR: too few MHC-bearing rows ({len(mhc_rows)}) for "
              f"{SAMPLE_SIZE}-row sample")
        return 2

    sample = mhc_rows.shuffle(seed=SAMPLE_SEED).select(range(SAMPLE_SIZE))

    print(f"[preflight] loading tokenizer ({BASE_MODEL})")
    tok = get_tokenizer(BASE_MODEL)
    sep_id = int(tok.eos_token_id)
    special_ids = _collect_special_token_ids(tok)
    special_ids.add(sep_id)

    all_pass = True
    summary: list[tuple[str, int, int, int]] = []  # (variant, n_expect, n_fail, n_empty)
    for tcr_v, mhc_v in VARIANTS:
        n_expect = 0
        n_fail = 0
        n_empty_projection = 0
        for row in sample:
            text, kinds = project_canonical_row_to_run_input(row, tcr_v, mhc_v)
            if not text:
                n_empty_projection += 1
                continue
            prepped = _preprocess_for_eos_strategy(text, EOS_STRATEGY)
            enc = tok(prepped, add_special_tokens=True, truncation=True, max_length=1024)
            input_ids = list(enc["input_ids"])
            regions = annotate_token_regions(
                text,
                row,
                kinds,
                input_ids,
                separator_token_id=sep_id,
                special_token_ids=special_ids,
            )
            pocket_count = sum(1 for r in regions if r == "pocket")
            if _has_pocket(row):
                n_expect += 1
                if pocket_count == 0:
                    n_fail += 1
        summary.append((f"({tcr_v},{mhc_v})", n_expect, n_fail, n_empty_projection))
        verdict = "PASS" if n_fail <= PASS_THRESHOLD else "FAIL"
        if n_fail > PASS_THRESHOLD:
            all_pass = False
        print(
            f"  variant=({tcr_v:>5},{mhc_v:>14})  "
            f"n_expecting_pocket={n_expect:>3}  "
            f"failures={n_fail:>2}  "
            f"empty_projection={n_empty_projection:>2}  "
            f"[{verdict}]"
        )

    print()
    if all_pass:
        print("[preflight] PASS — all variants under failure threshold; safe to launch training.")
        return 0
    print(
        f"[preflight] FAIL — at least one variant exceeded threshold "
        f"({PASS_THRESHOLD} failures / {SAMPLE_SIZE} sample). "
        "Debug `identify_segment_regions` MHC overlay in cpt_training_lib.py "
        "(lines ~254-290) before launching Stage 3 training."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
