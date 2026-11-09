#!/usr/bin/env python3
"""Write the held-out peptide list that keeps a benchmark task out of the CPT corpus.

Policy and scope: `quest/data/benchmark_exclusion.py`. The held-out parts are named explicitly;
there is no default, because which parts are isolated is a decision and is recorded with the list.

Usage:
    python3 build_cpt_benchmark_exclusion.py --splits-dir data/benchmark_v2/splits \
        --task as_paired_i --heldout-parts val test_iid test_novel_pep test_level4 \
        --min-edit-distance 2 --out heldout_peptides.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quest.data.benchmark_exclusion import build_exclusion, write_exclusion  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--heldout-parts", nargs="+", required=True)
    ap.add_argument("--min-edit-distance", type=int, choices=[1, 2], required=True,
                    help="smallest edit distance a corpus peptide may have to a held-out one; "
                         "1 = exact matches only, 2 = also one-edit neighbours")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows, meta = build_exclusion(a.splits_dir, a.task, a.heldout_parts, a.min_edit_distance)
    digest = write_exclusion(rows, meta, a.out)
    print(json.dumps({**meta, "list_sha256": digest, "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
