#!/usr/bin/env python3
"""Check CDR completeness across standardized_again parquet files."""

import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path

DATA_DIR = Path("data/standardized_again")

results = []

for db_dir in sorted(DATA_DIR.iterdir()):
    if not db_dir.is_dir():
        continue
    db_name = db_dir.name
    parquet_files = sorted(db_dir.glob("part_*.parquet"))
    if not parquet_files:
        continue

    table = pq.read_table(parquet_files, columns=[
        "tra", "tra_cdr1", "tra_cdr2", "tra_cdr3", "tra_full",
        "trb", "trb_cdr1", "trb_cdr2", "trb_cdr3", "trb_full",
    ])

    n = table.num_rows

    def count_non_null(col_name):
        col = table.column(col_name)
        return pc.sum(pc.invert(pc.is_null(col, nan_is_null=True))).as_py()

    def count_non_null_and(base_col, other_col):
        """Count rows where both base_col and other_col are non-null."""
        base = table.column(base_col)
        other = table.column(other_col)
        base_valid = pc.invert(pc.is_null(base, nan_is_null=True))
        other_valid = pc.invert(pc.is_null(other, nan_is_null=True))
        return pc.sum(pc.and_(base_valid, other_valid)).as_py()

    tra_count = count_non_null("tra")
    tra_cdr1 = count_non_null_and("tra", "tra_cdr1")
    tra_cdr2 = count_non_null_and("tra", "tra_cdr2")
    tra_cdr3 = count_non_null_and("tra", "tra_cdr3")
    tra_full = count_non_null_and("tra", "tra_full")

    trb_count = count_non_null("trb")
    trb_cdr1 = count_non_null_and("trb", "trb_cdr1")
    trb_cdr2 = count_non_null_and("trb", "trb_cdr2")
    trb_cdr3 = count_non_null_and("trb", "trb_cdr3")
    trb_full = count_non_null_and("trb", "trb_full")

    results.append((db_name, tra_count, tra_cdr1, tra_cdr2, tra_cdr3, tra_full,
                     trb_count, trb_cdr1, trb_cdr2, trb_cdr3, trb_full))

# Print table
def pct(num, denom):
    if denom == 0:
        return "N/A"
    return f"{100.0 * num / denom:.1f}%"

header = (
    f"{'database':<16} "
    f"{'tra_rec':>9} {'cdr1':>8} {'cdr2':>8} {'cdr3':>8} {'full':>8}  "
    f"{'trb_rec':>9} {'cdr1':>8} {'cdr2':>8} {'cdr3':>8} {'full':>8}"
)
print(header)
print("-" * len(header))

totals = [0] * 10
for row in results:
    db = row[0]
    vals = row[1:]
    for i in range(10):
        totals[i] += vals[i]
    tra_c = vals[0]
    trb_c = vals[5]
    print(
        f"{db:<16} "
        f"{tra_c:>9,} {pct(vals[1], tra_c):>8} {pct(vals[2], tra_c):>8} {pct(vals[3], tra_c):>8} {pct(vals[4], tra_c):>8}  "
        f"{trb_c:>9,} {pct(vals[6], trb_c):>8} {pct(vals[7], trb_c):>8} {pct(vals[8], trb_c):>8} {pct(vals[9], trb_c):>8}"
    )

print("-" * len(header))
tra_t = totals[0]
trb_t = totals[5]
print(
    f"{'TOTAL':<16} "
    f"{tra_t:>9,} {pct(totals[1], tra_t):>8} {pct(totals[2], tra_t):>8} {pct(totals[3], tra_t):>8} {pct(totals[4], tra_t):>8}  "
    f"{trb_t:>9,} {pct(totals[6], trb_t):>8} {pct(totals[7], trb_t):>8} {pct(totals[8], trb_t):>8} {pct(totals[9], trb_t):>8}"
)
