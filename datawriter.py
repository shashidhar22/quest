#!/usr/bin/env python


import argparse, os, sys, time, uuid
from itertools import combinations
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, Features, Value, concatenate_datasets
import s3fs 
import fsspec

import pyarrow.dataset as pds
from tqdm import tqdm
from pathlib import Path
from collections import Counter

###############################################################################
# 1. CLI
###############################################################################
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate combo_ids from an S3 Parquet corpus"
    )
    p.add_argument("--s3-prefix",  nargs="+", required=True,
                   help="Top-level S3 prefix containing *.parquet shards "
                        "(wildcard **/*.parquet is appended automatically)")
    p.add_argument("--model-name", required=True,
                   choices=["bert", "protbert", "esm", "llama"],
                   help="Concatenation style for combo_id")
    p.add_argument("--output", required=True,
                   help="Destination path (local folder or s3://...)")
    p.add_argument("--batch-size", type=int, default=50_000,
                   help="Rows to stream from Arrow at a time (default=50 000)")
    p.add_argument("--s3-key", default=os.getenv("AWS_ACCESS_KEY_ID"),
                   help="AWS key (env AWS_ACCESS_KEY_ID is fallback)")
    p.add_argument("--s3-secret", default=os.getenv("AWS_SECRET_ACCESS_KEY"),
                   help="AWS secret (env AWS_SECRET_ACCESS_KEY is fallback)")
    p.add_argument("--s3-token", default=os.getenv("AWS_SESSION_TOKEN"),
                   help="AWS session token (optional)")
    return p.parse_args()


###############################################################################
# 2. Combo / join helpers
###############################################################################
FIELDS = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]

START_STYLE = {
    # model-specific starting characters for combo_id
    "bert":      lambda vals: "[CLS] " + vals,
    "protbert":  lambda vals: "[CLS] " + vals,
}

JOIN_STYLE = {
    # model-specific separators or ordering rules
    "bert":      lambda vals: " [SEP] ".join(vals),
    "protbert":  lambda vals: " [SEP] ".join(vals),
}

END_STYLE = {
    "bert":     lambda vals: f"{vals} [SEP]",
    "protbert": lambda vals: f"{vals} [SEP]",
}

def explode_row(row: Dict[str, str], join_fn, start_fn, end_fn) -> List[Dict[str, object]]:
    """Return list of {combo_id, combo_feats} for one source row."""
    present = [(row[f], f) for f in FIELDS if row[f]]
    combos = []
    for r in range(1, len(present) + 1):
        for combo in combinations(present, r):
            vals, feats = zip(*combo)
            combos.append({
                "combo_id":   end_fn(start_fn(join_fn(vals))),
                "combo_feats": list(feats),
            })
    return combos

def harmonise(table: pa.Table) -> pa.Table:
    want = {
        "mhc_one_id":  pa.string(),
        "mhc_two_id":  pa.string(),
        "mhc_two":     pa.string(),
        "trad_gene":   pa.string(),
        "mhc_restriction": pa.string(),   # keep for old files
    }
    for col, dtype in want.items():
        if col not in table.column_names:
            table = table.append_column(col, pa.nulls(len(table), type=dtype))
    return table

###############################################################################
# 3. Main driver
###############################################################################
def main() -> None:
    args = parse_args()
    join_fn = JOIN_STYLE[args.model_name]
    start_fn = START_STYLE[args.model_name]
    end_fn = END_STYLE[args.model_name]
    # storage_options for datasets, s3fs & fsspec
    storage = {"key": args.s3_key, "secret": args.s3_secret}
    if args.s3_token:
        storage["token"] = args.s3_token

    # Run aws configure to set up credentials
    aws_access_key_id = args.s3_key or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = args.s3_secret or os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token = args.s3_token or os.getenv("AWS_SESSION_TOKEN")
    fs = s3fs.S3FileSystem(
        key=aws_access_key_id,
        secret=aws_secret_access_key,
        token=aws_session_token
    )

    def count_rows(prefix: str) -> int:
        base = prefix.replace("s3://", "").rstrip("/")
        return pds.dataset(base, format="parquet", filesystem=fs).count_rows()

    total_rows = sum(count_rows(p) for p in args.s3_prefix)
    print(f"[INFO] Total source rows across prefixes: {total_rows:,}")

    wildcards = [
        prefix.rstrip("/") + "/**/*.parquet"
        for prefix in args.s3_prefix            # now a list
    ]
    print(f"[INFO] Streaming from {wildcards}")


    union_cols  = set()
    all_files = [f                          # one flat list
             for pattern in args.s3_prefix   # iterate prefixes
             for f in sorted(fs.glob(f'{pattern.rstrip('/')}/**/*.parquet'))]

    for f in all_files:
        schema = pds.dataset(f's3://{f}').schema             # no data read
        union_cols.update(schema.names)

    # deterministic order → HF keeps it
    union_cols  = sorted(union_cols)

    # Build a Feature spec (all nullable strings here; adjust types if you like)
    features = Features({c: Value("string") for c in union_cols})

    def load_one(path):
        """Return a HF Dataset with the full column set (missing ⇒ null)."""

        if fs.isdir(path):
            file_list = [f"s3://{p}" for p in fs.glob(f"{path.rstrip('/')}/**/*.parquet")]
            if not file_list:
                raise ValueError(f"{path} is a dir but no *.parquet inside")
            data_files = file_list
        else:
            data_files = f"s3://{path}"
        ds = load_dataset("parquet", data_files=data_files, split="train", features=features)
        


        # (Optional) drop columns you truly don't care about
        # ds = ds.remove_columns(["mhc_restriction", "trad_gene"])

        return ds

    # Load & stitch
    chunks = [load_one(f) for f in all_files]
    ds = concatenate_datasets(chunks)


    # Keep only needed fields (fill missing with "")
    ds = ds.map(lambda ex: {k: ex.get(k, "") for k in FIELDS},
                batched=True,
                batch_size=args.batch_size,)

    # Arrow writer setup
    schema = pa.schema([
        ("combo_id", pa.string()),
        ("combo_feats", pa.list_(pa.string())),
    ])
    feat_counts = Counter()

    tmp_local = f"/tmp/{uuid.uuid4()}.parquet"
    writer = pq.ParquetWriter(tmp_local, schema, compression="zstd")

    rows_in, rows_out = 0, 0
    t0 = time.time()

    pbar = tqdm(total=total_rows, unit="rows")


    for batch in ds.iter(batch_size=args.batch_size):
        n_src = len(batch["tra"])
        rows_in += n_src
        out_batch = {"combo_id": [], "combo_feats": []}

        # explode inside Python
        for idx in range(n_src):
            row = {f: batch[f][idx] for f in FIELDS}
            for combo in explode_row(row, join_fn, start_fn, end_fn):
                out_batch["combo_id"].append(combo["combo_id"])
                out_batch["combo_feats"].append(combo["combo_feats"])
                rows_out += 1
                feat_counts[tuple(combo["combo_feats"])] += 1
        pbar.update(n_src)
        pbar.set_postfix(combos=rows_out, refresh=False)
        writer.write_table(pa.Table.from_pydict(out_batch, schema=schema))
        
    writer.close()
    pbar.close()
    # Copy to final destination
    if args.output.startswith("s3://"):
        dst = args.output.rstrip("/") + "/combos.parquet"
        fs.put(tmp_local, dst)
        os.remove(tmp_local)
        print(f"[DONE] Wrote {rows_out:,} combos to {dst}")
    else:
        os.rename(tmp_local, args.output)
        print(f"[DONE] Wrote {rows_out:,} combos to {args.output}")

    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed/3600:.2f} h  |  "
          f"{rows_in:,} source rows  |  "
          f"{rows_out:,} combos")
    print("\nCounts per combo_feats:")
    for feats, n in sorted(feat_counts.items(), key=lambda x: (-len(x[0]), x[0])):
        feat_label = ",".join(feats)
        print(f"{feat_label:<30} : {n:,}")

if __name__ == "__main__":
    main()
