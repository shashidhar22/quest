#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# 0.  Silence HF’s own progress bars *before* importing the library
###############################################################################
import os, argparse, uuid
os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"

from itertools     import combinations
from pathlib       import Path
from typing        import Dict, List

from datasets      import (load_dataset, concatenate_datasets,
                           Features, Value, Dataset, DatasetDict)
from tqdm          import tqdm
import s3fs, pyarrow.dataset as pds
import multiprocessing as mp


###############################################################################
# 1.  CLI
###############################################################################
def parse_args():
    p = argparse.ArgumentParser("Generate exploded combo DatasetDict")
    p.add_argument("--path",        nargs="+", required=True,
                   help="Local paths or s3:// prefixes with Parquet shards")
    p.add_argument("--model-name",  required=True,
                   choices=["bert", "protbert", "esm", "llama"])
    p.add_argument("--output",      required=True,
                   help="Destination directory (local) for save_to_disk()")
    p.add_argument("--batch-size",  type=int,  default=2_000_000)
    p.add_argument("--s3-key",      default=os.getenv("AWS_ACCESS_KEY_ID"))
    p.add_argument("--s3-secret",   default=os.getenv("AWS_SECRET_ACCESS_KEY"))
    p.add_argument("--s3-token",    default=os.getenv("AWS_SESSION_TOKEN"))
    return p.parse_args()


###############################################################################
# 2.  Combo-helper functions
###############################################################################
FIELDS = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]

START_STYLE = {
    "bert":     lambda s: "[CLS] " + s,
    "protbert": lambda s: "[CLS] " + s,
    "esm":      lambda s: s,
    "llama":    lambda s: s,
}
JOIN_STYLE  = {
    "bert":     lambda t: " [SEP] ".join(t),
    "protbert": lambda t: " [SEP] ".join(t),
    "esm":      lambda t: " ".join(t),
    "llama":    lambda t: " ".join(t),
}
END_STYLE   = {
    "bert":     lambda s: f"{s} [SEP]",
    "protbert": lambda s: f"{s} [SEP]",
    "esm":      lambda s: s,
    "llama":    lambda s: s,
}

def explode_example(ex, join_fn, start_fn, end_fn):
    present = [(ex.get(f, None), f) for f in FIELDS if ex.get(f, None)]
    combo_ids, combo_feats = [], []
    for r in range(1, len(present) + 1):
        for combo in combinations(present, r):
            vals, feats = zip(*combo)
            combo_ids.append(end_fn(start_fn(join_fn(vals))))
            combo_feats.append(feats)
    return {"combo_id": combo_ids, "combo_feats": combo_feats}


# def explode_example(batch, join_fn, start_fn, end_fn):
#     """
#     Expand one source row into N combo rows.
#     Returns dict of lists so HF duplicates rows automatically.
#     """
#     combos_ids = []
#     combo_feats = []
#     for i in range(len(batch["tra"])):
#         example = {f: batch[f][i] for f in batch.keys()}
#         present = [(example[f], f) for f in example.keys() if example[f]]
#         for r in range(1, len(present) + 1):
#             for combo in combinations(present, r):
#                 vals, feats = zip(*combo)
#                 combo_id = end_fn(start_fn(join_fn(vals)))
#                 combos_ids.append(combo_id)
#                 combo_feats.append(list(feats))
#     return {
#         "combo_id":   combos_ids,
#         "combo_feats": combo_feats,
#     }
    



    #     if i >= len(ex["tra"]):
    #         break
    # present = [(ex[f], f) for f in ex.keys() if ex[f]]

    # combos = []
    # for r in range(1, len(present) + 1):
    #     for combo in combinations(present, r):
    #         vals, feats = zip(*combo)
    #         combos.append({"combo_id": end_fn(start_fn(join_fn(vals))), 
    #                        "combo_feats": list(feats),
    #         })
    
    # return combos


###############################################################################
# 3.  Main driver
###############################################################################
def main():
    args = parse_args()
    is_remote = lambda p: p.startswith("s3://")

    # authenticated S3 client (noop for local)
    s3_fs = s3fs.S3FileSystem(key=args.s3_key,
                              secret=args.s3_secret,
                              token=args.s3_token)

    # helper ─ list every *.parquet under given prefix
    def list_parquet(pref: str) -> list[str]:
        """
        Return a flat list of *.parquet files.

        • If `pref` is itself a directory (local or S3) we expand it
        recursively and return every shard it contains.
        • If `pref` is a single *.parquet file we return it unchanged.
        """
        if is_remote(pref):
            key = pref[5:].rstrip("/")
            if s3_fs.isdir(key):                               # s3://folder.parquet/
                files = s3_fs.glob(f"{key}/**/*.parquet")
                return [f"s3://{p}" for p in files]
            return [pref]                                      # single object
        else:
            p = Path(pref)
            if p.is_dir():                                     # /path/to/folder.parquet/
                return [str(f) for f in p.rglob("*.parquet")]
            return [str(p)]                                    # single file


    # gather all files
    all_files = [p for pref in args.path for p in list_parquet(pref)]

    # union schema across shards
    union_cols = set()
    for f in all_files:
        schema = pds.dataset(f, format="parquet",
                             filesystem=s3_fs if is_remote(f) else None).schema
        union_cols.update(schema.names)

    features = Features({c: Value("string") for c in sorted(union_cols)})

    # load shards with a progress bar
    def load_one(path: str) -> Dataset:
        """
        Return a HF Dataset for *every* Parquet shard under `path`.

        `path` may be…
        • a single *.parquet file
        • a directory / prefix that contains many *.parquet shards
        • local or s3://
        """

        def _local_file_list(p: Path) -> list[str]:
            # /folder.parquet/  → expand recursively
            if p.is_dir():
                files = [str(f) for f in p.rglob("*.parquet")]
                if not files:
                    raise ValueError(f"{p} is a directory but no *.parquet inside")
                return files
            # /file.parquet
            return [str(p)]

        def _s3_file_list(uri: str) -> list[str]:
            key = uri[5:]                 # strip "s3://"
            if s3_fs.isdir(key):
                files = s3_fs.glob(f"{key.rstrip('/')}/**/*.parquet")
                if not files:
                    raise ValueError(f"s3://{key} is a prefix but no *.parquet inside")
                return [f"s3://{obj}" for obj in files]
            # s3://bucket/file.parquet
            return [uri if uri.startswith("s3://") else f"s3://{key}"]

        # ------------------------------------------------------------------ #
        if is_remote(path):
            data_files   = _s3_file_list(path)
            storage_opts = {"key": args.s3_key,
                            "secret": args.s3_secret,
                            "token": args.s3_token}
        else:
            data_files   = _local_file_list(Path(path))
            storage_opts = None

        return load_dataset(
            "parquet",
            data_files=data_files,     # always a list of actual files
            split="train",
            features=features,
            storage_options=storage_opts,
        )

    datasets = [load_one(f) for f in tqdm(all_files, desc="Loading shards")]
    ds       = concatenate_datasets(datasets).remove_columns(
                   [c for c in union_cols if c not in FIELDS])

    # ── OPTION A: two-step split → 80 / 10 / 10 ─────────────────────────────
    #
    # 1) pull out 10 % for *test*
    step1      = ds.train_test_split(test_size=0.10, shuffle=True, seed=42)
    # 2) from the remaining 90 %, pull 11.111 % ⇒ 10 % of original for *val*
    train_val  = step1["train"].train_test_split(test_size=0.111111,
                                                 shuffle=True, seed=42)

    raw_splits = {
        "train": train_val["train"],   # 80 %
        "val":   train_val["test"],    # 10 %
        "test":  step1["test"],        # 10 %
    }

    # explode each split
    join_fn, start_fn, end_fn = (JOIN_STYLE[args.model_name],
                                 START_STYLE[args.model_name],
                                 END_STYLE[args.model_name])

    def explode_split(subset: Dataset) -> Dataset:
        cols_to_remove = [c for c in subset.column_names if c not in FIELDS]
        explosion = subset.map(
            explode_example,
            fn_kwargs=dict(join_fn=join_fn, start_fn=start_fn, end_fn=end_fn),
            remove_columns=cols_to_remove,
            batched=False,
            num_proc= max(1, mp.cpu_count() - 2),  # use all cores
            desc="Exploding",
        )
        exploded_id  = [seq for val in explosion["combo_id"] for seq in val]
        exploded_feats = [tuple(feat) for val in explosion["combo_feats"] for feat in val]
        return Dataset.from_dict({
            "combo_id": exploded_id,
            "combo_feats": exploded_feats,
        })

   
    data_dict = dict()
    for name, subset in raw_splits.items():
        print(f"\n[INFO] Exploding {name} split with {len(subset):,} rows") 
        exploded = explode_split(subset)
        exploded = exploded.remove_columns([c for c in exploded.column_names if c not in ["combo_id", "combo_feats"]])
        data_dict[name] = exploded
   
    combos = DatasetDict(data_dict)

    # save to local disk (sync to S3 later if desired)
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    combos.save_to_disk(str(out_dir))
    print(f"\n[DONE] DatasetDict saved to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
