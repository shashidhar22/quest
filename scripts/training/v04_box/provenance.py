"""V04 run provenance: record what a run was, and refuse comparisons across a mismatch.

The first failure family in this programme is the invalid comparison: step 4000 against step 7800,
a balanced run against a proportional one under a different loss objective, a new distractor draw
against an old number. The cure that worked was making the invalid form unrepresentable in code --
the committed 4.3 comparison script reads both runs at a pinned step, so it could not have produced
the 4000-against-7800 reading. That principle covered STEP and nothing else.

This extends it to the rest of the tuple. Every run writes a manifest; every comparison calls
check() and dies on a mismatch in any key that is not explicitly waived. Stdlib only, so it runs on
a fresh clone and on the box.

Keys, and why each is here:
  objective          loss weighting; row-share is rejected for production and the flag defaults to
                     it, so a launcher that omits the flag trains the wrong thing silently
  token_budget       batching mode; it changes the per-region gradient weighting, not just speed
  bucket_boundaries  same
  steps              the 4000-vs-7800 failure
  early_stop         a stop on a noisy metric is what produced that failure
  draw               distractor pool construction; length_matched numbers are not comparable to
                     uniform ones
  pool_size          random Recall@10 is k/pool, so this is in every metric
  pool_id            hash of the candidate pool file. pool_size alone cannot tell a regenerated
                     pool from the original one, and the draw is built from this file, so a silent
                     pool change would look like a model effect
  dataset            the arm
  seed               training is non-deterministic at 1.31 %, so a seed set is part of a band
  cotenants          which GPUs ran alongside; a cross-arm band comparison carries this asymmetry
"""
import argparse
import hashlib
import json
import pathlib
import sys

KEYS = ("objective", "token_budget", "bucket_boundaries", "steps", "early_stop",
        "draw", "pool_size", "pool_id", "dataset", "seed", "cotenants")
# Keys a comparison is ALLOWED to differ on, per comparison kind. A waiver is never silent:
# check() reports which waived keys actually differed, so a tolerated difference appears in the
# run log next to the result it licensed.
WAIVED = {
    "band": ("seed", "cotenants"),          # a band is built FROM differing seeds
    "arms": ("dataset", "seed", "cotenants"),   # arm A vs arm D differ by dataset by construction
    "cells": ("cotenants",),    # GPU 3's neighbour differs between P1 and P2 by construction;
                                # everything else, dataset and seed included, must match
    "strict": (),
}


def write(path, **kv):
    m = {k: kv.get(k) for k in KEYS}
    m["manifest_version"] = 1
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, sort_keys=True))
    return m


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def check(a, b, kind="strict"):
    """Returns (offending keys, tolerated differences). Comparable when the first is empty."""
    waived = set(WAIVED[kind])
    bad, tolerated = [], []
    for k in KEYS:
        if a.get(k) == b.get(k):
            continue
        line = f"{k}: {a.get(k)!r} vs {b.get(k)!r}"
        (tolerated if k in waived else bad).append(line)
    return bad, tolerated


def pool_fingerprint(path):
    """Hash of the candidate pool file, so a silently regenerated pool is visible."""
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("write")
    w.add_argument("--out", required=True)
    for k in KEYS:
        w.add_argument(f"--{k.replace('_', '-')}", dest=k, default=None)
    c = sp.add_parser("check")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--kind", default="strict", choices=sorted(WAIVED))
    f = sp.add_parser("fingerprint")
    f.add_argument("--path", required=True)
    a = ap.parse_args()
    if a.cmd == "write":
        write(a.out, **{k: getattr(a, k) for k in KEYS})
        print(f"[prov] wrote {a.out}")
        return 0
    if a.cmd == "fingerprint":
        print(pool_fingerprint(a.path))
        return 0
    bad, tolerated = check(load(a.a), load(a.b), a.kind)
    if bad:
        print(f"[prov] REFUSED ({a.kind}): these runs are not comparable")
        for line in bad:
            print(f"[prov]   {line}")
        return 1
    print(f"[prov] comparable ({a.kind}): {a.a} vs {a.b}")
    for line in tolerated:
        print(f"[prov]   TOLERATED by kind={a.kind}: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
