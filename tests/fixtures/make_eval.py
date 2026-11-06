"""Stand in for `judge eval`: real ranks, real artifact digests, no model.

Used only by tests/conftest.py's judge stub. The launcher now derives each manifest from the
evaluation's own provenance block, so a stub that wrote bare ranks would exercise the refusal
path on every happy-path test and prove nothing about the path a real visit takes.

Everything here that could be faked is instead computed from the sandbox's real files through
the production code (`track_a.shared.artifacts`), so the digests are genuine digests of genuine
stand-in artifacts. The RANKS are copied from the recorded 2026-10-08 evaluation; nothing in
this file invents a measurement.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from track_a.shared import artifacts as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", required=True, help="a recorded eval.json to take ranks from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--task", default="as_paired_i")
    ap.add_argument("--draw", default="length_matched")
    a = ap.parse_args()

    d = json.loads(pathlib.Path(a.ranks).read_text())
    conds = list(d.get("conditions", {}))
    d["args"] = dict(d.get("args") or {}, task=a.task, distractor_draw=a.draw,
                     adapter=a.adapter, ckpt=a.ckpt, seed=42,
                     tcr_variant="cdr3", mhc_variant="full")
    d["provenance"] = {
        "splits_dir": str(A.splits_dir()),
        "pool_path": str(A.pool_path()),
        "pool_digest": A.file_digest(A.pool_path()),
        "split_digests": A.split_digests(a.task, conds) if conds else {},
        "ckpt_path": a.ckpt,
        "ckpt_digest": A.file_digest(a.ckpt),
        "device": "cuda:stub",
        "torch_version": "stub",
        "code_commit": A.code_commit(),
        "pool_size": d.get("pool_size"),
        "sft_seed": 42,
        **A.cpt_lineage(a.adapter),
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
