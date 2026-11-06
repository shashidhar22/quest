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
from track_a.esm_sft.scoring import SFT_OBJECTIVE, cotenants, sft_recipe  # noqa: E402
from track_a.shared import artifacts as A  # noqa: E402

# The recipe a judge-first SFT cell trains under: the CLI defaults of `judge train`. The stub
# does not train, so it states the recipe it is standing in for rather than inventing one.
STUB_SFT_ARGS = {"lr": 1e-4, "batch_size": 16, "epochs": 2, "patience": 1,
                 "max_train_pos": 50_000, "max_val_pos": 5_000, "num_workers": 4,
                 "seed": 42, "max_len": 1024}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", required=True, help="a recorded eval.json to take ranks from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--task", default="as_paired_i")
    ap.add_argument("--draw", default="length_matched")
    ap.add_argument("--max-len", dest="max_len", type=int, default=1024)
    a = ap.parse_args()

    d = json.loads(pathlib.Path(a.ranks).read_text())
    conds = list(d.get("conditions", {}))
    args = dict(d.get("args") or {})
    args.update(task=a.task, distractor_draw=a.draw, adapter=a.adapter, ckpt=a.ckpt,
                tcr_variant="cdr3", mhc_variant="full", max_len=a.max_len)
    # `judge eval` has no --seed, so a real evaluation records none in its args. The stub must
    # not record one either: a stand-in that is MORE complete than the real thing is how the
    # structurally-null sft_seed went unnoticed for a round.
    args.pop("seed", None)
    d["args"] = args
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
        "sft_objective": SFT_OBJECTIVE,
        "sft_seed": STUB_SFT_ARGS["seed"],
        "sft_max_len": STUB_SFT_ARGS["max_len"],
        "sft_recipe": sft_recipe(STUB_SFT_ARGS),
        "sft_train_digests": A.split_digests(a.task, ("train", "val")),
        "cotenants": cotenants(),
        **A.cpt_lineage(a.adapter),
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
