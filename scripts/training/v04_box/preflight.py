"""Does this box have what the run needs, and does it work? Checked on the CPU, before spending.

Audit finding 5, 2026-11-06. The judge-first preflight tested `-f` and `-d`: that paths exist.
That is not what a preflight is for. It passed on a box with no CUDA driver (the evaluator then
fell back to the CPU without a word), with peft absent, with an adapter directory containing no
adapter, and with the train/val splits missing -- the last of which surfaced only after the PAID
base evaluation, at the one moment the money was already gone.

Every check here runs on the CPU and costs seconds. Each prints what it found, not merely that
it looked, because a preflight whose output is "ok" cannot be reviewed afterwards.

Exit codes: 0 everything required is present and usable, 2 something is not. There is no middle
value, because a preflight has no scientific verdict to give -- it is all technical.
"""
import argparse
import importlib
import importlib.util
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from track_a.shared import artifacts as A  # noqa: E402
from track_a.esm_sft.scoring import (  # noqa: E402  (NumPy only)
    SFT_TRAIN_DEFAULTS, UNRECORDED, sft_recipe)

OK, ERROR = 0, 2
# What an adapter directory must actually contain to be an adapter. A directory that merely
# exists passed the old check; `checkpoints/.../final` with the weights missing would have been
# loaded as a fresh LoRA and silently evaluated as the base arm.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _torch():
    """torch, or None. `importlib.util.find_spec` is not safe here: it propagates whatever a
    meta-path finder raises, so one broken or deliberately blocking finder crashed the whole
    preflight instead of reporting that torch is unavailable. A preflight must survive the
    environment it is inspecting.
    """
    try:
        import torch
        return torch
    except Exception:                                 # noqa: BLE001 - any failure means "no"
        return None


def check_imports(names):
    """{name: version or None}. Imported, not merely found: a broken install resolves a spec."""
    out = {}
    for n in [x for x in names if x]:
        try:
            m = importlib.import_module(n)
            out[n] = str(getattr(m, "__version__", "present"))
        except Exception as e:                        # noqa: BLE001 - any import failure counts
            out[n] = f"FAILED: {type(e).__name__}: {e}"
    return out


def check_cuda():
    """What torch says about the GPUs, or why it cannot say."""
    torch = _torch()
    if torch is None:
        return {"available": False, "why": "torch is unavailable in this interpreter"}
    info = {"torch": torch.__version__, "available": bool(torch.cuda.is_available())}
    if info["available"]:
        info["devices"] = [torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
        info["capability"] = list(torch.cuda.get_device_capability(0))
    else:
        info["why"] = ("torch reports no CUDA device: the driver may not be up, or this is a "
                       "CPU-only build")
    return info


def check_adapter(path):
    if path in (None, "", "none", "base"):
        return {"adapter": "none", "ok": True, "note": "the base arm uses a fresh LoRA"}
    p = pathlib.Path(path)
    missing = [] if p.is_dir() else ["the directory itself"]
    missing += [f for f in ADAPTER_FILES if p.is_dir() and not (p / f).is_file()]
    out = {"adapter": str(p), "ok": not missing, "missing": missing,
           "digest": A.dir_digest(p)}
    out.update(A.cpt_lineage(p) if p.is_dir() else {})
    return out


def check_checkpoint(path, require_trainable=True):
    """Present, non-empty, and -- where torch is available -- actually loadable."""
    p = pathlib.Path(path)
    out = {"path": str(p), "exists": p.is_file(),
           "bytes": p.stat().st_size if p.is_file() else 0,
           "digest": A.file_digest(p)}
    out["ok"] = out["exists"] and out["bytes"] > 0
    if not out["ok"]:
        out["why"] = "missing or empty" if not out["exists"] else "zero bytes"
        return out
    torch = _torch()
    if torch is None:
        out["loaded"] = "not checked: torch is unavailable in this interpreter"
        return out
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:                            # noqa: BLE001
        out["ok"], out["why"] = False, f"torch could not load it: {type(e).__name__}: {e}"
        return out
    out["loaded"] = True
    out["recorded_args"] = {k: v for k, v in (ck.get("args") or {}).items()
                            if k in ("task", "tcr_variant", "mhc_variant", "adapter", "seed",
                                     "epochs", "lr", "batch_size", "max_len")}
    # The recipe the comparison will require. The arm-D cell trains under the judge's
    # defaults, so a base checkpoint trained otherwise, or one whose recipe is incomplete, would
    # otherwise be discovered only at C1 -- after the base re-score and the whole arm-D cell had
    # been paid for (pre-launch review, 2026-11-06).
    saved = ck.get("args") or {}
    out["sft_recipe"] = sft_recipe(saved)
    out["sft_recipe_expected"] = sft_recipe(SFT_TRAIN_DEFAULTS)
    out["recipe_problems"] = []
    if out["sft_recipe"] == UNRECORDED:
        out["recipe_problems"].append(
            "its SFT recipe is unrecorded (a training setting is missing from its args), so it "
            "cannot be shown to match the arm-D cell, and the comparison would refuse")
    elif out["sft_recipe"] != out["sft_recipe_expected"]:
        out["recipe_problems"].append(
            f"its SFT recipe {out['sft_recipe']} differs from the one the arm-D cell trains "
            f"under, {out['sft_recipe_expected']}, and the comparison would refuse")
    for k in ("seed", "max_len"):
        if saved.get(k) != SFT_TRAIN_DEFAULTS[k]:
            out["recipe_problems"].append(
                f"it was trained with {k}={saved.get(k)!r}; the arm-D cell uses "
                f"{SFT_TRAIN_DEFAULTS[k]!r}")
    keys = list((ck.get("trainable_state_dict") or {}).keys())
    out["n_trainable_keys"] = len(keys)
    out["trainable_key_prefixes"] = sorted({k.split(".")[0] for k in keys})
    if require_trainable and not keys:
        out["ok"] = False
        out["why"] = ("no trainable_state_dict: this file cannot restore a trained head, so an "
                      "evaluation of it would score freshly initialised weights")
    return out


def check_splits(task, conditions):
    digests = A.split_digests(task, conditions)
    missing = sorted(c for c, d in digests.items() if d == "MISSING")
    return {"splits_dir": str(A.splits_dir()), "digests": digests, "missing": missing,
            "ok": not missing}


def check_pool():
    p = A.pool_path()
    d = A.file_digest(p)
    return {"path": str(p), "digest": d, "ok": d != "MISSING"}


def run(args):
    report = {
        "commit": A.code_commit(),
        "imports": check_imports(args.require.split(",") if args.require else []),
        "cuda": check_cuda(),
        "pool": check_pool(),
        "splits": check_splits(args.task, args.conditions),
        "adapter": check_adapter(args.adapter),
    }
    if args.ckpt:
        report["checkpoint"] = check_checkpoint(args.ckpt)

    problems = []
    for n, v in report["imports"].items():
        if str(v).startswith("FAILED"):
            problems.append(f"import {n}: {v}")
    if args.require_cuda and not report["cuda"].get("available"):
        problems.append(f"CUDA: {report['cuda'].get('why')}. A GPU visit must not start on the "
                        f"CPU; that is a result nobody chose to buy.")
    if not report["pool"]["ok"]:
        problems.append(f"candidate pool: not found at {report['pool']['path']}")
    if not report["splits"]["ok"]:
        problems.append(
            f"split files missing from {report['splits']['splits_dir']}: "
            f"{report['splits']['missing']}. Required BEFORE the first paid step, because a "
            f"missing train split otherwise surfaces after the base evaluation is paid for.")
    if not report["adapter"]["ok"]:
        problems.append(f"adapter {report['adapter']['adapter']}: missing "
                        f"{report['adapter'].get('missing')}")
    if "checkpoint" in report and not report["checkpoint"]["ok"]:
        problems.append(f"checkpoint {args.ckpt}: {report['checkpoint'].get('why')}")
    for why in (report.get("checkpoint") or {}).get("recipe_problems") or []:
        problems.append(f"checkpoint {args.ckpt}: {why}")
    if args.require_cuda and "checkpoint" in report and \
            report["checkpoint"].get("loaded") is not True:
        problems.append(
            f"checkpoint {args.ckpt} was not loaded ({report['checkpoint'].get('loaded')}). On "
            f"a GPU visit that would leave the one artifact nothing else validates unvalidated")
    if report["commit"] == "unknown":
        problems.append(
            "the code commit is unknown (the git call failed). The comparison refuses 'unknown', "
            "so finding it there would cost the whole visit")

    report["problems"] = problems
    report["ok"] = not problems
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
    for p in problems:
        print(f"[preflight] PROBLEM: {p}", file=sys.stderr)
    print(f"[preflight] {'ok' if report['ok'] else 'FAILED'}: "
          f"{len(problems)} problem(s); commit {report['commit']}", file=sys.stderr)
    return OK if report["ok"] else ERROR


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--task", default="as_paired_i")
    ap.add_argument("--conditions", nargs="+",
                    default=["train", "val", "test_level4", "test_novel_pep", "test_iid"])
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--require", default="torch,peft,transformers",
                    help="comma-separated modules that must import")
    ap.add_argument("--require-cuda", dest="require_cuda", action="store_true")
    ap.add_argument("--out", default=None)
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
