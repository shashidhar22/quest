"""V04 run provenance: record what a run WAS, and refuse comparisons across a mismatch.

The first failure family in this programme is the invalid comparison: step 4000 against step 7800,
a balanced run against a proportional one under a different loss objective, a new distractor draw
against an old number. The cure that worked was making the invalid form unrepresentable in code --
the committed 4.3 comparison script reads both runs at a pinned step, so it could not have produced
the 4000-against-7800 reading. That principle covered STEP and nothing else.

This extends it to the rest of the tuple. Every run writes a manifest; every comparison calls
check() and dies on a mismatch in any key that is not explicitly waived. Stdlib only, so it runs
on a fresh clone and on the box.

WHAT THE 2026-11-06 AUDIT CHANGED, and why each part was necessary:

1. **Absence used to read as agreement.** `check` compared `a.get(k) == b.get(k)`, so two
   manifests that merely agreed on absence were comparable -- `check({}, {})` reported nothing
   wrong. A manifest is now validated against a SCHEMA: every key must be present, and the
   required ones must carry a value. A field nobody recorded is now the loudest thing in the file.

2. **One overloaded key set described two different things.** A CPT pretraining run and an SFT
   judge cell were both written through the same eleven keys, so half of an SFT cell's manifest
   was the string "n/a" and half of a CPT run's was irrelevant. Two schemas now, named in the
   manifest, and a comparison across schemas refuses rather than comparing "n/a" with "n/a".

3. **Lineage was whatever the running script declared.** Both judge-first manifests said
   `objective=legacy_row_share_cpt`. That is wrong twice over: the base arm has no CPT at all,
   and arm D's own `trainer_state.json` records a fixed batch of 16 with accumulation 16 and no
   token budget, which is the fixed-batch objective, not row share. CPT lineage is now read from
   the adapter by `track_a.shared.artifacts.cpt_lineage`, SFT settings are their own fields, and
   `write --from-eval` takes the artifact digests from the evaluation's OWN record of what it
   read. A launcher can no longer name a pool the evaluator never opened.

4. **A run could be compared with itself.** Nothing checked that the two arms were two models.
   A reused stale evaluation therefore produced a confident paired comparison of one arm against
   itself. A treatment comparison in which neither the adapter nor the SFT checkpoint differs is
   now refused.

Keys, and why each is here:
  CPT RUN (run_v04.sh's six re-baselines)
    objective          loss weighting; row-share is rejected for production and the flag defaults
                       to it, so a launcher that omits the flag trains the wrong thing silently
    token_budget       batching mode; it changes the per-region gradient weighting, not just speed
    bucket_boundaries  same
    steps              the 4000-vs-7800 failure
    early_stop         a stop on a noisy metric is what produced that failure
    draw, pool_size, pool_id, dataset, seed, cotenants  as below
  SFT CELL (the judge cells, including judge_first.sh)
    task, dataset      the arm and the split family
    draw               distractor pool construction; length_matched numbers are not comparable
                       to uniform ones
    pool_size          random Recall@10 is k/pool, so this is in every metric
    pool_digest        hash of the pool file the EVALUATOR resolved and read, not the one the
                       launcher named. pool_size alone cannot tell a regenerated pool from the
                       original, and the draw is built from this file
    split_digests      per-condition hashes of the parquet files actually read
    sft_ckpt_digest    which trained checkpoint produced these scores
    sft_seed           the SFT run's seed
    sft_objective      what the JUDGE optimised; it is not a CPT objective
    sft_max_len        the length the CHECKPOINT was trained at, from its own recorded args
    sft_recipe         lr, batch, epochs, patience, caps and workers, from the same place; any
                       missing field makes the whole recipe `unrecorded`, which is refused,
                       because a half-known recipe cannot be shown equal to another
    eval_max_len       the length the EVALUATION serialized to, which is a different thing: two
                       arms at 1024 and 128 each matched their own checkpoint while scoring
                       different inputs
    cpt_*              read from the adapter: its digest, its own recorded objective, its run id,
                       completed steps, seed and token budget. "none" for a base arm, which has
                       no CPT history, and "unrecorded" when the artifact itself is silent
    device             a silent CPU fallback produces numbers nobody wanted to buy
    code_commit        two arms scored by different code are not a paired comparison
    cotenants          which GPUs ran alongside; a cross-arm band comparison carries this
                       asymmetry
"""
import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
try:
    from track_a.shared import artifacts as A
except ImportError:                 # pragma: no cover - a checkout without track_a
    A = None

MANIFEST_VERSION = 2

CPT_RUN_KEYS = ("objective", "token_budget", "bucket_boundaries", "steps", "early_stop",
                "draw", "pool_size", "pool_id", "dataset", "seed", "cotenants")

SFT_CELL_KEYS = ("task", "dataset", "draw", "pool_size", "pool_digest", "split_digests",
                 "sft_ckpt_digest", "sft_seed", "sft_objective", "sft_max_len", "sft_recipe",
                 "eval_max_len",
                 "cpt_adapter", "cpt_adapter_digest", "cpt_objective", "cpt_run_id",
                 "cpt_steps", "cpt_seed", "cpt_token_budget",
                 "device", "code_commit", "cotenants")

SCHEMAS = {"cpt_run": CPT_RUN_KEYS, "sft_cell": SFT_CELL_KEYS}

# Keys that must carry a VALUE, not merely exist. The rest may be null, because null is
# sometimes the honest record: a base arm has no CPT steps, and inventing a number for it is
# the behaviour this module exists to prevent.
REQUIRED = {
    "cpt_run": CPT_RUN_KEYS,
    "sft_cell": ("task", "dataset", "draw", "pool_size", "pool_digest", "split_digests",
                 "sft_ckpt_digest", "sft_seed", "sft_objective", "sft_max_len", "sft_recipe",
                 "eval_max_len",
                 "cpt_adapter", "cpt_adapter_digest", "cpt_objective", "cpt_run_id",
                 "device", "code_commit", "cotenants"),
}
# A digest of "MISSING" is a faithful record of a file that was not there, and an invalid basis
# for a verdict, so these may not be MISSING even though they are present.
NO_MISSING = ("pool_digest", "split_digests", "sft_ckpt_digest")
# The absence trap one level in: `None` is caught by required-presence, but the STRING
# "unrecorded" is a value, so two arms that both failed to record their SFT recipe would agree
# with each other. The SFT recipe is exactly what a paired comparison needs held equal, so not
# knowing it is fatal. `cpt_objective` is deliberately NOT here: "unrecorded" there is an honest
# statement about a legacy adapter's own files and does not stop a comparison.
NO_SENTINEL = ("sft_recipe", "sft_max_len", "sft_seed", "eval_max_len")
SENTINELS = ("unrecorded", "unknown", "n/a", "none")

# Keys a comparison is ALLOWED to differ on, per schema and comparison kind. A waiver is never
# silent: check() reports which waived keys actually differed, so a tolerated difference appears
# in the run log next to the result it licensed.
CPT_LINEAGE = ("cpt_adapter", "cpt_adapter_digest", "cpt_objective", "cpt_run_id",
               "cpt_steps", "cpt_seed", "cpt_token_budget")
WAIVED = {
    "cpt_run": {
        "band": ("seed", "cotenants"),              # a band is built FROM differing seeds
        "arms": ("dataset", "seed", "cotenants"),   # arm A vs arm D differ by dataset
        "cells": ("cotenants",),    # GPU 3's neighbour differs between P1 and P2 by construction;
                                    # everything else, dataset and seed included, must match
        "strict": (),
    },
    "sft_cell": {
        # The treatment IS the CPT adapter, so those differ by construction -- and so does the
        # SFT checkpoint trained on top of it. Everything else must match.
        # Only the CPT init and the checkpoint trained on top of it differ. The SFT recipe --
        # seed, length, lr, batch, epochs, caps -- is fixed across arms by construction, and
        # waiving the seed (as this did until round two of the 2026-11-06 review) let two arms
        # differ in the one setting a paired comparison most needs held.
        "cells": ("cotenants", "sft_ckpt_digest") + CPT_LINEAGE,
        "arms": ("cotenants", "dataset", "sft_ckpt_digest") + CPT_LINEAGE,
        # A seed BAND is the one comparison built from differing seeds, so it says so.
        "band": ("cotenants", "sft_seed", "sft_ckpt_digest"),
        "strict": (),
    },
}
# What must actually DIFFER for a treatment comparison to be a comparison at all.
TREATMENT = {"sft_cell": ("cpt_adapter_digest", "sft_ckpt_digest")}
EXPECTS_A_DIFFERENCE = ("cells", "arms")


def _empty(v):
    return v is None or (isinstance(v, (str, list, dict, tuple)) and len(v) == 0)


def schema_of(m):
    """The schema a manifest declares, or None. Version 1 manifests predate the field; they are
    not silently adopted, because a v1 manifest is exactly the under-specified thing being
    fixed."""
    kind = m.get("manifest_kind")
    return kind if kind in SCHEMAS else None


def validate(m, label="manifest"):
    """Problems with ONE manifest, as printable strings. Empty means well formed."""
    kind = schema_of(m)
    if kind is None:
        return [f"{label}: manifest_kind is missing or unknown "
                f"({m.get('manifest_kind')!r}); expected one of {sorted(SCHEMAS)}"]
    out = []
    if m.get("manifest_version") != MANIFEST_VERSION:
        out.append(f"{label}: manifest_version {m.get('manifest_version')!r}, "
                   f"expected {MANIFEST_VERSION}")
    for k in SCHEMAS[kind]:
        if k not in m:
            out.append(f"{label}: {k} is missing; a field nobody recorded cannot match one")
        elif k in REQUIRED[kind] and _empty(m[k]):
            out.append(f"{label}: {k} is empty ({m[k]!r}) but required by the {kind} schema")
        elif k in NO_MISSING and "MISSING" in json.dumps(m[k]):
            out.append(f"{label}: {k} records MISSING, so the artifact it describes was not "
                       f"there; that is a faithful record and an invalid basis for a verdict")
        elif k in NO_SENTINEL and str(m[k]).strip().lower() in SENTINELS:
            out.append(f"{label}: {k} is {m[k]!r}, which is a statement that nobody knows it. "
                       f"Two arms that both fail to record it would otherwise agree, and the "
                       f"SFT recipe is what a paired comparison needs held equal")
    return out


def write(path, **kv):
    kind = kv.pop("manifest_kind", None) or "cpt_run"
    if kind not in SCHEMAS:
        raise ValueError(f"unknown manifest_kind {kind!r}; expected one of {sorted(SCHEMAS)}")
    m = {k: kv.get(k) for k in SCHEMAS[kind]}
    m["manifest_kind"] = kind
    m["manifest_version"] = MANIFEST_VERSION
    problems = validate(m, label=str(path))
    if problems:
        raise ValueError("refusing to write an incomplete manifest:\n  " +
                         "\n  ".join(problems))
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, sort_keys=True))
    return m


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def from_eval_doc(d, **extra):
    """The artifact fields for an SFT-cell manifest, from an already-loaded eval document."""
    prov = d.get("provenance")
    if not isinstance(prov, dict) or not prov:
        raise ValueError(
            "this evaluation has no 'provenance' block, so what it read cannot be established. "
            "Re-run it with the current evaluator, which records the pool, split, checkpoint "
            "and adapter digests it actually used.")
    args = d.get("args") or {}
    m = {
        "manifest_kind": "sft_cell",
        "task": args.get("task"),
        "dataset": args.get("task"),
        "draw": args.get("distractor_draw"),
        "pool_size": d.get("pool_size"),
        # The length the EVALUATION serialized to, which is not the length the checkpoint was
        # trained at. The review's probe used 1024 against 128: each checkpoint matched its own
        # evaluation while the two arms scored different inputs.
        "eval_max_len": args.get("max_len"),
        "pool_digest": prov.get("pool_digest"),
        "split_digests": prov.get("split_digests"),
        "sft_ckpt_digest": prov.get("ckpt_digest"),
        # From the CHECKPOINT's recorded training args, by way of the evaluator -- not from the
        # eval's own CLI. `getattr(a, "seed", None)` in cmd_eval was structurally always None,
        # because --seed exists only on the train parser; the stub recorded 42 and so was more
        # complete than the real thing (round two, finding 4).
        "sft_seed": prov.get("sft_seed"),
        "sft_max_len": prov.get("sft_max_len"),
        "sft_recipe": prov.get("sft_recipe"),
        "sft_objective": prov.get("sft_objective"),
        "cotenants": prov.get("cotenants"),
        "device": prov.get("device"),
        "code_commit": prov.get("code_commit"),
    }
    for k in CPT_LINEAGE:
        m[k] = prov.get(k)
    m.update({k: v for k, v in extra.items() if v is not None})
    return m


def from_eval(eval_path, **extra):
    """The artifact fields for an SFT-cell manifest, taken from the evaluation's own record.

    This is the whole point of the repair: the evaluator writes down what it resolved and read,
    and the manifest is derived from THAT. An evaluation with no provenance block cannot be
    certified -- which is the correct answer for every file produced before the evaluator
    recorded one, including the 2026-10-08 ranks.
    """
    try:
        return from_eval_doc(load(eval_path), **extra)
    except ValueError as e:
        raise ValueError(f"{eval_path}: {e}") from None


def check(a, b, kind="strict"):
    """Returns (offending keys, tolerated differences). Comparable when the first is empty."""
    bad = validate(a, "arm A") + validate(b, "arm B")
    sa, sb = schema_of(a), schema_of(b)
    if sa and sb and sa != sb:
        bad.append(f"manifest_kind: {sa} vs {sb}; these describe different things and cannot "
                   f"be compared")
    if bad:
        return bad, []
    waived = set(WAIVED[sa].get(kind, ()))
    tolerated = []
    for k in SCHEMAS[sa]:
        if a.get(k) == b.get(k):
            continue
        line = f"{k}: {a.get(k)!r} vs {b.get(k)!r}"
        (tolerated if k in waived else bad).append(line)
    treatment = TREATMENT.get(sa, ())
    if kind in EXPECTS_A_DIFFERENCE and treatment and \
            all(a.get(k) == b.get(k) for k in treatment):
        bad.append(
            f"these two arms are the SAME model: {', '.join(treatment)} are identical, so a "
            f"{kind} comparison between them measures nothing. A reused or stale evaluation "
            f"looks exactly like this.")
    return bad, tolerated


def check_eval_provenance(a, b, kind="cells", conditions=()):
    """(offending, tolerated) for two eval documents, through the MANIFEST validator.

    Round two of the 2026-11-06 review, finding 3: this used to apply its own weaker rule --
    five required fields, and ANY difference among the treatment fields taken as evidence of two
    different models, including a difference in a run LABEL. Two evaluations with no checkpoint
    or adapter hashes at all, empty split hashes and different `cpt_run_id` strings were
    therefore compared and a result written, while the launcher's manifest validator would have
    refused all three. A contract that two entry points enforce differently is really the weaker
    of the two.

    So there is now one rule: derive an `sft_cell` manifest from each evaluation's own record and
    run exactly the validation the launcher runs. `conditions` additionally requires a split
    digest for each condition actually compared -- a hash covering a condition nobody compared
    proves nothing about the one that was.
    """
    bad = []
    for lbl, doc in (("arm A", a), ("arm B", b)):
        pv = doc.get("provenance")
        if not isinstance(pv, dict) or not pv:
            bad.append(f"{lbl} records no provenance block, so what it read cannot be "
                       f"established. Re-run it with the current evaluator.")
    if bad:
        return bad, []
    manifests = []
    for lbl, doc in (("arm A", a), ("arm B", b)):
        try:
            m = from_eval_doc(doc)
        except ValueError as e:
            bad.append(f"{lbl}: {e}")
            continue
        m["manifest_version"] = MANIFEST_VERSION
        problems = validate(m, label=lbl)
        if problems:
            bad.extend(problems)
        manifests.append(m)
        for cond in conditions:
            got = (m.get("split_digests") or {}).get(cond)
            if not got:
                bad.append(f"{lbl}: no split digest for {cond}, which this comparison reads, so "
                           f"the two arms cannot be shown to have scored the same data")
            elif got == "MISSING":
                bad.append(f"{lbl}: the split file for {cond} was not there when it ran")
    if bad or len(manifests) != 2:
        return bad, []
    more, tolerated = check(manifests[0], manifests[1], kind)
    return bad + more, tolerated


def verify_eval(eval_path, ckpt=None, adapter=None):
    """Re-hash the artifacts an evaluation says it read, and report what has changed since.

    The reuse check. An eval.json carries the digests of the pool, splits, checkpoint and
    adapter it used; this re-computes them from the files on disk now. A result may be reused
    only if nothing it depended on has moved. Without this, "an eval already exists, skipping"
    meant "trust a file of unknown origin", which is how a planted stale evaluation produced a
    certified comparison with no training and no checkpoint (audit finding 1, 2026-11-06).
    """
    if A is None:
        return ["this checkout has no track_a.shared.artifacts, so nothing can be re-hashed"]
    d = load(eval_path)
    prov = d.get("provenance")
    if not isinstance(prov, dict) or not prov:
        return [f"{eval_path} has no provenance block, so what it read cannot be re-checked"]
    task = (d.get("args") or {}).get("task")
    bad = []
    now = A.file_digest(A.pool_path())
    if prov.get("pool_digest") != now:
        bad.append(f"the candidate pool changed: recorded {prov.get('pool_digest')!r}, "
                   f"{A.pool_path()} now hashes to {now!r}")
    for cond, recorded in (prov.get("split_digests") or {}).items():
        if not task:
            bad.append("the eval does not record its task, so its splits cannot be located")
            break
        got = A.file_digest(A.split_path(task, cond))
        if got != recorded:
            bad.append(f"split {cond} changed: recorded {recorded!r}, now {got!r}")
    if ckpt is not None:
        got = A.file_digest(ckpt)
        if got != prov.get("ckpt_digest"):
            bad.append(f"the checkpoint changed: recorded {prov.get('ckpt_digest')!r}, "
                       f"{ckpt} now hashes to {got!r}")
    if adapter is not None and adapter not in ("none", "base", ""):
        got = A.dir_digest(adapter)
        if got != prov.get("cpt_adapter_digest"):
            bad.append(f"the CPT adapter changed: recorded "
                       f"{prov.get('cpt_adapter_digest')!r}, {adapter} now hashes to {got!r}")
    return bad


def pool_fingerprint(path):
    """Hash of a named file. Prefer `resolved_pool_fingerprint`: hashing a path the launcher
    names while the evaluator reads another is the disconnect the 2026-11-06 audit found."""
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def resolved_pool_fingerprint():
    """Hash of the candidate pool the SPLIT LOADER would open, by the one resolution rule."""
    if A is None:
        return "MISSING"
    return A.file_digest(A.pool_path())


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("write")
    w.add_argument("--out", required=True)
    w.add_argument("--manifest-kind", dest="manifest_kind", default=None, choices=sorted(SCHEMAS))
    w.add_argument("--from-eval", dest="from_eval", default=None,
                   help="derive the artifact fields from this eval.json's own provenance block")
    for k in sorted(set(CPT_RUN_KEYS) | set(SFT_CELL_KEYS)):
        w.add_argument(f"--{k.replace('_', '-')}", dest=k, default=None)
    c = sp.add_parser("check")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--kind", default="strict", choices=sorted(WAIVED["cpt_run"]))
    v = sp.add_parser("verify-eval")
    v.add_argument("--eval", dest="ev", required=True)
    v.add_argument("--ckpt", default=None)
    v.add_argument("--adapter", default=None)
    f = sp.add_parser("fingerprint")
    f.add_argument("--path", default=None)
    f.add_argument("--pool", action="store_true",
                   help="hash the pool the split loader resolves, not a path given here")
    a = ap.parse_args()

    if a.cmd == "fingerprint":
        if a.pool:
            print(resolved_pool_fingerprint())
        elif a.path:
            print(pool_fingerprint(a.path))
        else:
            print("fingerprint needs --pool or --path", file=sys.stderr)
            return 2
        return 0

    if a.cmd == "verify-eval":
        bad = verify_eval(a.ev, a.ckpt, a.adapter)
        if bad:
            print(f"[prov] REFUSED: {a.ev} cannot be reused; its inputs have changed")
            for line in bad:
                print(f"[prov]   {line}")
            return 1
        print(f"[prov] verified: every artifact {a.ev} recorded still hashes the same")
        return 0

    if a.cmd == "write":
        kv = {k: getattr(a, k) for k in set(CPT_RUN_KEYS) | set(SFT_CELL_KEYS)}
        kv = {k: v for k, v in kv.items() if v is not None}
        if kv.get("split_digests") and isinstance(kv["split_digests"], str):
            kv["split_digests"] = json.loads(kv["split_digests"])
        kind = a.manifest_kind
        if a.from_eval:
            try:
                derived = from_eval(a.from_eval, **kv)
            except ValueError as e:
                print(f"[prov] REFUSED: {e}")
                return 1
            kind = kind or derived.pop("manifest_kind")
            derived.pop("manifest_kind", None)
            kv = derived
        try:
            write(a.out, manifest_kind=kind, **kv)
        except ValueError as e:
            print(f"[prov] REFUSED: {e}")
            return 1
        print(f"[prov] wrote {a.out}")
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
