"""Stage 4.6 step 1 — does pocket_ppl track how often the MHC was seen in training?

Question (plans/stage4-6.md): is `pocket_ppl` pocket biochemistry, or recall of frequently seen
HLA chains? If loss falls steeply as an MHC's training frequency rises, the metric is largely
memorization and must stop being quoted as biochemistry.

Deviation from the plan, found 2026-10-07: the saved Stage 4.1 results hold only stratum
aggregates, not per-token or per-allele losses, so they cannot be re-bucketed. This script
re-scores the canonical eval per row instead. It calls the unmodified `region_restricted_eval`
with an added row-index bucket column, so masking, batching and seeds are exactly the Stage 4.1
eval's. Training rows carry no allele names, so "frequency" is the number of training rows that
contain the eval row's exact MHC segment(s) under that arm's own serialization — the quantity a
memorizing model would key on.

Three subcommands, per arm (A = run01 cdr3 x pocket, D = run05 cdr3 x full, F = run11 cdr123 x full):
  freq   --arm X   CPU: per eval row, training-row count of its MHC segment(s).
  loss   --arm X   GPU or Apple MPS: per-row pocket loss; verifies the pocket aggregate
                   reproduces the stored Stage 4.1 result (masked-token count exact, ppl ~1 %).
  report           joins both, bins by frequency, writes the slope table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

Q = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Q), str(Q / "scripts" / "training")]   # repo root for `quest.*`, lib dir for cpt_training_lib
OUT = Q / "docs" / "cpt_stage4.6" / "step1"

ARMS = {
    "A": dict(run="run01_cdr3_pocket_proportional_10M", tcr="cdr3", mhc="pocket",
              adapter="cpt_stage41_pointA_run01", results="pointA_run01"),
    "D": dict(run="run05_cdr3_fullmhc_proportional_10M", tcr="cdr3", mhc="full",
              adapter="cpt_stage41_pointD_run05", results="pointD_run05"),
    "F": dict(run="run11_cdr123_fullmhc_proportional_10M", tcr="cdr123", mhc="full",
              adapter="cpt_stage41_pointF_run11", results="pointF_run11"),
}
EVAL_DIR = Q / "data" / "cpt_canonical_eval_v2"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _eval_ds():
    from datasets import load_from_disk
    ds = load_from_disk(str(EVAL_DIR))
    return ds.add_column("row_i", list(range(len(ds))))


# ------------------------------------------------------------------ freq (CPU)
def cmd_freq(a) -> None:
    import duckdb
    import pyarrow as pa
    from cpt_training_lib import EOS_LITERAL, project_canonical_row_to_run_input

    arm = ARMS[a.arm]
    ds = _eval_ds()
    rows = []
    for i in range(len(ds)):
        r = ds[i]
        text, kinds = project_canonical_row_to_run_input(r, arm["tcr"], arm["mhc"])
        pieces = text.split(EOS_LITERAL) if text else []
        if len(pieces) != len(kinds):
            raise RuntimeError(f"row {i}: {len(pieces)} pieces vs {len(kinds)} kinds")
        mhc = [p for p, k in zip(pieces, kinds) if k.startswith("mhc")]
        rows.append(dict(row_i=i, mhc_class=_repaired_class(r), mhc_one_allele=r["mhc_one_allele"],
                         mhc_two_allele=r["mhc_two_allele"], mhc_segs=mhc))
    ev = pd.DataFrame(rows)
    segs = sorted({s for L in ev.mhc_segs for s in L})
    _log(f"arm {a.arm}: {len(ev):,} eval rows, {int((ev.mhc_segs.str.len() > 0).sum()):,} with MHC, "
         f"{len(segs):,} distinct MHC segments")

    tables = []
    for f in sorted((Q / "data" / "cpt_datasets" / arm["run"] / "train").glob("data-*.arrow")):
        with pa.memory_map(str(f)) as src:
            tables.append(pa.ipc.open_stream(src).read_all().select(["input_text"]))
    train = pa.concat_tables(tables)
    _log(f"arm {a.arm}: {train.num_rows:,} training rows")
    con = duckdb.connect()
    con.register("train", train)
    con.register("want", pa.table({"seg": segs}))
    counts = con.execute(f"""
        WITH t AS (SELECT row_number() OVER () AS rid, input_text FROM train),
             s AS (SELECT DISTINCT rid, unnest(string_split(input_text, '{EOS_LITERAL}')) AS seg FROM t)
        SELECT s.seg, count(*) AS n FROM s JOIN want USING (seg) GROUP BY s.seg
    """).df()
    cmap = dict(zip(counts.seg, counts.n))
    ev["seg_counts"] = ev.mhc_segs.apply(lambda L: [int(cmap.get(s, 0)) for s in L])
    # A row's frequency is its RAREST MHC segment: a class-II row is familiar only if both chains are.
    ev["train_freq"] = ev.seg_counts.apply(lambda c: min(c) if c else np.nan)
    ev["n_mhc_segs"] = ev.mhc_segs.str.len()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"freq_{a.arm}.parquet"
    ev.drop(columns=["mhc_segs"]).to_parquet(out)
    has = ev[ev.n_mhc_segs > 0]
    _log(f"wrote {out}: MHC rows {len(has):,}; never-seen in training {int((has.train_freq == 0).sum()):,}; "
         f"median freq {has.train_freq.median():,.0f}")


# ------------------------------------------------------------------ ident (CPU)
# Added 2026-10-07 after `freq` showed only 214 of 4,531 MHC eval rows (none of the 23 class I)
# have an EXACT sequence match in run05 training: training chains differ from eval chains by a
# few residues, i.e. are different alleles. Exact counts are therefore ~0 for 95 % of rows. The
# memorization question becomes: is pocket loss lower when the nearest training chain is closer?
def cmd_ident(a) -> None:
    import duckdb
    import pyarrow as pa
    from Bio import Align

    arm = ARMS[a.arm]
    ev = pd.read_parquet(OUT / f"freq_{a.arm}.parquet")
    from cpt_training_lib import EOS_LITERAL, project_canonical_row_to_run_input
    ds = _eval_ds()
    segs_by_row = {}
    for i in ev.row_i:
        text, kinds = project_canonical_row_to_run_input(ds[int(i)], arm["tcr"], arm["mhc"])
        pieces = text.split(EOS_LITERAL) if text else []
        segs_by_row[int(i)] = [p for p, k in zip(pieces, kinds) if k.startswith("mhc")]
    want = sorted({s for L in segs_by_row.values() for s in L})
    # No absolute floor: class-II pocket pseudo-sequences are 19 aa (class I 37). Same-length
    # peptides/CDRs become candidates too, but cannot reach the 0.97 band and lose the max to a
    # real pocket neighbour.
    lo = int(min(map(len, want)) * 0.9) if want else 1

    tables = []
    for f in sorted((Q / "data" / "cpt_datasets" / arm["run"] / "train").glob("data-*.arrow")):
        with pa.memory_map(str(f)) as src:
            tables.append(pa.ipc.open_stream(src).read_all().select(["input_text"]))
    con = duckdb.connect()
    con.register("train", pa.concat_tables(tables))
    # Distinct candidate chains (length filter drops CDRs and peptides) with the number of
    # training rows that carry each one.
    cand = con.execute(f"""
        WITH t AS (SELECT row_number() OVER () AS rid, input_text FROM train),
             s AS (SELECT DISTINCT rid, unnest(string_split(input_text, '{EOS_LITERAL}')) AS seg FROM t)
        SELECT seg, count(*) AS n FROM s WHERE length(seg) >= {lo} GROUP BY seg
    """).df()
    _log(f"arm {a.arm}: {len(want)} eval MHC chains vs {len(cand):,} distinct training chains (len >= {lo})")

    al = Align.PairwiseAligner(mode="global", match_score=1, mismatch_score=0,
                               open_gap_score=0, extend_gap_score=0)
    cs, cn = cand.seg.tolist(), cand.n.to_numpy()
    best = {}
    for w in want:
        ids = np.array([al.score(w, c) / max(len(w), len(c)) if abs(len(c) - len(w)) <= 0.1 * len(w) else 0.0
                        for c in cs])
        best[w] = (float(ids.max()) if len(ids) else 0.0, int(cn[ids >= 0.97].sum()))
    ev["nn_identity"] = ev.row_i.map(lambda i: min((best[s][0] for s in segs_by_row[int(i)]), default=np.nan))
    ev["near97_rows"] = ev.row_i.map(lambda i: min((best[s][1] for s in segs_by_row[int(i)]), default=np.nan))
    ev[["row_i", "nn_identity", "near97_rows"]].to_parquet(OUT / f"ident_{a.arm}.parquet")
    has = ev[ev.n_mhc_segs > 0]
    _log(f"wrote ident_{a.arm}.parquet: nearest-identity quartiles "
         f"{has.nn_identity.quantile([.25, .5, .75]).round(4).tolist()}")


# ------------------------------------------------------------------ loss (GPU / MPS)
def cmd_loss(a) -> None:
    import torch
    from peft import PeftModel
    from cpt_training_lib import get_tokenizer, load_base_model, region_restricted_eval

    arm = ARMS[a.arm]
    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    _log(f"arm {a.arm}: device {dev}")
    base = load_base_model("esmc_300m")
    model = PeftModel.from_pretrained(base, str(Q / "checkpoints" / arm["adapter"] / "final"))
    model = model.to(dev).eval()
    tok = get_tokenizer("esmc_300m")
    ds = _eval_ds()
    t0 = time.time()
    # Same arguments as the Stage 4.1 eval except the extra row_i bucket; regions/buckets only
    # change what is accumulated, never the masking RNG stream.
    res = region_restricted_eval(model, ds, arm["tcr"], arm["mhc"], tok, device=dev,
                                 eos_strategy="raw", regions=("overall", "pocket"),
                                 bucket_cols=("row_i",), max_length=1024, batch_size=16)
    _log(f"arm {a.arm}: eval {time.time() - t0:.0f}s")

    per = [dict(row_i=int(k), loss_sum=v["loss"] * v["n_masked_tokens"], n_tok=v["n_masked_tokens"])
           for k, v in res["pocket"]["row_i"].items()]
    df = pd.DataFrame(per)
    stored = json.load(open(Q / "docs" / "cpt_stage4.1" / "results" / arm["results"]
                            / "canonical_v2_results.json"))["result"]["pocket"]["overall"]["_"]
    mine = res["pocket"]["overall"]["_"]
    rel = abs(mine["perplexity"] - stored["perplexity"]) / stored["perplexity"] * 100
    check = dict(arm=a.arm, device=str(dev),
                 tokens_mine=mine["n_masked_tokens"], tokens_stored=stored["n_masked_tokens"],
                 ppl_mine=mine["perplexity"], ppl_stored=stored["perplexity"], ppl_rel_pct=rel,
                 masks_identical=mine["n_masked_tokens"] == stored["n_masked_tokens"],
                 ppl_within_1pct=rel <= 1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"loss_{a.arm}.parquet")
    (OUT / f"check_{a.arm}.json").write_text(json.dumps(check, indent=2))
    _log(f"REPRO {'PASS' if check['masks_identical'] and check['ppl_within_1pct'] else 'FAIL'} arm {a.arm}: "
         f"tokens {mine['n_masked_tokens']} vs {stored['n_masked_tokens']}, "
         f"pocket ppl {mine['perplexity']:.4f} vs {stored['perplexity']:.4f} ({rel:.2f} %)")


# ------------------------------------------------------------------ report
BINS = [(-1, 0.93, "<0.93"), (0.93, 0.97, "0.93-0.97"), (0.97, 0.99999, "0.97-<1"), (0.99999, 2, "exact")]
# Decision rule, fixed 2026-10-07 BEFORE any per-row loss was seen (carries stage4-6.md's 2x):
# within an MHC class, pocket ppl of the farthest bin / the exact bin > 2 = pocket_ppl is largely
# recall of seen chains. Ratio near 1 = closeness to training does not drive it. Classes with no
# exact-match rows, or a bin under 30 rows, are reported but not judged.
RULE_RATIO, MIN_BIN_ROWS = 2.0, 30


def _bin_label(x: float) -> str:
    for lo, hi, lab in BINS:
        if lo <= x < hi:
            return lab
    return "?"


def _repaired_class(row) -> str:
    """`mhc_class` with the class-II-alpha-in-`mhc_one` repair applied.

    cpt_canonical_eval_v2 labels 157 rows 'none' though they carry a class-II alpha chain (145
    DQA1, 7 DRA, 5 DPA1) in `mhc_one`. `cpt_training_lib._relabel_none_stratum` has fixed this
    inside region_restricted_eval since 2026-10-03; this script built its own labels and so
    bypassed it, leaving its strata out of step with every other Stage 4 report (found 2026-10-15).
    """
    from cpt_training_lib import _relabel_none_stratum
    cls = row["mhc_class"]
    if cls != "none":
        return cls
    return _relabel_none_stratum(row["mhc_one_allele"], row["mhc_two_allele"]) or cls


def _ppl(g: pd.DataFrame) -> float:
    return float(np.exp(g.loss_sum.sum() / g.n_tok.sum())) if g.n_tok.sum() else float("nan")


def _ppl_ci(g: pd.DataFrame, n_boot: int = 5000, seed: int = 0):
    """Row-level bootstrap CI on a token-weighted perplexity.

    The 2026-10-07 report judged point ratios with no uncertainty, on an 'exact' bin of 86 rows.
    Added 2026-10-15 so the rule is read against a CI; all three arms still clear it.
    """
    if not len(g) or not g.n_tok.sum():
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    ls, nt = g.loss_sum.to_numpy(), g.n_tok.to_numpy()
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    b = np.exp(ls[idx].sum(1) / nt[idx].sum(1))
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def _ratio_ci(far: pd.DataFrame, ex: pd.DataFrame, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    def draw(g):
        ls, nt = g.loss_sum.to_numpy(), g.n_tok.to_numpy()
        i = rng.integers(0, len(g), (n_boot, len(g)))
        return np.exp(ls[i].sum(1) / nt[i].sum(1))
    b = draw(far) / draw(ex)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def cmd_report(a) -> None:
    labs = [b[2] for b in BINS]
    lines = ["# Stage 4.6 step 1 — does pocket perplexity track closeness to a training MHC?", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/analysis/stage46_allele_freq.py`.", "",
             "**Measure.** For each eval row, the identity of its MHC chain (rarest/farthest chain for "
             "two-chain rows) to the nearest MHC chain in that arm's own training set, by global "
             "alignment. Pocket perplexity is token-weighted within each bin, from a per-row re-score "
             "that reproduces the stored Stage 4.1 result (check per arm below).", "",
             "**Why not frequency deciles as planned.** Only 214 of 4,531 MHC eval rows have an exact "
             "training match (none of the 23 class I), so ~95 % of rows share frequency 0.", "",
             f"**Rule, fixed before results:** within a class, ppl(farthest bin) / ppl(exact) > {RULE_RATIO} "
             f"= largely recall. Bins under {MIN_BIN_ROWS} rows are shown but not judged. Ratios carry a "
             "row-level bootstrap CI (added 2026-10-15) and are judged on the whole interval.", "",
             "**Two limits on how this table may be read** (2026-10-15):", "",
             "1. **The class-I stratum is 23 rows and ~130 masked tokens**, 11 alleles, 11 of them "
             "HLA-B\\*08:03, and the eval holds **no HLA-A\\*02:01 at all**. No class-I perplexity here "
             "is quotable without its n, and none of it speaks to an A\\*02:01 benchmark such as "
             "IMMREP25. See `quest-cpt/findings.md` §9.7.", "",
             "2. **Ratios are not comparable across arms.** Each arm's identity is measured against its "
             "own training set in its own serialization — pocket pseudo-sequences for A, full chains for "
             "D and F — so the identity bins mean different things per arm. Read each arm against the "
             "rule, never against another arm.", ""]
    summ = []
    for arm in ARMS:
        need = [OUT / f"{k}_{arm}.parquet" for k in ("freq", "ident", "loss")] + [OUT / f"check_{arm}.json"]
        if not all(p.exists() for p in need):
            lines += [f"## Arm {arm}: inputs missing", ""]
            continue
        chk = json.loads(need[3].read_text())
        d = (pd.read_parquet(need[0]).merge(pd.read_parquet(need[1]), on="row_i")
             .merge(pd.read_parquet(need[2]), on="row_i"))
        d = d[(d.n_tok > 0) & (d.n_mhc_segs > 0)].copy()
        # Parquets written before 2026-10-15 hold unrepaired labels; repair on read.
        d["mhc_class"] = d.apply(_repaired_class, axis=1)
        d["bin"] = d.nn_identity.map(_bin_label)
        repro = "PASS" if chk["masks_identical"] and chk["ppl_within_1pct"] else "FAIL"
        lines += [f"## Arm {arm} — {ARMS[arm]['tcr']} x {ARMS[arm]['mhc']}, trained on `{ARMS[arm]['run']}`", "",
                  f"Reproduction of the stored Stage 4.1 pocket result: **{repro}** — masked tokens "
                  f"{chk['tokens_mine']} vs {chk['tokens_stored']}, ppl {chk['ppl_mine']:.4f} vs "
                  f"{chk['ppl_stored']:.4f} ({chk['ppl_rel_pct']:.2f} %), {chk['device']}.", "",
                  "| class | " + " | ".join(labs) + " | far/exact |", "|---" * (len(labs) + 2) + "|"]
        for cls, g in d.groupby("mhc_class"):
            cells, vals = [], {}
            for lab in labs:
                gb = g[g.bin == lab]
                vals[lab] = (_ppl(gb), len(gb))
                cells.append(f"{vals[lab][0]:.3f} (n={len(gb)})" if len(gb) else "—")
            far = next((lab for lab in labs if vals[lab][1] >= MIN_BIN_ROWS), None)
            ok = vals["exact"][1] >= MIN_BIN_ROWS and far and far != "exact"
            ratio = vals[far][0] / vals["exact"][0] if ok else float("nan")
            if ok:
                lo, hi = _ratio_ci(g[g.bin == far], g[g.bin == "exact"])
                # Judge on the CI, not the point estimate: clears the rule only if its whole
                # interval does.
                verdict = "recall" if lo > RULE_RATIO else ("not recall" if hi <= RULE_RATIO
                                                            else "inconclusive")
                cell = f"{ratio:.2f} [{lo:.2f}, {hi:.2f}] {verdict}"
            else:
                lo = hi = float("nan")
                verdict, cell = "not judged", "not judged"
            lines.append(f"| {cls} | " + " | ".join(cells) + f" | {cell} |")
            summ.append(dict(arm=arm, mhc_class=cls, rows=len(g), tokens=int(g.n_tok.sum()),
                             ratio=ratio, ci_lo=lo, ci_hi=hi, verdict=verdict, repro=repro))
        # class-adjusted rank correlation: per-row loss minus its class's token-weighted mean loss
        d["row_loss"] = d.loss_sum / d.n_tok
        cm = d.groupby("mhc_class").apply(lambda g: g.loss_sum.sum() / g.n_tok.sum())
        d["resid"] = d.row_loss - d.mhc_class.map(cm)
        rho = d.resid.corr(d.nn_identity, method="spearman")
        lines += ["", f"Class-adjusted Spearman(per-row pocket loss, identity) = **{rho:.3f}** "
                  f"over {len(d):,} rows (negative = closer to training, lower loss).", ""]
        summ.append(dict(arm=arm, mhc_class="ALL (class-adjusted rho)", rows=len(d), ratio=rho,
                         verdict="", repro=repro))
    s = pd.DataFrame(summ)
    if "tokens" in s:   # integer counts must not print as 131.000 beside the ratios
        s["tokens"] = s.tokens.map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    lines[9:9] = ["## Summary", "", s.to_markdown(index=False, floatfmt=".3f") if len(s) else "(none)", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("freq", "ident", "loss"):
        p = sp.add_parser(c)
        p.add_argument("--arm", choices=list(ARMS), required=True)
    sp.add_parser("report")
    a = ap.parse_args()
    {"freq": cmd_freq, "ident": cmd_ident, "loss": cmd_loss, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
