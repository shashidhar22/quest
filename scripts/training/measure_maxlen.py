#!/usr/bin/env python3
"""Measure the exact no-truncation token length of CPT datasets.

Method (rigorous, not a sample estimate):
  Every token spans >= 1 character of input_text, so token_count <= char_count
  for any row. Therefore:
    1. Compute char length for ALL rows (cheap, vectorised over Arrow).
    2. Tokenise the top-N rows by char length, without truncation -> max_tok.
    3. Confirm that every row with char_len >= max_tok is inside that top-N set.
       If it is, no unexamined row can exceed max_tok, so max_tok is the exact
       maximum. If it is not, widen N and repeat.
  A failure looks like step 3 reporting uncovered rows -- then the printed
  maximum is only a lower bound and N must grow.

Also reports percentiles from a random sample, for choosing bucket boundaries.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, "/home/ubuntu/quest")
from scripts.training.cpt_training_lib import get_tokenizer  # noqa: E402

TOP_N = 20_000
SAMPLE_N = 50_000
BASE_MODEL = "esmc_300m"


def tok_lengths(tok, texts: list[str], batch: int = 512) -> np.ndarray:
    out = np.empty(len(texts), dtype=np.int32)
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        enc = tok(chunk, add_special_tokens=True, truncation=False, padding=False)
        out[i : i + len(chunk)] = [len(x) for x in enc["input_ids"]]
    return out


def measure(path: Path, tok) -> dict:
    ds = load_from_disk(str(path))
    splits = sorted(ds.keys())
    res = {"path": path.name, "splits": {}}
    overall_max = 0
    for sp in splits:
        d = ds[sp]
        texts_col = d.data.column("input_text")
        char_len = np.array([len(v) for v in texts_col.to_pylist()], dtype=np.int32)
        n = len(char_len)

        order = np.argsort(-char_len)[:TOP_N]
        top_texts = [d[int(i)]["input_text"] for i in order]
        top_tok = tok_lengths(tok, top_texts)
        max_tok = int(top_tok.max())

        # Coverage proof: every row that could possibly exceed max_tok must be
        # in the examined set.
        threshold_char = char_len[order].min()
        n_at_or_above = int((char_len >= max_tok).sum())
        covered = n_at_or_above <= len(order) or max_tok > threshold_char
        uncovered = max(0, n_at_or_above - len(order)) if not covered else 0

        rs = np.random.default_rng(42)
        idx = rs.choice(n, size=min(SAMPLE_N, n), replace=False)
        samp_tok = tok_lengths(tok, [d[int(i)]["input_text"] for i in idx])

        res["splits"][sp] = {
            "rows": n,
            "max_tokens": max_tok,
            "max_chars": int(char_len.max()),
            "coverage_ok": bool(covered),
            "uncovered_rows": uncovered,
            "examined_min_char": int(threshold_char),
            "sample_p50": int(np.percentile(samp_tok, 50)),
            "sample_p95": int(np.percentile(samp_tok, 95)),
            "sample_p99": int(np.percentile(samp_tok, 99)),
            "sample_max": int(samp_tok.max()),
        }
        overall_max = max(overall_max, max_tok)
    res["dataset_max_tokens"] = overall_max
    return res


def main() -> None:
    tok = get_tokenizer(BASE_MODEL)
    print(f"tokenizer: {type(tok).__name__}\n", flush=True)
    results = []
    for p in sys.argv[1:]:
        path = Path(p)
        print(f"=== {path.name} ===", flush=True)
        r = measure(path, tok)
        for sp, s in r["splits"].items():
            flag = "OK" if s["coverage_ok"] else f"INCOMPLETE ({s['uncovered_rows']} rows unexamined)"
            print(
                f"  {sp:11s} rows={s['rows']:>9,}  MAX_TOK={s['max_tokens']:>5}  "
                f"max_chars={s['max_chars']:>5}  p50={s['sample_p50']:>4} "
                f"p95={s['sample_p95']:>4} p99={s['sample_p99']:>4}  coverage={flag}",
                flush=True,
            )
        print(f"  -> dataset max tokens: {r['dataset_max_tokens']}\n", flush=True)
        results.append(r)
    if results:
        gmax = max(r["dataset_max_tokens"] for r in results)
        allok = all(s["coverage_ok"] for r in results for s in r["splits"].values())
        print(f"GLOBAL MAX TOKENS = {gmax}   (coverage complete: {allok})")
        for cand in (256, 384, 512, 640, 768, 1024, 1280, 1536, 2048):
            if cand >= gmax:
                print(f"SMALLEST SAFE max_length (power-of-2-ish) = {cand}")
                break


if __name__ == "__main__":
    main()
