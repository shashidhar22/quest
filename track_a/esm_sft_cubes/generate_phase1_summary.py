"""Phase 1 summary: 4 tables (CSV+md) + 4 heatmaps + 2 scatter figures.

Reads per-cell metric JSONs from <root>/<ckpt>__<cube>/cell_<i>__<pool>__metrics.json
and writes:

  <root>/_SUMMARY/recall_at_10__<cube>__<pool>.csv
  <root>/_SUMMARY/recall_at_10__<cube>__<pool>.md
  <root>/_SUMMARY/heatmap__<cube>__<pool>.png
  <root>/_SUMMARY/pocket_ppl_vs_recall10__<cube>.png

The pocket_ppl x-axis is read from each checkpoint's canonical_v2_results.json
(class-I MHC class breakdown).

Usage:
    python -m track_a.esm_sft_cubes.generate_phase1_summary \
        --root /home/ubuntu/quest/checkpoints/sft_retrieval/phase1_crossenc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CKPT_TO_RESULTS = {
    "A": "/home/ubuntu/quest/checkpoints/cpt_stage4_pointA_run01/final/canonical_v2_results.json",
    "B": "/home/ubuntu/quest/checkpoints/cpt_stage4_pointB_run10/final/canonical_v2_results.json",
    "C": "/home/ubuntu/quest/checkpoints/cpt_stage4_pointC_run09/final/canonical_v2_results.json",
    "D": "/home/ubuntu/quest/checkpoints/cpt_stage4_pointD_run05/final/canonical_v2_results.json",
    "E": "/home/ubuntu/quest/checkpoints/cpt_stage2_run03_full/final/canonical_v2_results.json",
}
CKPTS = ["BASELINE", "A", "D", "E", "B", "C"]
CUBES = ["cube_a_trb", "cube_c_paired"]
POOLS = ["fullpool", "n100"]
CELLS = [f"cell_{i}" for i in range(1, 9)]

# Match the cell-severity color coding in scripts/antigenic_specificity_class_I REPORT.md §2
CELL_COLOR = {
    "cell_1": "yellow", "cell_2": "yellow", "cell_5": "yellow", "cell_6": "yellow",
    "cell_3": "orange", "cell_4": "red", "cell_7": "red", "cell_8": "red",
}


def _read_metric(root: Path, ckpt: str, cube: str, cell: str, pool: str) -> dict | None:
    p = root / f"{ckpt}__{cube}" / f"{cell}__{pool}__metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _recall_at_10(metric: dict, pool: str) -> float:
    if metric is None: return float("nan")
    # fullpool: from quest.metrics.retrieval_metrics → 'retrieval_recall_at_10'
    # n100:    from normalized_recall_k → 'recall@10'
    if pool == "fullpool":
        for k in ("retrieval_recall_at_10", "recall_at_10", "recall@10"):
            if k in metric and isinstance(metric[k], (int, float)):
                return float(metric[k])
    else:
        for k in ("recall@10", "recall_at_10"):
            if k in metric and isinstance(metric[k], (int, float)):
                return float(metric[k])
    return float("nan")


def _n_test(metric: dict) -> int:
    if metric is None: return 0
    return int(metric.get("n_test", 0))


def _pocket_ppl_class_i(results_path: str) -> float:
    """Extract the class-I MHC-pocket perplexity from canonical_v2_results.json."""
    try:
        d = json.loads(Path(results_path).read_text())
    except Exception:
        return float("nan")
    # Look for class-I pocket perplexity in any nested structure
    # Common paths: by_mhc_class.I.pocket.perplexity, or overall.I.pocket.perplexity
    def _walk(obj, depth=0):
        if depth > 6 or not isinstance(obj, dict): return
        for k, v in obj.items():
            if isinstance(v, dict):
                if k == "pocket" and "perplexity" in v:
                    yield (depth, k, float(v["perplexity"]))
                yield from _walk(v, depth + 1)

    # Prefer class-I-conditioned pocket_ppl if accessible; fall back to overall
    best = None
    # Heuristic: try class-I keys first
    for path in ["by_mhc_class", "mhc_class", "I", "i", "class_i", "overall"]:
        node = d
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None; break
        if isinstance(node, dict):
            for depth, key, ppl in _walk(node):
                if best is None or depth < best[0]:
                    best = (depth, ppl)
    if best is not None:
        return best[1]
    # Fall back to a flat walk
    for depth, key, ppl in _walk(d):
        return ppl
    return float("nan")


def build_recall_table(root: Path, cube: str, pool: str) -> pd.DataFrame:
    """Rows = checkpoints, cols = cells. Value = Recall@10, plus *_n_test for context."""
    rows = []
    for ckpt in CKPTS:
        row = {"ckpt": ckpt}
        for cell in CELLS:
            m = _read_metric(root, ckpt, cube, cell, pool)
            row[cell] = _recall_at_10(m, pool)
            row[f"{cell}__n"] = _n_test(m)
        rows.append(row)
    return pd.DataFrame(rows)


def write_md_table(df: pd.DataFrame, out_md: Path, cube: str, pool: str) -> None:
    """Render markdown table with n_test footnote and severity coloring (text marker)."""
    lines = [f"# {cube} — {pool} — Recall@10\n",
             f"5 checkpoints × 8 cells. Values in parentheses are n_test (low-power cells marked '*').\n",
             "| ckpt | " + " | ".join(CELLS) + " |",
             "|---|" + "|".join(["---:"] * len(CELLS)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for cell in CELLS:
            val = r[cell]; n = int(r[f"{cell}__n"])
            star = "*" if n < 50 else ""
            val_s = f"{val:.3f}" if not (isinstance(val, float) and (val != val)) else "—"
            cells.append(f"{val_s} ({n}{star})")
        lines.append(f"| {r['ckpt']} | " + " | ".join(cells) + " |")
    lines.append("\n_* n_test < 50 — low-power; values reported but should be treated as uncertain._")
    out_md.write_text("\n".join(lines))


def write_heatmap(df: pd.DataFrame, out_png: Path, cube: str, pool: str) -> None:
    import matplotlib.pyplot as plt
    # Build matrix
    mat = df[CELLS].to_numpy().astype(float)
    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = 0.8 if pool == "n100" else 0.4
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(CELLS))); ax.set_xticklabels(CELLS, rotation=30, ha="right")
    ax.set_yticks(range(len(CKPTS))); ax.set_yticklabels(CKPTS)
    for i, ckpt in enumerate(CKPTS):
        for j, cell in enumerate(CELLS):
            v = mat[i, j]
            n = int(df[f"{cell}__n"].iloc[i])
            txt = f"{v:.3f}\nn={n}" if not (isinstance(v, float) and (v != v)) else "—"
            color = "white" if v < vmax * 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    ax.set_title(f"Recall@10  —  {cube}  ({pool})")
    fig.colorbar(im, ax=ax, label="Recall@10")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)


def write_pocket_ppl_scatter(root: Path, cube: str, out_png: Path) -> None:
    """8-panel grid: per-cell scatter of pocket_ppl (x) vs Recall@10 (y), 5 points per panel."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    pocket_ppls = {ckpt: _pocket_ppl_class_i(p) for ckpt, p in CKPT_TO_RESULTS.items()}

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for j, cell in enumerate(CELLS):
        ax = axes[j // 4, j % 4]
        xs, ys, lbls = [], [], []
        for ckpt in CKPTS:
            m = _read_metric(root, ckpt, cube, cell, "fullpool")
            x = pocket_ppls.get(ckpt, float("nan"))
            y = _recall_at_10(m, "fullpool")
            xs.append(x); ys.append(y); lbls.append(ckpt)
        xs = np.array(xs); ys = np.array(ys)
        valid = ~(np.isnan(xs) | np.isnan(ys))
        if valid.sum() >= 3:
            rho, _ = spearmanr(xs[valid], ys[valid])
        else:
            rho = float("nan")
        ax.scatter(xs, ys, s=80)
        for x, y, l in zip(xs, ys, lbls):
            ax.annotate(l, (x, y), fontsize=10, xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("CPT pocket_ppl (log)")
        ax.set_ylabel("SFT Recall@10")
        ax.set_title(f"{cell}  ρ={rho:.2f}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{cube} — CPT pocket_ppl vs SFT Recall@10 (fullpool)", y=1.02)
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()

    root = Path(args.root)
    summary = root / "_SUMMARY"; summary.mkdir(parents=True, exist_ok=True)

    for cube in CUBES:
        for pool in POOLS:
            df = build_recall_table(root, cube, pool)
            df.to_csv(summary / f"recall_at_10__{cube}__{pool}.csv", index=False)
            write_md_table(df, summary / f"recall_at_10__{cube}__{pool}.md", cube, pool)
            try:
                write_heatmap(df, summary / f"heatmap__{cube}__{pool}.png", cube, pool)
            except Exception as e:
                print(f"heatmap {cube}__{pool} FAILED: {e}")
        try:
            write_pocket_ppl_scatter(root, cube, summary / f"pocket_ppl_vs_recall10__{cube}.png")
        except Exception as e:
            print(f"scatter {cube} FAILED: {e}")

    print(f"Wrote summary to {summary}/")


if __name__ == "__main__":
    main()
