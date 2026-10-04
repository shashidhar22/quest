#!/usr/bin/env python3
"""Stage 4.1: generate cpt_stage41_SUMMARY.md.

Same generator as build_stage4_summary.py, re-pointed at the three Stage 4.1
arms. All three ran at a FIXED max_length=1024 and a FIXED effective batch of
1024, so unlike Stage 4 the per-region numbers are on a common footing.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

QUEST_ROOT = Path("/home/ubuntu/quest")
SUMMARY_PATH = QUEST_ROOT / "checkpoints/cpt_stage41_SUMMARY.md"

FIXED_MAX_LENGTH = 1024
FIXED_EFFECTIVE_BATCH = 1024

POINTS = [
    {
        "key": "A",
        "name": "pointA_run01",
        "label": "cdr3 + pocket",
        "tcr": "cdr3", "mhc": "pocket",
        "result_file": QUEST_ROOT / "checkpoints/cpt_stage41_pointA_run01/final/canonical_v2_results.json",
        "done_file": QUEST_ROOT / "checkpoints/cpt_stage41_pointA_run01/_done",
    },
    {
        "key": "D",
        "name": "pointD_run05",
        "label": "cdr3 + full MHC",
        "tcr": "cdr3", "mhc": "full",
        "result_file": QUEST_ROOT / "checkpoints/cpt_stage41_pointD_run05/final/canonical_v2_results.json",
        "done_file": QUEST_ROOT / "checkpoints/cpt_stage41_pointD_run05/_done",
    },
    {
        "key": "F",
        "name": "pointF_run11",
        "label": "cdr123 + full MHC",
        "tcr": "cdr123", "mhc": "full",
        "result_file": QUEST_ROOT / "checkpoints/cpt_stage41_pointF_run11/final/canonical_v2_results.json",
        "done_file": QUEST_ROOT / "checkpoints/cpt_stage41_pointF_run11/_done",
    },
]

REGIONS = [
    ("overall_ppl", "overall"),
    ("cdr3_ppl", "cdr3"),
    ("cdr12_ppl", "cdr12"),
    ("framework_ppl", "framework"),
    ("peptide_ppl", "peptide"),
    ("pocket_ppl (primary)", "pocket"),
    ("pocket_contact_only_ppl", "pocket_contact_only"),
]


def _load_result(pt: dict) -> dict | None:
    if not pt["result_file"].exists():
        return None
    return json.load(open(pt["result_file"]))


def _region_overall(payload: dict, region: str):
    r = payload.get("result", {}).get(region, {})
    blk = r.get("overall", {}).get("_")
    if not blk:
        return None
    return blk.get("perplexity")


def _region_mhc_breakdown(payload: dict, region: str) -> dict:
    r = payload.get("result", {}).get(region, {})
    by_class = r.get("mhc_class", {})
    return {cls: blk.get("perplexity") for cls, blk in by_class.items()}


def _fmt(x):
    if x is None:
        return "—"
    if not isinstance(x, (int, float)):
        return str(x)
    return f"{x:.3f}"


def _read_done_field(pt: dict, field: str) -> str | None:
    if not pt["done_file"].exists():
        return None
    for line in pt["done_file"].read_text().splitlines():
        if line.startswith(f"{field}="):
            return line.split("=", 1)[1]
    return None


def main() -> None:
    payloads: dict[str, dict | None] = {pt["key"]: _load_result(pt) for pt in POINTS}

    lines: list[str] = []
    lines.append("# TCRBench v3 — CPT Stage 4.1 SUMMARY (Stage 4 re-run at one fixed max_length)")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append("ESM-C 300M LoRA MLM trained on three serialization arms with `max_length=1024` and effective batch 1024 held fixed across all of them, so serialization is the only variable — the control Stage 4 lacked. Canonical evaluation is region-restricted on `data/cpt_canonical_eval_v2`; primary decision metric is `pocket_ppl` (MHC-bearing rows only, by construction).")
    lines.append("")

    lines.append("## 1. Setup")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append("| Base model | `esmc_300m` (EvolutionaryScale ESM Cambrian) |")
    lines.append("| LoRA | r=32, α=64, dropout=0.05; auto-detected ESM-C targets |")
    lines.append("| Canonical eval | `/home/ubuntu/quest/data/cpt_canonical_eval_v2/` (19,708 rows, 9 strata) |")
    lines.append("| EOS strategy | `raw` |")
    lines.append("| Early stop | `pocket_ppl` plateau, 3-eval patience, 1% rel improvement |")
    lines.append("| WandB project | `tcrbench-v3-cpt-stage41` |")
    lines.append("")

    lines.append("## 2. Per-point execution")
    lines.append("")
    lines.append("| Point | TCR | MHC | Dataset | Status | Wall (hr) |")
    lines.append("|---|---|---|---|---|---:|")
    # Derive the dataset from each point's own name ("pointF_run11" -> "run11") rather than a
    # parallel dict. The inherited Stage 4 dict was keyed A/B/C/D/E and raised KeyError: 'F'
    # on the new cdr123 x full arm (2026-10-03).
    for pt in POINTS:
        elapsed = _read_done_field(pt, "elapsed_hr")
        present = payloads[pt["key"]] is not None
        status = "✓" if present else "MISSING"
        dataset = pt["name"].split("_")[-1]
        lines.append(
            f"| {pt['key']} | {pt['tcr']} | {pt['mhc']} | {dataset} | {status} | {elapsed or '—'} |"
        )
    lines.append("")

    lines.append("## 3. Region-restricted canonical eval (centerpiece)")
    lines.append("")
    lines.append("Perplexity per region, overall. Lower is better. `pocket_ppl` is the primary decision metric.")
    lines.append("")
    header = "| Region | " + " | ".join(f"{pt['key']} ({pt['label']})" for pt in POINTS) + " |"
    sep = "|---|" + ":---:|" * len(POINTS)
    lines.append(header)
    lines.append(sep)
    for label, region in REGIONS:
        row = [label]
        for pt in POINTS:
            payload = payloads[pt["key"]]
            row.append(_fmt(_region_overall(payload, region) if payload else None))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 4. Per-mhc_class breakdown for primary metric (`pocket_ppl`)")
    lines.append("")
    mhc_classes: set[str] = set()
    for pt in POINTS:
        payload = payloads[pt["key"]]
        if payload:
            mhc_classes.update(_region_mhc_breakdown(payload, "pocket").keys())
    if mhc_classes:
        ordered = sorted(mhc_classes)
        header = "| mhc_class | " + " | ".join(pt["key"] for pt in POINTS) + " |"
        sep = "|---|" + ":---:|" * len(POINTS)
        lines.append(header)
        lines.append(sep)
        for cls in ordered:
            row = [cls]
            for pt in POINTS:
                payload = payloads[pt["key"]]
                if payload:
                    bd = _region_mhc_breakdown(payload, "pocket")
                    row.append(_fmt(bd.get(cls)))
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    else:
        lines.append("_(no payloads available)_")
        lines.append("")

    # Driven by POINTS, not by hardcoded letters. The inherited Stage 4 version read
    # payloads["B"/"C"/"E"], which do not exist in Stage 4.1 (arms are A, D, F) and would
    # raise KeyError once the earlier KeyError: 'F' was fixed (2026-10-03).
    lines.append("## 5. Serialization comparison (one fixed max_length across all arms)")
    lines.append("")
    pocket = {
        pt["key"]: (_region_overall(payloads[pt["key"]], "pocket") if payloads[pt["key"]] else None)
        for pt in POINTS
    }
    labels = {pt["key"]: pt["label"] for pt in POINTS}

    def pct(new, base):
        if new is None or base is None or base == 0:
            return None
        return 100.0 * (new - base) / base

    def fmt_pct(x):
        if x is None:
            return "—"
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.1f}%"

    base_key = POINTS[0]["key"]
    base_val = pocket[base_key]
    lines.append(f"| Arm | serialization | `pocket_ppl` | change vs {base_key} |")
    lines.append("|---|---|---:|---:|")
    lines.append(f"| {base_key} (baseline) | {labels[base_key]} | {_fmt(base_val)} | — |")
    for pt in POINTS[1:]:
        k = pt["key"]
        lines.append(
            f"| {k} | {labels[k]} | {_fmt(pocket[k])} | {fmt_pct(pct(pocket[k], base_val))} |"
        )
    lines.append("")

    # Determine winner
    candidates = [(pt["key"], pt["label"], _region_overall(payloads[pt["key"]], "pocket")) for pt in POINTS if payloads[pt["key"]]]
    finite = [(k, lab, v) for k, lab, v in candidates if v is not None]
    if finite:
        winner_key, winner_lab, winner_val = min(finite, key=lambda x: x[2])
        lines.append(f"**Best `pocket_ppl`**: Point {winner_key} ({winner_lab}) = {winner_val:.3f}")
        lines.append("")

    lines.append("## 6. Cross-stage synthesis")
    lines.append("")
    lines.append("- Stage 2 (TCR-axis sweep, pocket MHC held): CDR3 won on `cdr3_ppl`; full TCR lost on multi-segment rows.")
    lines.append("- Stage 3 (MHC-axis sweep, CDR3 TCR held): full MHC pushed `pocket_ppl` down dramatically vs pocket-only (Run 05 vs Stage 2 Run 01).")
    lines.append("- Stage 4 reported those gains, but set `max_length` per point (64/256/1024/1024), so representation and context window moved together and each point was scored on its own truncated view of the eval set.")
    lines.append("- **Stage 4.1 (this run) holds `max_length` at 1024 and effective batch at 1024 across every arm**, so the serialization comparison above is the first one in this programme that attributes its own result. Arm D is a reproducibility control against Stage 4 Point D (`pocket_ppl` 2.001).")
    lines.append("")

    lines.append("## 7. Diagnostics")
    lines.append("")
    for pt in POINTS:
        payload = payloads[pt["key"]]
        if not payload:
            lines.append(f"- **Point {pt['key']}** ({pt['label']}): result file missing")
            continue
        diag = payload.get("result", {}).get("_diagnostics") or payload.get("config", {})
        n_total = diag.get("n_rows_total") if isinstance(diag, dict) else None
        n_empty = diag.get("n_rows_empty") if isinstance(diag, dict) else None
        n_fail = diag.get("n_rows_annot_fail") if isinstance(diag, dict) else None
        eos = diag.get("eos_strategy") if isinstance(diag, dict) else None
        lines.append(
            f"- **Point {pt['key']}** ({pt['label']}): tcr={pt['tcr']} mhc={pt['mhc']} "
            f"eos_strategy={eos} rows={n_total} empty={n_empty} annot_fail={n_fail}"
        )
    lines.append("")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[stage4_summary] wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
