#!/usr/bin/env python3
"""Phase 4 of the Stage 4 combined pipeline: generate cpt_stage4_SUMMARY.md.

Reads the 5 canonical_v2_results.json files and emits the side-by-side
summary table along with per-mhc_class breakdowns and a coupling
decision matrix.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

QUEST_ROOT = Path("/home/ubuntu/quest")
SUMMARY_PATH = QUEST_ROOT / "checkpoints/cpt_stage4_SUMMARY.md"

POINTS = [
    {
        "key": "A",
        "name": "pointA_run01",
        "label": "cdr3 + pocket",
        "tcr": "cdr3", "mhc": "pocket",
        "result_file": QUEST_ROOT / "checkpoints/cpt_stage4_pointA_run01/final/canonical_v2_results.json",
        "done_file": QUEST_ROOT / "checkpoints/cpt_stage4_pointA_run01/_done",
    },
    {
        "key": "B",
        "name": "pointB_run10",
        "label": "cdr123 + pocket+contact",
        "tcr": "cdr123", "mhc": "pocket_contact",
        "result_file": QUEST_ROOT / "checkpoints/cpt_stage4_pointB_run10/final/canonical_v2_results.json",
        "done_file": QUEST_ROOT / "checkpoints/cpt_stage4_pointB_run10/_done",
    },
    {
        "key": "C",
        "name": "pointC_run09",
        "label": "full + full",
        "tcr": "full", "mhc": "full",
        "result_file": QUEST_ROOT / "checkpoints/cpt_stage4_pointC_run09/final/canonical_v2_results.json",
        "done_file": QUEST_ROOT / "checkpoints/cpt_stage4_pointC_run09/_done",
    },
    {
        "key": "D",
        "name": "pointD_run05",
        "label": "cdr3 + full MHC",
        "tcr": "cdr3", "mhc": "full",
        "result_file": QUEST_ROOT / "checkpoints/cpt_stage4_pointD_run05/final/canonical_v2_results.json",
        "done_file": QUEST_ROOT / "checkpoints/cpt_stage4_pointD_run05/_done",
    },
    {
        "key": "E",
        "name": "pointE_run03_reeval",
        "label": "full + pocket (reused)",
        "tcr": "full", "mhc": "pocket",
        "result_file": QUEST_ROOT / "checkpoints/cpt_stage4_pointE_run03_reeval/canonical_v2_results.json",
        "done_file": QUEST_ROOT / "checkpoints/cpt_stage4_pointE_run03_reeval/_done",
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
    lines.append("# TCRBench v3 — CPT Stage 4 SUMMARY (Coupled TCR×MHC Representation Sweep)")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append("ESM-C 300M LoRA MLM trained on five coupling-scope points to test whether jointly enriching the TCR and MHC representations beats the Stage 2 CDR3-only winner. Canonical evaluation is region-restricted on `data/cpt_canonical_eval_v2`; primary decision metric is `pocket_ppl` (MHC-bearing rows only, by construction).")
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
    lines.append("| WandB project | `tcrbench-v3-cpt-stage4` |")
    lines.append("")

    lines.append("## 2. Per-point execution")
    lines.append("")
    lines.append("| Point | TCR | MHC | Dataset | Status | Wall (hr) |")
    lines.append("|---|---|---|---|---|---:|")
    dataset_by_key = {"A": "run01", "B": "run10", "C": "run09", "D": "run05", "E": "run03 (reused ckpt)"}
    for pt in POINTS:
        elapsed = _read_done_field(pt, "elapsed_hr")
        present = payloads[pt["key"]] is not None
        status = "✓" if present else "MISSING"
        lines.append(
            f"| {pt['key']} | {pt['tcr']} | {pt['mhc']} | {dataset_by_key[pt['key']]} | {status} | {elapsed or '—'} |"
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

    lines.append("## 5. Coupling decision matrix")
    lines.append("")
    a = _region_overall(payloads["A"], "pocket") if payloads["A"] else None
    e = _region_overall(payloads["E"], "pocket") if payloads["E"] else None
    d = _region_overall(payloads["D"], "pocket") if payloads["D"] else None
    b = _region_overall(payloads["B"], "pocket") if payloads["B"] else None
    c = _region_overall(payloads["C"], "pocket") if payloads["C"] else None

    def pct(new, base):
        if new is None or base is None or base == 0:
            return None
        return 100.0 * (new - base) / base

    def fmt_pct(x):
        if x is None:
            return "—"
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.1f}%"

    lines.append("| Comparison | `pocket_ppl` change vs A | interpretation |")
    lines.append("|---|---:|---|")
    lines.append(f"| A (baseline) | {_fmt(a)} | reference (cdr3+pocket) |")
    lines.append(f"| B vs A | {fmt_pct(pct(b, a))} | cdr123 + pocket_contact effect |")
    lines.append(f"| C vs A | {fmt_pct(pct(c, a))} | full + full effect (max scope) |")
    lines.append(f"| D vs A | {fmt_pct(pct(d, a))} | MHC-side scope only (cdr3 + full MHC) |")
    lines.append(f"| E vs A | {fmt_pct(pct(e, a))} | TCR-side scope only (full + pocket) |")
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
    lines.append("- Stage 4 tests whether _coupled_ enrichment (rich TCR ∧ rich MHC) compounds the gains or saturates. See decision matrix above.")
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
