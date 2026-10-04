"""Generate /home/ubuntu/quest/checkpoints/cpt_stage2_SUMMARY.md from the
Stage 2 training outputs + Stage 1 retroactive eval results.

The centerpiece is a side-by-side per-region perplexity table across Runs 01,
02, 03. Also includes:
  - Per-run training trajectory (steps, wall time, throughput).
  - Cross-stage Stage 1 (ESM-2 150M Run 01) vs Stage 2 (ESM-C 300M Run 01)
    comparison at fixed representation.
  - Decision-gate interpretation per the Stage 2 plan.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT_PATH = "/home/ubuntu/quest/checkpoints/cpt_stage2_SUMMARY.md"
S2_CKPTS = [
    {"name": "Run 01 (CDR3)", "key": "run01_cdr3",
     "dir": "/home/ubuntu/quest/checkpoints/cpt_stage2_run01_cdr3"},
    {"name": "Run 02 (CDR1+2+3)", "key": "run02_cdr123",
     "dir": "/home/ubuntu/quest/checkpoints/cpt_stage2_run02_cdr123"},
    {"name": "Run 03 (Full TCR)", "key": "run03_full",
     "dir": "/home/ubuntu/quest/checkpoints/cpt_stage2_run03_full"},
]
S1_RETROACTIVE = [
    {"name": "Stage 1 Run 01 (ESM-2 150M proportional)",
     "ckpt": "/home/ubuntu/quest/checkpoints/cpt_stage1_run01_proportional/final"},
    {"name": "Stage 1 Run 06 (ESM-2 150M balanced)",
     "ckpt": "/home/ubuntu/quest/checkpoints/cpt_stage1_run06_balanced/final"},
]
REGIONS = ["overall", "cdr3", "cdr12", "pocket", "pocket_contact_only", "peptide", "framework"]


def _fmt(x, dp=3) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{dp}f}"
    except (TypeError, ValueError):
        return str(x)


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        try:
            cur = cur[k]
        except (KeyError, TypeError):
            return default
    return cur


def _read_canonical_v2_results(ckpt_dir: str) -> dict | None:
    """Returns a dict shaped like ``{"result": <region_eval_buckets>}``.

    Fallback order:
      1. ``final/canonical_v2_results.json`` (saved by trainer when
         region_eval=True). Returns its parsed content directly.
      2. ``checkpoint_best/trainer_state.json`` ``canonical_buckets`` field
         (legacy / Run 01 path before canonical_v2_results.json save was
         added). Wrap in a ``{"result": ...}`` envelope to match downstream
         consumers.
    """
    p = Path(ckpt_dir) / "final" / "canonical_v2_results.json"
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback: try checkpoint_best/trainer_state.json
    p2 = Path(ckpt_dir) / "checkpoint_best" / "trainer_state.json"
    if p2.exists():
        try:
            with open(p2) as f:
                ts = json.load(f)
            buckets = ts.get("canonical_buckets")
            if buckets is None:
                return None
            return {
                "config": {
                    "tcr_variant": ts.get("config", {}).get("tcr_variant"),
                    "mhc_variant": ts.get("config", {}).get("mhc_variant"),
                    "eos_strategy": ts.get("config", {}).get("eos_strategy"),
                },
                "global_step": ts.get("global_step"),
                "result": buckets,
            }
        except Exception:
            return None
    return None


def _read_trainer_state(ckpt_dir: str) -> dict | None:
    p = Path(ckpt_dir) / "final" / "trainer_state.json"
    if not p.exists():
        # try checkpoint_best
        p = Path(ckpt_dir) / "checkpoint_best" / "trainer_state.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _read_early_stop_history(ckpt_dir: str) -> dict | None:
    p = Path(ckpt_dir) / "early_stop_history.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    lines: list[str] = []
    lines.append("# TCRBench v3 — CPT Stage 2 SUMMARY (TCR Representation Sweep)\n")
    lines.append(
        "ESM-C 300M LoRA MLM trained on three 10M-row CPT datasets that vary the TCR "
        "representation (CDR3 / CDR1+2+3 / Full chain), all on the proportional "
        "distribution and the MHC pocket variant. Canonical evaluation is region-restricted "
        "on `data/cpt_canonical_eval_v2`; primary decision metric is `cdr3_ppl`.\n"
    )
    lines.append("## 1. Setup\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append("| Base model | `esmc_300m` (EvolutionaryScale ESM Cambrian) |")
    lines.append("| LoRA | r=32, α=64, dropout=0.05; targets: layernorm_qkv.1, out_proj, ffn.1, ffn.3 |")
    lines.append("| Canonical eval | `/home/ubuntu/quest/data/cpt_canonical_eval_v2/` (19,708 rows, 9 strata) |")
    lines.append("| EOS strategy | `raw` (ESM-C tokenizer natively recognizes `<eos>`) |")
    lines.append("| Early stop | CdrPplPlateauCallback on `cdr3_ppl`, 3-eval patience, 1% rel improvement |\n")

    # Per-run summary
    lines.append("## 2. Per-run summary\n")
    lines.append("| Run | TCR | MHC | Max len | Batch × Accum | Best `cdr3_ppl` | Final step | Stop trigger |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for c in S2_CKPTS:
        cv2 = _read_canonical_v2_results(c["dir"])
        ts = _read_trainer_state(c["dir"])
        es = _read_early_stop_history(c["dir"])
        best_cdr3 = _safe_get(cv2, "result", "cdr3", "overall", "_", "perplexity")
        if best_cdr3 is None:
            best_cdr3 = ts and ts.get("best_eval_loss")
        step = ts.get("global_step") if ts else None
        cfg = ts.get("config") if ts else {}
        # max_length
        ml = cfg.get("max_length") if cfg else None
        bs = cfg.get("batch_size") if cfg else None
        ga = cfg.get("gradient_accumulation_steps") if cfg else None
        bxa = f"{bs}×{ga}" if (bs and ga) else "—"
        tcr = cfg.get("tcr_variant") if cfg else "—"
        mhc = cfg.get("mhc_variant") if cfg else "—"
        stop = "early-stop" if es and any(
            (not h.get("improved", False) and h.get("metric") is not None)
            for h in (es.get("history") or [])[-3:]
        ) else "epoch end"
        lines.append(f"| {c['name']} | {tcr} | {mhc} | {ml} | {bxa} | {_fmt(best_cdr3)} | {step} | {stop} |")
    lines.append("")

    # Region-restricted side-by-side table
    lines.append("## 3. Region-restricted canonical eval (centerpiece)\n")
    lines.append("Perplexity per region, overall. Lower is better.\n")
    header = ["Region"] + [c["name"] for c in S2_CKPTS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    region_data = {}
    for c in S2_CKPTS:
        cv2 = _read_canonical_v2_results(c["dir"])
        region_data[c["key"]] = (cv2 or {}).get("result", {})
    for region in REGIONS:
        row = [region]
        for c in S2_CKPTS:
            m = _safe_get(region_data[c["key"]], region, "overall", "_", "perplexity")
            row.append(_fmt(m))
        # Mark primary metric.
        if region == "cdr3":
            row[0] = f"**{region}** (primary)"
        if region == "framework":
            row[0] = f"{region} (sanity)"
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-stratum cdr3_ppl
    lines.append("## 4. Per-stratum `cdr3_ppl` breakdown\n")
    # Collect stratum keys from any run.
    strata_keys: list[str] = []
    for c in S2_CKPTS:
        rd = _safe_get(region_data[c["key"]], "cdr3", "stratum_key")
        if rd:
            for k in rd.keys():
                if k not in strata_keys:
                    strata_keys.append(k)
    strata_keys.sort()
    if strata_keys:
        header2 = ["stratum_key"] + [c["name"] for c in S2_CKPTS] + ["nrows"]
        lines.append("| " + " | ".join(header2) + " |")
        lines.append("|" + "|".join(["---"] * len(header2)) + "|")
        for sk in strata_keys:
            row = [sk]
            nrows = ""
            for c in S2_CKPTS:
                m = _safe_get(region_data[c["key"]], "cdr3", "stratum_key", sk)
                if m is None:
                    row.append("—")
                else:
                    row.append(_fmt(m.get("perplexity")))
                    if not nrows:
                        nrows = str(m.get("n_rows", ""))
            row.append(nrows)
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("(no per-stratum data found)")
    lines.append("")

    # Cross-stage
    lines.append("## 5. Cross-stage comparison (model-size effect at fixed CDR3 rep)\n")
    lines.append("| Checkpoint | overall_ppl | cdr3_ppl |")
    lines.append("|---|---:|---:|")
    for s1 in S1_RETROACTIVE:
        p = Path(s1["ckpt"]) / "canonical_v2_results.json"
        if p.exists():
            try:
                with open(p) as f:
                    r = json.load(f)["result"]
                ov = _safe_get(r, "overall", "overall", "_", "perplexity")
                cd = _safe_get(r, "cdr3", "overall", "_", "perplexity")
                lines.append(f"| {s1['name']} | {_fmt(ov)} | {_fmt(cd)} |")
            except Exception:
                lines.append(f"| {s1['name']} | (parse error) | — |")
        else:
            lines.append(f"| {s1['name']} | (file missing: {p}) | — |")
    # Add Stage 2 Run 01 for comparison
    rc = _read_canonical_v2_results(S2_CKPTS[0]["dir"])
    if rc:
        ov = _safe_get(rc, "result", "overall", "overall", "_", "perplexity")
        cd = _safe_get(rc, "result", "cdr3", "overall", "_", "perplexity")
        lines.append(f"| Stage 2 Run 01 (ESM-C 300M, CDR3) | {_fmt(ov)} | {_fmt(cd)} |")
    lines.append("")

    # Decision gates
    lines.append("## 6. Decision gates\n")
    try:
        run01_cdr3 = _safe_get(region_data["run01_cdr3"], "cdr3", "overall", "_", "perplexity")
        run02_cdr3 = _safe_get(region_data["run02_cdr123"], "cdr3", "overall", "_", "perplexity")
        run03_cdr3 = _safe_get(region_data["run03_full"], "cdr3", "overall", "_", "perplexity")
        # nseg2+ cdr3_ppl (multi-segment rows)
        ns2_run01 = _safe_get(region_data["run01_cdr3"], "cdr3", "n_segments", "2", "perplexity")
        ns2_run03 = _safe_get(region_data["run03_full"], "cdr3", "n_segments", "2", "perplexity")
        if run01_cdr3 and run03_cdr3:
            improvement = (run01_cdr3 - run03_cdr3) / run01_cdr3 * 100
            lines.append(f"- Overall `cdr3_ppl` improvement Run 03 vs Run 01: **{improvement:+.1f}%**")
        if ns2_run01 and ns2_run03:
            improvement_ns2 = (ns2_run01 - ns2_run03) / ns2_run01 * 100
            lines.append(f"- `nseg2_cdr3_ppl` (multi-segment rows) Run 03 vs Run 01: **{improvement_ns2:+.1f}%**")
        if run02_cdr3 and run03_cdr3:
            r02_v_r03 = (run02_cdr3 - run03_cdr3) / run03_cdr3 * 100
            lines.append(f"- Run 02 vs Run 03 `cdr3_ppl`: {r02_v_r03:+.1f}% (negative means Run 02 wins)")

        lines.append("")
        lines.append("**Recommendation** (apply the plan's decision rules):")
        if run01_cdr3 and run03_cdr3 and ns2_run01 and ns2_run03:
            ns2_pct = (ns2_run01 - ns2_run03) / ns2_run01 * 100
            if ns2_pct > 10:
                lines.append(
                    f"- Run 03 (full TCR) beats Run 01 (CDR3) by **{ns2_pct:.1f}%** on `nseg2_cdr3_ppl` "
                    f"(threshold: >10%). **Use full TCR for production; keep Run 09 in Stage 6.**"
                )
            elif ns2_pct < 5:
                lines.append(
                    f"- Improvement only **{ns2_pct:.1f}%** on `nseg2_cdr3_ppl` (threshold: <5%). "
                    f"**Stick with CDR3 for production; drop Run 09 (~$50 saved).**"
                )
            else:
                lines.append(
                    f"- Improvement **{ns2_pct:.1f}%** is in the indeterminate zone [5%, 10%]. "
                    f"Look at run02 vs run03 to decide CDR1+2+3 vs Full TCR."
                )
    except Exception as e:
        lines.append(f"_(decision gates: parse error — {e})_")
    lines.append("")

    # Diagnostics
    lines.append("## 7. Diagnostics\n")
    for c in S2_CKPTS:
        cv2 = _read_canonical_v2_results(c["dir"])
        if cv2:
            diag = _safe_get(cv2, "result", "_diagnostics", default={})
            lines.append(f"- **{c['name']}**: tcr={diag.get('tcr_variant')} mhc={diag.get('mhc_variant')} "
                         f"eos_strategy={diag.get('eos_strategy')} "
                         f"rows={diag.get('n_rows_tokenized')}/{diag.get('n_rows_total')} "
                         f"empty={diag.get('n_rows_empty')} annot_fail={diag.get('n_rows_annot_fail')}")
    lines.append("")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
