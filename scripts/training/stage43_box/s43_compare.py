"""Stage 4.3 verdicts, written to run.log by run_s43.sh (also runnable on a laptop against the
published logs: Q=<dir holding checkpoints/> O=<s43_out> python s43_compare.py all).
  ab      EVAL_AB      old vs new region eval of one adapter must be EXACTLY equal.
  equiv   TRAIN_EQUIV  pre-registered: overall and cdr3 at step 2000 within 2 % of Stage 4.1 arm 2;
                       pocket reported, flagged > 5 %.
  resume  RESUME_LOG   the log must show the resume happened at global_step=1000;
          RESUME       the run resumed from the equivalence run's step-1000 checkpoint must reach
                       a step-2000 eval within 2 % (overall, cdr3) of that uninterrupted run's;
                       pocket flagged > 5 %.
Both runs evaluate every 500 steps, so entry 3 of early_stop_history.json is step 2000 in each —
for the resumed run too, because resume restores the plateau history along with the optimizer.
"""
import json, os, re, sys
O = os.environ.get("O", "/home/ubuntu/s43_out")
Q = os.path.join(os.environ.get("Q", "/home/ubuntu/quest"), "checkpoints")


def regions(p):
    d = json.load(open(p)); return d.get("regions", d)


def eval_ab():
    try:
        a, b = regions(f"{O}/eval_old.json"), regions(f"{O}/eval_new.json")
        diffs = []
        def walk(x, y, path):
            if isinstance(x, dict):
                if set(x) != set(y): diffs.append((path, "keys differ")); return
                for k in x: walk(x[k], y[k], f"{path}/{k}")
            elif x != y: diffs.append((path, x, y))
        walk(a, b, "")
        print(f"EVAL_AB {'PASS (exactly equal)' if not diffs else 'FAIL'}; {len(diffs)} differing leaves; first: {diffs[:3]}")
    except Exception as e:
        print("EVAL_AB ERROR", repr(e))


def at_step(log, step):
    for line in open(log, errors="ignore"):
        m = re.search(rf"canonical_v2 eval @ step {step}\] overall_ppl=([\d.]+)\s+cdr3_ppl=([\d.]+)", line)
        if m: return float(m.group(1)), float(m.group(2))


def pocket_at_4th_eval(run_dir):
    return json.load(open(f"{run_dir}/early_stop_history.json"))["history"][3]["metric"]


rel = lambda x, y: abs(x - y) / y * 100


def compare(label, new_dir, ref_dir):
    try:
        ref, new = at_step(f"{ref_dir}/training.log", 2000), at_step(f"{new_dir}/training.log", 2000)
        hr, hn = pocket_at_4th_eval(ref_dir), pocket_at_4th_eval(new_dir)
        ok = rel(new[0], ref[0]) <= 2 and rel(new[1], ref[1]) <= 2
        print(f"{label} {'PASS' if ok else 'FAIL'} @2000: overall {new[0]} vs {ref[0]} ({rel(new[0],ref[0]):.2f}%), "
              f"cdr3 {new[1]} vs {ref[1]} ({rel(new[1],ref[1]):.2f}%), pocket {hn:.4f} vs {hr:.4f} "
              f"({rel(hn,hr):.2f}%{' FLAG >5%' if rel(hn,hr) > 5 else ''})")
    except Exception as e:
        print(f"{label} ERROR", repr(e))


def resume_happened(run_dir):
    try:
        lines = [l for l in open(f"{run_dir}/training.log", errors="ignore") if "Resumed from" in l]
        ok = any("global_step=1000" in l for l in lines)
        print(f"RESUME_LOG {'PASS' if ok else 'FAIL'}: {lines[-1].strip() if lines else 'no Resumed-from line'}")
    except Exception as e:
        print("RESUME_LOG ERROR", repr(e))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("ab", "all"):
        eval_ab()
    if what in ("equiv", "all"):
        compare("TRAIN_EQUIV", f"{Q}/s43_equiv_pointD", f"{Q}/cpt_stage41_pointD_run05")
    if what in ("equiv_v2", "all"):
        # v2 sampler (mixed-composition steps + row-weighted loss) vs Stage 4.1 — the real test —
        # and vs the v1 run for the record (how much the fix moved things).
        compare("TRAIN_EQUIV_V2", f"{Q}/s43_equiv_v2_pointD", f"{Q}/cpt_stage41_pointD_run05")
        compare("V2_VS_V1", f"{Q}/s43_equiv_v2_pointD", f"{Q}/s43_equiv_pointD")
    if what in ("resume", "all"):
        resume_happened(f"{Q}/s43_resume_pointD")
        compare("RESUME", f"{Q}/s43_resume_pointD", f"{Q}/s43_equiv_pointD")
