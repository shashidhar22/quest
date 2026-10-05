"""Stage 4.3 verdicts. Eval A/B must be EXACT; training is pre-registered: overall and cdr3
within 2 % of Stage 4.1 arm 2 at step 2000, pocket reported (flag > 5 %)."""
import json, re
O = "/home/ubuntu/s43_out"; Q = "/home/ubuntu/quest/checkpoints"
def regions(p):
    d = json.load(open(p)); return d.get("regions", d)
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
try:
    ref = at_step(f"{Q}/cpt_stage41_pointD_run05/training.log", 2000)
    new = at_step(f"{Q}/s43_equiv_pointD/training.log", 2000)
    hr = json.load(open(f"{Q}/cpt_stage41_pointD_run05/early_stop_history.json"))["history"][3]["metric"]
    hn = json.load(open(f"{Q}/s43_equiv_pointD/early_stop_history.json"))["history"][3]["metric"]
    rel = lambda x, y: abs(x - y) / y * 100
    ok = rel(new[0], ref[0]) <= 2 and rel(new[1], ref[1]) <= 2
    print(f"TRAIN_EQUIV {'PASS' if ok else 'FAIL'} @2000: overall {new[0]} vs {ref[0]} ({rel(new[0],ref[0]):.2f}%), "
          f"cdr3 {new[1]} vs {ref[1]} ({rel(new[1],ref[1]):.2f}%), pocket {hn:.4f} vs {hr:.4f} "
          f"({rel(hn,hr):.2f}%{' FLAG >5%' if rel(hn,hr) > 5 else ''})")
except Exception as e:
    print("TRAIN_EQUIV ERROR", repr(e))
