"""Stage 4.2 readability predicate — can a condition resolve a base-versus-CPT difference?

This is the SECOND gate condition, and it is not the one `s42_check.py gate` already applies.
That one asks whether retrieval beats chance, positioned against the random rate. Passing it does
not license reading a comparison: `test_level4` clears chance decisively and its interval is still
0.46 wide, which cannot separate two arms whose difference is small. The two questions need
different power and are therefore emitted separately, and a comparison must refuse to print when
this one fails.

Three things this module does that the shipped CI path got wrong at least once each:

1. **The resampling unit is the peptide.** Molecules nest within peptides, outcomes are near-binary
   per peptide, and resampling queries as if independent is the error `findings.md` §9.6 retired.
   All of a peptide's molecules move together, and the estimator resampled is the same
   molecule-weighted mean the eval reports.
2. **The comparison is paired.** Both arms score the same peptides against the same pool, so the
   variance that matters is that of the per-peptide difference and peptide difficulty cancels. An
   MDE read off one arm's marginal interval is far too pessimistic. With only one arm's ranks the
   cross-arm peptide-level correlation rho is unmeasurable, so every MDE is reported AS A FUNCTION
   of rho. rho = 0 is the conservative bound and coincides with an unpaired comparison.
3. **The metric is a parameter.** Recall@k thresholds away the rank information. The same ranks
   support MRR and normalised rank, which may resolve an effect recall@k cannot — in which case the
   fix is the metric, not a task switch. Metrics are judged against ONE underlying effect, declared
   in `quest-cpt/box-jobs/prereg-4.2-effect.md` as a multiplicative rank improvement lambda.

Imports stdlib only, deliberately: the judge pulls in the trainer and therefore `peft`, so anything
living there cannot be exercised on a laptop. The verdicts that cost this programme visits were all
produced by code that could not be run away from the box.
"""
import math
import random

POWER_Z = 2.80          # 80 % power, two-sided alpha 0.05: 1.96 + 0.84
N_BOOT = 4000


# --- metrics on a rank: lower rank is better, rank 1 means the true peptide ranked first ---

def recall_at_k(rank, pool, k=10):
    return 1.0 if rank <= k else 0.0


def mrr(rank, pool):
    return 1.0 / rank


def norm_rank(rank, pool):
    """Concordance: the share of distractors the true peptide outranks. 0.5 under random."""
    return (pool - rank) / float(pool - 1)


METRICS = {"recall_at_10": recall_at_k, "mrr": mrr, "norm_rank": norm_rank}


def random_baseline(metric, pool, k=10):
    if metric == "recall_at_10":
        return k / float(pool)
    if metric == "mrr":
        return sum(1.0 / r for r in range(1, pool + 1)) / pool
    if metric == "norm_rank":
        return 0.5
    raise ValueError(metric)


# --- the estimator and its cluster-bootstrap standard error ---

def weighted_mean(values):
    return sum(values) / float(len(values)) if values else float("nan")


def by_peptide(ranks, peptides):
    """{peptide: [rank, ...]}. `ranks` and `peptides` are parallel per-molecule sequences."""
    out = {}
    for r, p in zip(ranks, peptides):
        out.setdefault(p, []).append(r)
    return out


def cluster_bootstrap(ranks, peptides, pool, metric="recall_at_10", k=10,
                      n_boot=N_BOOT, seed=42):
    """SE and percentile interval of the molecule-weighted mean, resampling PEPTIDES.

    Returns (point, se, lo, hi, n_molecules, n_peptides).
    """
    f = METRICS[metric]
    groups = by_peptide(ranks, peptides)
    keys = list(groups)
    vals = {p: [f(r, pool) if metric != "recall_at_10" else f(r, pool, k)
                for r in rs] for p, rs in groups.items()}
    point = weighted_mean([v for p in keys for v in vals[p]])
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pooled = []
        for _ in range(len(keys)):
            pooled.extend(vals[keys[rng.randrange(len(keys))]])
        draws.append(weighted_mean(pooled))
    draws.sort()
    mean = sum(draws) / len(draws)
    se = math.sqrt(sum((d - mean) ** 2 for d in draws) / (len(draws) - 1))
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return point, se, lo, hi, len(ranks), len(keys)


def naive_bootstrap_se(ranks, pool, metric="recall_at_10", k=10, n_boot=N_BOOT, seed=42):
    """The WRONG one, kept so the inflation factor is measurable rather than asserted."""
    f = METRICS[metric]
    vals = [f(r, pool) if metric != "recall_at_10" else f(r, pool, k) for r in ranks]
    rng = random.Random(seed)
    draws = [weighted_mean([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(n_boot)]
    mean = sum(draws) / len(draws)
    return math.sqrt(sum((d - mean) ** 2 for d in draws) / (len(draws) - 1))


# --- the paired MDE, as a function of rho ---

def paired_mde(se_base, rho, power_z=POWER_Z):
    """Smallest detectable mean difference, assuming both arms share the base arm's variance.

    Var(diff) = 2 V (1 - rho) under Var(cpt) = Var(base) = V, so SE(diff) = sqrt(2(1-rho)) SE_base.
    rho = 0 reproduces the unpaired case and is the conservative bound; rho -> 1 means peptide
    difficulty cancels completely. rho is NOT measurable from one arm, which is why it stays a
    parameter here instead of being quietly fixed.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    return power_z * math.sqrt(2.0 * (1.0 - rho)) * se_base


def breakeven_rho(se_base, effect, power_z=POWER_Z):
    """The rho at which MDE(rho) == effect. Retires most of the exposure to assuming rho.

    Solving power_z * sqrt(2(1-rho)) * se == effect gives
        rho = 1 - (effect / (power_z * se))^2 / 2
    Below 0 the pair is readable even unpaired, so rho buys nothing and the answer is robust.
    Above 1 it is unreadable at any correlation. In between it is the exact failure point, which
    is a different object from "probably tolerates a lot of error".
    """
    if se_base <= 0 or effect <= 0:
        return None
    return 1.0 - (effect / (power_z * se_base)) ** 2 / 2.0


def stratify(ranks, peptides, pool, metric="recall_at_10", k=10, key=len):
    """The metric by a per-peptide key, default peptide LENGTH.

    Recall tracks peptide length against a pool that is 76 % 9-mers (findings.md 9.6), so a
    condition with enough peptides to stratify should be stratified in the same pass rather than
    later. Returns {key: (n_molecules, n_peptides, value)}.
    """
    f = METRICS[metric]
    buckets = {}
    for r, p in zip(ranks, peptides):
        buckets.setdefault(key(p), ([], set()))
        v = f(r, pool, k) if metric == "recall_at_10" else f(r, pool)
        buckets[key(p)][0].append(v)
        buckets[key(p)][1].add(p)
    return {kk: (len(vs), len(ps), weighted_mean(vs)) for kk, (vs, ps) in sorted(buckets.items())}


# --- expressing ONE effect on every metric: the pre-registered lambda mechanism ---

def apply_lambda(rank, lam):
    return max(1, int(round(rank / lam)))


def lambda_for_recall_delta(ranks, pool, delta, k=10, hi=60.0, tol=1e-4):
    """Smallest lambda whose rank improvement lifts Recall@k by at least `delta`."""
    base = weighted_mean([recall_at_k(r, pool, k) for r in ranks])
    target = base + delta
    if weighted_mean([recall_at_k(apply_lambda(r, hi), pool, k) for r in ranks]) < target:
        return None                      # unreachable even at an implausible lambda
    lo = 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        got = weighted_mean([recall_at_k(apply_lambda(r, mid), pool, k) for r in ranks])
        if got >= target:
            hi = mid
        else:
            lo = mid
    return hi


def effect_under_lambda(ranks, peptides, pool, lam, metric="recall_at_10", k=10):
    """The change in `metric`'s molecule-weighted mean when every rank improves by lambda."""
    f = METRICS[metric]
    def m(rs):
        return weighted_mean([f(r, pool) if metric != "recall_at_10" else f(r, pool, k)
                              for r in rs])
    return m([apply_lambda(r, lam) for r in ranks]) - m(list(ranks))


# --- the predicate itself ---

def readability(ranks, peptides, pool, prereg_recall_delta, rhos=(0.0, 0.5, 0.8, 0.9, 0.95),
                metrics=("recall_at_10", "mrr", "norm_rank"), k=10, n_boot=N_BOOT, seed=42):
    """Per metric: the effect under the pre-registered lambda, and the MDE at each rho.

    `verdict` is READABLE when some metric's effect clears its MDE at a rho of 0.9 or below,
    UNREADABLE when no metric clears it even at rho = 0.95.
    """
    lam = lambda_for_recall_delta(ranks, pool, prereg_recall_delta, k)
    out = {"n_molecules": len(ranks), "n_peptides": len(set(peptides)), "pool": pool,
           "lambda": lam, "prereg_recall_delta": prereg_recall_delta, "metrics": {}}
    if lam is None:
        out["verdict"] = "UNREACHABLE"
        return out
    best_rho = None
    for name in metrics:
        point, se, lo, hi, nm, npep = cluster_bootstrap(ranks, peptides, pool, name, k,
                                                        n_boot, seed)
        eff = effect_under_lambda(ranks, peptides, pool, lam, name, k)
        mdes = {rho: paired_mde(se, rho) for rho in rhos}
        clears = sorted([rho for rho, m in mdes.items() if eff >= m])
        out["metrics"][name] = {
            "point": point, "se": se, "ci95": [lo, hi],
            "random": random_baseline(name, pool, k),
            "effect_under_lambda": eff, "mde": mdes,
            "min_rho_that_clears": clears[0] if clears else None,
            "breakeven_rho": breakeven_rho(se, eff),
            "naive_se": naive_bootstrap_se(ranks, pool, name, k, n_boot, seed),
        }
        if clears and (best_rho is None or clears[0] < best_rho):
            best_rho = clears[0]
    out["min_rho_any_metric"] = best_rho
    out["verdict"] = ("READABLE" if best_rho is not None and best_rho <= 0.9
                      else "MARGINAL" if best_rho is not None
                      else "UNREADABLE")
    return out
