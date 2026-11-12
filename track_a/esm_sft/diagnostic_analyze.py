"""Describe judge-diagnostic results: score stability, then receptor dependence; no verdict.

Reads `diagnostic_eval` outputs and the manifest they were scored from. NumPy and stdlib only,
so the read-out runs on a laptop without torch.
"""
import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from track_a.esm_sft.diagnostic_eval import validate_manifest

POOL = 100
DONOR_SETS = ('all_donors', 'excluding_known_positive')
READING = ('Exploratory pilot, not a confirmatory test. Donors are independent replacements within '
           'MHC and CDR3-length strata, not a permutation test. The +/-0.02 margin is a draft; '
           'intervals are described against it geometrically and no useful/negligible label is '
           'assigned. Scored at native precision only.')


def _finite(s):
    v = float(s)
    if not math.isfinite(v): raise ValueError(f'non-finite number in input: {s}')
    return v


def _load(path):
    def refuse(s): raise ValueError(f'non-finite number in input: {s}')
    with open(path) as f: return json.load(f, parse_float=_finite, parse_constant=refuse)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm_rank(rank, pool=POOL):
    # The same formula paired_compare._val("norm_rank", ...) uses (readability.norm_rank), copied
    # rather than imported so track_a does not depend on the box scripts.
    return (pool - rank) / float(pool - 1)


def cluster_interval(deltas, clusters, n_boot, seed):
    """95% percentile interval of the mean, resampling clusters and pooling their queries."""
    by = {}
    for d, c in zip(deltas, clusters): by.setdefault(c, []).append(d)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    sums = np.array([sum(by[k]) for k in keys]); sizes = np.array([len(by[k]) for k in keys])
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def versus_margin(lo, hi, margin):
    if lo > margin: return 'above'
    if hi < -margin: return 'below'
    if lo >= -margin and hi <= margin: return 'inside'
    return 'straddles'


def _check(results, manifest, manifest_sha):
    if not isinstance(results, dict) or results.get('protocol') != 'judge-diagnostic-results-v1':
        raise ValueError('results protocol is not judge-diagnostic-results-v1')
    if results.get('manifest_sha256') != manifest_sha:
        raise ValueError('results were scored from a different manifest')
    rows = results.get('rows')
    keys = [(r.get('condition'), r.get('row_id')) for r in rows] if isinstance(rows, list) else None
    want = {(q['condition'], q['row_id']): q for q in manifest['rows']}
    if keys is None or len(keys) != len(set(keys)) or set(keys) != set(want):
        raise ValueError("results rows are not exactly the manifest's (condition, row_id) set")
    for r in rows:
        q = want[(r['condition'], r['row_id'])]
        for s in [r['original'], r['repeat'], r['reverse']] + [d['result'] for d in r['donors'] if d['result']]:
            if len(s['logits']) != POOL: raise ValueError(f'a scoring of {r["row_id"]} does not have {POOL} candidates')
            # The ranks are read from the file, so they are held to the logits beside them.
            a = np.asarray(s['logits'], dtype=np.float64)
            ties = int(np.sum(a == a[0]))
            if s['true_ties'] != ties or s['midrank'] != int(np.sum(a > a[0])) + (ties + 1) / 2:
                raise ValueError(f'a recorded midrank or tie count of {r["row_id"]} is not what its logits give')
        # Clusters set the resampling unit and donors the mean, so both are the manifest's: a
        # relabelled cluster or a repeated donor in the results file would change the interval.
        if any(r.get(k) != q[k] for k in ('peptide', 'pep_cluster')):
            raise ValueError(f'peptide or pep_cluster of {r["row_id"]} is not the manifest\'s')
        if [(d['seed'], d['result'] is None) for d in r['donors']] != [(d['seed'], d['row'] is None) for d in q['donors']]:
            raise ValueError(f'donors of {r["row_id"]} do not match the manifest')
    if not isinstance(results.get('runtime', {}).get('seconds'), (int, float)):
        raise ValueError('results record no runtime seconds')
    return rows


def stability(rows, seconds):
    out = {'n_queries': len(rows)}
    for name in ('repeat', 'reverse'):
        out[name] = {'max_abs_logit_diff': max(float(np.max(np.abs(np.subtract(r[name]['logits'], r['original']['logits'])))) for r in rows),
                     'queries_midrank_changed': sum(r[name]['midrank'] != r['original']['midrank'] for r in rows)}
    out['ties'] = {'queries_with_true_ties': sum(r['original']['true_ties'] > 1 for r in rows),
                   'queries_all_tied': sum(r['original']['true_ties'] == POOL for r in rows)}
    # How far apart the scores are at all. A ranking can repeat exactly and still rest on
    # differences at the numerical floor: both V04a judges put every candidate of every query
    # within 1e-4 of one logit (DG1, 2026-11-11), which ties alone did not show.
    logits = [np.asarray(r['original']['logits'], dtype=np.float64) for r in rows]
    spread = [float(a.max() - a.min()) for a in logits]
    level = [float(np.median(a)) for a in logits]
    distinct = [int(np.unique(a).size) for a in logits]
    three = lambda v: {'min': float(np.min(v)), 'median': float(np.median(v)), 'max': float(np.max(v))}
    out['scores'] = {'spread_across_candidates': three(spread), 'distinct_per_query': three(distinct),
                     'median_level': float(np.median(level)),
                     'level_range_across_queries': float(np.max(level) - np.min(level))}
    n = 3 * len(rows) + sum(d['result'] is not None for r in rows for d in r['donors'])
    out['seconds_per_scoring'] = seconds / n
    return out


def dependence(rows, known, n_boot, seed, margin):
    """Delta = original minus mean donor normalized rank (midrank), per query, then the mean."""
    out = {}
    for donor_set in DONOR_SETS:
        deltas, clusters, used, unmatched = [], [], 0, 0
        for r in rows:
            ds = [d['result'] for d in r['donors'] if d['result'] is not None
                  and (donor_set == 'all_donors' or not known[(r['condition'], r['row_id'], d['seed'])])]
            if not ds:
                unmatched += 1; continue
            used += len(ds)
            deltas.append(norm_rank(r['original']['midrank']) - sum(norm_rank(d['midrank']) for d in ds) / len(ds))
            clusters.append(r['pep_cluster'])
        k = len(set(clusters))
        interval = cluster_interval(deltas, clusters, n_boot, seed) if k >= 2 else None
        out[donor_set] = {'point': sum(deltas) / len(deltas) if deltas else None, 'interval': interval,
                          'interval_vs_margin': versus_margin(*interval, margin) if interval else 'unresolved',
                          'n_queries': len(rows), 'n_clusters': k, 'n_donor_scorings': used, 'n_unmatched': unmatched}
    return out


def analyze(manifest, manifest_sha, named, n_boot=4000, seed=42, margin=0.02):
    validate_manifest(manifest)
    known = {(q['condition'], q['row_id'], d['seed']): bool(d['known_positive_candidates'])
             for q in manifest['rows'] for d in q['donors']}
    report = {'protocol': 'judge-diagnostic-analysis-v1', 'manifest_sha256': manifest_sha,
              'settings': {'n_boot': n_boot, 'seed': seed, 'margin': margin, 'pool': POOL, 'rank': 'midrank',
                           'resampling_unit': 'pep_cluster (pooled across conditions in "pooled")'},
              'reading': READING, 'results': {}}
    for name, results in named.items():
        rows = _check(results, manifest, manifest_sha)
        conditions = sorted({r['condition'] for r in rows})
        report['results'][name] = {
            'checkpoint_sha256': results.get('checkpoint_sha256'), 'code_revision': results.get('code_revision'),
            'cpt_lineage': results.get('cpt_lineage'),
            'stability': stability(rows, results['runtime']['seconds']),
            'receptor_dependence': {c: dependence([r for r in rows if r['condition'] == c], known, n_boot, seed, margin)
                                    for c in conditions}}
        report['results'][name]['receptor_dependence']['pooled'] = dependence(rows, known, n_boot, seed, margin)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    # extend: a repeated --results must add to the list, not silently replace the earlier ones.
    p.add_argument('--results', required=True, nargs='+', action='extend', metavar='NAME=PATH')
    p.add_argument('--out', required=True)
    p.add_argument('--n-boot', type=int, default=4000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--margin', type=float, default=0.02)
    a = p.parse_args(argv)
    if Path(a.out).exists(): raise FileExistsError(a.out)
    if a.n_boot < 1 or not math.isfinite(a.margin) or a.margin < 0: raise ValueError('invalid settings')
    pairs = [x.split('=', 1) for x in a.results]
    if any(len(x) != 2 or not x[0] or not x[1] for x in pairs): raise ValueError('--results takes NAME=PATH')
    if len({n for n, _ in pairs}) != len(pairs): raise ValueError('duplicate results NAME')
    directory = Path(a.out).resolve().parent
    if not directory.is_dir() or not os.access(directory, os.W_OK): raise NotADirectoryError(f'output directory missing or not writable: {directory}')
    report = analyze(_load(a.manifest), _digest(a.manifest), {n: _load(path) for n, path in pairs}, a.n_boot, a.seed, a.margin)
    fd, tmp = tempfile.mkstemp(prefix='.analysis-', dir=directory)
    try:
        with os.fdopen(fd, 'w') as f: json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
        os.link(tmp, a.out)
    finally:
        os.unlink(tmp)
    for name, res in report['results'].items():
        st = res['stability']
        print(f"{name}: n={st['n_queries']} repeat_max={st['repeat']['max_abs_logit_diff']:.3g} "
              f"reverse_max={st['reverse']['max_abs_logit_diff']:.3g} ties>1={st['ties']['queries_with_true_ties']} "
              f"s/scoring={st['seconds_per_scoring']:.3g}")
        sc = st['scores']
        print(f"  {name} scores: level {sc['median_level']:+.5f} (range across queries {sc['level_range_across_queries']:.1e}); "
              f"spread across candidates min/median/max {sc['spread_across_candidates']['min']:.1e}/"
              f"{sc['spread_across_candidates']['median']:.1e}/{sc['spread_across_candidates']['max']:.1e}; "
              f"distinct of {POOL}: median {sc['distinct_per_query']['median']:.0f}")
        for cond, sets in res['receptor_dependence'].items():
            for donor_set, c in sets.items():
                point = 'NA' if c['point'] is None else f"{c['point']:+.4f}"
                ci = 'NA' if c['interval'] is None else f"[{c['interval'][0]:+.4f}, {c['interval'][1]:+.4f}]"
                print(f"  {name} {cond:15s} {donor_set:25s} delta={point} ci={ci} {c['interval_vs_margin']:10s} "
                      f"q={c['n_queries']} clusters={c['n_clusters']} donors={c['n_donor_scorings']} unmatched={c['n_unmatched']}")
    print(report['reading'])


if __name__ == '__main__': main()
