"""The diagnostic read-out on fake results: arithmetic, coverage accounting and refusals.

No model is involved. Results are produced by the real `evaluate_manifest` with a stand-in
scorer, so their shape is the scorer's own and cannot drift from what the box will write.
"""
import hashlib
import json
import os
import subprocess
import sys

import pytest

from track_a.esm_sft import diagnostic_analyze as A
from track_a.esm_sft import diagnostic_eval as E

ORIG = ('AAAA', 'BBBB')


def query(i, condition='test_iid', cluster=None, donors=((0, True, []),)):
    pep = f'Q{i}'
    row = {'row_id': str(i), 'peptide': pep, 'tra_cdr3': ORIG[0], 'trb_cdr3': ORIG[1],
           'mhc_one': 'M', 'mhc_two': None}
    cands = [pep] + [f'{pep}_d{j}' for j in range(99)]
    ds = [{'seed': s, 'row': ({**row, 'tra_cdr3': f'C{s}CC', 'trb_cdr3': f'D{s}DD'} if present else None),
           'known_positive_candidates': [cands[k] for k in known]} for s, present, known in donors]
    return {'row_id': str(i), 'condition': condition, 'peptide': pep, 'pep_cluster': cluster or f'c{i}',
            'original': row, 'candidates': {'uniform': cands, 'length_matched': list(cands)},
            'donors': ds, 'metadata': {}}


def manifest(queries):
    return {'protocol': 'judge-diagnostic-v1', 'input_hashes': {'pool': 'a' * 64}, 'rows': queries}


def dependent(rows):
    """The truth outranks everything with the original receptor and nothing with a donor."""
    return [(10. if r['tra_cdr3'] == ORIG[0] else -1.) if '_d' not in r['peptide'] else 0. for r in rows]


def independent(rows):
    return [10. if '_d' not in r['peptide'] else 0. for r in rows]


def write(tmp_path, m, scorer, name='res', seconds=8.0, mutate=None):
    mp = tmp_path / 'manifest.json'
    if not mp.exists():
        mp.write_text(json.dumps(m))
    res = {'protocol': 'judge-diagnostic-results-v1',
           'manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest(),
           'checkpoint_sha256': 'b' * 64, 'code_revision': 'abc123', 'cpt_lineage': {'run': name},
           'runtime': {'seconds': seconds}, 'rows': E.evaluate_manifest(m, scorer)}
    if mutate:
        mutate(res)
    rp = tmp_path / f'{name}.json'
    rp.write_text(json.dumps(res))
    return mp, rp


def run(tmp_path, mp, results, out='report.json', extra=()):
    args = ['--manifest', str(mp), '--out', str(tmp_path / out), '--n-boot', '200']
    for name, path in results:
        args += ['--results', f'{name}={path}']
    A.main(args + list(extra))
    return json.loads((tmp_path / out).read_text())


def cell(report, name, condition, donor_set='all_donors'):
    return report['results'][name]['receptor_dependence'][condition][donor_set]


def test_hand_computed_delta_with_ties_and_midrank(tmp_path):
    # Original: two distractors above the truth and one tied with it -> midrank 2 + 3/2 = 3.5.
    # Donor 0: truth alone on top -> midrank 1. Donor 1: all 100 tied -> midrank 50.5.
    # Donor 2 has no replacement and contributes nothing.
    def scorer(rows):
        out = []
        for r in rows:
            j = -1 if '_d' not in r['peptide'] else int(r['peptide'].split('_d')[1])
            if r['peptide'].startswith('Q2'):
                out.append(0.)
            elif r['tra_cdr3'] == ORIG[0]:
                out.append(5. if j in (-1, 2) else 6. if j in (0, 1) else 0.)
            elif r['tra_cdr3'] == 'C0CC':
                out.append(10. if j == -1 else 0.)
            else:
                out.append(0.)
        return out
    q = query(1, donors=((0, True, []), (1, True, []), (2, False, [])))
    m = manifest([q, query(2, cluster='c2')])
    mp, rp = write(tmp_path, m, scorer)
    rep = run(tmp_path, mp, [('base', rp)])
    nr = lambda r: (100 - r) / 99
    d1 = nr(3.5) - (nr(1) + nr(50.5)) / 2
    # Query 2: original all tied (midrank 50.5), donor 0 all tied -> delta 0.
    c = cell(rep, 'base', 'test_iid')
    assert c['n_queries'] == 2 and c['n_unmatched'] == 0 and c['n_donor_scorings'] == 3
    assert c['point'] == pytest.approx((d1 + 0.) / 2, abs=1e-12)
    st = rep['results']['base']['stability']
    assert st['n_queries'] == 2
    assert st['ties'] == {'queries_with_true_ties': 2, 'queries_all_tied': 1}
    assert st['repeat'] == {'max_abs_logit_diff': 0.0, 'queries_midrank_changed': 0}
    # 3 scorings per query plus one per donor with a result: 6 + 3 = 9.
    assert st['seconds_per_scoring'] == pytest.approx(8.0 / 9)


def test_receptor_independent_scorer_reads_zero_and_inside(tmp_path):
    m = manifest([query(i, cluster=f'c{i % 3}') for i in range(6)])
    mp, rp = write(tmp_path, m, independent)
    rep = run(tmp_path, mp, [('base', rp)])
    for cond in ('test_iid', 'pooled'):
        c = cell(rep, 'base', cond)
        assert c['point'] == 0.0 and c['interval'] == [0.0, 0.0]
        assert c['interval_vs_margin'] == 'inside'


def test_receptor_dependent_scorer_reads_above(tmp_path):
    m = manifest([query(i, cluster=f'c{i % 3}') for i in range(6)])
    mp, rp = write(tmp_path, m, dependent)
    rep = run(tmp_path, mp, [('armD', rp)])
    c = cell(rep, 'armD', 'test_iid')
    assert c['point'] == pytest.approx(1.0)
    assert c['interval_vs_margin'] == 'above'
    assert 'useful' not in json.dumps(rep['results']) and 'negligible' not in json.dumps(rep['results'])


def test_unmatched_queries_are_counted_and_excluded(tmp_path):
    qs = [query(i, cluster=f'c{i}') for i in range(3)]
    qs.append(query(9, cluster='c9', donors=((0, False, []), (1, False, []))))
    mp, rp = write(tmp_path, manifest(qs), dependent)
    c = cell(run(tmp_path, mp, [('base', rp)]), 'base', 'test_iid')
    assert c['n_queries'] == 4 and c['n_unmatched'] == 1
    assert c['n_clusters'] == 3 and c['point'] == pytest.approx(1.0)


def test_known_positive_exclusion_changes_the_donor_count(tmp_path):
    qs = [query(i, cluster=f'c{i}', donors=((0, True, []), (1, True, [5]))) for i in range(3)]
    qs.append(query(7, cluster='c7', donors=((0, True, [3]),)))
    mp, rp = write(tmp_path, manifest(qs), dependent)
    rep = run(tmp_path, mp, [('base', rp)])
    every, kept = cell(rep, 'base', 'test_iid'), cell(rep, 'base', 'test_iid', 'excluding_known_positive')
    assert every['n_donor_scorings'] == 7 and every['n_unmatched'] == 0
    assert kept['n_donor_scorings'] == 3 and kept['n_unmatched'] == 1


def test_fewer_than_two_clusters_is_unresolved(tmp_path):
    m = manifest([query(i, cluster='same') for i in range(3)] +
                 [query(i, condition='test_level4', cluster=f'l{i}') for i in range(3, 6)])
    mp, rp = write(tmp_path, m, dependent)
    rep = run(tmp_path, mp, [('base', rp)])
    c = cell(rep, 'base', 'test_iid')
    assert c['point'] == pytest.approx(1.0) and c['interval'] is None
    assert c['interval_vs_margin'] == 'unresolved'
    assert cell(rep, 'base', 'test_level4')['interval_vs_margin'] == 'above'


def test_cluster_bootstrap_is_seeded_and_wider_when_clusters_are_shared():
    deltas = [v for v in (0.3, -0.2, 0.1, -0.1, 0.25) for _ in range(6)]
    shared = [f'k{i // 6}' for i in range(30)]
    alone = [f's{i}' for i in range(30)]
    a = A.cluster_interval(deltas, shared, 2000, 42)
    assert a == A.cluster_interval(deltas, shared, 2000, 42)
    assert a != A.cluster_interval(deltas, shared, 2000, 7)
    b = A.cluster_interval(deltas, alone, 2000, 42)
    assert a[1] - a[0] > b[1] - b[0]


def test_margin_geometry():
    assert A.versus_margin(0.03, 0.05, 0.02) == 'above'
    assert A.versus_margin(-0.05, -0.03, 0.02) == 'below'
    assert A.versus_margin(-0.01, 0.02, 0.02) == 'inside'
    assert A.versus_margin(0.0, 0.05, 0.02) == 'straddles'


def _refuses(tmp_path, mp, results, match, exc=ValueError):
    with pytest.raises(exc, match=match):
        run(tmp_path, mp, results)
    assert not (tmp_path / 'report.json').exists()


def test_refusals(tmp_path):
    m = manifest([query(i) for i in range(3)])
    mp, rp = write(tmp_path, m, dependent)
    _, bad = write(tmp_path, m, dependent, name='proto', mutate=lambda r: r.update(protocol='x'))
    _refuses(tmp_path, mp, [('a', bad)], 'protocol')
    _, bad = write(tmp_path, m, dependent, name='sha', mutate=lambda r: r.update(manifest_sha256='0' * 64))
    _refuses(tmp_path, mp, [('a', bad)], 'manifest')
    _, bad = write(tmp_path, m, dependent, name='short', mutate=lambda r: r['rows'].pop())
    _refuses(tmp_path, mp, [('a', bad)], 'rows')
    _, bad = write(tmp_path, m, dependent, name='dup', mutate=lambda r: r['rows'].append(r['rows'][0]))
    _refuses(tmp_path, mp, [('a', bad)], 'rows')
    _refuses(tmp_path, mp, [('a', rp), ('a', rp)], 'duplicate')
    nan = tmp_path / 'nan.json'
    nan.write_text(rp.read_text().replace('"seconds": 8.0', '"seconds": NaN'))
    _refuses(tmp_path, mp, [('a', nan)], 'non-finite')
    big = tmp_path / 'big.json'
    big.write_text(rp.read_text().replace('"seconds": 8.0', '"seconds": 1e999'))
    _refuses(tmp_path, mp, [('a', big)], 'non-finite')


def test_writes_once_and_refuses_to_overwrite(tmp_path):
    mp, rp = write(tmp_path, manifest([query(i) for i in range(3)]), dependent)
    rep = run(tmp_path, mp, [('base', rp)])
    assert rep['protocol'] == 'judge-diagnostic-analysis-v1'
    assert rep['manifest_sha256'] == hashlib.sha256(mp.read_bytes()).hexdigest()
    assert rep['results']['base']['checkpoint_sha256'] == 'b' * 64
    assert rep['results']['base']['code_revision'] == 'abc123'
    assert 'not a permutation test' in rep['reading']
    before = (tmp_path / 'report.json').read_bytes()
    with pytest.raises(FileExistsError):
        run(tmp_path, mp, [('base', rp)])
    assert (tmp_path / 'report.json').read_bytes() == before


def test_help_without_torch_importable():
    code = ("import sys; sys.modules['torch'] = None; "
            "from track_a.esm_sft.diagnostic_analyze import main; main(['--help'])")
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=os.environ)
    assert r.returncode == 0, r.stderr
    assert '--margin' in r.stdout


def _row(i, cluster, original, donor):
    s = lambda midrank: {'midrank': midrank}
    return {'row_id': str(i), 'condition': 'c', 'pep_cluster': cluster, 'original': s(original),
            'donors': [{'seed': 0, 'result': s(donor)}]}


def test_the_read_out_resamples_clusters_not_queries():
    """Two clusters of ten identical queries: a cluster draw can only give 0, 0.5 or 1."""
    rows = [_row(i, 'a', 50, 50) for i in range(10)] + [_row(i, 'b', 1, 100) for i in range(10, 20)]
    known = {('c', r['row_id'], 0): False for r in rows}
    cell = A.dependence(rows, known, 2000, 42, 0.02)['all_donors']
    assert cell['point'] == pytest.approx(0.5) and cell['n_clusters'] == 2
    assert cell['interval'] == [0.0, 1.0]                 # resampling 20 queries would give about [0.3, 0.7]


def test_a_recorded_rank_that_its_own_logits_contradict_is_refused(tmp_path):
    m = manifest([query(1), query(2)])
    def lie(res): res['rows'][0]['original']['midrank'] = 99
    mp, rp = write(tmp_path, m, dependent, mutate=lie)
    with pytest.raises(ValueError, match='midrank'):
        run(tmp_path, mp, [('res', rp)])


def test_cluster_labels_and_donors_are_the_manifests_not_the_results_files(tmp_path):
    m = manifest([query(1, cluster='a'), query(2, cluster='a'), query(3, cluster='b')])
    def relabel(res): res['rows'][0]['pep_cluster'] = 'made_up'
    mp, rp = write(tmp_path, m, dependent, mutate=relabel)
    with pytest.raises(ValueError, match='pep_cluster'):
        run(tmp_path, mp, [('res', rp)])
    def duplicate(res): res['rows'][1]['donors'].append(dict(res['rows'][1]['donors'][0]))
    mp, rp = write(tmp_path, m, dependent, name='dup', mutate=duplicate)
    with pytest.raises(ValueError, match='donors'):
        run(tmp_path, mp, [('dup', rp)], out='r2.json')
