import numpy as np
import pandas as pd

from track_a.esm_sft.diagnostic_data import build_manifest


def row(i, peptide, receptor, allele='HLA-A*01:01'):
    return dict(row_id=i, peptide=peptide, tra_cdr3=receptor, trb_cdr3='BBB',
                mhc_one='MHC', mhc_two='MHC2', mhc_one_allele=allele, mhc_two_allele=None,
                pep_cluster=peptide)


def test_manifest_fixed_candidates_and_safe_donors():
    pool = [f'{i:09d}' for i in range(120)]
    a, b = row('a', pool[0], 'AAA'), row('b', pool[1], 'CCC')
    # A donor context already known positive for the query truth is unsafe.
    known = row('c', pool[0], 'CCC')
    splits = {'train': pd.DataFrame([known]), 'test_iid': pd.DataFrame([a, b])}
    m = build_manifest(splits, pool, max_queries=100)
    assert m == build_manifest(splits, pool, max_queries=100)
    q = next(x for x in m['rows'] if x['row_id'] == 'a')
    expected = [p for p in np.random.default_rng(int(__import__('hashlib').md5(b'a').hexdigest()[:8], 16)).choice(pool, size=108, replace=False) if p != pool[0]][:99]
    assert q['candidates']['uniform'] == [pool[0]] + expected
    assert all(d['row'] is None for d in q['donors'])
    assert q['metadata']['training_peptide_exposure'] is True
    assert q['metadata']['length_match_count']['length_matched'] == 99
    assert q['metadata']['class_evidence'] == 'named_class_i'


def test_missing_mhc_and_unknown_class_never_match_donors():
    pool = [f'{i:09d}' for i in range(120)]
    a, b = row('a', pool[0], 'AAA', ''), row('b', pool[1], 'CCC', '')
    a['mhc_two'] = b['mhc_two'] = None
    m = build_manifest({'train': pd.DataFrame([a]), 'test_iid': pd.DataFrame([a, b])}, pool)
    q = m['rows'][0]
    assert q['metadata']['class_evidence'] == 'unknown'
    assert all(d['row'] is None for d in q['donors'])


def test_unknown_class_observed_positives_and_candidate_training_exposure():
    pool = [f'{i:09d}' for i in range(120)]
    a = row('a', pool[0], 'AAA', '')
    a['mhc_two'] = None
    alt = dict(a, row_id='alt', peptide=pool[1])
    m = build_manifest({'train': pd.DataFrame([alt]), 'test_iid': pd.DataFrame([a])}, pool)
    q = m['rows'][0]
    assert pool[1] in q['metadata']['known_positive_candidates']
    mask = q['metadata']['training_candidate_exposure']['uniform']
    assert mask == [p == pool[1] for p in q['candidates']['uniform']]
    assert q['metadata']['training_candidate_exposure_count']['uniform'] == sum(mask)


def test_pool_validation_and_missing_cluster_fallback():
    import pytest
    pool = [f'{i:09d}' for i in range(120)]
    a = row('a', pool[0], 'AAA')
    a['pep_cluster'] = None
    splits = {'train': pd.DataFrame([a]), 'test_iid': pd.DataFrame([a])}
    assert build_manifest(splits, pool)['rows'][0]['pep_cluster'].startswith('peptide:')
    with pytest.raises(ValueError, match='unique'):
        build_manifest(splits, pool + [pool[0]])
    with pytest.raises(ValueError, match='nonempty'):
        build_manifest(splits, pool[:-1] + [''])


def test_candidate_draw_matches_judge_source():
    import ast
    from pathlib import Path
    from track_a.esm_sft.diagnostic_data import candidates
    tree = ast.parse(Path('track_a/esm_sft/judge.py').read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_qseed', '_pool_by_length', '_draw_distractors')]
    ns = {'np': np, 'hashlib': __import__('hashlib')}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'judge.py', 'exec'), ns)
    pool = [str(i) * (8 + i % 5) for i in range(120)]
    for truth in (pool[0], pool[35]):
        for mode in ('uniform', 'length_matched'):
            rng = np.random.default_rng(ns['_qseed']('id'))
            expected, _ = ns['_draw_distractors'](pool, ns['_pool_by_length'](pool), truth, rng, 99, mode)
            assert candidates(pool, truth, 'id', mode) == [truth] + expected
