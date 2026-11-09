"""Build a reproducible CPU-only diagnostic manifest from benchmark inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ('test_iid', 'test_novel_pep', 'test_level4')
SEEDS = range(5)


def clean(row):
    def value(v):
        if isinstance(v, (list, np.ndarray)):
            return [value(x) for x in v]
        if pd.isna(v):
            return None
        return v.item() if isinstance(v, np.generic) else v
    return {k: value(v) for k, v in row.items()}


def evidence(row):
    names = ' '.join(str(row.get(k) or '') for k in ('mhc_one_allele', 'mhc_two_allele')).upper()
    if re.search(r'(?:HLA-)?D[PRQ][A-Z0-9]*\*', names):
        return 'named_class_ii'
    if re.search(r'(?:HLA-)?[ABC]\*', names):
        return 'named_class_i'
    return 'unknown'


def context(row):
    a, b, m1, m2 = (row.get(k) for k in ('tra_cdr3', 'trb_cdr3', 'mhc_one', 'mhc_two'))
    if not a or not b or not m1:
        return None
    return a, b, m1, m2


def candidates(pool, truth, row_id, mode):
    rng = np.random.default_rng(int(hashlib.md5(str(row_id).encode()).hexdigest()[:8], 16))
    if mode == 'uniform':
        others = [p for p in rng.choice(pool, size=108, replace=False) if p != truth][:99]
    else:
        by_len = defaultdict(list)
        for p in pool:
            by_len[len(p)].append(p)
        same = [p for p in by_len.get(len(truth), ()) if p != truth]
        if len(same) >= 99:
            others = list(rng.choice(np.asarray(same, dtype=object), size=99, replace=False))
        else:
            others = list(rng.permutation(np.asarray(same, dtype=object))) if same else []
            for d in range(1, max(by_len, default=0) + 1):
                if len(others) >= 99:
                    break
                near = [p for length in (len(truth)-d, len(truth)+d)
                        for p in by_len.get(length, ()) if p != truth]
                if near:
                    others += list(rng.choice(np.asarray(near, dtype=object), size=min(99-len(others), len(near)), replace=False))
    return [truth] + others


def build_manifest(splits, pool, max_queries=100):
    if any(not isinstance(p, str) or not p.strip() for p in pool):
        raise ValueError('pool must contain nonempty strings')
    if len(set(pool)) != len(pool):
        raise ValueError('pool peptides must be unique')
    required = {'row_id', 'peptide', 'tra_cdr3', 'trb_cdr3', 'mhc_one', 'mhc_two', 'mhc_one_allele', 'mhc_two_allele'}
    for name, frame in splits.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{name} missing required columns: {sorted(missing)}')
    if max_queries < 1 or len(pool) < 108:
        raise ValueError('max_queries must be positive and pool must contain at least 108 entries')
    records = {}
    for name, frame in splits.items():
        f = frame[frame.peptide.notna() & frame.trb_cdr3.notna()].drop_duplicates('row_id')
        records[name] = [clean(r) for r in f.to_dict('records')]
    positives = defaultdict(set)
    for rows in records.values():
        for row in rows:
            ctx = context(row)
            if ctx:
                positives[ctx].add(row['peptide'])
    train_peptides = {r['peptide'] for r in records.get('train', [])}
    rows_out = []
    counts = {}
    summary = {}
    for condition in CONDITIONS:
        if condition not in records:
            continue
        eligible = [r for r in records[condition] if r['peptide'] in set(pool) and r.get('tra_cdr3') and r.get('trb_cdr3')]
        # Round-robin clusters with seeded stable ordering, to avoid a dominant cluster.
        groups = defaultdict(list)
        for r in eligible:
            groups[str(r.get('pep_cluster') or r['peptide'])].append(r)
        rng = np.random.default_rng(42)
        keys = sorted(groups)
        rng.shuffle(keys)
        for key in keys:
            rng.shuffle(groups[key])
        selected = []
        while keys and len(selected) < max_queries:
            next_keys = []
            for key in keys:
                if len(selected) < max_queries:
                    selected.append(groups[key].pop())
                if groups[key]:
                    next_keys.append(key)
            keys = next_keys
        counts[condition] = 0
        start = len(rows_out)
        for row in selected:
            ctx = context(row)
            donor_pool = []
            if ctx and (ctx[3] or evidence(row) == 'named_class_i'):
                for other in records[condition]:
                    oc = context(other)
                    if oc and (oc[3] or evidence(other) == 'named_class_i') and oc[2:] == ctx[2:] and tuple(map(len, oc[:2])) == tuple(map(len, ctx[:2])) and oc[:2] != ctx[:2] and row['peptide'] not in positives[oc]:
                        donor_pool.append(other)
            donor_pool.sort(key=lambda r: str(r['row_id']))
            donors = []
            for seed in SEEDS:
                digest = hashlib.sha256(f"{condition}|{row['row_id']}|{seed}".encode()).hexdigest()
                donors.append({'seed': seed, 'row': donor_pool[int(digest, 16) % len(donor_pool)] if donor_pool else None})
            if not donor_pool:
                counts[condition] += 1
            cs = {mode: candidates(pool, row['peptide'], row['row_id'], mode) for mode in ('uniform', 'length_matched')}
            for donor in donors:
                donor['known_positive_candidates'] = sorted(positives.get(context(donor['row']), set()).intersection(cs['uniform'] + cs['length_matched'])) if donor['row'] else []
            rows_out.append({'row_id': row['row_id'], 'condition': condition, 'peptide': row['peptide'],
                'pep_cluster': row.get('pep_cluster') or 'peptide:' + row['peptide'], 'original': row, 'candidates': cs, 'donors': donors,
                'metadata': {'class_evidence': evidence(row), 'known_positive_candidates': sorted(positives[ctx].intersection(cs['uniform'] + cs['length_matched'])) if ctx else None,
                    'training_peptide_exposure': row['peptide'] in train_peptides,
                    'training_candidate_exposure': {k: [p in train_peptides for p in v] for k, v in cs.items()},
                    'training_candidate_exposure_count': {k: sum(p in train_peptides for p in v) for k, v in cs.items()},
                    'length_match_count': {k: sum(len(p) == len(row['peptide']) for p in v[1:]) for k, v in cs.items()}}})
        chosen = rows_out[start:]
        summary[condition] = {'eligible': len(eligible), 'selected': len(chosen),
            'class_evidence': {label: sum(r['metadata']['class_evidence'] == label for r in chosen) for label in ('named_class_i', 'named_class_ii', 'unknown')},
            'queries_with_known_alternative_positive': sum(any(p != r['peptide'] for p in (r['metadata']['known_positive_candidates'] or [])) for r in chosen),
            'unshuffleable': counts[condition],
            'selected_distinct_clusters': len({r['pep_cluster'] for r in chosen}),
            'eligible_class_evidence': {label: sum(evidence(r) == label for r in eligible) for label in ('named_class_i', 'named_class_ii', 'unknown')},
            'eligible_context_unavailable': sum(context(r) is None for r in eligible),
            'eligible_with_known_alternative_positive': sum(bool(positives[context(r)] - {r['peptide']}) for r in eligible if context(r) is not None)}
    return {'summary': summary, 'protocol': 'judge-diagnostic-v1', 'input_hashes': {}, 'max_queries_per_condition': max_queries,
            'donor_method': 'independent replacement within exact MHC sequence pair and CDR3 length strata; not a permutation',
            'unshuffleable_counts': counts, 'rows': rows_out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-queries', type=int, default=100)
    args = parser.parse_args()
    paths = {s: args.splits_dir / f'as_paired_i_{s}.parquet' for s in ('train', 'val', *CONDITIONS)}
    pool_path = args.splits_dir / 'as_class_i_candidate_pool.tsv'
    if args.out.exists():
        parser.error(f'output already exists: {args.out}')
    for path in [*paths.values(), pool_path]:
        if not path.is_file():
            parser.error(f'missing required input: {path}')
    pool_df = pd.read_csv(pool_path, sep='\t', keep_default_na=False)
    pool = (pool_df['peptide'] if 'peptide' in pool_df else pool_df.iloc[:, 0]).astype(str).tolist()
    manifest = build_manifest({s: pd.read_parquet(p) for s, p in paths.items()}, pool, args.max_queries)
    manifest['input_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths.values(), pool_path]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as out:
        json.dump(manifest, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(manifest['summary'], indent=2))


if __name__ == '__main__':
    main()
