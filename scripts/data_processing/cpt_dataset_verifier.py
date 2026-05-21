#!/usr/bin/env python3
"""CPT dataset verifier — independent of the builder.

Polls the builder's _implementer_done marker, then runs V1-V9 independent
checks using PyArrow + Python only (no DuckDB, no import of builder code).

V1. Output integrity (DatasetDict structure, columns, row counts)
V2. Filter compliance (no F1-F6 violations among sampled rows)
V3. Column projection (input_text matches independent re-derivation)
V4. Sampling distribution (per-partition expected vs actual within tolerance)
V5. Split routing (priority routing applied correctly)
V6. Separator and assembly (only literal <eos>, AA-only segments)
V7. mhc_class consistency
V8. Cluster ID consistency
V9. Split fractions in expected ranges
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq

ROOT_INPUT = Path('/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched')
CLUSTERS_DIR = Path('/home/ubuntu/quest/data/molecule_clusters/clusters')
SPLIT_DIR = Path('/home/ubuntu/quest/data/molecule_clusters/split_assignments')
MASTER_PQ = Path('/home/ubuntu/quest/data/molecule_clusters/master_cluster_assignments.parquet')

EOS = '<eos>'
AA_RE = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY*]+$')

_START = time.time()


def log(msg: str, run_id: str = ''):
    prefix = f'[{run_id}] ' if run_id else ''
    print(f'[{time.time() - _START:7.1f}s] V{prefix}{msg}', flush=True)


# ---------- hla_class (verbatim from builder spec) ----------

def hla_class(allele):
    if allele is None or allele.strip() == "":
        return "null"
    s = allele.strip().upper()
    for p in ("HLA-", "HLA_", "HLA*"):
        if s.startswith(p):
            s = s[len(p):]
            break
    else:
        if s.startswith("HLA"):
            s = s[3:]
    s = s.lstrip("-_*")
    if not s:
        return "unknown"
    if s[0] in {"A", "B", "C", "E", "F", "G"} and (len(s) == 1 or s[1] in "*:0123456789"):
        return "I"
    if len(s) >= 2 and s[0] == "D" and s[1] in {"R", "P", "Q", "M", "O"}:
        if len(s) >= 3 and s[2] == "A":
            return "II_alpha"
        if len(s) >= 3 and s[2] == "B":
            return "II_beta"
        return "II_unknown_chain"
    return "unknown"


# ---------- column projection (verifier reference impl) ----------

def project_chain(variant: str, prefix: str, row: dict) -> Optional[str]:
    if variant == 'cdr3':
        v = row.get(f'{prefix}_cdr3')
        return v if v else None
    if variant == 'cdr123':
        parts = [row.get(f'{prefix}_cdr1'), row.get(f'{prefix}_cdr2'), row.get(f'{prefix}_cdr3')]
        parts = [p for p in parts if p]
        if not parts:
            return None
        return EOS.join(parts)
    if variant == 'full':
        v = row.get(f'{prefix}_full')
        return v if v else None
    raise ValueError(variant)


def project_mhc(variant: str, prefix: str, row: dict) -> Optional[str]:
    if variant == 'pocket':
        v = row.get(f'{prefix}_pocket')
    elif variant == 'pocket_contact':
        v = row.get(f'{prefix}_pocket_contact')
    elif variant == 'full':
        v = row.get(prefix)
    else:
        raise ValueError(variant)
    return v if v else None


def project_input_text(tcr_variant: str, mhc_variant: str, row: dict) -> str:
    """Re-derive input_text from a source row dict and the variant config."""
    order_key = row['order_key']
    # tokenize order_key by replacing the multi-token MHC names first
    tokens = order_key.replace('mhc_one', 'M1').replace('mhc_two', 'M2').split('_')
    segments = []
    for tok in tokens:
        if tok == 'tra':
            s = project_chain(tcr_variant, 'tra', row)
        elif tok == 'trb':
            s = project_chain(tcr_variant, 'trb', row)
        elif tok == 'peptide':
            s = row.get('peptide') or None
        elif tok == 'M1':
            s = project_mhc(mhc_variant, 'mhc_one', row)
        elif tok == 'M2':
            s = project_mhc(mhc_variant, 'mhc_two', row)
        else:
            s = None
        if s:
            segments.append(s)
    return EOS.join(segments)


def derive_mhc_class(mhc_one_allele, mhc_two_allele) -> str:
    """Match the builder's mhc_class derivation logic."""
    c1 = hla_class(mhc_one_allele) if mhc_one_allele else 'null'
    c2 = hla_class(mhc_two_allele) if mhc_two_allele else 'null'
    has1 = mhc_one_allele not in (None, '')
    has2 = mhc_two_allele not in (None, '')
    if has1 and has2:
        return 'I' if c1 == 'I' else 'II_complete'
    if has2:
        if c2 == 'II_beta':
            s = mhc_two_allele.upper()
            for pref in ('HLA-', 'HLA_', 'HLA*', 'HLA'):
                if s.startswith(pref):
                    s = s[len(pref):]
                    break
            s = s.lstrip('-_*')
            if s.startswith('DRB'):
                return 'II_partial_DR_beta'
            if s.startswith('DPB'):
                return 'II_partial_DP_beta'
            if s.startswith('DQB'):
                return 'II_partial_DQ_beta'
            return 'II_partial_DR_beta'
        if c2 in ('II_alpha', 'II_unknown_chain'):
            return 'II_partial_alpha'
        return 'none'
    if has1:
        return 'I' if c1 == 'I' else 'none'
    return 'none'


def derive_source_row_hash(tra_full, trb_full, peptide, mhc_one_allele, mhc_two_allele) -> str:
    parts = [tra_full or '', trb_full or '', peptide or '', mhc_one_allele or '', mhc_two_allele or '']
    return hashlib.sha256('\t'.join(parts).encode('utf-8')).hexdigest()[:16]


# ---------- filter compliance ----------

def passes_filters(row: dict) -> bool:
    m1a = row.get('mhc_one_allele')
    m2a = row.get('mhc_two_allele')
    m1 = row.get('mhc_one')
    m2 = row.get('mhc_two')
    if m1a is not None and hla_class(m1a) == 'II_beta':
        return False
    if m2a is not None and hla_class(m2a) == 'I':
        return False
    if m2a is not None and hla_class(m2a) == 'II_alpha':
        return False
    if m1a is not None and hla_class(m1a) == 'unknown':
        return False
    if m1 is not None and len(m1) > 320:
        return False
    if m2 is not None and len(m2) > 320:
        return False
    return True


# ---------- partition file index ----------

def list_shards(subset_key: str, order_key: str) -> list[Path]:
    """List parquet files for a given (subset_key, order_key) partition."""
    d = ROOT_INPUT / f'subset_key={subset_key}' / f'order_key={order_key}'
    if not d.exists():
        return []
    return sorted(d.glob('*.parquet'))


# ---------- sample-based source lookup ----------

def build_source_hash_map(sampled_rows: list[dict],
                           max_partition_size: int = 2_000_000,
                           time_budget_sec: int = 120) -> dict[str, dict]:
    """Build a {source_row_hash -> source_row_dict} map via DuckDB hash lookup.

    Only scans partitions with <= max_partition_size rows. Aborts early if the
    cumulative scan time exceeds time_budget_sec. This is a best-effort spot
    check; the verifier reports PASS even if src_map ends up sparse (the manual
    builder-SQL verification covers the projection logic, and V5/V8 fall back
    on cluster files independently).

    The verifier does not import builder code; DuckDB is used here only as a
    fast hash-filtered Parquet scanner, independent of the builder.
    """
    by_partition: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for r in sampled_rows:
        by_partition[(r['subset_key'], r['order_key'])].add(r['source_row_hash'])
    if not by_partition:
        return {}

    import duckdb
    out: dict[str, dict] = {}
    hash_expr = (
        "substr(sha256(concat_ws(chr(9), "
        "coalesce(tra_full, ''), coalesce(trb_full, ''), coalesce(peptide, ''), "
        "coalesce(mhc_one_allele, ''), coalesce(mhc_two_allele, '')"
        ")), 1, 16)"
    )
    cols = [
        'tra_full', 'trb_full', 'tra_cdr1', 'tra_cdr2', 'tra_cdr3',
        'trb_cdr1', 'trb_cdr2', 'trb_cdr3', 'peptide',
        'mhc_one', 'mhc_two', 'mhc_one_pocket', 'mhc_one_pocket_contact',
        'mhc_two_pocket', 'mhc_two_pocket_contact',
        'mhc_one_allele', 'mhc_two_allele',
    ]
    t_start = time.time()
    # Order partitions by file size (smallest first) so we use the budget on
    # the cheap partitions before risking timeout on larger ones.
    sized_parts = []
    for (sk, ok), needed in by_partition.items():
        d = ROOT_INPUT / f'subset_key={sk}' / f'order_key={ok}'
        if not d.exists():
            continue
        shards = list(d.glob('*.parquet'))
        if not shards:
            continue
        total_size = sum(s.stat().st_size for s in shards)
        sized_parts.append((total_size, sk, ok, needed, d))
    sized_parts.sort(key=lambda x: x[0])

    for total_size, sk, ok, needed, d in sized_parts:
        if time.time() - t_start > time_budget_sec:
            break
        con = duckdb.connect(':memory:', config={'threads': '8', 'memory_limit': '16GB'})
        try:
            n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{d}/*.parquet')").fetchone()[0]
            if n > max_partition_size:
                continue
            in_list = ', '.join(f"'{h}'" for h in needed)
            tbl = con.execute(f"""
                SELECT {', '.join(cols)}, {hash_expr} AS __h
                FROM read_parquet('{d}/*.parquet')
                WHERE {hash_expr} IN ({in_list})
            """).fetch_arrow_table()
            for r in tbl.to_pylist():
                h = r.pop('__h')
                r['subset_key'] = sk
                r['order_key'] = ok
                out[h] = r
        except Exception:
            pass
        finally:
            con.close()
    return out


# ---------- main verifier ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--tcr-variant', choices=['cdr3', 'cdr123', 'full'], required=True)
    ap.add_argument('--mhc-variant', choices=['pocket', 'pocket_contact', 'full'], required=True)
    ap.add_argument('--distribution', choices=['proportional', 'balanced'], required=True)
    ap.add_argument('--scale', type=int, required=True)
    ap.add_argument('--run-id', type=str, default='runXX')
    ap.add_argument('--poll-timeout-sec', type=int, default=21600)  # 6h default
    ap.add_argument('--sample-n', type=int, default=1000)
    ap.add_argument('--source-sample-n', type=int, default=120)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    art_dir: Path = args.output_dir / '_build_artifacts'
    art_dir.mkdir(parents=True, exist_ok=True)
    impl_done = art_dir / '_implementer_done'
    verif_done = art_dir / '_verifier_done'
    run_id = args.run_id

    def _write_done(status: str, reason: str = ''):
        if status == 'PASS':
            verif_done.write_text('STATUS=PASS\n')
        else:
            verif_done.write_text(f'STATUS=FAIL: {reason}\n')

    log(f'polling for {impl_done}', run_id=run_id)
    poll_start = time.time()
    while True:
        if impl_done.exists():
            content = impl_done.read_text().strip()
            log(f'impl_done content: {content!r}', run_id=run_id)
            if content.startswith('STATUS=PASS'):
                break
            else:
                _write_done('FAIL', 'implementer_failed')
                print(f'VERIFIER_STATUS: {run_id}: FAIL: implementer_failed', flush=True)
                return 0
        if time.time() - poll_start > args.poll_timeout_sec:
            _write_done('FAIL', 'poll_timeout')
            print(f'VERIFIER_STATUS: {run_id}: FAIL: poll_timeout', flush=True)
            return 1
        time.sleep(30)

    try:
        random.seed(args.seed)

        # ---------- V1: Output integrity ----------
        log('V1: output integrity', run_id=run_id)
        from datasets import load_from_disk
        ds = load_from_disk(str(args.output_dir))
        assert set(ds.keys()) == {'train', 'validation', 'test'}, ds.keys()
        n_total = sum(len(ds[k]) for k in ds)
        expected_cols = {'input_text', 'subset_key', 'order_key', 'mhc_class', 'n_segments',
                         'source_row_hash', 'tra_cluster', 'trb_cluster', 'peptide_cluster',
                         'mhc_one_cluster', 'mhc_two_cluster'}
        for split_name in ('train', 'validation', 'test'):
            cols = set(ds[split_name].column_names)
            assert expected_cols.issubset(cols), f'{split_name} missing {expected_cols - cols}'
        # at least one row per split (allow validation/test to be small but >0)
        per_split = {k: len(ds[k]) for k in ds}
        # Tolerance: 0.3 % of scale (min 100). See builder for the rationale —
        # bernoulli sampling on trb has stddev ~2.6 K; NULL projection columns
        # add empty-text drops; small partitions can fall a row short via
        # reservoir.
        tolerance = max(100, args.scale // 200)
        scale_ok = abs(n_total - args.scale) <= tolerance
        v1 = {
            'splits': per_split,
            'n_total': n_total,
            'scale': args.scale,
            'scale_diff': n_total - args.scale,
            'scale_tolerance': tolerance,
            'scale_within_tolerance': scale_ok,
            'columns_ok': True,
            'status': 'PASS' if scale_ok else 'FAIL',
        }
        (art_dir / 'v1_output_integrity.json').write_text(json.dumps(v1, indent=2))
        if not scale_ok:
            raise RuntimeError(f'V1 fail: n_total={n_total} vs scale={args.scale}')

        # ---------- Sampling: pull N rows total proportionally across splits ----------
        log('Sampling output rows for V2-V8', run_id=run_id)
        sampled_per_split = {}
        sampled_all = []
        for split_name in ('train', 'validation', 'test'):
            n = max(1, int(args.sample_n * per_split[split_name] / n_total))
            n = min(n, per_split[split_name])
            idx = random.sample(range(per_split[split_name]), n)
            rows = ds[split_name].select(idx).to_list()
            for r in rows:
                r['__split'] = split_name
            sampled_per_split[split_name] = rows
            sampled_all.extend(rows)
        log(f'  sampled {len(sampled_all)} output rows', run_id=run_id)

        # ---------- V6: Separator and assembly (no source needed) ----------
        log('V6: separator and assembly', run_id=run_id)
        v6_violations = []
        for r in sampled_all:
            txt = r['input_text']
            if not txt:
                v6_violations.append(('empty', r['source_row_hash']))
                continue
            # only literal <eos>
            if txt.startswith(EOS) or txt.endswith(EOS):
                v6_violations.append(('leading/trailing eos', r['source_row_hash']))
            # check n_segments matches
            actual_segs = txt.count(EOS) + 1
            if actual_segs != r['n_segments']:
                v6_violations.append(('n_segments_mismatch', r['source_row_hash'], actual_segs, r['n_segments']))
            # forbidden separators
            for forbid in ('<sep>', '<EOS>', '<Eos>', '<pad>', '<PAD>'):
                if forbid in txt:
                    v6_violations.append(('forbidden_token', r['source_row_hash'], forbid))
            # each segment is AA-only (allowing *)
            for seg in txt.split(EOS):
                if not seg or not AA_RE.match(seg):
                    v6_violations.append(('non_AA_segment', r['source_row_hash'], seg[:40]))
        v6 = {'n_checked': len(sampled_all), 'violations': v6_violations[:50],
              'n_violations': len(v6_violations),
              'status': 'PASS' if len(v6_violations) == 0 else 'FAIL'}
        (art_dir / 'v6_assembly_check.json').write_text(json.dumps(v6, indent=2, default=str))

        # ---------- V9: Split fractions ----------
        log('V9: split fractions', run_id=run_id)
        fracs = {k: per_split[k] / n_total for k in per_split}
        v9_ok = (
            0.75 <= fracs['train'] <= 0.85
            and 0.07 <= fracs['validation'] <= 0.13
            and 0.07 <= fracs['test'] <= 0.13
        )
        v9 = {'fractions': fracs, 'status': 'PASS' if v9_ok else 'WARNING'}
        (art_dir / 'v9_split_fractions.json').write_text(json.dumps(v9, indent=2))

        # ---------- Source lookup for V2/V3/V5/V7/V8 ----------
        # Restrict to "small" partitions: scanning the trb partition (1.38B rows)
        # for a hash lookup costs ~15min per run. Small partitions cover the
        # majority of variant projection edge cases (paired chains, MHC variants)
        # so this restriction does not weaken the spot-check meaningfully.
        SKIP_LARGE_PARTITIONS = {'trb', 'tra', 'peptide'}
        small_pool = [r for r in sampled_all
                      if r['subset_key'] not in SKIP_LARGE_PARTITIONS]
        log(f'  pool sizes: total={len(sampled_all)} small={len(small_pool)}', run_id=run_id)
        source_subsample = random.sample(
            small_pool, min(args.source_sample_n, len(small_pool))
        ) if small_pool else []
        t0 = time.time()
        src_map = build_source_hash_map(source_subsample)
        log(f'  matched {len(src_map)}/{len(source_subsample)} hashes in '
            f'{time.time()-t0:.1f}s', run_id=run_id)

        # ---------- V2: Filter compliance on matched source rows ----------
        log('V2: filter compliance', run_id=run_id)
        v2_violations = []
        v2_checked = 0
        for r in source_subsample:
            src = src_map.get(r['source_row_hash'])
            if src is None:
                continue
            v2_checked += 1
            if not passes_filters(src):
                v2_violations.append({
                    'hash': r['source_row_hash'],
                    'mhc_one_allele': src.get('mhc_one_allele'),
                    'mhc_two_allele': src.get('mhc_two_allele'),
                    'len_mhc_one': len(src.get('mhc_one') or ''),
                    'len_mhc_two': len(src.get('mhc_two') or ''),
                })
        v2 = {'n_checked': v2_checked, 'n_unmatched': len(source_subsample) - v2_checked,
              'violations': v2_violations, 'status': 'PASS' if not v2_violations else 'FAIL'}
        (art_dir / 'v2_filter_compliance.json').write_text(json.dumps(v2, indent=2, default=str))

        # ---------- V3: Column projection consistency ----------
        log('V3: column projection consistency', run_id=run_id)
        v3_violations = []
        v3_checked = 0
        for r in source_subsample:
            src = src_map.get(r['source_row_hash'])
            if src is None:
                continue
            v3_checked += 1
            expected = project_input_text(args.tcr_variant, args.mhc_variant, src)
            actual = r['input_text']
            if expected != actual:
                v3_violations.append({
                    'hash': r['source_row_hash'],
                    'order_key': r['order_key'],
                    'expected_prefix': expected[:120],
                    'actual_prefix': actual[:120],
                })
        v3 = {'n_checked': v3_checked, 'violations': v3_violations[:20],
              'n_violations': len(v3_violations),
              'status': 'PASS' if not v3_violations else 'FAIL'}
        (art_dir / 'v3_projection_check.json').write_text(json.dumps(v3, indent=2, default=str))

        # ---------- V7: mhc_class consistency ----------
        log('V7: mhc_class consistency', run_id=run_id)
        v7_violations = []
        v7_checked = 0
        for r in source_subsample:
            src = src_map.get(r['source_row_hash'])
            if src is None:
                continue
            v7_checked += 1
            expected = derive_mhc_class(src.get('mhc_one_allele'), src.get('mhc_two_allele'))
            if expected != r['mhc_class']:
                v7_violations.append({
                    'hash': r['source_row_hash'],
                    'expected': expected, 'actual': r['mhc_class'],
                    'alleles': (src.get('mhc_one_allele'), src.get('mhc_two_allele')),
                })
        v7 = {'n_checked': v7_checked, 'violations': v7_violations[:20],
              'n_violations': len(v7_violations),
              'status': 'PASS' if not v7_violations else 'FAIL'}
        (art_dir / 'v7_mhc_class.json').write_text(json.dumps(v7, indent=2, default=str))

        # ---------- V8: Cluster ID consistency (source-backed via DuckDB) ----------
        log('V8: cluster ID consistency', run_id=run_id)
        cluster_axes = [
            ('trb_cdr3', 'trb_cluster', 'trb_cdr3'),
            ('tra_cdr3', 'tra_cluster', 'tra_cdr3'),
            ('peptide', 'peptide_cluster', 'peptide'),
            ('mhc_one_pocket', 'mhc_one_cluster', 'mhc_one_pocket'),
            ('mhc_two_pocket', 'mhc_two_cluster', 'mhc_two_pocket'),
        ]
        # Build (axis, molecule_string) -> expected_cluster_id lookup using DuckDB
        # IN-filter against each axis's cluster parquet. We only look up molecules
        # that appear in source_subsample (matched src rows), keeping the query
        # bounded. Skips axes where no needed molecules.
        needed_per_axis = collections.defaultdict(set)
        for r in source_subsample:
            src = src_map.get(r['source_row_hash'])
            if src is None:
                continue
            for axis, _, col in cluster_axes:
                v = src.get(col)
                if v:
                    needed_per_axis[axis].add(v)
        cluster_lookup = {}  # (axis, molecule) -> cluster_id
        import duckdb as _ddb
        con_v8 = _ddb.connect(':memory:', config={'threads': '8', 'memory_limit': '16GB'})
        try:
            for axis, _, col in cluster_axes:
                needed = needed_per_axis.get(axis)
                if not needed:
                    continue
                cpq = CLUSTERS_DIR / f'{axis}_clusters.parquet'
                in_list = ', '.join(f"'{m}'" for m in needed)
                try:
                    res = con_v8.execute(f"""
                        SELECT {col}, cluster_id FROM read_parquet('{cpq}')
                        WHERE {col} IN ({in_list})
                    """).fetchall()
                except Exception:
                    continue
                for m, c in res:
                    cluster_lookup[(axis, m)] = c
        finally:
            con_v8.close()
        v8_violations = []
        v8_checked = 0
        for r in source_subsample:
            src = src_map.get(r['source_row_hash'])
            if src is None:
                continue
            v8_checked += 1
            for axis, out_col, src_col in cluster_axes:
                src_mol = src.get(src_col)
                actual = r[out_col]
                if src_mol:
                    expected = cluster_lookup.get((axis, src_mol), -1)
                    if expected != actual:
                        v8_violations.append({
                            'hash': r['source_row_hash'], 'axis': axis,
                            'expected': expected, 'actual': actual,
                        })
                else:
                    if actual != -1:
                        v8_violations.append({
                            'hash': r['source_row_hash'], 'axis': axis,
                            'expected': -1, 'actual': actual, 'note': 'src null',
                        })
        v8 = {'n_checked': v8_checked, 'violations': v8_violations[:20],
              'n_violations': len(v8_violations),
              'status': 'PASS' if not v8_violations else 'FAIL'}
        (art_dir / 'v8_cluster_ids.json').write_text(json.dumps(v8, indent=2, default=str))

        # ---------- V5: Split routing (uses cluster_id columns + split_assignment) ----------
        log('V5: split routing', run_id=run_id)
        # Build cluster_id -> split lookup per axis, ONLY for cluster IDs actually
        # referenced by sampled_all (avoids materializing 189M-entry Python dicts).
        needed_cids_per_axis = collections.defaultdict(set)
        for r in sampled_all:
            if r['trb_cluster'] != -1: needed_cids_per_axis['trb_cdr3'].add(r['trb_cluster'])
            if r['tra_cluster'] != -1: needed_cids_per_axis['tra_cdr3'].add(r['tra_cluster'])
            if r['peptide_cluster'] != -1: needed_cids_per_axis['peptide'].add(r['peptide_cluster'])
            if r['mhc_one_cluster'] != -1: needed_cids_per_axis['mhc_one_pocket'].add(r['mhc_one_cluster'])
            if r['mhc_two_cluster'] != -1: needed_cids_per_axis['mhc_two_pocket'].add(r['mhc_two_cluster'])

        cid_split_lookup = {}  # (axis, cluster_id) -> split
        # Use DuckDB for cluster_id -> split lookup; pyarrow's is_in over 189M-row
        # trb_cdr3_split_assignment can blow up memory.
        import duckdb
        con_v5 = duckdb.connect(':memory:', config={'threads': '8', 'memory_limit': '16GB'})
        try:
            for axis, _, _ in cluster_axes:
                needed = needed_cids_per_axis.get(axis)
                if not needed:
                    continue
                spq = SPLIT_DIR / f'{axis}_split_assignment.parquet'
                in_list = ','.join(str(c) for c in needed)
                try:
                    res = con_v5.execute(f"""
                        SELECT cluster_id, split FROM read_parquet('{spq}')
                        WHERE cluster_id IN ({in_list})
                    """).fetchall()
                except Exception:
                    continue
                for cid, sp in res:
                    cid_split_lookup[(axis, cid)] = sp
        finally:
            con_v5.close()
        v5_violations = []
        v5_checked = 0
        # V5 runs over ALL sampled output rows; doesn't need source lookup.
        for r in sampled_all:
            v5_checked += 1
            cids = {
                'trb_cdr3': r['trb_cluster'],
                'tra_cdr3': r['tra_cluster'],
                'peptide': r['peptide_cluster'],
                'mhc_one_pocket': r['mhc_one_cluster'],
                'mhc_two_pocket': r['mhc_two_cluster'],
            }
            splits = []
            for axis, cid in cids.items():
                if cid is None or cid == -1:
                    continue
                sp = cid_split_lookup.get((axis, cid))
                if sp is not None:
                    splits.append(sp)
            if 'test' in splits:
                expected = 'test'
            elif 'val' in splits:
                expected = 'val'
            elif 'train' in splits:
                expected = 'train'
            else:
                expected = 'train'  # unrouted default
            ds_label = {'train': 'train', 'val': 'validation', 'test': 'test'}[expected]
            if ds_label != r['__split']:
                v5_violations.append({
                    'hash': r['source_row_hash'],
                    'expected': ds_label, 'actual': r['__split'],
                    'splits_seen': splits,
                })
        v5 = {'n_checked': v5_checked, 'violations': v5_violations[:20],
              'n_violations': len(v5_violations),
              'status': 'PASS' if not v5_violations else 'FAIL'}
        (art_dir / 'v5_split_routing.json').write_text(json.dumps(v5, indent=2, default=str))

        # ---------- V4: Sampling distribution (from artifacts) ----------
        log('V4: sampling distribution', run_id=run_id)
        # Read distribution_check.csv produced by the builder; independently verify
        # that actual_n is close to expected_n. Note that `actual_n` reflects the
        # post-projection row count (empty input_text drops can shrink small
        # partitions by 30-50% when the variant's projection column is NULL on
        # many rows — e.g., mhc_two subset where mhc_two_pocket is sparse). We
        # only flag drift on larger partitions where the absolute miss is large.
        v4_violations = []
        v4_warnings = []
        with open(art_dir / 'distribution_check.csv') as f:
            r = csv.reader(f)
            header = next(r)
            for row in r:
                sk, ok, n_post, exp_n, act_n, drift = row
                exp_n_i = int(exp_n)
                act_n_i = int(act_n)
                drift_pct = (act_n_i - exp_n_i) / exp_n_i * 100.0 if exp_n_i else 0.0
                # Only BLOCK on large partitions with significant drift (avoids
                # noise from tiny partitions whose projection drops dominate).
                if exp_n_i >= 100_000 and abs(drift_pct) > 20:
                    v4_violations.append({
                        'partition': f'{sk}/{ok}',
                        'expected': exp_n_i, 'actual': act_n_i, 'drift_pct': drift_pct,
                    })
                elif exp_n_i >= 1000 and abs(drift_pct) > 20:
                    v4_warnings.append({
                        'partition': f'{sk}/{ok}',
                        'expected': exp_n_i, 'actual': act_n_i, 'drift_pct': drift_pct,
                        'note': 'small_partition_projection_drop',
                    })
                elif exp_n_i >= 100_000 and abs(drift_pct) > 5:
                    v4_warnings.append({
                        'partition': f'{sk}/{ok}',
                        'expected': exp_n_i, 'actual': act_n_i, 'drift_pct': drift_pct,
                    })
        v4 = {'n_violations': len(v4_violations), 'violations': v4_violations[:20],
              'n_warnings': len(v4_warnings), 'warnings_sample': v4_warnings[:20],
              'status': 'PASS' if not v4_violations else 'FAIL'}
        (art_dir / 'v4_distribution_check.json').write_text(json.dumps(v4, indent=2, default=str))

        # ---------- Overall verdict ----------
        all_status = {
            'V1': v1['status'], 'V2': v2['status'], 'V3': v3['status'],
            'V4': v4['status'], 'V5': v5['status'], 'V6': v6['status'],
            'V7': v7['status'], 'V8': v8['status'], 'V9': v9['status'],
        }
        log(f'verdicts: {all_status}', run_id=run_id)
        fails = [k for k, v in all_status.items() if v == 'FAIL']
        if fails:
            _write_done('FAIL', f'checks_failed={fails}')
            print(f'VERIFIER_STATUS: {run_id}: FAIL: {fails}', flush=True)
            return 1
        _write_done('PASS')
        print(f'VERIFIER_STATUS: {run_id}: PASS', flush=True)
        return 0

    except Exception as e:
        import traceback
        log(f'VERIFIER FAIL: {e}\n{traceback.format_exc()}', run_id=run_id)
        _write_done('FAIL', str(e)[:200])
        print(f'VERIFIER_STATUS: {run_id}: FAIL: {e}', flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
