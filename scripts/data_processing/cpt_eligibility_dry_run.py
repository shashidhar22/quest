#!/usr/bin/env python3
"""Corpus repair step 5: dry-run the eligibility policy difference before any sampling.

Compares `legacy-320-v0` with `reference-aware-v1` on the source corpus and writes counts only.
It samples nothing, builds nothing and never writes to the corpus.

Method. Eligibility depends only on (mhc_one, mhc_one_allele, mhc_two, mhc_two_allele), so the
source is collapsed to its distinct four-column combinations with row counts per subset_key.
Each combination is then judged three ways:

  legacy      the legacy SQL predicate, with its F1-F6 flags
  new (SQL)   the production semi-join, `filter_where_sql('reference-aware-v1')`
  new (py)    `mhc_eligibility.row_decision`, which also gives the reason

The two "new" answers must agree on every combination; a disagreement is a defect in the filter,
is reported, and makes the exit code non-zero.

Counts are reported both as expanded rows (ordering variants included) and as distinct
combinations / distinct chain sequences. Neither is a count of independent molecules.

Not available at this layer: the source database of a row (no such column), so nothing here is
broken down by source.

Usage:
    python3 cpt_eligibility_dry_run.py --source-root <dir with subset_key=*/...> \
        --hla-reference hla_prot.fasta --out-dir <new dir>
    python3 cpt_eligibility_dry_run.py --footer-scan <dir or s3://...>   # row/null totals only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quest.data.mhc_eligibility import (  # noqa: E402
    MHC_ONE_ROLES,
    MHC_TWO_ROLES,
    accepted_pairs_table,
    build_index_from_fasta,
    normalize_sequence,
    pair_group_of,
    row_decision,
)

LEGACY_LENGTH_LIMIT = 320
COLS = ('mhc_one', 'mhc_one_allele', 'mhc_two', 'mhc_two_allele')


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _none(value):
    """DuckDB/pandas hand back NaN or NA for NULL; the decision code wants None."""
    return value if isinstance(value, str) else None


def scan_combinations(con, source_root: Path):
    """Distinct eligibility inputs with expanded-row counts per subset_key."""
    src = f"read_parquet('{source_root}/**/*.parquet', hive_partitioning=1)"
    return con.execute(f"""
        SELECT CAST(subset_key AS VARCHAR) AS subset_key,
               mhc_one, mhc_one_allele, mhc_two, mhc_two_allele, COUNT(*) AS n_rows
        FROM {src} GROUP BY ALL
    """).df()


def judge(con, combos, index):
    """Add legacy flags, the SQL decision and the Python decision to each combination."""
    from scripts.data_processing.cpt_dataset_builder import (
        LEGACY_POLICY, REFERENCE_AWARE_POLICY, filter_where_sql, hsql,
    )
    keys = combos[list(COLS)].drop_duplicates().reset_index(drop=True)
    keys['combo_id'] = range(len(keys))
    con.register('keys_df', keys)
    con.execute('CREATE OR REPLACE TABLE keys AS SELECT * FROM keys_df')

    pairs = [(_none(s), _none(a)) for s, a in zip(keys.mhc_one, keys.mhc_one_allele)]
    pairs += [(_none(s), _none(a)) for s, a in zip(keys.mhc_two, keys.mhc_two_allele)]
    con.execute('CREATE OR REPLACE TABLE dry_elig (seq_norm VARCHAR, allele_norm VARCHAR, '
                'chain_role VARCHAR, pair_group VARCHAR)')
    accepted = accepted_pairs_table(index, pairs)
    if accepted:
        con.executemany('INSERT INTO dry_elig VALUES (?, ?, ?, ?)',
                        [(r['seq_norm'], r['allele_norm'], r['chain_role'], r['pair_group'])
                         for r in accepted])

    def predicate(policy, **kw):
        text = filter_where_sql(policy, **kw).strip()
        assert text.startswith('WHERE'), text[:40]
        return text[len('WHERE'):]

    h1, h2 = hsql('mhc_one_allele'), hsql('mhc_two_allele')
    flags = con.execute(f"""
        SELECT combo_id,
          ({predicate(LEGACY_POLICY)}) AS legacy_pass,
          ({predicate(REFERENCE_AWARE_POLICY, eligibility_table='dry_elig')}) AS new_sql_pass,
          (mhc_one_allele IS NOT NULL AND ({h1}) = 'II_beta')  AS F1,
          (mhc_two_allele IS NOT NULL AND ({h2}) = 'I')        AS F2,
          (mhc_two_allele IS NOT NULL AND ({h2}) = 'II_alpha') AS F3,
          (mhc_one_allele IS NOT NULL AND ({h1}) = 'unknown')  AS F4,
          (mhc_one IS NOT NULL AND length(mhc_one) > {LEGACY_LENGTH_LIMIT}) AS F5,
          (mhc_two IS NOT NULL AND length(mhc_two) > {LEGACY_LENGTH_LIMIT}) AS F6
        FROM keys ORDER BY combo_id
    """).df()
    keys = keys.merge(flags, on='combo_id')

    new_pass, reason, name_only, described = [], [], [], []
    for r in keys.itertuples():
        m1, a1, m2, a2 = (_none(getattr(r, c)) for c in COLS)
        ok, why = row_decision(index, m1, a1, m2, a2)
        new_pass.append(ok)
        reason.append(why)
        # Would the row pass on sequence evidence alone? If so and it is rejected, the rejection
        # rests entirely on an allele name, which is the class of error worth reading by hand.
        name_only.append((not ok) and row_decision(index, m1, None, m2, None)[0])
        described.append(_describe(index, m1, a1, m2, a2))
    keys['new_pass'], keys['new_reason'], keys['rejected_on_name_only'] = new_pass, reason, name_only
    for col in described[0] if described else ():
        keys[col] = [d[col] for d in described]
    return combos.merge(keys, on=list(COLS), how='left'), keys


def _describe(index, m1, a1, m2, a2) -> dict:
    out = {}
    for slot, seq, allele in (('one', m1, a1), ('two', m2, a2)):
        norm = normalize_sequence(seq)
        d = index.decide(seq, allele) if norm else None
        out[f'{slot}_sha'] = _sha(norm) if norm else ''
        out[f'{slot}_len'] = len(norm)
        out[f'{slot}_allele'] = (allele or '')[:40] if (allele is None or len(allele) < 60) \
            else f'<sequence-like:{len(allele)}>'
        out[f'{slot}_decision'] = f'{d.decision}:{d.reason}' if d else ''
        out[f'{slot}_class'] = (d.mhc_class or '') if d else ''
        out[f'{slot}_role'] = (d.chain_role or '') if d else ''
        out[f'{slot}_loci'] = '/'.join(d.loci) if d else ''
        out[f'{slot}_classical'] = bool(d.classical) if d else False
        out[f'{slot}_evidence'] = d.allele_evidence if d else ''
        out[f'{slot}_group'] = (pair_group_of(d.mhc_class, d.loci) or '') if d else ''
    return out


def summarize(rows, keys) -> dict:
    """`rows` is one line per (subset_key, combination); `keys` one line per combination."""
    def tally(frame, by):
        g = frame.groupby(by, dropna=False)
        return {(' | '.join(map(str, k)) if isinstance(k, tuple) else str(k)):
                {'rows': int(v.n_rows.sum()), 'combinations': int(v.combo_id.nunique())}
                for k, v in g}

    total = int(rows.n_rows.sum())
    legacy_fail = rows[~rows.legacy_pass]
    length_only = legacy_fail[~(legacy_fail.F1 | legacy_fail.F2 | legacy_fail.F3 | legacy_fail.F4)]
    accepted = rows[rows.new_pass]
    lost = rows[rows.legacy_pass & ~rows.new_pass]
    gained = rows[~rows.legacy_pass & rows.new_pass]
    disagree = keys[keys.new_pass != keys.new_sql_pass]

    chains = defaultdict(set)
    for r in keys[keys.new_pass].itertuples():
        for slot in ('one', 'two'):
            if getattr(r, f'{slot}_sha'):
                chains[(getattr(r, f'{slot}_class'), getattr(r, f'{slot}_loci'),
                        getattr(r, f'{slot}_len') > LEGACY_LENGTH_LIMIT)].add(getattr(r, f'{slot}_sha'))

    with_mhc = rows[(rows.one_sha != '') | (rows.two_sha != '')]
    return {
        'rows_scanned': total,
        'rows_with_any_mhc_chain': int(with_mhc.n_rows.sum()),
        'distinct_combinations': int(len(keys)),
        'transition_legacy_to_new': tally(rows, ['legacy_pass', 'new_pass']),
        'by_subset_key': {
            str(k): {'rows': int(v.n_rows.sum()),
                     'legacy_pass_rows': int(v.n_rows[v.legacy_pass].sum()),
                     'new_pass_rows': int(v.n_rows[v.new_pass].sum())}
            for k, v in rows.groupby('subset_key')},
        'legacy_flag_rows': {f: int(rows.n_rows[rows[f]].sum()) for f in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')},
        'legacy_length_only_rejections': {'rows': int(length_only.n_rows.sum()),
                                          'combinations': int(length_only.combo_id.nunique())},
        'new_reason': tally(rows, 'new_reason'),
        'new_accepted_mhc_one': tally(accepted[accepted.one_sha != ''],
                                      ['one_class', 'one_classical', 'one_loci']),
        'new_accepted_mhc_one_over_320': tally(
            accepted[accepted.one_len > LEGACY_LENGTH_LIMIT], ['one_class', 'one_loci']),
        'new_accepted_mhc_two': tally(accepted[accepted.two_sha != ''], ['two_class', 'two_loci']),
        'new_accepted_allele_evidence_mhc_one': tally(accepted[accepted.one_sha != ''], 'one_evidence'),
        'new_accepted_distinct_chain_sequences': {
            f'{c} | {loci} | {"over" if over else "upto"}_{LEGACY_LENGTH_LIMIT}': len(s)
            for (c, loci, over), s in sorted(chains.items())},
        'lost_legacy_pass_new_fail': tally(lost, 'new_reason'),
        'gained_legacy_fail_new_pass': {'rows': int(gained.n_rows.sum()),
                                        'combinations': int(gained.combo_id.nunique())},
        'rejected_on_allele_name_only': tally(rows[rows.rejected_on_name_only], 'new_reason'),
        'sql_python_disagreements': int(len(disagree)),
        'acceptance': {
            'sql_and_python_agree_on_every_combination': bool(len(disagree) == 0),
            'reference_supported_role_consistent_rows_rejected': int(
                rows.n_rows[rows.rejected_on_name_only].sum()),
            'note': ('The second number is rows whose chains are all reference-exact, in an '
                     'allowed role and pair-compatible, yet rejected because an allele NAME '
                     'contradicts the sequence. The plan requires these to be explained, not '
                     'necessarily zero. No rule in the new policy reads chain length.'),
        },
    }


def footer_scan(root: str) -> dict:
    """Row totals and MHC null counts per subset_key from parquet footers only (no data read)."""
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    fs, path = pafs.FileSystem.from_uri(root) if '://' in root else (pafs.LocalFileSystem(), root)
    out = defaultdict(Counter)
    for info in fs.get_file_info(pafs.FileSelector(path, recursive=True)):
        if not info.path.endswith('.parquet'):
            continue
        subset = next((p.split('=', 1)[1] for p in info.path.split('/') if p.startswith('subset_key=')), '?')
        md = pq.ParquetFile(fs.open_input_file(info.path)).metadata
        out[subset]['files'] += 1
        out[subset]['rows'] += md.num_rows
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            for i in range(rg.num_columns):
                col = rg.column(i)
                if col.path_in_schema in ('mhc_one', 'mhc_two'):
                    st = col.statistics
                    if st is None or not st.has_null_count:
                        out[subset][f'{col.path_in_schema}_stats_missing_groups'] += 1
                    else:
                        out[subset][f'{col.path_in_schema}_non_null'] += rg.num_rows - st.null_count
    return {k: dict(v) for k, v in sorted(out.items())}


def run(source_root: Path, hla_reference: Path, out_dir: Path) -> dict:
    import duckdb
    out_dir.mkdir(parents=True, exist_ok=False)
    ref_sha = hashlib.sha256(hla_reference.read_bytes()).hexdigest()
    index = build_index_from_fasta(hla_reference, reference_version=f'{hla_reference.name}:{ref_sha[:12]}')
    con = duckdb.connect()
    try:
        combos = scan_combinations(con, source_root)
        rows, keys = judge(con, combos, index)
    finally:
        con.close()
    summary = {'policy_old': 'legacy-320-v0', 'policy_new': 'reference-aware-v1',
               'reference_sha256': ref_sha, 'reference_distinct_sequences': len(index),
               'reference_full_length_by_locus': dict(sorted(index.full_lengths.items())),
               'source_root': str(source_root), 'slot_roles': {'mhc_one': MHC_ONE_ROLES, 'mhc_two': MHC_TWO_ROLES},
               **summarize(rows, keys)}
    # Sequences are replaced by hashes and lengths: the tables are for reading, not for reuse.
    per_combo = rows.groupby('combo_id').n_rows.sum().rename('n_rows')
    table = keys.drop(columns=list(COLS)).merge(per_combo, on='combo_id').sort_values('n_rows', ascending=False)
    table.to_csv(out_dir / 'combinations.csv', index=False)
    table[table.new_pass != table.new_sql_pass].to_csv(out_dir / 'sql_python_disagreements.csv', index=False)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source-root', type=Path)
    ap.add_argument('--hla-reference', type=Path)
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--footer-scan', default=None, help='directory or s3:// URI; prints totals and exits')
    a = ap.parse_args()
    if a.footer_scan:
        print(json.dumps(footer_scan(a.footer_scan), indent=2))
        return 0
    if not (a.source_root and a.hla_reference and a.out_dir):
        ap.error('--source-root, --hla-reference and --out-dir are required')
    summary = run(a.source_root, a.hla_reference, a.out_dir)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary['sql_python_disagreements'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
