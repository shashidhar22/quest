"""Opt-in, evaluation-only judge diagnostics; never modifies the locked evaluator."""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import tempfile
import time
from pathlib import Path

import numpy as np


def validate_manifest(m):
    if not isinstance(m, dict) or m.get('protocol') != 'judge-diagnostic-v1' or not isinstance(m.get('input_hashes'), dict):
        raise ValueError('invalid diagnostic manifest header')
    if not m['input_hashes'] or any(not isinstance(k, str) or not k or not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for k, v in m['input_hashes'].items()):
        raise ValueError('invalid input hashes')
    rows = m.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('manifest rows must be nonempty')
    seen = set()
    for q in rows:
        if not isinstance(q, dict) or any(not isinstance(q.get(k), str) or not q[k] for k in ('row_id', 'condition', 'peptide', 'pep_cluster')):
            raise ValueError('invalid query identifiers')
        key = (q['condition'], q['row_id'])
        if key in seen: raise ValueError('duplicate query')
        seen.add(key)
        if not isinstance(q.get('original'), dict) or q['original'].get('peptide') != q['peptide'] or str(q['original'].get('row_id')) != q['row_id']:
            raise ValueError('original peptide mismatch')
        if not isinstance(q.get('metadata'), dict): raise ValueError('missing metadata')
        candidates = q.get('candidates')
        if not isinstance(candidates, dict): raise ValueError('missing candidates')
        for name in ('uniform', 'length_matched'):
            c = candidates.get(name)
            if not isinstance(c, list) or len(c) != 100 or any(not isinstance(x, str) or not x for x in c) or c[0] != q['peptide'] or len(set(c)) != len(c):
                raise ValueError('invalid candidates')
        original = q['original']
        if any(not isinstance(original.get(k), str) or not original[k] for k in ('tra_cdr3', 'trb_cdr3')) or not original.get('mhc_one'):
            raise ValueError('missing original receptor or MHC')
        donors = q.get('donors')
        if not isinstance(donors, list) or not donors: raise ValueError('missing donors')
        seeds = set()
        for donor in donors:
            if not isinstance(donor, dict) or not isinstance(donor.get('seed'), int) or donor['seed'] in seeds or 'row' not in donor:
                raise ValueError('invalid donor')
            seeds.add(donor['seed'])
            known = donor.get('known_positive_candidates')
            if not isinstance(known, list) or any(not isinstance(x, str) for x in known) or q['peptide'] in known:
                raise ValueError('invalid donor known positives')
            r = donor['row']
            if r is not None:
                if not isinstance(r, dict) or any(not isinstance(r.get(k), str) or not r[k] or len(r[k]) != len(original[k]) for k in ('tra_cdr3', 'trb_cdr3')):
                    raise ValueError('invalid donor receptor')
                if (r['tra_cdr3'], r['trb_cdr3']) == (original['tra_cdr3'], original['trb_cdr3']) or any(r.get(k) != original.get(k) for k in ('mhc_one', 'mhc_two')):
                    raise ValueError('donor receptor or MHC mismatch')
    return m


def _stats(scores):
    a = np.asarray(scores, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all(): raise ValueError('invalid or non-finite logits')
    greater = int(np.sum(a > a[0])); ties = int(np.sum(a == a[0]))
    return {'logits': a.tolist(), 'true_ties': ties, 'conservative_rank': greater + ties,
            'midrank': greater + (ties + 1) / 2}


def evaluate_manifest(manifest, scorer, batch_size=100, max_seconds=None, draw='length_matched', start=None):
    validate_manifest(manifest)
    if batch_size < 1 or draw not in ('uniform', 'length_matched') or (max_seconds is not None and max_seconds <= 0): raise ValueError('invalid scoring settings')
    start = time.monotonic() if start is None else start
    def score(rows):
        values = []
        for i in range(0, len(rows), batch_size):
            if max_seconds is not None and time.monotonic() - start >= max_seconds: raise TimeoutError('diagnostic time limit exceeded')
            chunk = np.asarray(scorer(rows[i:i+batch_size]), dtype=np.float64)
            if chunk.shape != (len(rows[i:i+batch_size]),) or not np.isfinite(chunk).all(): raise ValueError('invalid or non-finite logits')
            values.extend(chunk.tolist())
            if max_seconds is not None and time.monotonic() - start >= max_seconds: raise TimeoutError('diagnostic time limit exceeded')
        return values
    result = []
    for q in manifest['rows']:
        c = q['candidates'][draw]
        rows = [{**q['original'], 'peptide': p} for p in c]
        original = score(rows); repeat = score(rows); reverse = score(rows[::-1])[::-1]
        donors = []
        for d in q['donors']:
            if d['row'] is None:
                donors.append({'seed': d['seed'], 'result': None}); continue
            swap = [{**r, 'tra_cdr3': d['row']['tra_cdr3'], 'trb_cdr3': d['row']['trb_cdr3']} for r in rows]
            donors.append({'seed': d['seed'], 'result': _stats(score(swap))})
        result.append({'row_id': q['row_id'], 'condition': q['condition'], 'peptide': q['peptide'], 'pep_cluster': q['pep_cluster'], 'candidates': c, 'original': _stats(original), 'repeat': _stats(repeat), 'reverse': _stats(reverse), 'donors': donors})
    return result


def _digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'ckpt', 'out', 'adapter', 'task', 'tcr_variant', 'mhc_variant'): p.add_argument('--' + name, required=True)
    p.add_argument('--max_len', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--max-seconds', type=float, default=3600)
    p.add_argument('--draw', choices=['length_matched', 'uniform'], default='length_matched')
    p.add_argument('--precision', choices=['native', 'fp32'], default='native')
    p.add_argument('--allow-cpu', action='store_true')
    a = p.parse_args(argv)
    start = time.monotonic()
    if (a.task, a.tcr_variant, a.mhc_variant) != ('as_paired_i', 'cdr3', 'full'):
        raise ValueError('diagnostic requires as_paired_i, cdr3, full')
    if Path(a.out).exists(): raise FileExistsError(a.out)
    directory = Path(a.out).resolve().parent
    # Checked before scoring: discovering this at write time would discard the whole run.
    if not directory.is_dir() or not os.access(directory, os.W_OK): raise NotADirectoryError(f'output directory missing or not writable: {directory}')
    with open(a.manifest) as f: manifest = validate_manifest(json.load(f))
    if a.max_len < 1 or a.batch_size < 1 or a.max_seconds <= 0: raise ValueError('invalid limits')
    if a.precision == 'fp32': raise ValueError('fp32 encoder mode is not implemented; native mode uses the judge precision')
    manifest_digest = _digest(a.manifest); checkpoint_digest = _digest(a.ckpt)
    from track_a.esm_sft import judge
    import torch
    dev = torch.device(judge.select_device(torch.cuda.is_available(), allow_cpu=a.allow_cpu))
    model, tok = judge.build_model(a.adapter, dev)
    ck = judge.load_checkpoint(model, a.ckpt, dev, want=vars(a))
    model.eval()
    def scorer(rows):
        seqs = []
        for row in rows:
            # serialized_or_refuse already refuses a row that serializes to nothing or whose
            # peptide survived no token; max_len + 1 is requested so a row AT the limit is
            # visible here as over it rather than silently truncated.
            ids, pep = judge.serialized_or_refuse(judge._clean(row), a.tcr_variant, a.mhc_variant,
                                                  tok, a.max_len + 1, "diagnostic row")
            if len(ids) > a.max_len: raise ValueError('empty or truncated serialization')
            seqs.append((ids, pep))
        ids, mask, pep = judge.pad(seqs, int(tok.pad_token_id))
        with torch.no_grad():
            return model(ids.to(dev), mask.to(dev), pep.to(dev)).float().cpu().numpy()
    results = evaluate_manifest(manifest, scorer, a.batch_size, a.max_seconds, a.draw, start)
    from track_a.shared import artifacts as A
    output = {'protocol': 'judge-diagnostic-results-v1', 'manifest_sha256': manifest_digest, 'checkpoint_sha256': checkpoint_digest, 'code_revision': A.code_commit(), 'diagnostic_code_sha256': _digest(__file__), 'checkpoint_args': ck.get('args'), 'checkpoint_data_digests': ck.get('data_digests'), 'cpt_lineage': A.cpt_lineage(a.adapter), 'settings': vars(a), 'runtime': {'device': str(dev), 'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'seconds': time.monotonic()-start}, 'rows': results}
    if time.monotonic() - start >= a.max_seconds: raise TimeoutError('diagnostic time limit exceeded')
    fd, tmp = tempfile.mkstemp(prefix='.diagnostic-', dir=directory)
    try:
        with os.fdopen(fd, 'w') as f: json.dump(output, f, allow_nan=False); f.write('\n')
        os.link(tmp, a.out)
    finally:
        os.unlink(tmp)


if __name__ == '__main__': main()
