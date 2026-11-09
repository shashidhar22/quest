import copy
import numpy as np
import pytest
from track_a.esm_sft import diagnostic_eval as d


def manifest():
    row = {'row_id': '1', 'peptide': 'AAA', 'tra_cdr3': 'A', 'trb_cdr3': 'B', 'mhc_one': 'M', 'mhc_two': None}
    candidates = ['AAA', 'BBB', 'CCC'] + [f'P{i}' for i in range(97)]
    return {'protocol': 'judge-diagnostic-v1', 'input_hashes': {'pool': 'a'*64}, 'rows': [
        {'row_id': '1', 'condition': 'test', 'peptide': 'AAA', 'pep_cluster': 'c',
         'candidates': {'uniform': candidates, 'length_matched': candidates.copy()},
         'original': row, 'donors': [{'seed': 1, 'row': {**row, 'tra_cdr3': 'X', 'trb_cdr3': 'Y'}, 'known_positive_candidates': []}], 'metadata': {}}]}


def test_order_ties_and_receptor_swap():
    seen = []
    def scorer(rows):
        seen.extend(copy.deepcopy(rows))
        return np.array([{'AAA': 2., 'BBB': 2.}.get(r['peptide'], 1.) for r in rows])
    out = d.evaluate_manifest(manifest(), scorer, batch_size=2)
    q = out[0]
    assert q['original']['logits'] == q['reverse']['logits'] == [2., 2.] + [1.]*98
    assert q['original']['true_ties'] == 2
    assert q['original']['conservative_rank'] == 2
    assert q['original']['midrank'] == 1.5
    assert all(r['mhc_one'] == 'M' for r in seen)
    assert seen[-1]['tra_cdr3'] == 'X' and seen[-1]['trb_cdr3'] == 'Y'


def test_invalid_and_nonfinite():
    m = manifest(); m['rows'][0]['candidates']['length_matched'][0] = 'BAD'
    with pytest.raises(ValueError): d.validate_manifest(m)
    with pytest.raises(ValueError): d.evaluate_manifest(manifest(), lambda rows: [float('nan')]*len(rows))


def test_cli_refuses_invalid_manifest_without_output(tmp_path):
    import json
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps({'protocol': 'wrong'}))
    out = tmp_path / 'result.json'
    with pytest.raises(ValueError):
        d.main(['--manifest', str(path), '--ckpt', 'unused', '--out', str(out), '--adapter', 'none', '--task', 'as_paired_i', '--tcr_variant', 'cdr3', '--mhc_variant', 'full'])
    assert not out.exists()


def test_cli_help_without_judge_import():
    import os
    import subprocess
    import sys
    code = "import sys; from track_a.esm_sft.diagnostic_eval import main; main(['--help'])"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=os.environ)
    assert result.returncode == 0
    assert '--precision' in result.stdout


def test_main_publish_and_failures(tmp_path, monkeypatch):
    import contextlib
    import json
    import sys
    import types
    from track_a.esm_sft import diagnostic_eval
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(manifest()))
    ckpt = tmp_path / 'ckpt'; ckpt.write_text('checkpoint')
    out = tmp_path / 'result.json'
    args = ['--manifest', str(path), '--ckpt', str(ckpt), '--out', str(out), '--adapter', 'none', '--task', 'as_paired_i', '--tcr_variant', 'cdr3', '--mhc_variant', 'full']
    class Tensor:
        def __init__(self, value): self.value = value
        def to(self, dev): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return np.array(self.value)
    class Model:
        def eval(self): pass
        def __call__(self, ids, mask): return Tensor([1.] * len(ids.value))
    torch = types.SimpleNamespace(device=lambda x: x, cuda=types.SimpleNamespace(is_available=lambda: True), tensor=lambda x, dtype: Tensor(x), long='long', no_grad=contextlib.nullcontext, __version__='fake')
    judge = types.SimpleNamespace(select_device=lambda *a, **kw: 'cuda', build_model=lambda *a: (Model(), types.SimpleNamespace(pad_token_id=0)), load_checkpoint=lambda *a, **kw: {'args': {'seed': 7}, 'data_digests': {'train': 'digest'}}, serialize=lambda *a: [1, 2], _clean=lambda r: r, pad=lambda seqs, pid: (Tensor(seqs), Tensor(seqs)))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    import track_a.esm_sft
    monkeypatch.setattr(track_a.esm_sft, 'judge', judge, raising=False)
    diagnostic_eval.main(args)
    result = json.loads(out.read_text())
    assert result['checkpoint_args']['seed'] == 7 and len(result['rows'][0]['original']['logits']) == 100
    with pytest.raises(FileExistsError): diagnostic_eval.main(args)
    out.unlink()
    judge.serialize = lambda *a: [1] * 1025
    with pytest.raises(ValueError, match='truncated'): diagnostic_eval.main(args)
    assert not out.exists()
    judge.serialize = lambda *a: [1]
    monkeypatch.setattr(Model, '__call__', lambda self, ids, mask: Tensor([float('nan')] * len(ids.value)))
    with pytest.raises(ValueError, match='non-finite'): diagnostic_eval.main(args)
    assert not out.exists()


def test_timeout():
    with pytest.raises(TimeoutError):
        d.evaluate_manifest(manifest(), lambda rows: [1.] * len(rows), max_seconds=1, start=0)
