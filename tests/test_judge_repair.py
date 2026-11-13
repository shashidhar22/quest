"""DG1 integration: real tokenizer, serialization, scorer and checkpoint I/O; tiny encoder."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from esm.tokenization import EsmSequenceTokenizer
from track_a.esm_sft import judge as J
from track_a.esm_sft.encoders import LoadedEncoder
from track_a.esm_sft.model import ESMScorer


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 16)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def model():
    return ESMScorer(LoadedEncoder(Encoder(), None, 16, False),
                     head_dim=8, dropout=0, peptide_pool=True)


def row(order='trb_peptide_mhc_one'):
    return dict(row_id='r', trb_cdr3='CASS', peptide='NLV', mhc_one='MMMMM',
                mhc_one_allele='HLA-A*02:01', order_key=order)


@pytest.mark.parametrize('order', ['trb_peptide_mhc_one', 'peptide_mhc_one_trb', 'mhc_one_trb_peptide'])
def test_real_tokenizer_locates_only_peptide_across_orders(order):
    tok = EsmSequenceTokenizer()
    ids, pep = J.serialized_or_refuse(row(order), 'cdr3', 'full', tok, 1024, 'test')
    assert tok.decode([i for i, p in zip(ids, pep) if p]).replace(' ', '') == 'NLV'
    assert sum(pep) == 3


def test_truncation_cannot_silently_remove_peptide():
    with pytest.raises(J.ScoreError, match='no peptide token'):
        J.serialized_or_refuse(row('mhc_one_trb_peptide'), 'cdr3', 'full',
                               EsmSequenceTokenizer(), 5, 'test')


def test_checkpoint_preserves_peptide_pool_and_normalization_and_batch_independence(tmp_path):
    torch.manual_seed(42)
    m = model().train()
    tok = EsmSequenceTokenizer()
    rows = [J.serialize({**row(), 'peptide': p}, 'cdr3', 'full', tok, 1024)
            for p in ('NLV', 'AAA', 'GGG', 'LLL')]
    batch = J.pad(rows, tok.pad_token_id)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.01)
    for _ in range(3):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(m(*batch), torch.tensor([1., 0., 0., 0.])).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        expected = m(*batch)
    state = {k: v for k, v in m.state_dict().items() if any(t in k for t in J.TRAINABLE_KEYS)}
    path = tmp_path / 'judge.pt'
    torch.save({'trainable_state_dict': state}, path)
    restored = model()
    restored.encoder.load_state_dict(m.encoder.state_dict())
    J.load_checkpoint(restored, path, 'cpu')
    with torch.no_grad():
        torch.testing.assert_close(restored(*batch), expected, rtol=0, atol=0)
        singles = torch.cat([restored(*(x[i:i+1] for x in batch)) for i in range(4)])
        torch.testing.assert_close(singles, expected, atol=1e-6, rtol=1e-5)
    del state['head.0.running_mean']
    torch.save({'trainable_state_dict': state}, path)
    with pytest.raises(J.ScoreError, match='missing'):
        J.load_checkpoint(restored, path, 'cpu')


def test_fixture_generator_uses_current_serializer(monkeypatch):
    path = Path(__file__).parent / 'fixtures' / 'make_judge_pooled_fixture.py'
    spec = importlib.util.spec_from_file_location('fixture_generator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    df = pd.DataFrame([{**row(), 'label': 1}, {**row(), 'row_id': 's', 'peptide': 'AAA', 'label': 0}])
    monkeypatch.setattr(J, 'load_split', lambda *a: df)
    monkeypatch.setattr(J, 'with_negatives', lambda frame, seed: frame)
    m, tok = model().eval(), EsmSequenceTokenizer()
    monkeypatch.setattr(J, 'build_model', lambda *a: (m, tok))
    vectors, labels, molecules = module.pooled_peptide_vectors('train', 2, 43, 1, 'cpu', 2)
    assert vectors.shape == (2, 16)
    assert sorted(labels.tolist()) == [0, 1]
    assert len(np.unique(molecules)) == 2
    with torch.no_grad():
        expected = m.encoder.embedding(torch.tensor(tok('NLV')['input_ids'][1:-1])).mean(0).numpy()
    np.testing.assert_allclose(vectors[labels == 1][0], expected, atol=1e-6)


def train_args(tmp_path, batch_size):
    return SimpleNamespace(seed=42, allow_cpu=True, out_dir=str(tmp_path), smoke=False,
                           max_train_pos=4, max_val_pos=4, task='as_paired_i', adapter='none',
                           tcr_variant='cdr3', mhc_variant='full', max_len=1024,
                           batch_size=batch_size, num_workers=0, lr=1e-4, epochs=1, patience=1)


def training_inputs(monkeypatch):
    df = pd.DataFrame([{**row(), 'row_id': str(i), 'trb_cdr3': 'CASS' + aa, 'peptide': aa * 3}
                       for i, aa in enumerate('ALGV')])
    monkeypatch.setattr(J, 'load_split', lambda *a: df)
    monkeypatch.setattr(J, 'build_model', lambda *a: (model(), EsmSequenceTokenizer()))
    monkeypatch.setattr(J.A, 'split_digests', lambda *a: {'train': 'test', 'val': 'test'})
    monkeypatch.setattr(J, 'cotenants', lambda: {})


@pytest.mark.parametrize('batch_size', [1, 128])
def test_invalid_batch_cannot_publish_untrained_checkpoint(tmp_path, monkeypatch, batch_size):
    training_inputs(monkeypatch)
    with pytest.raises(J.ScoreError, match='batch'):
        J.cmd_train(train_args(tmp_path, batch_size))
    assert not (tmp_path / 'best.pt').exists()


def test_train_entrypoint_saves_reloadable_repaired_judge(tmp_path, monkeypatch):
    training_inputs(monkeypatch)
    trained, tok = model(), EsmSequenceTokenizer()
    monkeypatch.setattr(J, 'build_model', lambda *a: (trained, tok))
    # 24 validation rows yield batches of 10, 10, 4: loss must be row-weighted.
    J.cmd_train(train_args(tmp_path, 5))
    checkpoint = J.load_checkpoint(model(), tmp_path / 'best.pt', 'cpu')
    assert checkpoint['trainable_state_dict']['head.0.num_batches_tracked'].item() > 0
    assert 'pool_peptide.score.weight' in checkpoint['trainable_state_dict']
    assert np.isfinite(checkpoint['val_loss'])
    import json
    summary = json.loads((tmp_path / 'train_summary.json').read_text())
    epoch = summary['history'][0]
    assert epoch['val_loss'] == checkpoint['val_loss']
    rows = J.with_negatives(J.load_split('as_paired_i', 'val'), 44)
    dataset = J.RowsDS(rows, 'cdr3', 'full', tok, 1024)
    ids, am, pep, labels = J.collate(tok.pad_token_id)([dataset[i] for i in range(len(dataset))])
    trained.eval()
    with torch.no_grad():
        expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(trained(ids, am, pep), labels).item()
    assert epoch['val_loss'] == pytest.approx(expected_loss, abs=1e-6)
    assert epoch['val_prior_loss'] == pytest.approx(0.45056120886630463)
    assert epoch['val_loss_beats_prior'] == (epoch['val_loss'] < epoch['val_prior_loss'])


def test_training_passes_full_split_reference_before_subsampling(tmp_path, monkeypatch):
    training_inputs(monkeypatch)
    original = J.with_negatives
    seen = []
    def checked(panel, seed, known_positives=None):
        assert len(panel) == 2
        assert len(known_positives) == 4
        seen.append(seed)
        return original(panel, seed, known_positives=known_positives)
    monkeypatch.setattr(J, 'with_negatives', checked)
    args = train_args(tmp_path, 2)
    args.max_train_pos = args.max_val_pos = 2
    J.cmd_train(args)
    assert seen == [43, 44]
