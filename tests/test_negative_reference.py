"""Subsampled fits must not label known full-split positives as negatives."""
import pandas as pd
from track_b.shared.negative_sampling import sample_negatives


def test_full_reference_excludes_positive_outside_sample():
    panel = pd.DataFrame([
        dict(row_id='a', trb_cdr3='T1', peptide='P1', mhc_one_allele='A'),
        dict(row_id='b', trb_cdr3='T2', peptide='P2', mhc_one_allele='A'),
        dict(row_id='c', trb_cdr3='T3', peptide='P3', mhc_one_allele='A'),
    ])
    reference = pd.concat([panel, pd.DataFrame([
        dict(row_id='outside', trb_cdr3='T1', peptide='P2', mhc_one_allele='A'),
        dict(row_id='pool-only', trb_cdr3='T4', peptide='P4', mhc_one_allele='A'),
    ])], ignore_index=True)
    old = sample_negatives(panel, seed=43)
    assert ((old.trb_cdr3 == 'T1') & (old.peptide == 'P2')).any()
    actual = sample_negatives(panel, seed=43, known_positives=reference)
    assert not ((actual.trb_cdr3 == 'T1') & (actual.peptide == 'P2')).any()
    assert set(actual.peptide) <= set(panel.peptide)
    assert set(actual.row_id) <= set(panel.row_id)
    assert len(actual) == 15


def test_explicit_same_reference_preserves_seeded_rows():
    panel = pd.DataFrame(dict(trb_cdr3=['A','B','C'], peptide=['X','Y','Z'], mhc_one_allele=['H']*3))
    pd.testing.assert_frame_equal(sample_negatives(panel), sample_negatives(panel, known_positives=panel))
