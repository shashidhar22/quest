#!/usr/bin/env python3
"""Generate a brainstorm-friendly report with examples from each CPT dataset.

Writes to data/cpt_datasets/CPT_DATASET_REPORT.md with:
  - Per-run config, counts, split fractions
  - mhc_class distribution + n_segments distribution
  - input_text length quantiles
  - 5-8 example rows pulled from diverse subset_keys (covering single-mol,
    paired-chain, and TCR-pMHC-MHC rows where present)
  - Cross-variant comparison table
  - Open questions / suggested next experiments
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from datasets import load_from_disk

DATA_ROOT = Path('/home/ubuntu/quest/data/cpt_datasets')
RUNS = [
    ('run01', 'cdr3', 'pocket', 'proportional', 10_000_000, 'run01_cdr3_pocket_proportional_10M'),
    ('run02', 'cdr123', 'pocket', 'proportional', 10_000_000, 'run02_cdr123_pocket_proportional_10M'),
    ('run03', 'full', 'pocket', 'proportional', 10_000_000, 'run03_full_pocket_proportional_10M'),
    ('run04', 'cdr3', 'pocket_contact', 'proportional', 10_000_000, 'run04_cdr3_pocketcontact_proportional_10M'),
    ('run05', 'cdr3', 'full', 'proportional', 10_000_000, 'run05_cdr3_fullmhc_proportional_10M'),
    ('run06', 'cdr3', 'pocket', 'balanced', 10_000_000, 'run06_cdr3_pocket_balanced_10M'),
    ('run06b', 'cdr3', 'pocket', 'balanced', 1_000_000, 'run06b_cdr3_pocket_balanced_1M'),
    ('run06c', 'cdr3', 'pocket', 'balanced', 100_000_000, 'run06c_cdr3_pocket_balanced_100M'),
    ('run07', 'cdr3', 'pocket', 'proportional', 1_000_000, 'run07_cdr3_pocket_proportional_1M'),
    ('run08', 'cdr3', 'pocket', 'proportional', 100_000_000, 'run08_cdr3_pocket_proportional_100M'),
    ('run09', 'full', 'full', 'proportional', 10_000_000, 'run09_full_fullmhc_proportional_10M'),
]

# Subset-key "personas" we want to pull examples from (in priority order)
DIVERSE_KEYS = [
    'trb',                                  # bulk-repertoire TRB only
    'tra',                                  # bulk-repertoire TRA only
    'peptide',                              # peptide-only
    'tra_trb',                              # paired chains, no antigen
    'trb_peptide',                          # TRB + peptide
    'peptide_mhc_one',                      # peptide-MHC-I (no TCR)
    'peptide_mhc_two',                      # peptide-MHC-II (no TCR)
    'tra_trb_peptide_mhc_one',              # full αβ-pMHC-I
    'tra_trb_peptide_mhc_one_mhc_two',      # everything together
    'mhc_one',                              # MHC alone
    'mhc_two',                              # MHC alone
]


def gather_examples(ds_dict, max_per_subset=2, total_cap=8):
    """Return list of (subset_key, order_key, input_text, mhc_class, n_segs, hash)."""
    train = ds_dict['train']
    # Sample 5K rows for fast subset_key indexing
    n_sample = min(5000, len(train))
    idx = list(range(0, len(train), max(1, len(train) // n_sample)))[:n_sample]
    rows = train.select(idx).to_list()
    by_subset = collections.defaultdict(list)
    for r in rows:
        by_subset[r['subset_key']].append(r)
    out = []
    for key in DIVERSE_KEYS:
        if key in by_subset:
            for r in by_subset[key][:max_per_subset]:
                out.append(r)
                if len(out) >= total_cap:
                    return out
    # if we didn't get enough, top up with whatever subset_keys exist
    for key, rs in by_subset.items():
        if key in DIVERSE_KEYS:
            continue
        for r in rs[:1]:
            out.append(r)
            if len(out) >= total_cap:
                return out
    return out


def truncate(s: str, n: int = 140) -> str:
    if not s:
        return ''
    if len(s) <= n:
        return s
    return s[:n] + '…'


def fmt_int(n):
    return f'{n:,}'


def main():
    lines = []
    lines.append('# TCRBench v3 CPT Dataset Sweep — Brainstorm Report')
    lines.append('')
    lines.append('All 11 variants of the {TCR rep × MHC rep × distribution × scale} sweep,')
    lines.append('with examples drawn from diverse `subset_key` partitions inside each dataset.')
    lines.append('Examples are taken from the **train** split (the largest pool in each run).')
    lines.append('')
    lines.append('| Run | TCR | MHC | Dist | Scale | Total rows | mean_len | Probes... |')
    lines.append('|---|---|---|---|---:|---:|---:|---|')

    per_run = {}
    for run_id, tcr, mhc, dist, scale, suffix in RUNS:
        path = DATA_ROOT / suffix
        manifest = json.loads((path / '_build_artifacts' / 'build_manifest.json').read_text())
        ds = load_from_disk(str(path))
        per_run[run_id] = (manifest, ds, suffix, tcr, mhc, dist, scale)
        total = sum(manifest['split_counts'].values())
        probes = []
        if tcr == 'cdr3' and mhc == 'pocket' and dist == 'proportional':
            probes.append('baseline')
        elif tcr == 'cdr123':
            probes.append('CDR1+2+3 vs CDR3')
        elif tcr == 'full':
            probes.append('full-TCR signal')
        if mhc == 'pocket_contact':
            probes.append('contact residues')
        elif mhc == 'full':
            probes.append('full-MHC signal')
        if dist == 'balanced':
            probes.append('balanced sampling')
        if scale != 10_000_000:
            probes.append(f'scale={scale//1_000_000}M')
        lines.append(
            f"| {run_id} | {tcr} | {mhc} | {dist} | {fmt_int(scale)} | "
            f"{fmt_int(total)} | {manifest['input_text_length']['mean']:.0f} | "
            f"{', '.join(probes) or '—'} |"
        )
    lines.append('')

    # ---- Per-run sections ----
    for run_id, tcr, mhc, dist, scale, suffix in RUNS:
        manifest, ds, _, _, _, _, _ = per_run[run_id]
        lines.append(f'## {run_id} — {tcr} / {mhc} / {dist} / {scale//1_000_000 if scale>=1_000_000 else scale}M')
        lines.append('')
        lines.append(f'**Output**: `data/cpt_datasets/{suffix}/`')
        lines.append('')
        total = sum(manifest['split_counts'].values())
        lines.append('### Split sizes')
        lines.append('')
        lines.append('| Split | Rows | Fraction |')
        lines.append('|---|---:|---:|')
        for sn, sk in [('train', 'train'), ('validation', 'validation'), ('test', 'test')]:
            n = manifest['split_counts'].get(sk, 0)
            f = manifest['split_fractions'].get(sk, 0)
            lines.append(f'| {sn} | {fmt_int(n)} | {f*100:.2f}% |')
        lines.append(f'| **total** | **{fmt_int(total)}** | **100.00%** |')
        lines.append('')

        # mhc_class distribution
        lines.append('### `mhc_class` distribution')
        lines.append('')
        lines.append('| Class | Count | Fraction |')
        lines.append('|---|---:|---:|')
        mhc_dist = manifest.get('mhc_class_distribution', {})
        for cls in sorted(mhc_dist.keys(), key=lambda c: -mhc_dist[c]):
            n = mhc_dist[cls]
            lines.append(f'| `{cls}` | {fmt_int(n)} | {n/total*100:.2f}% |')
        lines.append('')

        # n_segments
        nseg = manifest.get('n_segments_distribution', {})
        if nseg:
            lines.append('### `n_segments` distribution')
            lines.append('')
            lines.append('| Segments | Count |')
            lines.append('|---:|---:|')
            for k in sorted(nseg.keys(), key=lambda k: int(k)):
                lines.append(f'| {k} | {fmt_int(nseg[k])} |')
            lines.append('')

        # length stats
        lq = manifest['input_text_length']
        lines.append('### `input_text` length')
        lines.append('')
        lines.append(f"- mean: **{lq['mean']:.1f}** chars")
        lines.append(f"- p50: {lq['p50']}, p95: {lq['p95']}, p99: {lq['p99']}, max: {lq['max']}")
        lines.append('')

        # examples
        lines.append('### Examples')
        lines.append('')
        examples = gather_examples(ds)
        for i, r in enumerate(examples, 1):
            lines.append(f"**Example {i}** — `subset_key={r['subset_key']}`, "
                         f"`order_key={r['order_key']}`")
            lines.append('')
            lines.append('```')
            lines.append(truncate(r['input_text'], 180))
            lines.append('```')
            lines.append(f"`mhc_class={r['mhc_class']}` `n_segments={r['n_segments']}` "
                         f"`tra_cluster={r['tra_cluster']}` `trb_cluster={r['trb_cluster']}` "
                         f"`peptide_cluster={r['peptide_cluster']}` "
                         f"`mhc_one_cluster={r['mhc_one_cluster']}` "
                         f"`mhc_two_cluster={r['mhc_two_cluster']}`")
            lines.append('')
        lines.append('---')
        lines.append('')

    # ---- Cross-variant comparison ----
    lines.append('## Cross-variant comparison')
    lines.append('')
    lines.append('### Length × variant matrix')
    lines.append('')
    lines.append('| Variant | mean | p50 | p95 | max |')
    lines.append('|---|---:|---:|---:|---:|')
    for run_id, *_ in RUNS:
        m = per_run[run_id][0]
        lq = m['input_text_length']
        lines.append(
            f"| {run_id} ({m['config']['tcr_variant']}/{m['config']['mhc_variant']}/"
            f"{m['config']['distribution']}/{m['config']['scale']//1_000_000}M) | "
            f"{lq['mean']:.1f} | {lq['p50']} | {lq['p95']} | {lq['max']} |"
        )
    lines.append('')

    # subset_key share comparison (proportional vs balanced at 10M)
    lines.append('### subset_key share (Proportional 10M vs Balanced 10M)')
    lines.append('')
    lines.append('Diversity of the rows by source partition — Balanced runs intentionally over-sample minority subsets.')
    lines.append('')
    runs_for_compare = ['run01', 'run06']
    by_run_subset = {}
    for r in runs_for_compare:
        _, ds, _, _, _, _, _ = per_run[r]
        counts = collections.Counter()
        for batch in ds['train'].select(range(min(50000, len(ds['train'])))):
            counts[batch['subset_key']] += 1
        by_run_subset[r] = counts
    all_subsets = sorted(set(by_run_subset['run01'].keys()) | set(by_run_subset['run06'].keys()),
                          key=lambda k: -by_run_subset['run01'].get(k, 0))
    lines.append('| subset_key | run01 prop % (50K sample) | run06 bal % (50K sample) |')
    lines.append('|---|---:|---:|')
    for sk in all_subsets[:15]:
        n1 = by_run_subset['run01'].get(sk, 0)
        n2 = by_run_subset['run06'].get(sk, 0)
        lines.append(f'| {sk} | {n1/50000*100:.2f} | {n2/50000*100:.2f} |')
    lines.append('')

    # ---- Brainstorm prompts ----
    lines.append('## Open questions to brainstorm with Claude')
    lines.append('')
    lines.append('1. **Where does Run 1 baseline end?** Proportional 10M of cdr3+pocket is dominated by TRB-only bulk repertoire (~94%). MLM val perplexity on this corpus is mostly a measure of TRB CDR3 modeling. Should we report Run 1 val perplexity bucketed by `mhc_class` to separate signal from MHC-restricted vs. bulk samples?')
    lines.append('')
    lines.append('2. **CDR1+2+3 vs full TCR** (Run 2 vs Run 3): Run 2 averages 33 chars, Run 3 averages 266 chars. Run 3 carries V-region information beyond the loops — but at 10× compute cost. Run 3/9 val molecules share V-region context with train (Option B caveat). Is the V-region gain worth the cost, and how do we deconfound that from data-leak risk?')
    lines.append('')
    lines.append('3. **Pocket vs pocket+contact vs full MHC** (Run 1 vs 4 vs 5): the lengths are nearly identical here (15/15/17). Either pocket and contact extractors are giving very similar strings, or the variants are pulling from the same partition mix. Worth a head-to-head MLM perplexity check.')
    lines.append('')
    lines.append('4. **Balanced shows different split fractions** (Run 6: 76.9/11.4/11.7 vs Run 1: 79.9/10.0/10.1). Small subsets carry more val/test cluster mass per-row, so Balanced inflates val/test. For paper-level metrics, do we want to report balanced-train + balanced-val, or balanced-train + proportional-val?')
    lines.append('')
    lines.append('5. **Run 9 (full TCR + full MHC) is the interaction probe**: ~267 chars per example, contains tra_full + trb_full + peptide + mhc_one_full (+ mhc_two_full for class II). With ESM-2 650M\'s 1024-token limit, the full-everything rows fit. But how does a model trained on Run 9 compare to one trained on Run 3 (full TCR, pocket MHC) for downstream binding prediction?')
    lines.append('')
    lines.append('6. **Scale curve** (Runs 7, 1, 8 = 1M, 10M, 100M): with the same cdr3+pocket+proportional config, what does MLM val perplexity look like as a function of scale? Where does the curve flatten?')
    lines.append('')
    lines.append('7. **Run 6c (100M Balanced) — diversity at scale**: with 31 subset_keys forced to ~3.2 M each (modulo deficit redistribution), this dataset has the strongest representation of paired/labeled molecules per training step. Worth checking if it gives better downstream binding-prediction scores vs. Run 8 (proportional same scale).')
    lines.append('')
    lines.append('8. **Antigenic-specificity test filter** is currently `NOT_APPLIED` (per each `build_manifest.json`). Decide on SFT splits first; then plan a single retroactive pass that removes any rows whose `(tra_cdr3, trb_cdr3, peptide)` triple appears in the SFT test set.')
    lines.append('')

    out_path = DATA_ROOT / 'CPT_DATASET_REPORT.md'
    out_path.write_text('\n'.join(lines))
    print(f'wrote {out_path} ({len(lines)} lines, {out_path.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
