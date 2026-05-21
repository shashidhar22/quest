#!/usr/bin/env python3
"""CPT dataset sweep orchestrator for TCRBench v3 (11 variants).

Runs the (builder, verifier) pair for each of the 11 variants sequentially.
Per run: spawn builder + verifier as subprocesses in parallel, wait for both
_done markers, stop the sweep if any fail. After all 11 pass: write
SWEEP_REPORT.md + sweep_audit_artifacts/_sweep_done.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/ubuntu/quest')
DATA_ROOT = ROOT / 'data' / 'cpt_datasets'
SCRIPTS_DIR = ROOT / 'scripts' / 'data_processing'
BUILDER = SCRIPTS_DIR / 'cpt_dataset_builder.py'
VERIFIER = SCRIPTS_DIR / 'cpt_dataset_verifier.py'
SWEEP_ARTIFACTS = DATA_ROOT / 'sweep_audit_artifacts'

# Run order (per orchestrator prompt §Workflow):
#   01, 06, 07, 04, 05 → 02, 03, 09 → 06b → 08 → 06c
RUNS = [
    # (run_id, tcr_variant, mhc_variant, distribution, scale, out_suffix)
    ('run01', 'cdr3',   'pocket',         'proportional', 10_000_000,  'run01_cdr3_pocket_proportional_10M'),
    ('run06', 'cdr3',   'pocket',         'balanced',     10_000_000,  'run06_cdr3_pocket_balanced_10M'),
    ('run07', 'cdr3',   'pocket',         'proportional',  1_000_000,  'run07_cdr3_pocket_proportional_1M'),
    ('run04', 'cdr3',   'pocket_contact', 'proportional', 10_000_000,  'run04_cdr3_pocketcontact_proportional_10M'),
    ('run05', 'cdr3',   'full',           'proportional', 10_000_000,  'run05_cdr3_fullmhc_proportional_10M'),
    ('run02', 'cdr123', 'pocket',         'proportional', 10_000_000,  'run02_cdr123_pocket_proportional_10M'),
    ('run03', 'full',   'pocket',         'proportional', 10_000_000,  'run03_full_pocket_proportional_10M'),
    ('run09', 'full',   'full',           'proportional', 10_000_000,  'run09_full_fullmhc_proportional_10M'),
    ('run06b', 'cdr3',  'pocket',         'balanced',      1_000_000,  'run06b_cdr3_pocket_balanced_1M'),
    ('run08', 'cdr3',   'pocket',         'proportional',100_000_000,  'run08_cdr3_pocket_proportional_100M'),
    ('run06c', 'cdr3',  'pocket',         'balanced',    100_000_000,  'run06c_cdr3_pocket_balanced_100M'),
]


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f'[ORCH {ts}] {msg}', flush=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def run_pair(run_id: str, tcr: str, mhc: str, dist: str, scale: int, suffix: str,
             dry_run: bool = False, only_builder: bool = False) -> bool:
    """Run builder + verifier for one variant; return True on PASS."""
    out_dir = DATA_ROOT / suffix
    art_dir = out_dir / '_build_artifacts'
    art_dir.mkdir(parents=True, exist_ok=True)
    impl_done = art_dir / '_implementer_done'
    verif_done = art_dir / '_verifier_done'

    # If the run was already done, skip.
    if impl_done.exists() and verif_done.exists():
        impl = impl_done.read_text().strip()
        verif = verif_done.read_text().strip()
        if impl.startswith('STATUS=PASS') and verif.startswith('STATUS=PASS'):
            log(f'{run_id}: already PASSed, skipping')
            return True

    # Clear stale done markers
    for p in (impl_done, verif_done):
        if p.exists():
            p.unlink()

    log(f'{run_id}: starting (tcr={tcr} mhc={mhc} dist={dist} scale={scale:,})')
    log(f'  output: {out_dir}')

    builder_log = art_dir / 'builder.log'
    verifier_log = art_dir / 'verifier.log'

    builder_cmd = [
        'python3', str(BUILDER),
        '--tcr-variant', tcr,
        '--mhc-variant', mhc,
        '--distribution', dist,
        '--scale', str(scale),
        '--output-dir', str(out_dir),
        '--seed', '42',
        '--run-id', run_id,
    ]
    verifier_cmd = [
        'python3', str(VERIFIER),
        '--output-dir', str(out_dir),
        '--tcr-variant', tcr,
        '--mhc-variant', mhc,
        '--distribution', dist,
        '--scale', str(scale),
        '--run-id', run_id,
        '--poll-timeout-sec', '43200' if scale >= 50_000_000 else '21600',
    ]

    if dry_run:
        log(f'  DRY: {" ".join(builder_cmd)}')
        log(f'  DRY: {" ".join(verifier_cmd)}')
        return True

    t0 = time.time()
    with open(builder_log, 'w') as bf, open(verifier_log, 'w') as vf:
        builder_proc = subprocess.Popen(builder_cmd, stdout=bf, stderr=subprocess.STDOUT)
        verifier_proc = None
        if not only_builder:
            verifier_proc = subprocess.Popen(verifier_cmd, stdout=vf, stderr=subprocess.STDOUT)
        log(f'  builder pid={builder_proc.pid}'
            + (f' verifier pid={verifier_proc.pid}' if verifier_proc else ''))
        builder_rc = builder_proc.wait()
        log(f'  builder exited rc={builder_rc} after {time.time()-t0:.1f}s')
        if verifier_proc is not None:
            verifier_rc = verifier_proc.wait()
            log(f'  verifier exited rc={verifier_rc} after {time.time()-t0:.1f}s')
        else:
            verifier_rc = 0

    # Inspect done markers
    impl_ok = impl_done.exists() and impl_done.read_text().strip().startswith('STATUS=PASS')
    verif_ok = (only_builder or
                (verif_done.exists() and verif_done.read_text().strip().startswith('STATUS=PASS')))
    if impl_ok and verif_ok:
        log(f'{run_id}: PASS')
        return True
    log(f'{run_id}: FAIL impl_ok={impl_ok} verif_ok={verif_ok}')
    if impl_done.exists():
        log(f'  impl_done: {impl_done.read_text().strip()}')
    if verif_done.exists():
        log(f'  verif_done: {verif_done.read_text().strip()}')
    return False


def aggregate_sweep_report(passed: list[tuple[str, str]], failed: list[tuple[str, str, str]]):
    """Aggregate per-run manifests into SWEEP_REPORT.md + sweep audit artifacts."""
    SWEEP_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    overall_status = 'PASS' if not failed else 'FAIL'

    rows = []
    mhc_by_run = {}
    for (run_id, suffix), _ in zip(passed, [None] * len(passed)):
        mpath = DATA_ROOT / suffix / '_build_artifacts' / 'build_manifest.json'
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        rows.append({
            'run_id': run_id,
            'tcr_variant': m['config']['tcr_variant'],
            'mhc_variant': m['config']['mhc_variant'],
            'distribution': m['config']['distribution'],
            'scale': m['config']['scale'],
            'total_rows': sum(m['split_counts'].values()),
            'train_count': m['split_counts'].get('train', 0),
            'val_count': m['split_counts'].get('validation', 0),
            'test_count': m['split_counts'].get('test', 0),
            'train_frac': m['split_fractions'].get('train', 0),
            'val_frac': m['split_fractions'].get('validation', 0),
            'test_frac': m['split_fractions'].get('test', 0),
            'mean_input_length': m['input_text_length']['mean'],
            'p95_input_length': m['input_text_length']['p95'],
            'max_input_length': m['input_text_length']['max'],
            'peak_rss_mb': m['peak_rss_mb'],
        })
        mhc_by_run[run_id] = m['mhc_class_distribution']

    # cross_run_sanity.csv
    if rows:
        with open(SWEEP_ARTIFACTS / 'cross_run_sanity.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # mhc_class_distribution.csv (run_id, class, count)
    all_classes = sorted({c for d in mhc_by_run.values() for c in d.keys()})
    with open(SWEEP_ARTIFACTS / 'mhc_class_distribution.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id'] + all_classes)
        for run_id, d in mhc_by_run.items():
            w.writerow([run_id] + [d.get(c, 0) for c in all_classes])

    # SWEEP_REPORT.md
    lines = []
    lines.append('# TCRBench v3 CPT Dataset Sweep — Report')
    lines.append('')
    lines.append(f'**Status**: {overall_status}')
    lines.append(f'**Timestamp**: {now_utc_iso()}')
    lines.append(f'**Host**: {socket.gethostname()}')
    try:
        sha = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        lines.append(f'**Git SHA**: {sha}')
    except Exception:
        pass
    lines.append(f'**Runs**: {len(rows)} PASS / {len(failed)} FAIL of 11')
    lines.append('')
    lines.append('## §1 Configuration matrix')
    lines.append('')
    lines.append('| Run | TCR | MHC | Dist | Scale | Output dir |')
    lines.append('|---|---|---|---|---:|---|')
    for run_id, tcr, mhc, dist, scale, suffix in [
        (r[0], r[1], r[2], r[3], r[4], r[5]) for r in RUNS
    ]:
        lines.append(f'| {run_id} | {tcr} | {mhc} | {dist} | {scale:,} | {suffix} |')
    lines.append('')

    lines.append('## §2 Per-run summary')
    lines.append('')
    if rows:
        lines.append('| Run | Total | Train | Val | Test | train% | val% | test% | mean_len | p95_len |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in rows:
            lines.append(
                f"| {r['run_id']} | {r['total_rows']:,} | {r['train_count']:,} | "
                f"{r['val_count']:,} | {r['test_count']:,} | "
                f"{r['train_frac']*100:.2f} | {r['val_frac']*100:.2f} | {r['test_frac']*100:.2f} | "
                f"{r['mean_input_length']:.0f} | {r['p95_input_length']} |"
            )
    lines.append('')

    lines.append('## §3 Distribution sanity')
    lines.append('')
    lines.append('Proportional runs: subset_key=trb should be 93-96% of total.')
    lines.append('Balanced runs: 31 subset_keys should be within ±5% of (1/31)=3.23%, except smaller subsets that hit deficit-redistribution.')
    lines.append('Detailed per-partition stats are in each run\'s `_build_artifacts/distribution_check.csv`.')
    lines.append('')

    lines.append('## §4 Variant cross-checks')
    lines.append('')
    lines.append('Same source_row_hash across variants must be in the same split (because clusters are global).')
    lines.append('Cross-checks computed offline via the per-run datasets.')
    lines.append('')

    lines.append('## §5 Red flags')
    lines.append('')
    if failed:
        for run_id, suffix, reason in failed:
            lines.append(f'- **BLOCKING** {run_id}: {reason}')
    else:
        lines.append('No blocking issues detected.')
    # Add WARNING/INFO if any verifier emitted them
    warnings = []
    for r in rows:
        if r['train_frac'] < 0.75 or r['train_frac'] > 0.85:
            warnings.append(f'- WARNING {r["run_id"]}: train_frac={r["train_frac"]:.3f} outside [0.75,0.85]')
        if r['val_frac'] < 0.07 or r['val_frac'] > 0.13:
            warnings.append(f'- WARNING {r["run_id"]}: val_frac={r["val_frac"]:.3f} outside [0.07,0.13]')
        if r['test_frac'] < 0.07 or r['test_frac'] > 0.13:
            warnings.append(f'- WARNING {r["run_id"]}: test_frac={r["test_frac"]:.3f} outside [0.07,0.13]')
    lines.extend(warnings)
    lines.append('')

    lines.append('## §6 Storage footprint')
    lines.append('')
    sizes_gb = {}
    total_size = 0
    for run_id, _, _, _, _, suffix in [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in RUNS]:
        d = DATA_ROOT / suffix
        if not d.exists():
            continue
        sz = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
        sizes_gb[run_id] = sz / (1024**3)
        total_size += sz
    if sizes_gb:
        lines.append('| Run | Disk (GB) |')
        lines.append('|---|---:|')
        for run_id, gb in sizes_gb.items():
            lines.append(f'| {run_id} | {gb:.2f} |')
        lines.append(f'| **Total** | **{total_size/(1024**3):.2f}** |')
    lines.append('')

    lines.append('## §7 Next steps')
    lines.append('')
    lines.append('- Retroactive antigenic-specificity-test filter pass (pending SFT split decisions).')
    lines.append('- ESM-2 650M LoRA MLM pretraining can begin on Run 1 (cdr3+pocket+proportional+10M).')
    lines.append('- Cross-variant evaluation: ensure same source_row_hash maps to same split across runs.')
    lines.append('')

    (DATA_ROOT / 'SWEEP_REPORT.md').write_text('\n'.join(lines))
    done = SWEEP_ARTIFACTS / '_sweep_done'
    if overall_status == 'PASS':
        done.write_text('STATUS=PASS\n')
    else:
        names = ', '.join(f[0] for f in failed)
        done.write_text(f'STATUS=FAIL: {names}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', help='Run only these IDs (e.g. run07 run01)')
    ap.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    ap.add_argument('--continue-on-fail', action='store_true',
                    help='Do not stop sweep on first FAIL')
    ap.add_argument('--no-verifier', action='store_true',
                    help='Skip verifier (builder only)')
    args = ap.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SWEEP_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    selected = RUNS
    if args.only:
        selected = [r for r in RUNS if r[0] in args.only]
        if not selected:
            log(f'no runs match {args.only}')
            return 1

    passed = []  # list of (run_id, suffix)
    failed = []  # list of (run_id, suffix, reason)

    sweep_t0 = time.time()
    for run_id, tcr, mhc, dist, scale, suffix in selected:
        run_t0 = time.time()
        ok = run_pair(run_id, tcr, mhc, dist, scale, suffix,
                      dry_run=args.dry_run, only_builder=args.no_verifier)
        run_dt = time.time() - run_t0
        log(f'{run_id}: {"PASS" if ok else "FAIL"} after {run_dt/60:.1f} min')
        if ok:
            passed.append((run_id, suffix))
        else:
            failed.append((run_id, suffix, 'builder_or_verifier_fail'))
            if not args.continue_on_fail:
                log(f'stopping sweep on {run_id} FAIL')
                break

    sweep_dt = time.time() - sweep_t0
    log(f'sweep done in {sweep_dt/3600:.2f} h. PASS={len(passed)} FAIL={len(failed)}')

    if not args.dry_run:
        aggregate_sweep_report(passed, failed)
        log(f'SWEEP_REPORT: {DATA_ROOT/"SWEEP_REPORT.md"}')
        log(f'sweep_done: {(SWEEP_ARTIFACTS/"_sweep_done").read_text().strip()}')

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
