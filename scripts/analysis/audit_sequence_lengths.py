#!/usr/bin/env python3
"""
Audit sequence token lengths across cardinality groups to assess ESM2 vs ESM-C
context window requirements.

Determines whether the 1024-token context window of ESM2 is a bottleneck for
multi-molecule sequences (especially quintets: tra+trb+peptide+mhc_one+mhc_two).
After MHC resolution, MHC alleles become full protein sequences (hundreds of AAs),
which may push quintets beyond 1024 tokens.

Usage:
    python -m scripts.analysis.audit_sequence_lengths \
        --input-dir data/splits/mlm_full \
        --sample-size 10000 \
        --seed 42 \
        --num-workers 16
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# Reuse from sibling modules
from scripts.analysis.summarize_dedup_output import (
    CARDINALITY_LABELS,
    parse_permutation_key,
)
from scripts.data_processing.esm_tokenizer import (
    build_ascii_lookup_table,
    load_tokenizer,
)

# Try importing matplotlib for histograms
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Try importing ESM-C
try:
    from esm.tokenization import EsmSequenceTokenizer

    HAS_ESMC = True
except ImportError:
    HAS_ESMC = False


# ---------------------------------------------------------------------------
# Pass 1: Scan permutation keys and classify by cardinality
# ---------------------------------------------------------------------------


def _scan_batch_keys(args: Tuple[str, List[int]]) -> Dict[int, List[int]]:
    """Read permutation_key column for specific row indices, return cardinality->row_idx mapping."""
    batch_path, row_indices = args
    try:
        table = pq.read_table(batch_path, columns=["permutation_key"])
        pk_col = table.column("permutation_key")
        num_rows = len(pk_col)

        cardinality_map = defaultdict(list)
        for row_idx in row_indices:
            if row_idx >= num_rows:
                continue
            pk = pk_col[row_idx].as_py()
            fields = parse_permutation_key(pk)
            card = len(fields)
            cardinality_map[card].append(row_idx)

        return dict(cardinality_map)
    except Exception as e:
        print(f"WARNING: Failed to scan {batch_path}: {e}", file=sys.stderr)
        return {}


def pass1_scan_and_sample(
    index: np.ndarray,
    resolved_dir: Path,
    sample_size: int,
    seed: int,
    num_workers: int,
) -> Dict[int, Dict[int, List[int]]]:
    """
    Pass 1: Scan permutation keys, classify by cardinality, sample per group.

    Returns:
        Dict[cardinality -> Dict[batch_idx -> List[row_idx]]]
    """
    print("\n" + "=" * 70)
    print("PASS 1: SCANNING PERMUTATION KEYS")
    print("=" * 70)

    # Group index entries by batch_idx
    batch_groups = defaultdict(list)
    for batch_idx, row_idx in index:
        batch_groups[int(batch_idx)].append(int(row_idx))

    print(f"  Index entries: {len(index):,}")
    print(f"  Unique batches: {len(batch_groups):,}")

    # Prepare worker args
    worker_args = []
    for batch_idx, row_indices in sorted(batch_groups.items()):
        batch_path = resolved_dir / f"batch_{batch_idx:06d}.parquet"
        if not batch_path.exists():
            print(
                f"WARNING: Missing batch file {batch_path.name}", file=sys.stderr
            )
            continue
        worker_args.append((str(batch_path), row_indices))

    # Parallel scan
    # Accumulate: cardinality -> list of (batch_idx, row_idx)
    card_all = defaultdict(list)

    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_scan_batch_keys, worker_args),
            total=len(worker_args),
            desc="Scanning keys",
        ):
            # Recover batch_idx from the args (results are unordered, but we
            # can extract batch_idx from the file path embedded in result context)
            # Actually we need to pair results back. Let's use imap instead.
            pass

    # Re-do with imap to preserve ordering with args
    card_all = defaultdict(list)
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_scan_batch_keys, worker_args),
                total=len(worker_args),
                desc="Scanning keys",
            )
        )

    for (batch_path_str, _), result in zip(worker_args, results):
        # Extract batch_idx from path
        batch_idx = int(Path(batch_path_str).stem.split("_")[1])
        for card, row_indices in result.items():
            for row_idx in row_indices:
                card_all[card].append((batch_idx, row_idx))

    # Report counts
    print("\n  Cardinality distribution:")
    for card in sorted(card_all.keys()):
        label = CARDINALITY_LABELS.get(card, f"CARD-{card}")
        print(f"    {label:<12} {len(card_all[card]):>12,}")

    # Sample per cardinality group
    rng = np.random.default_rng(seed)
    sampled = {}
    for card in sorted(card_all.keys()):
        entries = card_all[card]
        n = min(sample_size, len(entries))
        if n == 0:
            continue
        indices = rng.choice(len(entries), size=n, replace=False)
        selected = [entries[i] for i in indices]

        # Group by batch_idx for efficient I/O in pass 2
        by_batch = defaultdict(list)
        for batch_idx, row_idx in selected:
            by_batch[batch_idx].append(row_idx)
        sampled[card] = dict(by_batch)

        label = CARDINALITY_LABELS.get(card, f"CARD-{card}")
        print(f"  Sampled {n:,} from {label}")

    return sampled


# ---------------------------------------------------------------------------
# Pass 2: Load sequences for sampled indices
# ---------------------------------------------------------------------------


def _load_sequences(
    args: Tuple[str, List[int]],
) -> List[Tuple[int, str]]:
    """Read sequence column for specific row indices."""
    batch_path, row_indices = args
    try:
        table = pq.read_table(batch_path, columns=["sequence"])
        seq_col = table.column("sequence")
        num_rows = len(seq_col)

        results = []
        for row_idx in row_indices:
            if row_idx >= num_rows:
                continue
            seq = seq_col[row_idx].as_py()
            results.append((row_idx, seq))
        return results
    except Exception as e:
        print(f"WARNING: Failed to load {batch_path}: {e}", file=sys.stderr)
        return []


def pass2_load_sequences(
    sampled: Dict[int, Dict[int, List[int]]],
    resolved_dir: Path,
    num_workers: int,
) -> Dict[int, List[str]]:
    """
    Pass 2: Load sequences for sampled indices.

    Returns:
        Dict[cardinality -> List[sequence_str]]
    """
    print("\n" + "=" * 70)
    print("PASS 2: LOADING SEQUENCES")
    print("=" * 70)

    # Collect all unique (batch_idx, row_indices) pairs across cardinalities,
    # then read each batch file once
    batch_requests = defaultdict(set)  # batch_idx -> set of row_indices
    # Track which cardinality each (batch_idx, row_idx) belongs to
    membership = defaultdict(list)  # (batch_idx, row_idx) -> [cardinality, ...]

    for card, by_batch in sampled.items():
        for batch_idx, row_indices in by_batch.items():
            for row_idx in row_indices:
                batch_requests[batch_idx].add(row_idx)
                membership[(batch_idx, row_idx)].append(card)

    # Prepare worker args
    worker_args = []
    for batch_idx in sorted(batch_requests.keys()):
        batch_path = resolved_dir / f"batch_{batch_idx:06d}.parquet"
        row_indices = sorted(batch_requests[batch_idx])
        worker_args.append((str(batch_path), row_indices))

    print(f"  Batch files to read: {len(worker_args)}")
    total_seqs = sum(len(v) for v in batch_requests.values())
    print(f"  Total sequences to load: {total_seqs:,}")

    # Parallel load
    all_sequences = {}  # (batch_idx, row_idx) -> sequence
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_load_sequences, worker_args),
                total=len(worker_args),
                desc="Loading sequences",
            )
        )

    for (batch_path_str, _), result in zip(worker_args, results):
        batch_idx = int(Path(batch_path_str).stem.split("_")[1])
        for row_idx, seq in result:
            all_sequences[(batch_idx, row_idx)] = seq

    # Distribute sequences to cardinality groups
    card_sequences = defaultdict(list)
    for (batch_idx, row_idx), cards in membership.items():
        seq = all_sequences.get((batch_idx, row_idx))
        if seq is not None:
            for card in cards:
                card_sequences[card].append(seq)

    for card in sorted(card_sequences.keys()):
        label = CARDINALITY_LABELS.get(card, f"CARD-{card}")
        print(f"  {label}: {len(card_sequences[card]):,} sequences loaded")

    return dict(card_sequences)


# ---------------------------------------------------------------------------
# Token length computation
# ---------------------------------------------------------------------------


def compute_token_lengths(
    sequences: List[str], sep_char: str
) -> np.ndarray:
    """Compute token lengths using fast path: len(seq.replace(' ', sep)) + 2."""
    lengths = np.empty(len(sequences), dtype=np.int32)
    for i, seq in enumerate(sequences):
        lengths[i] = len(seq.replace(" ", sep_char)) + 2  # CLS + EOS
    return lengths


def validate_fast_path(
    sequences: List[str],
    tokenizer,
    sep_char: str,
    n_validate: int = 100,
    seed: int = 42,
) -> int:
    """Validate fast path against actual tokenizer.encode() on a random subset."""
    rng = np.random.default_rng(seed)
    n = min(n_validate, len(sequences))
    if n == 0:
        return 0

    indices = rng.choice(len(sequences), size=n, replace=False)
    mismatches = 0

    for idx in indices:
        seq = sequences[idx]
        # Fast path
        fast_len = len(seq.replace(" ", sep_char)) + 2

        # Actual tokenizer encode
        tokenized_seq = seq.replace(" ", sep_char)
        encoded = tokenizer.encode(tokenized_seq)
        actual_len = len(encoded)

        if fast_len != actual_len:
            mismatches += 1

    return mismatches


def compute_stats(lengths: np.ndarray) -> Dict:
    """Compute summary statistics for an array of lengths."""
    if len(lengths) == 0:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "p5": 0,
            "p25": 0,
            "p75": 0,
            "p95": 0,
            "p99": 0,
        }
    return {
        "count": int(len(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "p5": float(np.percentile(lengths, 5)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
    }


# ---------------------------------------------------------------------------
# Histogram generation
# ---------------------------------------------------------------------------


def plot_histograms(
    card_lengths: Dict[int, np.ndarray],
    max_context: int,
    output_dir: Path,
    tokenizer_name: str,
):
    """Plot per-cardinality length distributions with context window markers."""
    n_cards = len(card_lengths)
    if n_cards == 0:
        return

    fig, axes = plt.subplots(
        n_cards, 1, figsize=(12, 3.5 * n_cards), squeeze=False
    )

    for i, card in enumerate(sorted(card_lengths.keys())):
        ax = axes[i, 0]
        lengths = card_lengths[card]
        label = CARDINALITY_LABELS.get(card, f"CARD-{card}")

        ax.hist(lengths, bins=100, alpha=0.7, color="steelblue", edgecolor="none")
        ax.axvline(
            x=max_context, color="red", linestyle="--", linewidth=1.5, label=f"{max_context} (ESM2)"
        )
        ax.axvline(
            x=2048, color="orange", linestyle="--", linewidth=1.5, label="2048"
        )
        pct_over = float(np.mean(lengths > max_context) * 100)
        ax.set_title(
            f"{label} (n={len(lengths):,}, >{max_context}: {pct_over:.1f}%)"
        )
        ax.set_xlabel("Token length")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / f"sequence_length_histograms_{tokenizer_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Histogram saved: {out_path}")


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------


def print_table(
    card_lengths: Dict[int, np.ndarray],
    max_context: int,
    tokenizer_name: str,
):
    """Print a formatted summary table."""
    header = f"{'Cardinality':<14} {'N':>8} {'Mean':>8} {'Median':>8} {'p95':>8} {'p99':>8} {'Max':>8} {'>' + str(max_context):>8}"
    print(f"\n  [{tokenizer_name}]")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for card in sorted(card_lengths.keys()):
        lengths = card_lengths[card]
        label = CARDINALITY_LABELS.get(card, f"CARD-{card}")
        n = len(lengths)
        if n == 0:
            print(f"  {label:<14} {0:>8}")
            continue

        mean = np.mean(lengths)
        median = np.median(lengths)
        p95 = np.percentile(lengths, 95)
        p99 = np.percentile(lengths, 99)
        mx = np.max(lengths)
        pct_over = np.mean(lengths > max_context) * 100

        print(
            f"  {label:<14} {n:>8,} {mean:>8.0f} {median:>8.0f} "
            f"{p95:>8.0f} {p99:>8.0f} {mx:>8} {pct_over:>7.1f}%"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Audit sequence token lengths across cardinality groups (ESM2 vs ESM-C)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root splits dir (contains phase0a_resolved/ and phase3_subsets/)",
    )
    parser.add_argument(
        "--index-file",
        type=str,
        default="proportional_5M.npy",
        help="Index file name within phase3_subsets/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir for JSON report + histograms (default: {input-dir}/analysis)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Rows per cardinality group to sample",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=1024,
        help="Token threshold for overflow analysis",
    )
    parser.add_argument(
        "--esm2-model",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="ESM2 model name for tokenizer",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count // 2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-histograms",
        action="store_true",
        help="Skip histogram generation",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    resolved_dir = input_dir / "phase0a_resolved"
    index_path = input_dir / "phase3_subsets" / args.index_file
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"
    num_workers = args.num_workers or (cpu_count() // 2)

    # Validate paths
    if not resolved_dir.is_dir():
        sys.exit(f"ERROR: Resolved parquet dir not found: {resolved_dir}")
    if not index_path.is_file():
        sys.exit(f"ERROR: Index file not found: {index_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SEQUENCE LENGTH AUDIT")
    print("=" * 70)
    print(f"  Input dir:    {input_dir}")
    print(f"  Index file:   {index_path.name}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Sample size:  {args.sample_size:,} per cardinality")
    print(f"  Max context:  {args.max_context}")
    print(f"  Workers:      {num_workers}")
    print(f"  Seed:         {args.seed}")

    # Load index
    index = np.load(str(index_path))
    print(f"  Index shape:  {index.shape} ({index.shape[0]:,} entries)")

    # -----------------------------------------------------------------------
    # Pass 1: Scan and sample
    # -----------------------------------------------------------------------
    sampled = pass1_scan_and_sample(
        index, resolved_dir, args.sample_size, args.seed, num_workers
    )

    # -----------------------------------------------------------------------
    # Pass 2: Load sequences
    # -----------------------------------------------------------------------
    card_sequences = pass2_load_sequences(sampled, resolved_dir, num_workers)

    # -----------------------------------------------------------------------
    # Tokenizer analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOKEN LENGTH ANALYSIS")
    print("=" * 70)

    report = {
        "index_file": args.index_file,
        "total_index_entries": int(index.shape[0]),
        "sample_size_per_cardinality": args.sample_size,
        "max_context": args.max_context,
        "seed": args.seed,
        "tokenizers": {},
    }

    # Determine which tokenizers to run
    tokenizer_configs = [("esm2", args.esm2_model, "-")]
    if HAS_ESMC:
        tokenizer_configs.append(("esmc", None, "|"))
    else:
        print("\n  NOTE: ESM-C package not installed. Running ESM2-only analysis.")
        print("  Install with: pip install esm")

    for model_type, model_name, sep_char in tokenizer_configs:
        print(f"\n  Loading tokenizer: {model_type}...")
        try:
            tokenizer, default_sep = load_tokenizer(model_type, model_name or "")
        except Exception as e:
            print(f"  WARNING: Failed to load {model_type} tokenizer: {e}")
            continue

        sep = sep_char or default_sep

        # Validate fast path on first non-empty group
        validation_mismatches = 0
        for card in sorted(card_sequences.keys()):
            seqs = card_sequences[card]
            if seqs:
                mismatches = validate_fast_path(
                    seqs, tokenizer, sep, n_validate=100, seed=args.seed
                )
                validation_mismatches += mismatches
                break

        print(
            f"  Fast-path validation: {validation_mismatches} mismatches (100 samples)"
        )

        # Compute lengths per cardinality
        card_lengths = {}
        tokenizer_report = {
            "separator": sep,
            "validation_mismatches": validation_mismatches,
            "cardinalities": {},
        }

        for card in sorted(card_sequences.keys()):
            seqs = card_sequences[card]
            if not seqs:
                continue
            lengths = compute_token_lengths(seqs, sep)
            card_lengths[card] = lengths

            stats = compute_stats(lengths)
            pct_over = float(np.mean(lengths > args.max_context) * 100)
            pct_over_2048 = float(np.mean(lengths > 2048) * 100)
            stats["pct_over_max_context"] = pct_over
            stats["pct_over_2048"] = pct_over_2048

            label = CARDINALITY_LABELS.get(card, f"CARD-{card}")
            tokenizer_report["cardinalities"][label] = stats

        report["tokenizers"][model_type] = tokenizer_report

        # Print table
        print_table(card_lengths, args.max_context, model_type.upper())

        # Histograms
        if not args.no_histograms and HAS_MATPLOTLIB and card_lengths:
            plot_histograms(card_lengths, args.max_context, output_dir, model_type)
        elif not args.no_histograms and not HAS_MATPLOTLIB:
            print("  NOTE: matplotlib not installed, skipping histograms.")

    # -----------------------------------------------------------------------
    # Cross-tokenizer comparison
    # -----------------------------------------------------------------------
    tokenizer_names = list(report["tokenizers"].keys())
    if len(tokenizer_names) == 2:
        print("\n" + "=" * 70)
        print("CROSS-TOKENIZER COMPARISON")
        print("=" * 70)

        t1, t2 = tokenizer_names
        for card_label in report["tokenizers"][t1]["cardinalities"]:
            s1 = report["tokenizers"][t1]["cardinalities"][card_label]
            s2 = report["tokenizers"][t2]["cardinalities"].get(card_label)
            if s2 is None:
                continue
            diff = s2["mean"] - s1["mean"]
            print(
                f"  {card_label:<12} mean diff ({t2}-{t1}): {diff:+.1f} tokens"
            )

        report["cross_tokenizer_comparison"] = {
            "note": f"Positive diff means {t2} produces longer sequences than {t1}"
        }

    # -----------------------------------------------------------------------
    # Conclusion
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Use ESM2 stats for the conclusion (primary tokenizer)
    esm2_report = report["tokenizers"].get("esm2", {})
    quintet_stats = esm2_report.get("cardinalities", {}).get("QUINTETS", {})
    quartet_stats = esm2_report.get("cardinalities", {}).get("QUARTETS", {})

    conclusion = {}
    if quintet_stats:
        pct = quintet_stats.get("pct_over_max_context", 0)
        conclusion["quintet_pct_over_1024"] = pct
        conclusion["quintet_count"] = quintet_stats.get("count", 0)
        conclusion["quintet_max"] = quintet_stats.get("max", 0)
        conclusion["quintet_p99"] = quintet_stats.get("p99", 0)
        conclusion["esmc_advantage"] = pct > 10.0
        conclusion["threshold_pct"] = 10.0

        if pct > 10.0:
            print(
                f"  QUINTETS: {pct:.1f}% exceed {args.max_context} tokens "
                f"(>{conclusion['threshold_pct']:.0f}% threshold)"
            )
            print("  => ESM-C has a STRUCTURAL ADVANTAGE for quintet sequences")
        else:
            print(
                f"  QUINTETS: {pct:.1f}% exceed {args.max_context} tokens "
                f"(<={conclusion['threshold_pct']:.0f}% threshold)"
            )
            print("  => ESM2 context window is SUFFICIENT for most quintets")
    else:
        conclusion["quintet_pct_over_1024"] = None
        conclusion["esmc_advantage"] = None
        print("  No quintet sequences found in sample.")

    if quartet_stats:
        pct = quartet_stats.get("pct_over_max_context", 0)
        conclusion["quartet_pct_over_1024"] = pct
        print(f"  QUARTETS: {pct:.1f}% exceed {args.max_context} tokens")

    report["conclusion"] = conclusion

    # Save JSON report
    report_path = output_dir / "sequence_length_audit.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
