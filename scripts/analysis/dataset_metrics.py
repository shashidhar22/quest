#!/usr/bin/env python3
"""
Comprehensive dataset metrics for foundation model training data.

Analyzes deduplicated (parquet) and tokenized (arrow) datasets, computing:
- Basic statistics (file counts, sizes, row counts)
- Sequence length distributions
- Token analysis and vocabulary coverage
- Permutation key breakdown
- Training readiness metrics (shard balance, memory estimation)

Outputs: Console summary, JSON report, and histogram plots.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be skipped.")

# Try to import HuggingFace datasets
try:
    from datasets import load_from_disk, Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets not installed. Tokenized analysis will use fallback method.")


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def format_size(mb: float) -> str:
    """Format size in appropriate units."""
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


def compute_percentiles(data: np.ndarray, percentiles: List[int] = [5, 25, 50, 75, 95, 99]) -> Dict[str, float]:
    """Compute percentiles for a numpy array."""
    return {f"p{p}": float(np.percentile(data, p)) for p in percentiles}


def analyze_deduplicated(data_path: Path, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze deduplicated parquet files.

    Returns dict with:
    - file_count, total_size_mb
    - total_sequences
    - sequence_lengths (array for histograms)
    - length_stats (min, max, mean, median, std, percentiles)
    - permutation_counts (Counter)
    - length_by_permutation (dict of arrays)
    """
    print("\n" + "=" * 80)
    print("ANALYZING DEDUPLICATED DATASET")
    print("=" * 80)

    parquet_files = sorted(data_path.glob("*.parquet"))
    file_count = len(parquet_files)

    if file_count == 0:
        print(f"No parquet files found in {data_path}")
        return {}

    print(f"Found {file_count} parquet files")

    # Collect file sizes
    file_sizes = [get_file_size_mb(f) for f in parquet_files]
    total_size_mb = sum(file_sizes)

    # Collect sequence data
    all_lengths = []
    permutation_counts = Counter()
    length_by_permutation = defaultdict(list)
    total_sequences = 0
    rows_per_file = []

    # If sampling, select random files
    files_to_process = parquet_files
    if sample_size and sample_size < file_count:
        print(f"Sampling {sample_size} files out of {file_count}")
        indices = np.random.choice(file_count, sample_size, replace=False)
        files_to_process = [parquet_files[i] for i in sorted(indices)]

    for pq_file in tqdm(files_to_process, desc="Reading parquet files"):
        table = pq.read_table(pq_file)
        num_rows = table.num_rows
        rows_per_file.append(num_rows)
        total_sequences += num_rows

        # Get columns
        if 'sequence' in table.column_names:
            sequences = table.column('sequence').to_pylist()
            lengths = [len(s) for s in sequences]
            all_lengths.extend(lengths)

        if 'permutation_key' in table.column_names:
            perm_keys = table.column('permutation_key').to_pylist()
            for i, key in enumerate(perm_keys):
                permutation_counts[key] += 1
                if 'sequence' in table.column_names:
                    length_by_permutation[key].append(lengths[i])

    # Scale up if we sampled
    scale_factor = file_count / len(files_to_process) if sample_size else 1.0
    if scale_factor > 1.0:
        total_sequences = int(total_sequences * scale_factor)
        print(f"Scaled sequence count: {format_number(total_sequences)} (extrapolated)")

    # Compute length statistics
    lengths_array = np.array(all_lengths)
    length_stats = {
        "min": int(np.min(lengths_array)),
        "max": int(np.max(lengths_array)),
        "mean": float(np.mean(lengths_array)),
        "median": float(np.median(lengths_array)),
        "std": float(np.std(lengths_array)),
        **compute_percentiles(lengths_array)
    }

    # Rows per file stats
    rows_array = np.array(rows_per_file)
    rows_stats = {
        "min": int(np.min(rows_array)),
        "max": int(np.max(rows_array)),
        "mean": float(np.mean(rows_array)),
    }

    results = {
        "file_count": file_count,
        "total_size_mb": total_size_mb,
        "total_sequences": total_sequences,
        "sequence_lengths": lengths_array,
        "length_stats": length_stats,
        "permutation_counts": dict(permutation_counts),
        "length_by_permutation": {k: np.array(v) for k, v in length_by_permutation.items()},
        "file_size_stats": {
            "min_mb": min(file_sizes),
            "max_mb": max(file_sizes),
            "mean_mb": np.mean(file_sizes),
            "total_mb": total_size_mb
        },
        "rows_per_file_stats": rows_stats,
        "sampled": sample_size is not None
    }

    return results


def analyze_tokenized(data_path: Path, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze tokenized arrow files (HuggingFace datasets format).

    Returns dict with:
    - shard_info (rows per shard)
    - token_lengths (array)
    - token_stats (min, max, mean, etc.)
    - token_id_counts (frequency of each token ID)
    - special_token_stats
    """
    print("\n" + "=" * 80)
    print("ANALYZING TOKENIZED DATASET")
    print("=" * 80)

    # Find all shards
    shard_dirs = sorted(data_path.glob("shard_*"))
    if not shard_dirs:
        # Maybe it's a single dataset without shards
        if (data_path / "dataset_info.json").exists():
            shard_dirs = [data_path]
        else:
            print(f"No shard directories found in {data_path}")
            return {}

    print(f"Found {len(shard_dirs)} shard(s)")

    # Collect shard information
    shard_info = {}
    all_token_lengths = []
    token_id_counts = Counter()
    total_rows = 0
    total_size_mb = 0

    # ESM-2 special tokens (typical values)
    SPECIAL_TOKENS = {
        0: "PAD",
        1: "UNK",
        2: "CLS/BOS",
        3: "SEP/EOS",
        32: "MASK"
    }
    special_token_counts = Counter()

    for shard_dir in tqdm(shard_dirs, desc="Processing shards"):
        shard_name = shard_dir.name if shard_dir != data_path else "root"

        # Get shard size from arrow files
        arrow_files = sorted(shard_dir.glob("*.arrow"))
        shard_size_mb = sum(get_file_size_mb(f) for f in arrow_files) if arrow_files else 0
        total_size_mb += shard_size_mb

        try:
            # Load dataset using HuggingFace datasets library
            if HAS_DATASETS:
                ds = load_from_disk(str(shard_dir))
                shard_rows = len(ds)
                total_rows += shard_rows

                # Sample for token analysis
                sample_count = min(50000, shard_rows)  # Sample up to 50k per shard
                if sample_size:
                    sample_count = min(sample_count, sample_size * 1000)

                # Get random indices for sampling
                if sample_count < shard_rows:
                    indices = np.random.choice(shard_rows, sample_count, replace=False)
                    sample_ds = ds.select(indices)
                else:
                    sample_ds = ds

                # Analyze input_ids
                if 'input_ids' in sample_ds.column_names:
                    for i, row in enumerate(sample_ds):
                        ids = row['input_ids']
                        all_token_lengths.append(len(ids))

                        # Count token IDs (sample for efficiency)
                        if i < 10000:
                            for tid in ids:
                                token_id_counts[tid] += 1
                                if tid in SPECIAL_TOKENS:
                                    special_token_counts[SPECIAL_TOKENS[tid]] += 1

                shard_info[shard_name] = {
                    "file_count": len(arrow_files),
                    "row_count": shard_rows,
                    "size_mb": shard_size_mb
                }
            else:
                # Fallback: just count files and estimate size
                shard_info[shard_name] = {
                    "file_count": len(arrow_files),
                    "row_count": 0,  # Unknown without datasets library
                    "size_mb": shard_size_mb
                }
                print(f"  {shard_name}: datasets library not available, skipping detailed analysis")

        except Exception as e:
            print(f"Error processing shard {shard_name}: {e}")
            shard_info[shard_name] = {
                "file_count": len(arrow_files),
                "row_count": 0,
                "size_mb": shard_size_mb,
                "error": str(e)
            }
            continue

    # Compute token length statistics
    if all_token_lengths:
        token_lengths_array = np.array(all_token_lengths)
        token_stats = {
            "min": int(np.min(token_lengths_array)),
            "max": int(np.max(token_lengths_array)),
            "mean": float(np.mean(token_lengths_array)),
            "median": float(np.median(token_lengths_array)),
            "std": float(np.std(token_lengths_array)),
            **compute_percentiles(token_lengths_array)
        }
    else:
        token_lengths_array = np.array([])
        token_stats = {}

    # Vocabulary coverage
    unique_tokens = len(token_id_counts)

    results = {
        "shard_count": len(shard_info),
        "shard_info": shard_info,
        "total_rows": total_rows,
        "total_size_mb": total_size_mb,
        "token_lengths": token_lengths_array,
        "token_stats": token_stats,
        "unique_tokens": unique_tokens,
        "special_token_counts": dict(special_token_counts),
        "top_tokens": token_id_counts.most_common(20),
        "sampled": sample_size is not None or total_rows > 0
    }

    return results


def compute_training_metrics(dedup_results: Dict, token_results: Dict) -> Dict[str, Any]:
    """
    Compute training readiness metrics.

    Returns:
    - Shard balance statistics
    - Memory estimates for various batch sizes
    - Recommended batch sizes for A100 40GB
    """
    print("\n" + "=" * 80)
    print("COMPUTING TRAINING READINESS METRICS")
    print("=" * 80)

    metrics = {}

    # Shard balance
    if token_results.get("shard_info"):
        shard_rows = [s["row_count"] for s in token_results["shard_info"].values()]
        if shard_rows:
            rows_array = np.array(shard_rows)
            metrics["shard_balance"] = {
                "min_rows": int(np.min(rows_array)),
                "max_rows": int(np.max(rows_array)),
                "mean_rows": float(np.mean(rows_array)),
                "std_rows": float(np.std(rows_array)),
                "imbalance_ratio": float(np.max(rows_array) / np.min(rows_array)) if np.min(rows_array) > 0 else 0
            }

    # Memory estimation for A100 40GB
    # Assumptions:
    # - ESM-2 650M model: ~2.6GB model weights in fp16/bf16
    # - Per-token memory: ~2 bytes (bf16) for embeddings
    # - Optimizer states: 2x model size for AdamW
    # - Activations: depends on sequence length and batch size

    if token_results.get("token_stats"):
        mean_tokens = token_results["token_stats"].get("mean", 512)
        max_tokens = token_results["token_stats"].get("max", 1024)

        # Rough memory estimation (very approximate)
        # Hidden dim for ESM-2 650M is 1280
        hidden_dim = 1280
        num_layers = 33
        bytes_per_param = 2  # bf16

        # Per-sequence activation memory (very rough)
        # activation_mem = batch_size * seq_len * hidden_dim * num_layers * bytes_per_param

        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        memory_estimates = {}

        for bs in batch_sizes:
            # Approximate memory per batch (in GB)
            # This is a rough estimate; actual usage depends on many factors
            activation_mem_gb = (bs * max_tokens * hidden_dim * num_layers * bytes_per_param) / (1024**3)
            total_approx_gb = 2.6 + (2.6 * 2) + activation_mem_gb  # model + optimizer + activations
            memory_estimates[f"batch_{bs}"] = {
                "activation_gb": round(activation_mem_gb, 2),
                "total_approx_gb": round(total_approx_gb, 2),
                "fits_a100_40gb": total_approx_gb < 38  # Leave some headroom
            }

        # Find recommended batch size
        recommended_bs = max([bs for bs in batch_sizes
                             if memory_estimates[f"batch_{bs}"]["fits_a100_40gb"]], default=1)

        metrics["memory_estimates"] = memory_estimates
        metrics["recommended_batch_size"] = recommended_bs
        metrics["max_sequence_length"] = max_tokens
        metrics["mean_sequence_length"] = mean_tokens

    # Token/sequence ratio
    if dedup_results.get("length_stats") and token_results.get("token_stats"):
        seq_mean = dedup_results["length_stats"]["mean"]
        tok_mean = token_results["token_stats"]["mean"]
        metrics["token_to_aa_ratio"] = round(tok_mean / seq_mean, 3) if seq_mean > 0 else 0

    return metrics


def generate_plots(dedup_results: Dict, token_results: Dict, training_metrics: Dict, output_dir: Path):
    """Generate histogram plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return

    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sequence length distribution
    if dedup_results.get("sequence_lengths") is not None and len(dedup_results["sequence_lengths"]) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        lengths = dedup_results["sequence_lengths"]
        ax.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Sequence Length (amino acids)")
        ax.set_ylabel("Count")
        ax.set_title("Sequence Length Distribution (Deduplicated)")
        ax.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
        ax.axvline(np.median(lengths), color='green', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "sequence_length_distribution.png", dpi=150)
        plt.close()
        print(f"  Saved: sequence_length_distribution.png")

    # 2. Token length distribution
    if token_results.get("token_lengths") is not None and len(token_results["token_lengths"]) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        lengths = token_results["token_lengths"]
        ax.hist(lengths, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel("Token Count")
        ax.set_ylabel("Count")
        ax.set_title("Token Length Distribution (Tokenized)")
        ax.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
        ax.axvline(np.median(lengths), color='green', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "token_length_distribution.png", dpi=150)
        plt.close()
        print(f"  Saved: token_length_distribution.png")

    # 3. Permutation key breakdown
    if dedup_results.get("permutation_counts"):
        fig, ax = plt.subplots(figsize=(10, 6))
        perm_counts = dedup_results["permutation_counts"]
        keys = list(perm_counts.keys())
        values = list(perm_counts.values())

        # Sort by count descending
        sorted_pairs = sorted(zip(keys, values), key=lambda x: -x[1])
        keys, values = zip(*sorted_pairs)

        bars = ax.barh(range(len(keys)), values, color='steelblue')
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys)
        ax.set_xlabel("Count")
        ax.set_title("Permutation Key Distribution")
        ax.invert_yaxis()

        # Add percentage labels
        total = sum(values)
        for i, (bar, val) in enumerate(zip(bars, values)):
            pct = val / total * 100
            ax.text(bar.get_width() + total*0.01, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / "permutation_key_breakdown.png", dpi=150)
        plt.close()
        print(f"  Saved: permutation_key_breakdown.png")

    # 4. Shard balance
    if token_results.get("shard_info"):
        fig, ax = plt.subplots(figsize=(10, 6))
        shard_info = token_results["shard_info"]
        shards = list(shard_info.keys())
        rows = [s["row_count"] for s in shard_info.values()]

        ax.bar(range(len(shards)), rows, color='purple', alpha=0.7)
        ax.set_xticks(range(len(shards)))
        ax.set_xticklabels(shards, rotation=45, ha='right')
        ax.set_xlabel("Shard")
        ax.set_ylabel("Row Count")
        ax.set_title("Shard Balance (Tokenized Dataset)")
        ax.axhline(np.mean(rows), color='red', linestyle='--', label=f'Mean: {np.mean(rows):,.0f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "shard_balance.png", dpi=150)
        plt.close()
        print(f"  Saved: shard_balance.png")


def print_report(dedup_results: Dict, token_results: Dict, training_metrics: Dict):
    """Print formatted console report."""

    print("\n")
    print("=" * 80)
    print("DATASET METRICS REPORT")
    print("=" * 80)

    # Deduplicated summary
    if dedup_results:
        print("\n### DEDUPLICATED DATASET ###")
        print("-" * 40)
        print(f"{'Files:':<25} {format_number(dedup_results['file_count'])}")
        print(f"{'Total Size:':<25} {format_size(dedup_results['total_size_mb'])}")
        print(f"{'Total Sequences:':<25} {format_number(dedup_results['total_sequences'])}")

        if dedup_results.get("length_stats"):
            stats = dedup_results["length_stats"]
            print(f"\nSequence Length Statistics:")
            print(f"  {'Min:':<20} {stats['min']}")
            print(f"  {'Max:':<20} {stats['max']}")
            print(f"  {'Mean:':<20} {stats['mean']:.1f}")
            print(f"  {'Median:':<20} {stats['median']:.1f}")
            print(f"  {'Std:':<20} {stats['std']:.1f}")
            print(f"  {'P5:':<20} {stats['p5']:.1f}")
            print(f"  {'P95:':<20} {stats['p95']:.1f}")
            print(f"  {'P99:':<20} {stats['p99']:.1f}")

        if dedup_results.get("permutation_counts"):
            print(f"\nPermutation Key Breakdown:")
            total = sum(dedup_results["permutation_counts"].values())
            sorted_counts = sorted(dedup_results["permutation_counts"].items(), key=lambda x: -x[1])
            for key, count in sorted_counts:
                pct = count / total * 100
                print(f"  {key:<25} {format_number(count):>15} ({pct:5.1f}%)")

    # Tokenized summary
    if token_results:
        print("\n### TOKENIZED DATASET ###")
        print("-" * 40)
        print(f"{'Shards:':<25} {token_results['shard_count']}")
        print(f"{'Total Size:':<25} {format_size(token_results['total_size_mb'])}")
        print(f"{'Total Rows:':<25} {format_number(token_results['total_rows'])}")
        print(f"{'Unique Tokens:':<25} {format_number(token_results['unique_tokens'])}")

        if token_results.get("token_stats"):
            stats = token_results["token_stats"]
            print(f"\nToken Length Statistics:")
            print(f"  {'Min:':<20} {stats['min']}")
            print(f"  {'Max:':<20} {stats['max']}")
            print(f"  {'Mean:':<20} {stats['mean']:.1f}")
            print(f"  {'Median:':<20} {stats['median']:.1f}")

        if token_results.get("special_token_counts"):
            print(f"\nSpecial Token Counts (sampled):")
            for token, count in sorted(token_results["special_token_counts"].items()):
                print(f"  {token:<20} {format_number(count)}")

        if token_results.get("shard_info"):
            print(f"\nShard Details:")
            for shard, info in sorted(token_results["shard_info"].items()):
                print(f"  {shard:<15} {info['file_count']:>5} files, {format_number(info['row_count']):>15} rows, {format_size(info['size_mb']):>10}")

    # Training metrics
    if training_metrics:
        print("\n### TRAINING READINESS ###")
        print("-" * 40)

        if training_metrics.get("shard_balance"):
            balance = training_metrics["shard_balance"]
            print(f"Shard Balance:")
            print(f"  {'Imbalance Ratio:':<25} {balance['imbalance_ratio']:.2f}x")
            print(f"  {'Min Rows:':<25} {format_number(balance['min_rows'])}")
            print(f"  {'Max Rows:':<25} {format_number(balance['max_rows'])}")

        if training_metrics.get("memory_estimates"):
            print(f"\nMemory Estimates (A100 40GB):")
            print(f"  {'Batch Size':<15} {'Activation':<15} {'Total Approx':<15} {'Fits?'}")
            for key, est in training_metrics["memory_estimates"].items():
                bs = key.replace("batch_", "")
                fits = "Yes" if est["fits_a100_40gb"] else "No"
                print(f"  {bs:<15} {est['activation_gb']:.1f} GB{' '*7} {est['total_approx_gb']:.1f} GB{' '*7} {fits}")

            print(f"\n  Recommended batch size: {training_metrics.get('recommended_batch_size', 'N/A')}")

        if training_metrics.get("token_to_aa_ratio"):
            print(f"  Token/AA ratio: {training_metrics['token_to_aa_ratio']}")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


def save_json_report(dedup_results: Dict, token_results: Dict, training_metrics: Dict, output_path: Path):
    """Save machine-readable JSON report."""

    # Convert numpy arrays to lists for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    report = {
        "deduplicated": {},
        "tokenized": {},
        "training_metrics": training_metrics
    }

    # Deduplicated (exclude large arrays)
    if dedup_results:
        report["deduplicated"] = {
            "file_count": dedup_results.get("file_count"),
            "total_size_mb": dedup_results.get("total_size_mb"),
            "total_sequences": dedup_results.get("total_sequences"),
            "length_stats": dedup_results.get("length_stats"),
            "permutation_counts": dedup_results.get("permutation_counts"),
            "file_size_stats": dedup_results.get("file_size_stats"),
            "rows_per_file_stats": dedup_results.get("rows_per_file_stats"),
            "sampled": dedup_results.get("sampled")
        }

    # Tokenized (exclude large arrays)
    if token_results:
        report["tokenized"] = {
            "shard_count": token_results.get("shard_count"),
            "shard_info": token_results.get("shard_info"),
            "total_rows": token_results.get("total_rows"),
            "total_size_mb": token_results.get("total_size_mb"),
            "token_stats": token_results.get("token_stats"),
            "unique_tokens": token_results.get("unique_tokens"),
            "special_token_counts": token_results.get("special_token_counts"),
            "top_tokens": token_results.get("top_tokens"),
            "sampled": token_results.get("sampled")
        }

    report = make_serializable(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved JSON report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute comprehensive metrics for foundation model training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis
  python dataset_metrics.py \\
      --deduplicated-path /path/to/deduplicated \\
      --tokenized-path /path/to/tokenized \\
      --output-dir /path/to/output

  # Quick sample analysis
  python dataset_metrics.py \\
      --deduplicated-path /path/to/deduplicated \\
      --sample-size 10
"""
    )

    parser.add_argument("--deduplicated-path", type=Path,
                       help="Path to deduplicated parquet directory")
    parser.add_argument("--tokenized-path", type=Path,
                       help="Path to tokenized arrow directory")
    parser.add_argument("--output-dir", type=Path, default=Path("./dataset_metrics"),
                       help="Output directory for reports and plots (default: ./dataset_metrics)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Number of files to sample for faster analysis")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")

    args = parser.parse_args()

    if not args.deduplicated_path and not args.tokenized_path:
        parser.error("At least one of --deduplicated-path or --tokenized-path is required")

    # Analyze datasets
    dedup_results = {}
    token_results = {}

    if args.deduplicated_path:
        if not args.deduplicated_path.exists():
            print(f"Error: Deduplicated path does not exist: {args.deduplicated_path}")
        else:
            dedup_results = analyze_deduplicated(args.deduplicated_path, args.sample_size)

    if args.tokenized_path:
        if not args.tokenized_path.exists():
            print(f"Error: Tokenized path does not exist: {args.tokenized_path}")
        else:
            token_results = analyze_tokenized(args.tokenized_path, args.sample_size)

    # Compute training metrics
    training_metrics = compute_training_metrics(dedup_results, token_results)

    # Print report
    print_report(dedup_results, token_results, training_metrics)

    # Save JSON report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json_report(dedup_results, token_results, training_metrics,
                    args.output_dir / "dataset_metrics.json")

    # Generate plots
    if not args.no_plots:
        generate_plots(dedup_results, token_results, training_metrics, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
