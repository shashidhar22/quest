#!/usr/bin/env python3
"""
Generate negative samples for peptide-MHC binding evaluation.

Since the dataset contains only positive (binding) pairs, we need to generate
negative samples for proper evaluation. This script implements two approaches:

1. MHC Shuffling (Soft Negatives):
   - Keep peptides, pair with random non-cognate MHC alleles
   - These are "soft" negatives since some may actually bind
   - Should be used with label smoothing or weighting in evaluation

2. Peptide Shuffling (Hard Negatives):
   - Keep MHC alleles, shuffle peptide amino acid sequences
   - These are "hard" negatives since shuffled sequences rarely bind
   - Can be used with standard evaluation metrics

Usage:
    # Generate both types of negatives with 1:5 ratio
    python scripts/evaluation/generate_negatives.py \\
        --input_dir data/eval/pmhc/class_one/peptide_mhc_one \\
        --output_dir data/eval/pmhc_with_negatives/class_one/peptide_mhc_one \\
        --negative_ratio 5 \\
        --seed 42

    # Generate only MHC-shuffled negatives
    python scripts/evaluation/generate_negatives.py \\
        --input_dir data/eval/pmhc/class_one/peptide_mhc_one \\
        --output_dir data/eval/pmhc_with_negatives/class_one/peptide_mhc_one \\
        --negative_types mhc_shuffle \\
        --negative_ratio 5

Author: Claude
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# Recognized test split patterns (ordered by expected difficulty)
TEST_SPLIT_PATTERNS = [
    'test_seen_motif',                  # Easy: Same binding motifs as training
    'test_unseen_peptide_seen_motif',   # Medium: New peptides, motifs seen
    'test_unseen_motif',                # Hard: Binding motifs not in training
    'test_unseen_allele',               # Hardest: MHC alleles not in training
    'test',                              # Fallback for simple split
]


def discover_test_splits(input_dir: Path) -> List[str]:
    """
    Auto-detect test split files in directory.

    Looks for parquet files matching recognized test split patterns.

    Args:
        input_dir: Directory to search for test splits

    Returns:
        List of discovered split names (without .parquet extension)
    """
    splits = []
    for pattern in TEST_SPLIT_PATTERNS:
        if (input_dir / f"{pattern}.parquet").exists():
            splits.append(pattern)

    return splits if splits else ['test']


def shuffle_peptide(peptide: str, seed: Optional[int] = None) -> str:
    """
    Shuffle the amino acids in a peptide sequence.

    Args:
        peptide: Original peptide sequence
        seed: Random seed for reproducibility

    Returns:
        Shuffled peptide sequence
    """
    if seed is not None:
        random.seed(seed)

    aa_list = list(peptide)
    random.shuffle(aa_list)
    return ''.join(aa_list)


def generate_mhc_shuffled_negatives(
    df: pd.DataFrame,
    allele_col: str = 'mhc_one_id',
    peptide_col: str = 'peptide',
    n_negatives_per_positive: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate soft negatives by pairing peptides with non-cognate MHC alleles.

    For each peptide, randomly selects n alleles that are NOT its cognate allele.

    Args:
        df: DataFrame with positive (peptide, allele) pairs
        allele_col: Column name for MHC allele
        peptide_col: Column name for peptide
        n_negatives_per_positive: Number of negatives per positive pair
        seed: Random seed

    Returns:
        DataFrame with negative pairs
    """
    np.random.seed(seed)

    # Get unique alleles
    all_alleles = df[allele_col].dropna().unique().tolist()
    n_alleles = len(all_alleles)

    if n_alleles < 2:
        print("Warning: Not enough unique alleles for MHC shuffling")
        return pd.DataFrame()

    # Get positive pairs for fast lookup
    positive_pairs: Set[Tuple[str, str]] = set(
        zip(df[peptide_col], df[allele_col])
    )

    # Generate negatives
    negatives = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating MHC-shuffled negatives"):
        peptide = row[peptide_col]
        cognate_allele = row[allele_col]

        # Get non-cognate alleles
        non_cognate = [a for a in all_alleles if a != cognate_allele]

        if len(non_cognate) == 0:
            continue

        # Sample alleles
        n_to_sample = min(n_negatives_per_positive, len(non_cognate))
        sampled_alleles = np.random.choice(non_cognate, size=n_to_sample, replace=False)

        for allele in sampled_alleles:
            # Skip if this pair exists as a positive
            if (peptide, allele) in positive_pairs:
                continue

            neg_row = row.copy()
            neg_row[allele_col] = allele
            neg_row['label'] = 0
            neg_row['negative_type'] = 'mhc_shuffle'
            negatives.append(neg_row)

    if not negatives:
        return pd.DataFrame()

    neg_df = pd.DataFrame(negatives)
    return neg_df


def generate_peptide_shuffled_negatives(
    df: pd.DataFrame,
    allele_col: str = 'mhc_one_id',
    peptide_col: str = 'peptide',
    n_negatives_per_positive: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate hard negatives by shuffling peptide sequences.

    For each (peptide, allele) pair, creates n shuffled versions of the peptide.

    Args:
        df: DataFrame with positive (peptide, allele) pairs
        allele_col: Column name for MHC allele
        peptide_col: Column name for peptide
        n_negatives_per_positive: Number of negatives per positive pair
        seed: Random seed

    Returns:
        DataFrame with negative pairs
    """
    np.random.seed(seed)

    # Get positive pairs and peptides for deduplication
    positive_pairs: Set[Tuple[str, str]] = set(
        zip(df[peptide_col], df[allele_col])
    )
    positive_peptides: Set[str] = set(df[peptide_col].unique())

    # Generate negatives
    negatives = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating peptide-shuffled negatives"):
        peptide = row[peptide_col]
        allele = row[allele_col]

        for i in range(n_negatives_per_positive):
            # Shuffle peptide with unique seed per iteration
            shuffled = shuffle_peptide(peptide, seed=seed + hash(peptide) + i)

            # Skip if shuffled is same as original or exists as positive
            if shuffled == peptide:
                continue
            if shuffled in positive_peptides:
                continue
            if (shuffled, allele) in positive_pairs:
                continue

            neg_row = row.copy()
            neg_row[peptide_col] = shuffled
            neg_row['label'] = 0
            neg_row['negative_type'] = 'peptide_shuffle'
            negatives.append(neg_row)

    if not negatives:
        return pd.DataFrame()

    neg_df = pd.DataFrame(negatives)
    return neg_df


def generate_random_peptides(
    length_distribution: List[int],
    n_peptides: int,
    seed: int = 42,
) -> List[str]:
    """
    Generate random peptide sequences.

    Uses natural amino acid frequency distribution.

    Args:
        length_distribution: List of peptide lengths to sample from
        n_peptides: Number of peptides to generate
        seed: Random seed

    Returns:
        List of random peptide sequences
    """
    np.random.seed(seed)

    # Natural amino acid frequencies (approximate)
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    aa_frequencies = [
        0.074, 0.025, 0.054, 0.054, 0.047,  # A, C, D, E, F
        0.074, 0.026, 0.068, 0.099, 0.058,  # G, H, I, K, L
        0.025, 0.045, 0.039, 0.034, 0.052,  # M, N, P, Q, R
        0.057, 0.051, 0.073, 0.013, 0.032,  # S, T, V, W, Y
    ]
    aa_frequencies = np.array(aa_frequencies) / sum(aa_frequencies)

    peptides = []
    lengths = np.random.choice(length_distribution, size=n_peptides, replace=True)

    for length in lengths:
        peptide = ''.join(np.random.choice(amino_acids, size=length, p=aa_frequencies))
        peptides.append(peptide)

    return peptides


def generate_random_negatives(
    df: pd.DataFrame,
    allele_col: str = 'mhc_one_id',
    peptide_col: str = 'peptide',
    n_negatives_per_positive: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate negatives using completely random peptides.

    Args:
        df: DataFrame with positive (peptide, allele) pairs
        allele_col: Column name for MHC allele
        peptide_col: Column name for peptide
        n_negatives_per_positive: Number of negatives per positive pair
        seed: Random seed

    Returns:
        DataFrame with negative pairs
    """
    np.random.seed(seed)

    # Get length distribution from positive peptides
    lengths = df[peptide_col].str.len().tolist()

    # Get unique alleles
    alleles = df[allele_col].dropna().unique().tolist()

    # Get positive pairs
    positive_pairs: Set[Tuple[str, str]] = set(
        zip(df[peptide_col], df[allele_col])
    )
    positive_peptides: Set[str] = set(df[peptide_col].unique())

    # Generate random peptides
    n_total = len(df) * n_negatives_per_positive
    random_peptides = generate_random_peptides(lengths, n_total * 2, seed)

    # Pair with random alleles
    negatives = []
    pep_idx = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating random negatives"):
        for _ in range(n_negatives_per_positive):
            if pep_idx >= len(random_peptides):
                break

            peptide = random_peptides[pep_idx]
            pep_idx += 1

            # Skip if exists as positive
            if peptide in positive_peptides:
                continue

            # Sample allele
            allele = np.random.choice(alleles)

            if (peptide, allele) in positive_pairs:
                continue

            neg_row = row.copy()
            neg_row[peptide_col] = peptide
            neg_row[allele_col] = allele
            neg_row['label'] = 0
            neg_row['negative_type'] = 'random'
            negatives.append(neg_row)

    if not negatives:
        return pd.DataFrame()

    return pd.DataFrame(negatives)


def combine_positives_and_negatives(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine positive and negative samples into evaluation dataset.

    Args:
        positives: DataFrame with positive samples
        negatives: DataFrame with negative samples

    Returns:
        Combined DataFrame
    """
    # Add label column to positives
    positives = positives.copy()
    positives['label'] = 1
    positives['negative_type'] = 'none'

    # Combine
    combined = pd.concat([positives, negatives], ignore_index=True)

    # Shuffle
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return combined


def load_parquet(path: Path) -> pd.DataFrame:
    """Load parquet file."""
    return pq.read_table(path).to_pandas()


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    print(f"Saved {len(df):,} rows to {path}")


def process_split(
    input_file: Path,
    output_dir: Path,
    split_name: str,
    negative_types: List[str],
    negative_ratio: int,
    allele_col: str,
    peptide_col: str,
    seed: int,
) -> dict:
    """
    Process a single data split.

    Args:
        input_file: Path to input parquet file
        output_dir: Output directory
        split_name: Name of the split (train, val, test)
        negative_types: Types of negatives to generate
        negative_ratio: Negatives per positive
        allele_col: Allele column name
        peptide_col: Peptide column name
        seed: Random seed

    Returns:
        Statistics dictionary
    """
    print(f"\n  Processing {split_name}...")

    # Load data
    positives = load_parquet(input_file)
    n_positives = len(positives)
    print(f"    Loaded {n_positives:,} positive samples")

    # Generate negatives
    all_negatives = []

    if 'mhc_shuffle' in negative_types:
        mhc_neg = generate_mhc_shuffled_negatives(
            positives,
            allele_col=allele_col,
            peptide_col=peptide_col,
            n_negatives_per_positive=negative_ratio,
            seed=seed,
        )
        all_negatives.append(mhc_neg)
        print(f"    Generated {len(mhc_neg):,} MHC-shuffled negatives")

    if 'peptide_shuffle' in negative_types:
        pep_neg = generate_peptide_shuffled_negatives(
            positives,
            allele_col=allele_col,
            peptide_col=peptide_col,
            n_negatives_per_positive=negative_ratio,
            seed=seed + 1,
        )
        all_negatives.append(pep_neg)
        print(f"    Generated {len(pep_neg):,} peptide-shuffled negatives")

    if 'random' in negative_types:
        rand_neg = generate_random_negatives(
            positives,
            allele_col=allele_col,
            peptide_col=peptide_col,
            n_negatives_per_positive=negative_ratio,
            seed=seed + 2,
        )
        all_negatives.append(rand_neg)
        print(f"    Generated {len(rand_neg):,} random negatives")

    # Combine negatives
    if all_negatives:
        negatives = pd.concat(all_negatives, ignore_index=True)
    else:
        negatives = pd.DataFrame()

    n_negatives = len(negatives)

    # Combine with positives
    combined = combine_positives_and_negatives(positives, negatives)

    # Save
    output_file = output_dir / f"{split_name}.parquet"
    save_parquet(combined, output_file)

    return {
        'split': split_name,
        'n_positives': n_positives,
        'n_negatives': n_negatives,
        'n_total': len(combined),
        'ratio': n_negatives / n_positives if n_positives > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate negative samples for peptide-MHC evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing train/val/test parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for data with negatives",
    )
    parser.add_argument(
        "--negative_types",
        type=str,
        nargs='+',
        choices=['mhc_shuffle', 'peptide_shuffle', 'random'],
        default=['mhc_shuffle', 'peptide_shuffle'],
        help="Types of negatives to generate (default: mhc_shuffle peptide_shuffle)",
    )
    parser.add_argument(
        "--negative_ratio",
        type=int,
        default=5,
        help="Number of negatives per positive (default: 5)",
    )
    parser.add_argument(
        "--allele_col",
        type=str,
        default="mhc_one_id",
        help="Column name for MHC allele (default: mhc_one_id)",
    )
    parser.add_argument(
        "--peptide_col",
        type=str,
        default="peptide",
        help="Column name for peptide (default: peptide)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=None,
        help="Splits to process (default: auto-detect all test splits)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be processed without actually generating negatives",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Auto-detect splits if not specified
    if args.splits is None:
        splits = discover_test_splits(input_dir)
        print(f"Auto-detected test splits: {splits}")
    else:
        splits = args.splits

    print("=" * 60)
    print("Negative Sample Generation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Negative types: {args.negative_types}")
    print(f"  Negative ratio: {args.negative_ratio}")
    print(f"  Allele column: {args.allele_col}")
    print(f"  Seed: {args.seed}")
    print(f"  Splits: {splits}")
    print(f"  Dry run: {args.dry_run}")

    # Dry run mode - just print what would be processed
    if args.dry_run:
        print("\n" + "=" * 60)
        print("Dry Run - Files that would be processed:")
        print("=" * 60)
        for split in splits:
            input_file = input_dir / f"{split}.parquet"
            output_file = output_dir / f"{split}.parquet"
            exists = "EXISTS" if input_file.exists() else "NOT FOUND"
            print(f"  {split}:")
            print(f"    Input:  {input_file} [{exists}]")
            print(f"    Output: {output_file}")
        print("\nNo files were modified (dry run mode).")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    stats = []

    for split in splits:
        input_file = input_dir / f"{split}.parquet"

        if not input_file.exists():
            print(f"\n  Skipping {split} (file not found: {input_file})")
            continue

        split_stats = process_split(
            input_file=input_file,
            output_dir=output_dir,
            split_name=split,
            negative_types=args.negative_types,
            negative_ratio=args.negative_ratio,
            allele_col=args.allele_col,
            peptide_col=args.peptide_col,
            seed=args.seed,
        )
        stats.append(split_stats)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if stats:
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
