#!/usr/bin/env python3
"""
Extract top TCRs based on attention weights from trained MIL model.

Runs inference on all repertoires, aggregates attention weights per unique TCR,
and outputs the top K most predictive sequences.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
from collections import defaultdict
import pyarrow.parquet as pq


class AttentionMIL(nn.Module):
    """Attention-based MIL model (must match training architecture)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_instance_weights: bool = True
    ):
        super().__init__()
        self.use_instance_weights = use_instance_weights

        # Instance encoder
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, instances, masks=None, instance_weights=None):
        """
        Args:
            instances: (B, N, D) bag of instances
            masks: (B, N) attention mask
            instance_weights: (B, N) instance weights (counts)

        Returns:
            logits: (B, num_classes)
            attention: (B, N) attention weights
        """
        B, N, D = instances.shape

        # Encode instances
        H = self.instance_encoder(instances)  # (B, N, hidden)

        # Gated attention
        A_V = torch.tanh(self.attention_V(H))
        A_U = torch.sigmoid(self.attention_U(H))
        A = self.attention_w(A_V * A_U).squeeze(-1)  # (B, N)

        # Apply instance weights (log scale)
        if self.use_instance_weights and instance_weights is not None:
            weights_norm = instance_weights / (instance_weights.sum(dim=1, keepdim=True) + 1e-8)
            A = A + torch.log(weights_norm + 1e-8)

        # Apply mask
        if masks is not None:
            A = A.masked_fill(~masks.bool(), -1e9)

        # Softmax attention
        attention = torch.softmax(A, dim=1)

        # Aggregate
        M = torch.bmm(attention.unsqueeze(1), H).squeeze(1)  # (B, hidden)

        # Classify
        logits = self.classifier(M)

        return logits, attention


def load_model(model_path: str, input_dim: int, hidden_dim: int = 256, device: str = 'cuda'):
    """Load trained MIL model."""
    model = AttentionMIL(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        use_instance_weights=True
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def extract_top_tcrs(
    model_path: str,
    clustered_pkl: str,
    repertoires_parquet: str,
    output_csv: str,
    top_k: int = 50000,
    hidden_dim: int = 256,
    device: str = 'cuda',
):
    """
    Extract top TCRs by attention weight aggregation.

    Strategy:
    1. For each repertoire, get attention weights for each cluster
    2. Map cluster attention back to original sequences (proportional to count)
    3. Aggregate attention for each unique TCR across all repertoires
    4. Sort and return top K

    Args:
        model_path: Path to trained model .pt file
        clustered_pkl: Pickle with clustered repertoires
        repertoires_parquet: Original Parquet with sequence_id, count per repertoire
        output_csv: Output CSV with top TCRs
        top_k: Number of top TCRs to extract
        hidden_dim: Model hidden dimension
        device: Device to run on
    """
    print("="*80)
    print("EXTRACTING TOP TCRs BY ATTENTION")
    print("="*80)

    # Load clustered data
    print(f"\nLoading clustered data from {clustered_pkl}")
    with open(clustered_pkl, 'rb') as f:
        clustered_data = pickle.load(f)
    print(f"Loaded {len(clustered_data)} repertoires")

    # Get input dimension
    input_dim = clustered_data[0]['instances'].shape[1]
    print(f"Input dimension: {input_dim}")

    # Load model
    print(f"\nLoading model from {model_path}")
    model = load_model(model_path, input_dim, hidden_dim, device)

    # Load original repertoire data for sequence mapping
    print(f"\nLoading repertoire sequences from {repertoires_parquet}")
    repertoires_df = pd.read_parquet(repertoires_parquet)
    print(f"Total sequence records: {len(repertoires_df):,}")

    # Aggregate attention per unique TCR
    print("\nRunning inference and aggregating attention...")
    tcr_attention = defaultdict(list)  # sequence_id -> list of (attention, count, label)

    for rep_data in tqdm(clustered_data, desc="Processing repertoires"):
        rep_id = rep_data['repertoire_id']
        label = rep_data['label']
        instances = torch.tensor(rep_data['instances'], dtype=torch.float32).unsqueeze(0).to(device)
        weights = torch.tensor(rep_data['instance_weights'], dtype=torch.float32).unsqueeze(0).to(device)

        # Get attention weights
        with torch.no_grad():
            logits, attention = model(instances, instance_weights=weights)
            attention = attention.squeeze(0).cpu().numpy()  # (N_clusters,)
            pred = torch.argmax(logits, dim=1).item()

        # Get sequences for this repertoire
        rep_sequences = repertoires_df[repertoires_df['repertoire_id'] == rep_id]

        # For now, distribute cluster attention to all sequences
        # This is an approximation - ideally we'd track which sequences went to which cluster
        # Simple approach: weight by sequence count
        total_count = rep_sequences['count'].sum()

        for _, row in rep_sequences.iterrows():
            seq_id = row['sequence_id']
            count = row['count']

            # Approximate attention as proportion of total count weighted by mean cluster attention
            # This is a simplification - better approach would track cluster assignments
            mean_attention = attention.mean()
            seq_attention = mean_attention * (count / total_count)

            tcr_attention[seq_id].append({
                'attention': seq_attention,
                'count': count,
                'label': label,
                'prediction': pred,
                'repertoire_id': rep_id,
            })

    print(f"Unique TCRs with attention scores: {len(tcr_attention):,}")

    # Aggregate attention per TCR
    print("\nAggregating attention scores...")
    tcr_scores = []

    for seq_id, entries in tqdm(tcr_attention.items(), desc="Aggregating"):
        # Aggregation strategies:
        # 1. Max attention across all appearances
        # 2. Mean attention weighted by count
        # 3. Sum of attention * count

        attentions = [e['attention'] for e in entries]
        counts = [e['count'] for e in entries]
        labels = [e['label'] for e in entries]

        max_attention = max(attentions)
        mean_attention = np.mean(attentions)
        total_count = sum(counts)
        weighted_attention = sum(a * c for a, c in zip(attentions, counts)) / total_count

        # Label association: more in positive or negative?
        pos_count = sum(c for c, l in zip(counts, labels) if l == 1)
        neg_count = sum(c for c, l in zip(counts, labels) if l == 0)
        label_ratio = pos_count / (pos_count + neg_count + 1e-8)

        tcr_scores.append({
            'sequence_id': seq_id,
            'max_attention': max_attention,
            'mean_attention': mean_attention,
            'weighted_attention': weighted_attention,
            'total_count': total_count,
            'num_repertoires': len(entries),
            'pos_count': pos_count,
            'neg_count': neg_count,
            'label_ratio': label_ratio,
        })

    # Sort by weighted attention (or max_attention)
    tcr_scores.sort(key=lambda x: -x['weighted_attention'])

    # Take top K
    top_tcrs = tcr_scores[:top_k]

    # Save to CSV
    print(f"\nSaving top {len(top_tcrs):,} TCRs to {output_csv}")
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

    df = pd.DataFrame(top_tcrs)
    df.to_csv(output_csv, index=False)

    # Statistics
    print(f"\nTop TCR statistics:")
    print(f"  Total unique TCRs: {len(tcr_attention):,}")
    print(f"  Top K extracted: {len(top_tcrs):,}")
    print(f"  Max attention range: [{df['max_attention'].min():.6f}, {df['max_attention'].max():.6f}]")
    print(f"  Mean label ratio: {df['label_ratio'].mean():.3f}")

    # Show top 10
    print(f"\nTop 10 TCRs:")
    for i, row in df.head(10).iterrows():
        print(f"  {i+1}. {row['sequence_id'][:20]}... attn={row['weighted_attention']:.6f} "
              f"count={row['total_count']:.0f} ratio={row['label_ratio']:.2f}")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract top TCRs based on attention weights"
    )
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model .pt file")
    parser.add_argument("--clustered_pkl", type=str, required=True,
                       help="Pickle with clustered repertoires")
    parser.add_argument("--repertoires_parquet", type=str, required=True,
                       help="Parquet with sequence_id, count per repertoire")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Output CSV file")
    parser.add_argument("--top_k", type=int, default=50000,
                       help="Number of top TCRs to extract (default: 50000)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Model hidden dimension (default: 256)")
    parser.add_argument("--device", type=str, default='cuda',
                       help="Device (default: cuda)")

    args = parser.parse_args()

    extract_top_tcrs(
        args.model_path,
        args.clustered_pkl,
        args.repertoires_parquet,
        args.output_csv,
        args.top_k,
        args.hidden_dim,
        args.device,
    )


if __name__ == "__main__":
    main()
