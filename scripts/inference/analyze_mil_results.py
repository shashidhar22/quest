#!/usr/bin/env python3
"""
analyze_mil_results.py
────────────────────────────────────────────────────────
Analyze and visualize MIL results for TCR repertoire classification.
"""

import argparse
import json
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir: str):
    """Load MIL results from output directory."""
    with open(f"{results_dir}/test_results.json", 'r') as f:
        metrics = json.load(f)
    
    with open(f"{results_dir}/repertoire_predictions.json", 'r') as f:
        predictions = json.load(f)
    
    return metrics, predictions


def print_summary(metrics: dict):
    """Print summary of model performance."""
    print("=" * 80)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    if metrics.get('auc'):
        print(f"ROC AUC:   {metrics['auc']:.4f}")
    print()


def analyze_top_tcrs(predictions: dict, label: int, top_n: int = 20):
    """Analyze most frequent TCRs in top instances for a specific label."""
    tcr_scores = []
    
    for rep_id, data in predictions.items():
        if data['true_label'] == label and data['predicted_label'] == label:
            # True positives for this label
            for inst in data['top_instances']:
                score = inst.get('attention_score', inst.get('importance_score', 0))
                tcr_scores.append((inst['sequence'], score))
    
    # Count occurrences
    tcr_counter = Counter([tcr for tcr, _ in tcr_scores])
    
    # Calculate average score for each TCR
    tcr_avg_scores = {}
    for tcr, score in tcr_scores:
        if tcr not in tcr_avg_scores:
            tcr_avg_scores[tcr] = []
        tcr_avg_scores[tcr].append(score)
    
    tcr_avg_scores = {tcr: np.mean(scores) for tcr, scores in tcr_avg_scores.items()}
    
    print(f"\nTop {top_n} Most Frequent TCRs in Label {label} (True Positives):")
    print("-" * 80)
    print(f"{'Rank':<6} {'TCR Sequence':<25} {'Count':<8} {'Avg Score':<12}")
    print("-" * 80)
    
    for i, (tcr, count) in enumerate(tcr_counter.most_common(top_n), 1):
        avg_score = tcr_avg_scores[tcr]
        print(f"{i:<6} {tcr:<25} {count:<8} {avg_score:<12.4f}")
    
    return tcr_counter


def analyze_misclassifications(predictions: dict):
    """Analyze misclassified repertoires."""
    false_positives = []
    false_negatives = []
    
    for rep_id, data in predictions.items():
        if data['predicted_label'] != data['true_label']:
            if data['predicted_label'] == 1:
                false_positives.append((rep_id, data))
            else:
                false_negatives.append((rep_id, data))
    
    print(f"\nMisclassifications:")
    print(f"  False Positives: {len(false_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")
    
    if false_positives:
        print(f"\nTop 5 False Positives (predicted positive, actually negative):")
        for i, (rep_id, data) in enumerate(false_positives[:5], 1):
            confidence = max(data['probabilities'])
            print(f"  {i}. {rep_id} (confidence: {confidence:.3f})")
    
    if false_negatives:
        print(f"\nTop 5 False Negatives (predicted negative, actually positive):")
        for i, (rep_id, data) in enumerate(false_negatives[:5], 1):
            confidence = max(data['probabilities'])
            print(f"  {i}. {rep_id} (confidence: {confidence:.3f})")


def export_top_tcrs_csv(predictions: dict, output_path: str, label: int = None):
    """Export top TCRs to CSV for further analysis."""
    rows = []
    
    for rep_id, data in predictions.items():
        if label is not None and data['true_label'] != label:
            continue
        
        for inst in data['top_instances']:
            rows.append({
                'repertoire_id': rep_id,
                'true_label': data['true_label'],
                'predicted_label': data['predicted_label'],
                'sequence': inst['sequence'],
                'rank': inst['rank'],
                'score': inst.get('attention_score', inst.get('importance_score', 0))
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Exported {len(df)} top TCRs to: {output_path}")


def plot_score_distributions(predictions: dict, output_path: str):
    """Plot distribution of attention/importance scores."""
    correct_scores = []
    incorrect_scores = []
    
    for rep_id, data in predictions.items():
        scores = [inst.get('attention_score', inst.get('importance_score', 0)) 
                  for inst in data['top_instances']]
        
        if data['predicted_label'] == data['true_label']:
            correct_scores.extend(scores)
        else:
            incorrect_scores.extend(scores)
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_scores, bins=50, alpha=0.6, label='Correct Predictions', density=True)
    plt.hist(incorrect_scores, bins=50, alpha=0.6, label='Incorrect Predictions', density=True)
    plt.xlabel('Attention/Importance Score')
    plt.ylabel('Density')
    plt.title('Distribution of TCR Importance Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved score distribution plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MIL results for TCR repertoire classification"
    )
    
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing MIL results")
    parser.add_argument("--export_csv", type=str, default=None,
                        help="Export top TCRs to CSV file")
    parser.add_argument("--label", type=int, default=None,
                        help="Filter by specific label (for CSV export)")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of top TCRs to show")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    metrics, predictions = load_results(args.results_dir)
    
    # Print summary
    print_summary(metrics)
    
    # Get unique labels
    labels = set(data['true_label'] for data in predictions.values())
    
    # Analyze top TCRs for each label
    for label in sorted(labels):
        analyze_top_tcrs(predictions, label, args.top_n)
    
    # Analyze misclassifications
    analyze_misclassifications(predictions)
    
    # Export to CSV if requested
    if args.export_csv:
        export_top_tcrs_csv(predictions, args.export_csv, args.label)
    
    # Plot score distributions
    plot_path = f"{args.results_dir}/score_distributions.png"
    plot_score_distributions(predictions, plot_path)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
