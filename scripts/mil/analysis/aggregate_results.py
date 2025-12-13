#!/usr/bin/env python3
"""
Aggregate MIL Results Across Datasets and Methods

Collects results from all Phase 3 training runs and generates:
1. Summary tables (CSV, LaTeX)
2. Comparison plots
3. Statistical significance tests

Usage:
    python aggregate_results.py --results_dir data/mil/results --output_dir data/mil/analysis
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_results(results_dir: Path) -> List[Dict]:
    """Load all results.json files from the results directory."""
    results = []
    
    # Pattern: results_dir/train_dataset_N/clustering_method/mil_method/results.json
    for dataset_dir in results_dir.iterdir():
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith('train_dataset_'):
            continue
        
        dataset_name = dataset_dir.name
        dataset_num = int(dataset_name.split('_')[-1])
        
        for clustering_dir in dataset_dir.iterdir():
            if not clustering_dir.is_dir():
                continue
            
            clustering_method = clustering_dir.name
            
            for method_dir in clustering_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                
                mil_method = method_dir.name
                results_file = method_dir / 'results.json'
                
                if results_file.exists():
                    try:
                        with open(results_file) as f:
                            result = json.load(f)
                        
                        results.append({
                            'dataset': dataset_name,
                            'dataset_num': dataset_num,
                            'clustering': clustering_method,
                            'mil_method': mil_method,
                            'test_auc': result.get('test_auc', np.nan),
                            'test_accuracy': result.get('test_accuracy', np.nan),
                            'test_f1': result.get('test_f1', np.nan),
                            'test_precision': result.get('test_precision', np.nan),
                            'test_recall': result.get('test_recall', np.nan),
                            'val_auc': result.get('best_val_auc', np.nan),
                            'train_time': result.get('train_time', np.nan),
                            'best_epoch': result.get('best_epoch', np.nan),
                            'results_path': str(results_file),
                        })
                    except Exception as e:
                        print(f"Error loading {results_file}: {e}")
    
    return results


def create_summary_table(results: List[Dict]) -> pd.DataFrame:
    """Create summary DataFrame from results."""
    df = pd.DataFrame(results)
    
    if df.empty:
        return df
    
    # Sort by dataset number, then clustering, then method
    df = df.sort_values(['dataset_num', 'clustering', 'mil_method'])
    
    return df


def create_pivot_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create pivot tables for different views of the data."""
    pivots = {}
    
    if df.empty:
        return pivots
    
    # AUC by method and dataset (averaged over clustering methods)
    pivots['auc_method_dataset'] = df.pivot_table(
        values='test_auc',
        index='mil_method',
        columns='dataset',
        aggfunc='mean'
    ).round(4)
    
    # AUC by clustering and dataset (averaged over MIL methods)
    pivots['auc_clustering_dataset'] = df.pivot_table(
        values='test_auc',
        index='clustering',
        columns='dataset',
        aggfunc='mean'
    ).round(4)
    
    # AUC by method and clustering (averaged over datasets)
    pivots['auc_method_clustering'] = df.pivot_table(
        values='test_auc',
        index='mil_method',
        columns='clustering',
        aggfunc='mean'
    ).round(4)
    
    # Full grid: method x (clustering, dataset)
    pivots['full_grid'] = df.pivot_table(
        values='test_auc',
        index='mil_method',
        columns=['clustering', 'dataset'],
        aggfunc='mean'
    ).round(4)
    
    return pivots


def compute_statistics(df: pd.DataFrame) -> Dict:
    """Compute statistical comparisons between methods and clustering approaches."""
    stats_results = {}
    
    if df.empty or len(df) < 2:
        return stats_results
    
    # Compare MIL methods (paired t-test across datasets)
    mil_methods = df['mil_method'].unique()
    method_comparisons = {}
    
    for i, method1 in enumerate(mil_methods):
        for method2 in mil_methods[i+1:]:
            scores1 = df[df['mil_method'] == method1]['test_auc'].dropna()
            scores2 = df[df['mil_method'] == method2]['test_auc'].dropna()
            
            if len(scores1) >= 2 and len(scores2) >= 2:
                # Use Mann-Whitney U test (non-parametric)
                stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                method_comparisons[f'{method1}_vs_{method2}'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'mean_diff': scores1.mean() - scores2.mean(),
                    f'{method1}_mean': scores1.mean(),
                    f'{method2}_mean': scores2.mean(),
                }
    
    stats_results['method_comparisons'] = method_comparisons
    
    # Compare clustering methods
    clustering_methods = df['clustering'].unique()
    if len(clustering_methods) >= 2:
        clustering_comparisons = {}
        for i, clust1 in enumerate(clustering_methods):
            for clust2 in clustering_methods[i+1:]:
                scores1 = df[df['clustering'] == clust1]['test_auc'].dropna()
                scores2 = df[df['clustering'] == clust2]['test_auc'].dropna()
                
                if len(scores1) >= 2 and len(scores2) >= 2:
                    stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                    clustering_comparisons[f'{clust1}_vs_{clust2}'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'mean_diff': scores1.mean() - scores2.mean(),
                        f'{clust1}_mean': scores1.mean(),
                        f'{clust2}_mean': scores2.mean(),
                    }
        
        stats_results['clustering_comparisons'] = clustering_comparisons
    
    # Overall best method
    method_means = df.groupby('mil_method')['test_auc'].mean()
    stats_results['best_method'] = method_means.idxmax()
    stats_results['best_method_auc'] = method_means.max()
    
    # Best clustering
    clustering_means = df.groupby('clustering')['test_auc'].mean()
    stats_results['best_clustering'] = clustering_means.idxmax()
    stats_results['best_clustering_auc'] = clustering_means.max()
    
    return stats_results


def generate_latex_table(pivot: pd.DataFrame, caption: str = "") -> str:
    """Generate LaTeX table from pivot DataFrame."""
    if pivot.empty:
        return ""
    
    latex = pivot.to_latex(float_format="%.4f", na_rep="-")
    
    if caption:
        # Add caption
        latex = latex.replace(
            "\\begin{tabular}",
            f"\\caption{{{caption}}}\n\\begin{tabular}"
        )
    
    return latex


def create_comparison_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Create comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not available, skipping plots")
        return
    
    if df.empty:
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Barplot: AUC by MIL method
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Method comparison
    method_means = df.groupby('mil_method')['test_auc'].agg(['mean', 'std']).reset_index()
    ax1 = axes[0, 0]
    bars = ax1.bar(method_means['mil_method'], method_means['mean'], 
                   yerr=method_means['std'], capsize=5)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    ax1.set_xlabel('MIL Method')
    ax1.set_ylabel('Test AUC')
    ax1.set_title('AUC by MIL Method (mean ± std)')
    ax1.legend()
    ax1.set_ylim(0, 1)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Clustering comparison
    clust_means = df.groupby('clustering')['test_auc'].agg(['mean', 'std']).reset_index()
    ax2 = axes[0, 1]
    bars = ax2.bar(clust_means['clustering'], clust_means['mean'],
                   yerr=clust_means['std'], capsize=5)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    ax2.set_xlabel('Clustering Method')
    ax2.set_ylabel('Test AUC')
    ax2.set_title('AUC by Clustering Method (mean ± std)')
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Dataset comparison
    dataset_means = df.groupby('dataset')['test_auc'].agg(['mean', 'std']).reset_index()
    ax3 = axes[1, 0]
    bars = ax3.bar(dataset_means['dataset'], dataset_means['mean'],
                   yerr=dataset_means['std'], capsize=5)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Test AUC')
    ax3.set_title('AUC by Dataset (mean ± std)')
    ax3.legend()
    ax3.set_ylim(0, 1)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Heatmap: Method x Dataset
    pivot = df.pivot_table(values='test_auc', index='mil_method', 
                           columns='dataset', aggfunc='mean')
    ax4 = axes[1, 1]
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                vmin=0.4, vmax=0.8, ax=ax4, cbar_kws={'label': 'Test AUC'})
    ax4.set_title('AUC Heatmap: Method x Dataset')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path / 'comparison_plots.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_path / 'comparison_plots.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. Detailed heatmap: Method x (Clustering, Dataset)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, clustering in enumerate(df['clustering'].unique()):
        subset = df[df['clustering'] == clustering]
        pivot = subset.pivot_table(values='test_auc', index='mil_method', 
                                   columns='dataset', aggfunc='mean')
        ax = axes[idx]
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                    vmin=0.4, vmax=0.8, ax=ax, cbar_kws={'label': 'Test AUC'})
        ax.set_title(f'{clustering.upper()} Clustering: Method x Dataset')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path / 'heatmap_by_clustering.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_path / 'heatmap_by_clustering.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate MIL Results")
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing results from Phase 3')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for aggregated analysis')
    parser.add_argument('--include_plots', action='store_true', default=True,
                        help='Generate comparison plots')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("AGGREGATE MIL RESULTS")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load all results
    print("\nLoading results...")
    results = load_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} result files")
    
    # Create summary DataFrame
    df = create_summary_table(results)
    
    # Save full results
    df.to_csv(output_dir / 'full_results.csv', index=False)
    print(f"\nFull results saved to {output_dir / 'full_results.csv'}")
    
    # Create pivot tables
    print("\nCreating pivot tables...")
    pivots = create_pivot_tables(df)
    
    for name, pivot in pivots.items():
        if not pivot.empty:
            pivot.to_csv(output_dir / f'pivot_{name}.csv')
            print(f"  Saved pivot_{name}.csv")
            
            # Generate LaTeX
            latex = generate_latex_table(pivot, caption=name.replace('_', ' ').title())
            with open(output_dir / f'pivot_{name}.tex', 'w') as f:
                f.write(latex)
    
    # Compute statistics
    print("\nComputing statistics...")
    stats_results = compute_statistics(df)
    
    with open(output_dir / 'statistics.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            return obj
        
        json.dump(convert_types(stats_results), f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n--- AUC by MIL Method ---")
    method_summary = df.groupby('mil_method')['test_auc'].agg(['mean', 'std', 'count'])
    method_summary = method_summary.sort_values('mean', ascending=False)
    print(method_summary.to_string())
    
    print("\n--- AUC by Clustering ---")
    clustering_summary = df.groupby('clustering')['test_auc'].agg(['mean', 'std', 'count'])
    print(clustering_summary.to_string())
    
    print("\n--- AUC by Dataset ---")
    dataset_summary = df.groupby('dataset')['test_auc'].agg(['mean', 'std', 'count'])
    print(dataset_summary.to_string())
    
    if stats_results:
        print(f"\nBest MIL Method: {stats_results.get('best_method', 'N/A')} "
              f"(AUC: {stats_results.get('best_method_auc', 'N/A'):.4f})")
        print(f"Best Clustering: {stats_results.get('best_clustering', 'N/A')} "
              f"(AUC: {stats_results.get('best_clustering_auc', 'N/A'):.4f})")
    
    # Generate plots
    if args.include_plots:
        print("\nGenerating plots...")
        create_comparison_plot(df, output_dir)
    
    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    

if __name__ == '__main__':
    main()
