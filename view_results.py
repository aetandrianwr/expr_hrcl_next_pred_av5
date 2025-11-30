"""
View and analyze benchmark results.

Usage:
    python view_results.py
    python view_results.py --top 10
    python view_results.py --metric test_acc@1
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='View benchmark results')
    parser.add_argument('--results', type=str, default='results/benchmark_results.csv',
                       help='Path to results CSV')
    parser.add_argument('--top', type=int, default=5,
                       help='Number of top results to show')
    parser.add_argument('--metric', type=str, default='test_acc@1',
                       help='Metric to sort by')
    args = parser.parse_args()
    
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"No results file found at: {results_file}")
        return
    
    # Load results
    df = pd.read_csv(results_file)
    
    if df.empty:
        print("No experiments logged yet.")
        return
    
    print("=" * 100)
    print(f"BENCHMARK RESULTS - Top {args.top} by {args.metric}")
    print("=" * 100)
    
    # Sort by metric
    df_sorted = df.sort_values(args.metric, ascending=False).head(args.top)
    
    # Display key columns
    columns_to_show = [
        'timestamp',
        'experiment_name',
        'd_model',
        'num_layers',
        'learning_rate',
        'test_acc@1',
        'test_f1',
        'test_mrr',
        'test_ndcg',
        'training_time'
    ]
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(df_sorted[columns_to_show].to_string(index=False))
    
    print("\n" + "=" * 100)
    print(f"Total experiments: {len(df)}")
    print(f"Best {args.metric}: {df[args.metric].max():.2f}%")
    print(f"Average {args.metric}: {df[args.metric].mean():.2f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
