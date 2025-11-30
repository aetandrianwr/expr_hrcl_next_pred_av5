"""
Results tracking and benchmarking system.
Maintains CSV log of all experiments with metrics.
"""

import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


class ResultsTracker:
    """
    Track and log experiment results to CSV.
    Appends new results after each run for easy benchmarking.
    """
    
    def __init__(self, results_file: str = "results/benchmark_results.csv"):
        """
        Initialize results tracker.
        
        Args:
            results_file: Path to CSV file for storing results
        """
        self.results_file = Path(results_file)
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Define CSV columns
        self.columns = [
            # Metadata
            'timestamp',
            'experiment_name',
            'dataset',
            'run_dir',
            
            # Model config
            'model_name',
            'd_model',
            'num_layers',
            'nhead',
            'dropout',
            'total_params',
            
            # Training config
            'batch_size',
            'learning_rate',
            'num_epochs',
            'epochs_trained',
            'optimizer',
            'scheduler',
            
            # Validation metrics
            'val_acc@1',
            'val_acc@3',
            'val_acc@5',
            'val_acc@10',
            'val_f1',
            'val_mrr',
            'val_ndcg',
            'best_epoch',
            
            # Test metrics
            'test_acc@1',
            'test_acc@3',
            'test_acc@5',
            'test_acc@10',
            'test_f1',
            'test_mrr',
            'test_ndcg',
            
            # System
            'device',
            'seed',
            'training_time',
        ]
        
        # Create file with headers if it doesn't exist
        if not self.results_file.exists():
            self._create_file()
    
    def _create_file(self):
        """Create CSV file with headers."""
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
    
    def log_result(self, 
                   config: Dict[str, Any],
                   val_metrics: Dict[str, float],
                   test_metrics: Dict[str, float],
                   training_info: Dict[str, Any]):
        """
        Log experiment results to CSV.
        
        Args:
            config: Configuration dictionary
            val_metrics: Validation metrics
            test_metrics: Test metrics
            training_info: Training information (epochs, time, etc.)
        """
        # Prepare result row
        result = {
            # Metadata
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'experiment_name': config.get('experiment', {}).get('name', 'unknown'),
            'dataset': config.get('experiment', {}).get('dataset', 'unknown'),
            'run_dir': str(training_info.get('run_dir', '')),
            
            # Model config
            'model_name': config.get('model', {}).get('name', 'unknown'),
            'd_model': config.get('model', {}).get('d_model', 0),
            'num_layers': config.get('model', {}).get('num_layers', 0),
            'nhead': config.get('model', {}).get('nhead', 0),
            'dropout': config.get('model', {}).get('dropout', 0),
            'total_params': training_info.get('total_params', 0),
            
            # Training config
            'batch_size': config.get('training', {}).get('batch_size', 0),
            'learning_rate': config.get('training', {}).get('learning_rate', 0),
            'num_epochs': config.get('training', {}).get('num_epochs', 0),
            'epochs_trained': training_info.get('epochs_trained', 0),
            'optimizer': config.get('training', {}).get('optimizer', 'unknown'),
            'scheduler': config.get('training', {}).get('scheduler', {}).get('type', 'unknown'),
            
            # Validation metrics
            'val_acc@1': val_metrics.get('acc@1', 0),
            'val_acc@3': val_metrics.get('acc@3', 0),
            'val_acc@5': val_metrics.get('acc@5', 0),
            'val_acc@10': val_metrics.get('acc@10', 0),
            'val_f1': val_metrics.get('f1', 0) * 100 if val_metrics.get('f1', 0) < 1 else val_metrics.get('f1', 0),
            'val_mrr': val_metrics.get('mrr', 0),
            'val_ndcg': val_metrics.get('ndcg', 0),
            'best_epoch': training_info.get('best_epoch', 0),
            
            # Test metrics
            'test_acc@1': test_metrics.get('acc@1', 0),
            'test_acc@3': test_metrics.get('acc@3', 0),
            'test_acc@5': test_metrics.get('acc@5', 0),
            'test_acc@10': test_metrics.get('acc@10', 0),
            'test_f1': test_metrics.get('f1', 0) * 100 if test_metrics.get('f1', 0) < 1 else test_metrics.get('f1', 0),
            'test_mrr': test_metrics.get('mrr', 0),
            'test_ndcg': test_metrics.get('ndcg', 0),
            
            # System
            'device': config.get('system', {}).get('device', 'unknown'),
            'seed': config.get('system', {}).get('seed', 0),
            'training_time': training_info.get('training_time', 0),
        }
        
        # Append to CSV
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(result)
        
        print(f"\n✓ Results logged to: {self.results_file}")
    
    def get_best_results(self, metric: str = 'test_acc@1', top_k: int = 5) -> pd.DataFrame:
        """
        Get top-k best results by specified metric.
        
        Args:
            metric: Metric to sort by
            top_k: Number of top results to return
        
        Returns:
            DataFrame with top results
        """
        if not self.results_file.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.results_file)
        return df.nlargest(top_k, metric)
    
    def get_all_results(self) -> pd.DataFrame:
        """Get all logged results."""
        if not self.results_file.exists():
            return pd.DataFrame()
        return pd.read_csv(self.results_file)
    
    def compare_experiments(self, exp_names: list) -> pd.DataFrame:
        """
        Compare specific experiments.
        
        Args:
            exp_names: List of experiment names to compare
        
        Returns:
            DataFrame with comparison
        """
        df = self.get_all_results()
        if df.empty:
            return df
        return df[df['experiment_name'].isin(exp_names)]
