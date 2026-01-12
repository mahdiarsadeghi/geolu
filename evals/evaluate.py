"""
Geolu - Algorithm Evaluation Module

Systematically evaluates prediction algorithms using historical data windows.

Copyright (c) 2026 Geolu
Licensed under Proprietary License with Educational Use
See LICENSE file for terms and conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory to path for algorithm imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from algorithms.Bachata import predict_all_assets as bachata_predict
from rythm import DataManager


class AlgorithmEvaluator:
    """Evaluates prediction algorithms using rolling historical windows."""
    
    def __init__(self, results_file='evaluation_results.json', csv_file='results.csv'):
        """Initialize the evaluator."""
        self.results_file = os.path.join(os.path.dirname(__file__), results_file)
        self.csv_file = os.path.join(os.path.dirname(__file__), csv_file)
        self.data_manager = DataManager()
        
        # Define evaluation windows (years back from present)
        self.evaluation_windows = [
            {'name': '4y-3y', 'train_end_years_ago': 3},
            {'name': '4y-2y', 'train_end_years_ago': 2},
            {'name': '4y-1y', 'train_end_years_ago': 1}
        ]
        
        # Prediction horizons
        self.horizons = {
            'weekly': 7,
            'monthly': 30,
            'yearly': 365
        }
        
        # Assets to evaluate
        self.assets = ['Gold', 'Bitcoin', 'Oil', 'S&P 500']
    
    def evaluate_algorithm(self, algorithm_name, algorithm_func):
        """
        Evaluate an algorithm across multiple time windows.
        
        Args:
            algorithm_name: Name of the algorithm (e.g., 'Bachata')
            algorithm_func: Function that takes (historical_data, forecast_days) and returns predictions
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING {algorithm_name.upper()}")
        print(f"{'='*70}\n")
        
        # Load full historical data
        full_data = self.data_manager.load_data()
        
        results = {
            'algorithm': algorithm_name,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'windows': {}
        }
        
        # Evaluate each time window
        for window in self.evaluation_windows:
            window_name = window['name']
            train_end_years_ago = window['train_end_years_ago']
            
            print(f"Window: {window_name}")
            print(f"  Training data: 4 years ago → {train_end_years_ago} years ago")
            
            # Calculate date boundaries
            today = full_data.index[-1]
            train_end_date = today - timedelta(days=train_end_years_ago * 365)
            train_start_date = today - timedelta(days=4 * 365)
            
            # Get training data window
            train_data = full_data[(full_data.index >= train_start_date) & 
                                   (full_data.index <= train_end_date)]
            
            if len(train_data) < 30:
                print(f"  ✗ Insufficient training data ({len(train_data)} days)\n")
                continue
            
            print(f"  Training period: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')} ({len(train_data)} days)")
            
            # Evaluate each horizon
            window_results = {}
            for horizon_name, forecast_days in self.horizons.items():
                print(f"    • {horizon_name} ({forecast_days} days)...")
                
                try:
                    # Get prediction from algorithm
                    predictions = algorithm_func(train_data, forecast_days=forecast_days)
                    
                    # Get actual values from truth data
                    actual_start_date = train_end_date + timedelta(days=1)
                    actual_end_date = train_end_date + timedelta(days=forecast_days)
                    actual_data = full_data[(full_data.index >= actual_start_date) & 
                                           (full_data.index <= actual_end_date)]
                    
                    if len(actual_data) == 0:
                        print(f"      ✗ No actual data available for evaluation")
                        continue
                    
                    # Calculate errors for each asset
                    horizon_results = {}
                    for asset in self.assets:
                        if asset in predictions and asset in actual_data.columns:
                            # Get predicted final price
                            predicted_price = predictions[asset]['prices'][-1]
                            
                            # Get actual final price
                            actual_price = float(actual_data[asset].iloc[-1])
                            
                            # Get last observed price (for normalization)
                            last_observed_price = float(train_data[asset].iloc[-1])
                            
                            # Calculate absolute deviation
                            absolute_error = abs(predicted_price - actual_price)
                            
                            # Calculate normalized error (percentage of last observed price)
                            normalized_error = (absolute_error / last_observed_price) * 100
                            
                            horizon_results[asset] = {
                                'predicted_price': round(predicted_price, 3),
                                'actual_price': round(actual_price, 3),
                                'last_observed_price': round(last_observed_price, 3),
                                'absolute_error': round(absolute_error, 3),
                                'normalized_error_percent': round(normalized_error, 3),
                                'confidence': predictions[asset]['confidence']
                            }
                            
                            print(f"      {asset}: Pred=${predicted_price:.2f}, Actual=${actual_price:.2f}, Error={normalized_error:.2f}%")
                    
                    window_results[horizon_name] = horizon_results
                
                except Exception as e:
                    print(f"      ✗ Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            results['windows'][window_name] = window_results
            print()
        
        # Calculate aggregate statistics
        results['aggregate'] = self._calculate_aggregate_stats(results['windows'])
        
        return results
    
    def _calculate_aggregate_stats(self, windows_data):
        """Calculate aggregate statistics across all windows."""
        aggregate = {}
        
        for horizon in self.horizons.keys():
            horizon_errors = {asset: [] for asset in self.assets}
            
            # Collect all normalized errors for this horizon
            for window_name, window_data in windows_data.items():
                if horizon in window_data:
                    for asset in self.assets:
                        if asset in window_data[horizon]:
                            error = window_data[horizon][asset]['normalized_error_percent']
                            horizon_errors[asset].append(error)
            
            # Calculate mean error for each asset
            aggregate[horizon] = {}
            for asset in self.assets:
                if len(horizon_errors[asset]) > 0:
                    mean_error = np.mean(horizon_errors[asset])
                    std_error = np.std(horizon_errors[asset])
                    aggregate[horizon][asset] = {
                        'mean_error_percent': round(mean_error, 3),
                        'std_error_percent': round(std_error, 3),
                        'num_evaluations': len(horizon_errors[asset])
                    }
        
        return aggregate
    
    def save_results(self, results):
        """Save evaluation results to JSON and CSV files."""
        # Load existing JSON results if available
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # Add or update results for this algorithm
        algorithm_name = results['algorithm']
        all_results[algorithm_name] = results
        
        # Save JSON
        with open(self.results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"✓ Results saved to {self.results_file}")
        
        # Save to CSV
        self._save_to_csv(results)
    
    def _save_to_csv(self, results):
        """Save results to CSV format."""
        algorithm_name = results['algorithm']
        windows_data = results['windows']
        
        # Prepare rows for CSV
        csv_rows = []
        
        for window_name, window_data in windows_data.items():
            for horizon in ['weekly', 'monthly', 'yearly']:
                if horizon in window_data:
                    row = {
                        'Algorithm': algorithm_name,
                        'Window': window_name,
                        'Prediction_Scale': horizon,
                        'Gold_Error_%': None,
                        'Bitcoin_Error_%': None,
                        'Oil_Error_%': None,
                        'Stock_Error_%': None
                    }
                    
                    # Fill in error values for each asset
                    for asset in self.assets:
                        if asset in window_data[horizon]:
                            error = window_data[horizon][asset]['normalized_error_percent']
                            # Round to 3 significant digits and format as percentage
                            error_formatted = f"{error:.3g}%"
                            if asset == 'Gold':
                                row['Gold_Error_%'] = error_formatted
                            elif asset == 'Bitcoin':
                                row['Bitcoin_Error_%'] = error_formatted
                            elif asset == 'Oil':
                                row['Oil_Error_%'] = error_formatted
                            elif asset == 'S&P 500':
                                row['Stock_Error_%'] = error_formatted
                    
                    csv_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(csv_rows)
        
        # Load existing CSV if available and append
        if os.path.exists(self.csv_file):
            df_existing = pd.read_csv(self.csv_file)
            # Remove old entries for this algorithm to avoid duplicates
            df_existing = df_existing[df_existing['Algorithm'] != algorithm_name]
            df = pd.concat([df_existing, df], ignore_index=True)
        
        # Save to CSV
        df.to_csv(self.csv_file, index=False)
        print(f"✓ Results saved to {self.csv_file}")
    
    def print_summary(self, results):
        """Print a summary of evaluation results."""
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY - {results['algorithm']}")
        print(f"{'='*70}\n")
        
        aggregate = results['aggregate']
        
        for horizon in ['weekly', 'monthly', 'yearly']:
            if horizon in aggregate:
                print(f"{horizon.upper()}:")
                for asset in self.assets:
                    if asset in aggregate[horizon]:
                        stats = aggregate[horizon][asset]
                        print(f"  {asset:10s}: {stats['mean_error_percent']:6.2f}% ± {stats['std_error_percent']:5.2f}% (n={stats['num_evaluations']})")
                print()


def main():
    """Main function to run evaluations."""
    evaluator = AlgorithmEvaluator()
    
    # Evaluate Bachata
    results = evaluator.evaluate_algorithm('Bachata', bachata_predict)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
