"""
Geolu - Where Algorithms Predict Value
Numerical prediction system for Gold, Bitcoin, Oil, and Stock markets.

Copyright (c) 2026 Geolu
Licensed under Proprietary License with Educational Use
See LICENSE file for terms and conditions.

The algorithms and predictive methodologies in this software are proprietary.
Commercial use requires permission from the copyright holder.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
import pytz
import sys
import shutil

# Add evals directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evals'))
from rythm import DataManager

# Import algorithms
from algorithms.Bachata import predict_all_assets as bachata_predict

warnings.filterwarnings('ignore')


class MarketPredictor:
    """Numerical prediction system for multiple market assets."""
    
    def __init__(self, 
                 guess_file='guess.csv',
                 log_file='predictions_log.json', 
                 web_data_file='docs/data.json'):
        """Initialize the market predictor."""
        self.guess_file = guess_file
        self.log_file = log_file
        self.web_data_file = web_data_file
        
        # Data manager for loading historical data
        self.data_manager = DataManager()
        
        # Asset names
        self.assets = ['Gold', 'Bitcoin', 'Oil', 'S&P 500']
        
        # Prediction periods
        self.periods = {
            'weekly': 7,
            'monthly': 30,
            'yearly': 365
        }
        
        # Algorithm registry
        self.algorithms = {
            'Bachata': bachata_predict
        }
    
    def run_prediction(self):
        """Run predictions for all assets using all algorithms."""
        print("\n" + "="*70)
        print("GEOLU MARKET PREDICTIONS")
        print("="*70)
        
        # Get timezone
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        timestamp = now_ny.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"Prediction Time: {timestamp}\n")
        
        # Load historical data
        print("Loading historical data...")
        historical_data = self.data_manager.load_data()
        print(f"  ✓ Loaded {len(historical_data)} data points from {historical_data.index[0].strftime('%Y-%m-%d')} to {historical_data.index[-1].strftime('%Y-%m-%d')}\n")
        
        # Get current prices
        current_prices = {}
        for asset in self.assets:
            current_prices[asset] = float(historical_data[asset].iloc[-1])
            print(f"  {asset}: ${current_prices[asset]:.2f}")
        
        print("\n" + "-"*70)
        print("RUNNING ALGORITHMS")
        print("-"*70 + "\n")
        
        # Run each algorithm
        all_predictions = {}
        for algo_name, algo_func in self.algorithms.items():
            print(f"Running {algo_name}...")
            
            algo_predictions = {}
            for period_name, forecast_days in self.periods.items():
                print(f"  • {period_name} ({forecast_days} days)...")
                
                try:
                    # Run algorithm
                    results = algo_func(historical_data, forecast_days=forecast_days)
                    
                    # Store results
                    algo_predictions[period_name] = {
                        'forecast_days': forecast_days,
                        'results': results
                    }
                    
                    # Print summary
                    for asset in self.assets:
                        if asset in results:
                            final_price = results[asset]['prices'][-1]
                            change_pct = ((final_price - current_prices[asset]) / current_prices[asset]) * 100
                            conf = results[asset]['confidence']
                            print(f"    {asset}: ${final_price:.2f} ({change_pct:+.2f}%, confidence: {conf:.2f})")
                
                except Exception as e:
                    print(f"    ✗ {algo_name} {period_name} failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            all_predictions[algo_name] = algo_predictions
            print()
        
        # Save predictions to guess.csv
        self.save_guess_csv(all_predictions, current_prices, timestamp)
        
        # Log prediction
        self.log_prediction(all_predictions, current_prices, timestamp)
        
        # Export for web
        self.export_web_data(all_predictions, current_prices, historical_data, timestamp)
        
        print("="*70)
        print("PREDICTION COMPLETE")
        print("="*70 + "\n")
        
        return all_predictions
    
    def save_guess_csv(self, predictions, current_prices, timestamp):
        """
        Save predictions to guess.csv in structured format.
        
        Format: Algorithm, Asset, Period, Days, Current_Price, Predicted_Price, Change_%, Confidence
        """
        rows = []
        
        for algo_name, algo_predictions in predictions.items():
            for period_name, period_data in algo_predictions.items():
                forecast_days = period_data['forecast_days']
                results = period_data['results']
                
                for asset in self.assets:
                    if asset in results:
                        current = current_prices[asset]
                        predicted = results[asset]['prices'][-1]
                        change_pct = ((predicted - current) / current) * 100
                        confidence = results[asset]['confidence']
                        
                        rows.append({
                            'Timestamp': timestamp,
                            'Algorithm': algo_name,
                            'Asset': asset,
                            'Period': period_name,
                            'Forecast_Days': forecast_days,
                            'Current_Price': round(current, 3),
                            'Predicted_Price': round(predicted, 3),
                            'Change_Percent': round(change_pct, 3),
                            'Confidence': round(confidence, 3)
                        })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Append to existing guess.csv if it exists
        if os.path.exists(self.guess_file):
            df_existing = pd.read_csv(self.guess_file)
            df = pd.concat([df_existing, df], ignore_index=True)
        
        # Save
        df.to_csv(self.guess_file, index=False)
        print(f"✓ Predictions saved to {self.guess_file}")
    
    def log_prediction(self, predictions, current_prices, timestamp):
        """Log prediction to predictions_log.json."""
        # Load existing log
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                log = json.load(f)
        else:
            log = []
        
        # Create log entry
        entry = {
            'timestamp': timestamp,
            'current_prices': current_prices,
            'predictions': {}
        }
        
        for algo_name, algo_predictions in predictions.items():
            entry['predictions'][algo_name] = {}
            for period_name, period_data in algo_predictions.items():
                entry['predictions'][algo_name][period_name] = {
                    'forecast_days': period_data['forecast_days'],
                    'results': period_data['results']
                }
        
        log.append(entry)
        
        # Save
        with open(self.log_file, 'w') as f:
            json.dump(log, f, indent=2)
        
        print(f"✓ Prediction logged to {self.log_file}")
    
    def export_web_data(self, predictions, current_prices, historical_data, timestamp):
        """Export data for web visualization."""
        # Ensure docs directory exists
        os.makedirs(os.path.dirname(self.web_data_file), exist_ok=True)
        
        # Prepare historical data for different time periods
        time_periods = {
            'weekly': historical_data.tail(7),
            'monthly': historical_data.tail(30),
            'yearly': historical_data.tail(365)
        }
        
        time_periods_for_web = {}
        for period_name, period_data in time_periods.items():
            time_periods_for_web[period_name] = {}
            for asset in self.assets:
                # Handle both 'S&P 500' and 'Stock' column names
                col_name = asset if asset in period_data.columns else 'Stock' if asset == 'S&P 500' else asset
                if col_name in period_data.columns:
                    time_periods_for_web[period_name][asset] = {
                        'dates': [d.strftime('%Y-%m-%d') for d in period_data.index],
                        'prices': [float(p) for p in period_data[col_name].values]
                    }
        
        # Prepare predictions for web (restructured for Bachata format)
        # Frontend expects: {Gold: {weekly: {...}, monthly: {...}}, Bitcoin: {...}}
        bachata_predictions = {}
        for asset in self.assets:
            bachata_predictions[asset] = {}
        
        for algo_name, algo_predictions in predictions.items():
            for period_name, period_data in algo_predictions.items():
                # Generate future dates
                last_date = historical_data.index[-1]
                forecast_days = period_data['forecast_days']
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )
                
                for asset in self.assets:
                    # Check both asset name and 'Stock' for S&P 500
                    result_key = asset
                    if asset == 'S&P 500' and asset not in period_data['results']:
                        result_key = 'Stock' if 'Stock' in period_data['results'] else None
                    
                    if result_key and result_key in period_data['results']:
                        bachata_predictions[asset][period_name] = {
                            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                            'prices': period_data['results'][result_key]['prices'],
                            'confidence': period_data['results'][result_key]['confidence']
                        }
        
        # Create web data structure
        web_data = {
            'last_updated': timestamp,
            'current_prices': current_prices,
            'time_periods': time_periods_for_web,
            'bachata_predictions': bachata_predictions,
            'prediction_history': self.get_prediction_history()
        }
        
        # Save
        with open(self.web_data_file, 'w') as f:
            json.dump(web_data, f, indent=2)
        
        print(f"✓ Web data exported to {self.web_data_file}")
        
        # Copy evaluation results if available
        eval_results_path = os.path.join('evals', 'results.csv')
        if os.path.exists(eval_results_path):
            dest_path = os.path.join('docs', 'results.csv')
            shutil.copy2(eval_results_path, dest_path)
            print(f"✓ Evaluation results copied to {dest_path}")
        
        # Copy guess.csv if available
        if os.path.exists(self.guess_file):
            dest_path = os.path.join('docs', 'guess.csv')
            shutil.copy2(self.guess_file, dest_path)
            print(f"✓ Predictions (guess.csv) copied to {dest_path}")
    
    def get_prediction_history(self):
        """Get prediction history from log file."""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []


def main():
    """Main function to run predictions."""
    predictor = MarketPredictor()
    predictor.run_prediction()


if __name__ == "__main__":
    main()
