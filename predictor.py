"""
Geolu - Where Algorithms Predict Value
Predicts gold prices weekly and logs predictions over time.
Supports multiple models for comparison and accuracy evaluation.

Copyright (c) 2026 Geolu
Licensed under Proprietary License with Educational Use
See LICENSE file for terms and conditions.

The algorithms and predictive methodologies in this software are proprietary.
Commercial use requires permission from the copyright holder.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import json
import os
import warnings
import pytz
from algorithms.Bachata import BachataPredictor
warnings.filterwarnings('ignore')


class GoldPricePredictor:
    def __init__(self, log_file='predictions_log.json', web_data_file='docs/data.json'):
        """Initialize the gold price predictor."""
        self.log_file = log_file
        self.web_data_file = web_data_file
        self.gold_ticker = 'GC=F'  # Gold futures
        self.bitcoin_ticker = 'BTC-USD'  # Bitcoin
        self.oil_ticker = 'CL=F'  # Crude Oil futures
        self.sp500_ticker = '^GSPC'  # S&P 500
        self.models = {}
        self.predictions = {}
    
    def fetch_multiple_assets(self, period='2y'):
        """Fetch historical data for multiple assets."""
        print(f"Fetching market data for multiple assets...")
        assets = {
            'Gold': self.gold_ticker,
            'Bitcoin': self.bitcoin_ticker,
            'Oil': self.oil_ticker,
            'S&P 500': self.sp500_ticker
        }
        
        asset_data = {}
        for name, ticker in assets.items():
            try:
                data = yf.download(ticker, period=period, progress=False)
                if not data.empty:
                    asset_data[name] = data
                    print(f"  ✓ {name} data fetched")
            except Exception as e:
                print(f"  ✗ {name} data failed: {e}")
        
        return asset_data
        
    def fetch_gold_data(self, period='2y'):
        """Fetch historical gold price data."""
        print(f"Fetching gold price data for the past {period}...")
        gold_data = yf.download(self.gold_ticker, period=period, progress=False)
        return gold_data
    
    def create_features(self, df):
        """Create features for prediction."""
        df = df.copy()
        
        # Technical indicators
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['MA_90'] = df['Close'].rolling(window=90).mean()
        df['Volatility'] = df['Close'].rolling(window=30).std()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        
        # Momentum indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_training_data(self, df):
        """Prepare data for training."""
        feature_columns = ['MA_7', 'MA_30', 'MA_90', 'Volatility', 
                          'Daily_Return', 'Price_Change', 'RSI']
        
        X = df[feature_columns].values
        y = df['Close'].values
        
        return X, y, feature_columns
    
    def initialize_models(self):
        """Initialize multiple prediction models."""
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
        }
    
    def train_models(self):
        """Train all prediction models."""
        print("Training prediction models...")
        
        # Fetch data
        gold_data = self.fetch_gold_data(period='2y')
        
        # Create features
        df_features = self.create_features(gold_data)
        
        # Prepare training data
        X, y, feature_columns = self.prepare_training_data(df_features)
        
        # Initialize models
        self.initialize_models()
        
        # Train all models
        for model_name, model in self.models.items():
            model.fit(X, y)
            print(f"  ✓ {model_name} trained")
        
        print("All models training completed!")
        return df_features, feature_columns
    
    def predict_next_week(self):
        """Make predictions for next week's gold price using all models."""
        # Train models with latest data
        df_features, feature_columns = self.train_models()
        
        # Get the most recent data point for prediction
        latest_features = df_features.iloc[-1]
        X_pred = latest_features[feature_columns].values.reshape(1, -1)
        
        # Get current price
        current_price = df_features['Close'].iloc[-1]
        
        # Make predictions with all models
        predictions = {}
        for model_name, model in self.models.items():
            pred_result = model.predict(X_pred)
            # Extract scalar value from numpy array
            predicted_price = float(pred_result.item() if hasattr(pred_result, 'item') else pred_result[0])
            price_change_percent = ((predicted_price - current_price) / current_price) * 100
            
            predictions[model_name] = {
                'predicted_price': predicted_price,
                'price_change_percent': float(price_change_percent)
            }
        
        # Get current time in New York timezone
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        
        return {
            'current_price': float(current_price),
            'predictions': predictions,
            'prediction_date': now_ny.strftime('%Y-%m-%d %H:%M:%S'),
            'target_date': (now_ny + timedelta(days=7)).strftime('%Y-%m-%d'),
            'historical_data': self.get_recent_historical_data(df_features)
        }
    
    def get_recent_historical_data(self, df, days=30):
        """Get recent historical data for visualization."""
        recent_data = df.tail(days)[['Close']].copy()
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in recent_data.index],
            'prices': [float(p.item()) if hasattr(p, 'item') else float(p) for p in recent_data['Close'].values]
        }
    
    def get_all_assets_historical_data(self, days=30):
        """Get recent historical data for all assets."""
        asset_data = self.fetch_multiple_assets(period='1y')
        all_historical = {}
        
        for asset_name, data in asset_data.items():
            if not data.empty:
                recent_data = data.tail(days)[['Close']].copy()
                all_historical[asset_name] = {
                    'dates': [d.strftime('%Y-%m-%d') for d in recent_data.index],
                    'prices': [float(p.item()) if hasattr(p, 'item') else float(p) for p in recent_data['Close'].values]
                }
        
        return all_historical
    
    def get_past_week_data(self):
        """Get past week (7 days) of historical data for all assets."""
        asset_data = self.fetch_multiple_assets(period='1mo')  # Fetch 1 month to ensure we have at least 7 days
        week_historical = {}
        
        for asset_name, data in asset_data.items():
            if not data.empty:
                # Get last 7 trading days
                week_data = data.tail(7)[['Close']].copy()
                week_historical[asset_name] = {
                    'dates': [d.strftime('%Y-%m-%d') for d in week_data.index],
                    'prices': [float(p.item()) if hasattr(p, 'item') else float(p) for p in week_data['Close'].values]
                }
        
        return week_historical
    
    def get_time_period_data(self, days, period):
        """Get historical data for a specific time period for all assets."""
        asset_data = self.fetch_multiple_assets(period=period)
        period_historical = {}
        
        for asset_name, data in asset_data.items():
            if not data.empty:
                period_data = data.tail(days)[['Close']].copy()
                period_historical[asset_name] = {
                    'dates': [d.strftime('%Y-%m-%d') for d in period_data.index],
                    'prices': [float(p.item()) if hasattr(p, 'item') else float(p) for p in period_data['Close'].values]
                }
        
        return period_historical
    
    def generate_simple_predictions(self, historical_data, forecast_days):
        """Generate simple smoothed predictions based on recent trends."""
        predictions = {}
        
        for asset_name, data in historical_data.items():
            if not data['prices']:
                continue
                
            prices = data['prices']
            # Calculate simple moving average trend
            recent_avg = sum(prices[-min(5, len(prices)):]) / min(5, len(prices))
            overall_avg = sum(prices) / len(prices)
            trend = (recent_avg - overall_avg) / overall_avg
            
            # Generate smoothed predictions
            last_price = prices[-1]
            predicted_prices = []
            for i in range(1, forecast_days + 1):
                # Gentle trend continuation with dampening
                predicted_change = trend * (1 - i / (forecast_days * 2))
                predicted_price = last_price * (1 + predicted_change * 0.5)
                predicted_prices.append(float(predicted_price))
            
            # Generate future dates
            from datetime import datetime, timedelta
            last_date = datetime.strptime(data['dates'][-1], '%Y-%m-%d')
            predicted_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                             for i in range(1, forecast_days + 1)]
            
            predictions[asset_name] = {
                'dates': predicted_dates,
                'prices': predicted_prices
            }
        
        return predictions
    
    def generate_bachata_predictions(self) -> Dict[str, Dict]:
        """
        Generate predictions using the Bachata Fourier analysis algorithm.
        
        Returns:
            Dictionary with predictions for all assets at different time scales
        """
        print("\nRunning Bachata Fourier Analysis...")
        
        # Fetch extended historical data for Bachata (5 years)
        asset_data_5y = self.fetch_multiple_assets(period='5y')
        
        if not asset_data_5y:
            print("  ✗ No data available for Bachata analysis")
            return {}
        
        # Initialize Bachata predictor
        bachata = BachataPredictor(window_months=6, lookback_years=5)
        
        try:
            # Fit the model with all asset data
            bachata.fit(asset_data_5y)
            print("  ✓ Bachata model fitted")
            
            # Generate predictions for each asset at different time scales
            all_predictions = {}
            time_scales = {
                'weekly': {'days': 7, 'points': 7, 'freq': 'D', 'label': 'daily'},      # 7 days (daily resolution)
                'monthly': {'days': 28, 'points': 4, 'freq': '7D', 'label': 'weekly'},   # 4 weeks (weekly resolution)
                'yearly': {'days': 365, 'points': 12, 'freq': '30D', 'label': 'monthly'} # 12 months (monthly resolution)
            }
            
            for asset_name in ['Gold', 'Bitcoin', 'Oil', 'S&P 500']:
                if asset_name not in asset_data_5y:
                    continue
                    
                asset_predictions = {}
                
                for scale_name, scale_config in time_scales.items():
                    # Get full prediction for the period
                    result = bachata.predict(asset_name, scale_config['days'])
                    
                    # Get current price
                    current_price = float(asset_data_5y[asset_name]['Close'].iloc[-1])
                    
                    # Generate dates at the appropriate frequency
                    last_date = asset_data_5y[asset_name].index[-1]
                    future_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=scale_config['points'],
                        freq=scale_config['freq']
                    )
                    
                    # Sample predictions at the appropriate intervals
                    predictions_array = result['predictions']
                    if isinstance(predictions_array, np.ndarray) and len(predictions_array) > 0:
                        # Sample at intervals to get the right number of points
                        interval = len(predictions_array) // scale_config['points']
                        if interval < 1:
                            interval = 1
                        sampled_predictions = predictions_array[::max(1, interval)][:scale_config['points']]
                        
                        # Predictions are price changes, apply them progressively
                        # Start from current price and add cumulative changes
                        predictions_with_base = np.zeros(scale_config['points'])
                        predictions_with_base[0] = current_price + sampled_predictions[0]
                        for i in range(1, scale_config['points']):
                            predictions_with_base[i] = predictions_with_base[i-1] + sampled_predictions[i]
                    else:
                        predictions_with_base = np.full(scale_config['points'], current_price)
                    
                    asset_predictions[scale_name] = {
                        'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                        'prices': [float(p) for p in predictions_with_base],
                        'confidence': float(result['confidence'][0]),
                        'dominant_frequencies': result['dominant_frequencies'][:3],  # Top 3
                        'resolution': scale_config['label']
                    }
                    
                    print(f"  ✓ {asset_name} {scale_name} predictions generated ({scale_config['points']} {scale_config['label']} points, confidence: {result['confidence'][0]:.2f})")
                
                all_predictions[asset_name] = asset_predictions
            
            return all_predictions
            
        except Exception as e:
            print(f"  ✗ Bachata prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def log_prediction(self, prediction):
        """Log the prediction to a JSON file."""
        # Load existing predictions
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                predictions = json.load(f)
        else:
            predictions = []
        
        # Add new prediction
        predictions.append(prediction)
        
        # Save updated predictions
        with open(self.log_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"Prediction logged to {self.log_file}")
    
    def get_prediction_history(self):
        """Retrieve prediction history."""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []
    
    def display_prediction(self, prediction):
        """Display prediction in a formatted way."""
        print("\n" + "="*70)
        print("GOLD PRICE PREDICTIONS - MULTI-MODEL COMPARISON")
        print("="*70)
        print(f"Prediction Date: {prediction['prediction_date']}")
        print(f"Target Date: {prediction['target_date']}")
        print(f"Current Price: ${prediction['current_price']:.2f}")
        print("-"*70)
        print(f"{'Model':<25} {'Predicted Price':>15} {'Change':>12}")
        print("-"*70)
        
        for model_name, pred in prediction['predictions'].items():
            print(f"{model_name:<25} ${pred['predicted_price']:>14.2f} {pred['price_change_percent']:>11.2f}%")
        
        print("="*70 + "\n")
    
    def export_web_data(self, prediction):
        """Export data for web visualization."""
        # Ensure docs directory exists
        os.makedirs(os.path.dirname(self.web_data_file), exist_ok=True)
        
        # Get historical data for all assets (30 days for context)
        all_assets_historical = self.get_all_assets_historical_data(days=30)
        
        # Get data for different time periods
        weekly_data = self.get_time_period_data(days=7, period='1mo')
        monthly_data = self.get_time_period_data(days=30, period='3mo')
        yearly_data = self.get_time_period_data(days=365, period='2y')
        
        # Generate simple baseline predictions
        weekly_predictions = self.generate_simple_predictions(weekly_data, forecast_days=7)
        monthly_predictions = self.generate_simple_predictions(monthly_data, forecast_days=30)
        yearly_predictions = self.generate_simple_predictions(yearly_data, forecast_days=365)
        
        # Generate Bachata algorithm predictions
        bachata_predictions = self.generate_bachata_predictions()
        
        # Prepare web data
        web_data = {
            'last_updated': prediction['prediction_date'],
            'current_price': prediction['current_price'],
            'target_date': prediction['target_date'],
            'historical': prediction['historical_data'],
            'all_assets': all_assets_historical,
            'past_week': weekly_data,  # Keep for backward compatibility
            'time_periods': {
                'weekly': weekly_data,
                'monthly': monthly_data,
                'yearly': yearly_data
            },
            'predictions_data': {
                'weekly': weekly_predictions,
                'monthly': monthly_predictions,
                'yearly': yearly_predictions
            },
            'bachata_predictions': bachata_predictions,  # Advanced Fourier predictions
            'predictions': prediction['predictions'],
            'prediction_history': self.get_prediction_history()
        }
        
        # Save web data
        with open(self.web_data_file, 'w') as f:
            json.dump(web_data, f, indent=2)
        
        print(f"Web data exported to {self.web_data_file}")
    
    def run_weekly_prediction(self):
        """Run the weekly prediction routine."""
        print("Starting Gold Price Weekly Prediction...")
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Make predictions with all models
        prediction = self.predict_next_week()
        
        # Display predictions
        self.display_prediction(prediction)
        
        # Log prediction
        self.log_prediction(prediction)
        
        # Export data for web visualization
        self.export_web_data(prediction)
        
        # Show prediction count
        history = self.get_prediction_history()
        print(f"Total predictions logged: {len(history)}")
        
        return prediction


def main():
    """Main function to run the predictor."""
    predictor = GoldPricePredictor()
    predictor.run_weekly_prediction()
    
    # Display recent prediction history
    history = predictor.get_prediction_history()
    if len(history) > 1:
        print("\nRecent Prediction History (Last 3):")
        print("-" * 70)
        for pred in history[-3:]:
            print(f"\n{pred['prediction_date']} | Target: {pred['target_date']}")
            print(f"Current Price: ${pred['current_price']:.2f}")
            if 'predictions' in pred:
                for model, data in pred['predictions'].items():
                    print(f"  {model}: ${data['predicted_price']:.2f} ({data['price_change_percent']:+.2f}%)")
            else:
                # Legacy format support
                print(f"  Predicted: ${pred.get('predicted_price', 0):.2f} ({pred.get('price_change_percent', 0):+.2f}%)")


if __name__ == "__main__":
    main()
