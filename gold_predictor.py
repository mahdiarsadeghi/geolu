"""
Gold Price Predictor
Predicts gold prices weekly and logs predictions over time.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import json
import os
import warnings
warnings.filterwarnings('ignore')


class GoldPricePredictor:
    def __init__(self, log_file='predictions_log.json'):
        """Initialize the gold price predictor."""
        self.log_file = log_file
        self.gold_ticker = 'GC=F'  # Gold futures ticker
        self.model = None
        
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
    
    def train_model(self):
        """Train the prediction model."""
        print("Training prediction model...")
        
        # Fetch data
        gold_data = self.fetch_gold_data(period='2y')
        
        # Create features
        df_features = self.create_features(gold_data)
        
        # Prepare training data
        X, y, feature_columns = self.prepare_training_data(df_features)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(X, y)
        
        print("Model training completed!")
        return df_features
    
    def predict_next_week(self):
        """Make prediction for next week's gold price."""
        # Train model with latest data
        df_features = self.train_model()
        
        # Get the most recent data point for prediction
        latest_features = df_features.iloc[-1]
        feature_columns = ['MA_7', 'MA_30', 'MA_90', 'Volatility', 
                          'Daily_Return', 'Price_Change', 'RSI']
        
        X_pred = latest_features[feature_columns].values.reshape(1, -1)
        
        # Make prediction
        predicted_price = self.model.predict(X_pred)[0]
        
        # Get current price
        current_price = df_features['Close'].iloc[-1]
        
        # Calculate prediction confidence (simplified)
        price_change_percent = ((predicted_price - current_price) / current_price) * 100
        
        return {
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'price_change_percent': float(price_change_percent),
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        }
    
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
        print("\n" + "="*60)
        print("GOLD PRICE PREDICTION")
        print("="*60)
        print(f"Prediction Date: {prediction['prediction_date']}")
        print(f"Target Date: {prediction['target_date']}")
        print(f"Current Price: ${prediction['current_price']:.2f}")
        print(f"Predicted Price (1 week): ${prediction['predicted_price']:.2f}")
        print(f"Expected Change: {prediction['price_change_percent']:+.2f}%")
        print("="*60 + "\n")
    
    def run_weekly_prediction(self):
        """Run the weekly prediction routine."""
        print("Starting Gold Price Weekly Prediction...")
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Make prediction
        prediction = self.predict_next_week()
        
        # Display prediction
        self.display_prediction(prediction)
        
        # Log prediction
        self.log_prediction(prediction)
        
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
        print("\nRecent Prediction History (Last 5):")
        print("-" * 60)
        for pred in history[-5:]:
            print(f"{pred['prediction_date']} | Current: ${pred['current_price']:.2f} | "
                  f"Predicted: ${pred['predicted_price']:.2f} | "
                  f"Change: {pred['price_change_percent']:+.2f}%")


if __name__ == "__main__":
    main()
