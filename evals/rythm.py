"""
Geolu - Data Fetching and Partitioning Module

Fetches historical market data and manages training/testing datasets.

Copyright (c) 2026 Geolu
Licensed under Proprietary License with Educational Use
See LICENSE file for terms and conditions.
"""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataManager:
    """Manages historical data fetching and train/test partitioning."""
    
    def __init__(self, data_file='truth.csv'):
        """Initialize the data manager."""
        self.data_file = os.path.join(os.path.dirname(__file__), data_file)
        self.tickers = {
            'Gold': 'GC=F',
            'Bitcoin': 'BTC-USD',
            'Oil': 'CL=F',
            'S&P 500': '^GSPC'
        }
    
    def fetch_and_save_data(self, years=4, force_refresh=False):
        """
        Fetch historical data for all assets and save to truth.csv.
        
        Args:
            years: Number of years of historical data to fetch
            force_refresh: If True, fetch data even if truth.csv exists
        
        Returns:
            DataFrame with columns: Date, Gold, Bitcoin, Oil, Stock
        """
        # Check if data already exists
        if os.path.exists(self.data_file) and not force_refresh:
            print(f"Loading existing data from {self.data_file}")
            return pd.read_csv(self.data_file, parse_dates=['Date'], index_col='Date')
        
        print(f"Fetching {years} years of historical data...")
        
        # Fetch data for all assets
        all_data = []
        period = f"{years}y"
        
        for asset_name, ticker in self.tickers.items():
            try:
                print(f"  Fetching {asset_name}...")
                data = yf.download(ticker, period=period, progress=False)
                if not data.empty:
                    # Round to 3 significant digits after decimal
                    asset_series = data['Close'].round(3)
                    asset_series.name = asset_name
                    all_data.append(asset_series)
                    print(f"  ✓ {asset_name}: {len(data)} data points")
            except Exception as e:
                print(f"  ✗ {asset_name} failed: {e}")
        
        # Combine all assets into single DataFrame
        df = pd.concat(all_data, axis=1)
        df.index.name = 'Date'
        
        # Remove any rows with missing data
        df = df.dropna()
        
        # Ensure columns have friendly names (in case they were saved as tickers)
        df.columns = ['Gold', 'Bitcoin', 'Oil', 'S&P 500']
        
        # Save to CSV
        df.to_csv(self.data_file)
        print(f"\n✓ Data saved to {self.data_file}")
        print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Total rows: {len(df)}")
        
        return df
    
    def load_data(self):
        """
        Load data from truth.csv. If file doesn't exist, fetch it first.
        
        Returns:
            DataFrame with historical data
        """
        if not os.path.exists(self.data_file):
            print("truth.csv not found. Fetching data...")
            return self.fetch_and_save_data()
        
        return pd.read_csv(self.data_file, parse_dates=['Date'], index_col='Date')
    
    def partition_data(self, train_ratio=0.8):
        """
        Partition data into training and testing sets.
        
        Args:
            train_ratio: Ratio of data to use for training (default 0.8)
        
        Returns:
            Tuple of (train_df, test_df)
        """
        df = self.load_data()
        
        # Calculate split point
        split_idx = int(len(df) * train_ratio)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"\nData partition:")
        print(f"  Training: {train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')} ({len(train_df)} rows)")
        print(f"  Testing:  {test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')} ({len(test_df)} rows)")
        
        return train_df, test_df
    
    def get_asset_data(self, asset_name, start_date=None, end_date=None):
        """
        Get data for a specific asset within a date range.
        
        Args:
            asset_name: Name of the asset ('Gold', 'Bitcoin', 'Oil', 'S&P 500')
            start_date: Start date (optional)
            end_date: End date (optional)
        
        Returns:
            Series with asset prices
        """
        df = self.load_data()
        
        if asset_name not in df.columns:
            raise ValueError(f"Asset '{asset_name}' not found. Available: {list(df.columns)}")
        
        asset_data = df[asset_name]
        
        if start_date:
            asset_data = asset_data[asset_data.index >= start_date]
        if end_date:
            asset_data = asset_data[asset_data.index <= end_date]
        
        return asset_data
    
    def get_training_window(self, end_date, window_days=365):
        """
        Get a training window ending at a specific date.
        
        Args:
            end_date: End date of the window
            window_days: Number of days in the window
        
        Returns:
            DataFrame with data for the window
        """
        df = self.load_data()
        
        # Filter data up to end_date
        df_filtered = df[df.index <= end_date]
        
        # Get last window_days of data
        return df_filtered.tail(window_days)


def main():
    """Main function to fetch and save data."""
    manager = DataManager()
    
    # Fetch and save data (4 years)
    df = manager.fetch_and_save_data(years=4, force_refresh=False)
    
    # Show partition example
    train_df, test_df = manager.partition_data(train_ratio=0.8)
    
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    print("\nSample data (last 5 rows):")
    print(df.tail())


if __name__ == "__main__":
    main()
