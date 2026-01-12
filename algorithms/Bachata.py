"""
Bachata - Fourier Series Analysis Prediction Algorithm

This algorithm uses multi-scale Fourier decomposition to predict asset prices.
It analyzes price movements across different frequency bands (hourly, daily, weekly, monthly)
and models how Fourier coefficients evolve over time.

Key Features:
- Independent Fourier analysis for each asset
- Covariance-based cross-asset influence weights
- Multi-scale frequency decomposition
- Coefficient evolution modeling with linear regression

Copyright (c) 2026 Geolu
Licensed under Proprietary License with Educational Use
See LICENSE file for terms and conditions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import LinearRegression


class BachataPredictor:
    """Fourier series-based price prediction using multi-scale analysis."""
    
    def __init__(self, window_months=6, lookback_years=5):
        """
        Initialize Bachata predictor.
        
        Args:
            window_months: Size of rolling analysis window in months
            lookback_years: How many years of historical data to analyze
        """
        self.window_months = window_months
        self.lookback_years = lookback_years
        
        # Define frequency bands (in days^-1)
        self.frequency_bands = {
            'hourly': (1/24, 1/12),    # Sub-daily variations
            'daily': (1/7, 1/2),       # 2-7 day cycles
            'weekly': (1/30, 1/7),     # Weekly to monthly cycles
            'monthly': (1/180, 1/30)   # Monthly to semi-annual cycles
        }
        
        # Storage for fitted models
        self.asset_models = {}
        self.fourier_covariance = {}
        self.influence_weights = {}
        
    def fit(self, asset_data):
        """
        Fit the Bachata model on historical asset data.
        
        Args:
            asset_data: Dictionary of {asset_name: DataFrame} with 'Close' prices
        """
        print("\nBachata: Analyzing Fourier components...")
        
        # Store asset names
        self.asset_names = list(asset_data.keys())
        
        # Analyze each asset independently
        for asset_name, data in asset_data.items():
            print(f"  • Processing {asset_name}...")
            self.asset_models[asset_name] = self._analyze_asset(data, asset_name)
        
        # Calculate Fourier covariance between assets
        print("  • Calculating Fourier covariance...")
        self._calculate_fourier_covariance()
        
        # Calculate influence weights
        print("  • Computing influence weights...")
        self._calculate_influence_weights()
        
        print("Bachata: Model fitting complete")
    
    def _analyze_asset(self, data, asset_name):
        """
        Analyze a single asset with Fourier decomposition.
        
        Args:
            data: DataFrame with 'Close' prices
            asset_name: Name of the asset
            
        Returns:
            Dictionary with Fourier analysis results
        """
        # Extract prices
        prices = data['Close'].values
        dates = data.index
        
        # Create rolling windows
        window_days = self.window_months * 30
        windows = []
        window_centers = []
        
        for i in range(len(prices) - window_days):
            window = prices[i:i+window_days]
            windows.append(window)
            window_centers.append(i + window_days // 2)
        
        # Extract Fourier coefficients for each window
        fourier_data = {
            'amplitudes': {band: [] for band in self.frequency_bands},
            'phases': {band: [] for band in self.frequency_bands},
            'window_centers': window_centers
        }
        
        for window in windows:
            coeffs = self._extract_fourier_coefficients(window)
            for band in self.frequency_bands:
                # Extract scalar values
                amp = coeffs[band]['amplitude']
                phase = coeffs[band]['phase']
                
                # Ensure they are scalars, not arrays
                if hasattr(amp, 'item'):
                    amp = amp.item()
                if hasattr(phase, 'item'):
                    phase = phase.item()
                
                fourier_data['amplitudes'][band].append(float(amp))
                fourier_data['phases'][band].append(float(phase))
        
        # Model coefficient evolution
        evolution_models = self._model_coefficient_evolution(fourier_data)
        
        return {
            'fourier_data': fourier_data,
            'evolution_models': evolution_models,
            'last_price': prices[-1],
            'window_days': window_days
        }
    
    def _extract_fourier_coefficients(self, signal):
        """
        Extract Fourier coefficients from a price signal.
        
        Args:
            signal: 1D array of prices
            
        Returns:
            Dictionary of coefficients by frequency band
        """
        # Normalize signal
        signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # Compute FFT
        fft_values = rfft(signal_normalized)
        fft_freqs = rfftfreq(len(signal), d=1.0)  # Daily frequency
        
        # Extract coefficients by frequency band
        coeffs = {}
        for band_name, (freq_min, freq_max) in self.frequency_bands.items():
            # Find frequencies in this band
            # Make sure mask length matches fft_freqs length
            mask = (fft_freqs >= freq_min) & (fft_freqs <= freq_max)
            
            # Apply mask to get coefficients in this band
            band_indices = np.where(mask)[0]
            
            if len(band_indices) > 0:
                # Use dominant coefficient in this band
                band_coeffs = fft_values[band_indices]
                amplitudes = np.abs(band_coeffs)
                max_idx = np.argmax(amplitudes)
                
                coeffs[band_name] = {
                    'amplitude': amplitudes[max_idx],
                    'phase': np.angle(band_coeffs[max_idx]),
                    'frequency': fft_freqs[band_indices[max_idx]]
                }
            else:
                coeffs[band_name] = {
                    'amplitude': 0.0,
                    'phase': 0.0,
                    'frequency': (freq_min + freq_max) / 2
                }
        
        return coeffs
    
    def _model_coefficient_evolution(self, fourier_data):
        """
        Model how Fourier coefficients change over time.
        
        Args:
            fourier_data: Dictionary with amplitude/phase time series
            
        Returns:
            Dictionary of LinearRegression models
        """
        evolution_models = {}
        X = np.array(fourier_data['window_centers']).reshape(-1, 1)
        
        for band in self.frequency_bands:
            # Model amplitude evolution
            y_amp = np.array(fourier_data['amplitudes'][band])
            amp_model = LinearRegression()
            amp_model.fit(X, y_amp)
            
            # Model phase evolution (with wrapping)
            y_phase = np.array(fourier_data['phases'][band])
            phase_model = LinearRegression()
            phase_model.fit(X, np.sin(y_phase))  # Model sin/cos separately
            
            evolution_models[band] = {
                'amplitude': amp_model,
                'phase': phase_model,
                'last_amplitude': y_amp[-1],
                'last_phase': y_phase[-1]
            }
        
        return evolution_models
    
    def _calculate_fourier_covariance(self):
        """
        Calculate covariance of Fourier components between assets.
        This measures how frequency components move together across assets.
        """
        # For each frequency band, calculate covariance of amplitudes
        for band in self.frequency_bands:
            # Collect amplitude time series for all assets
            amplitude_series = {}
            min_length = float('inf')
            
            # First pass: collect data and find minimum length
            for asset_name in self.asset_names:
                amplitudes = self.asset_models[asset_name]['fourier_data']['amplitudes'][band]
                amplitude_series[asset_name] = amplitudes
                min_length = min(min_length, len(amplitudes))
            
            # Second pass: truncate all to same length
            amplitude_matrix = []
            for asset_name in self.asset_names:
                amplitudes_truncated = amplitude_series[asset_name][:min_length]
                amplitude_matrix.append(amplitudes_truncated)
            
            # Convert to numpy array (assets x time_windows)
            amplitude_matrix = np.array(amplitude_matrix)
            
            # Calculate covariance matrix (assets x assets)
            if amplitude_matrix.shape[1] > 1:  # Need at least 2 time points
                cov_matrix = np.cov(amplitude_matrix)
            else:
                cov_matrix = np.eye(len(self.asset_names))
            
            # Store covariance for this frequency band
            self.fourier_covariance[band] = pd.DataFrame(
                cov_matrix,
                index=self.asset_names,
                columns=self.asset_names
            )
    
    def _calculate_influence_weights(self):
        """
        Calculate influence weight matrix from Fourier covariance.
        
        Each asset's prediction should be:
        - 70-85% explained by its own Fourier patterns
        - 15-30% influenced by other assets based on covariance strength
        """
        # Initialize influence weights
        for target_asset in self.asset_names:
            self.influence_weights[target_asset] = {}
            
            # Calculate total covariance with other assets across all frequency bands
            total_covariance = {}
            for source_asset in self.asset_names:
                cov_sum = 0
                for band in self.frequency_bands:
                    cov = self.fourier_covariance[band].loc[target_asset, source_asset]
                    cov_sum += abs(cov)
                total_covariance[source_asset] = cov_sum
            
            # Normalize to get influence weights
            # Self-weight should be dominant (85%), others share remaining 15%
            self_weight = 0.85
            cross_weight_budget = 1.0 - self_weight
            
            # Calculate cross-asset weights proportional to covariance
            other_assets = [a for a in self.asset_names if a != target_asset]
            if len(other_assets) > 0:
                other_cov_sum = sum(total_covariance[a] for a in other_assets)
                
                for source_asset in self.asset_names:
                    if source_asset == target_asset:
                        self.influence_weights[target_asset][source_asset] = self_weight
                    else:
                        if other_cov_sum > 0:
                            weight = cross_weight_budget * (total_covariance[source_asset] / other_cov_sum)
                        else:
                            weight = cross_weight_budget / len(other_assets)
                        self.influence_weights[target_asset][source_asset] = weight
            else:
                self.influence_weights[target_asset][target_asset] = 1.0
    
    def predict(self, asset_name, forecast_days):
        """
        Generate predictions for a specific asset.
        
        Args:
            asset_name: Name of the asset to predict
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with predictions and metadata
        """
        if asset_name not in self.asset_models:
            raise ValueError(f"Asset {asset_name} not in fitted models")
        
        model = self.asset_models[asset_name]
        
        # Project coefficients into future
        future_coeffs = self._project_coefficients(model, forecast_days)
        
        # Reconstruct signal from projected coefficients
        predictions = self._reconstruct_signal(future_coeffs, forecast_days)
        
        # Apply cross-asset adjustments based on influence weights
        predictions = self._apply_cross_asset_adjustments(
            asset_name, predictions, forecast_days
        )
        
        # Calculate confidence based on coefficient stability
        confidence = self._calculate_confidence(model)
        
        # Dominant frequencies for this asset
        dominant_freqs = []
        for band, band_model in model['evolution_models'].items():
            dominant_freqs.append({
                'band': band,
                'amplitude': float(band_model['last_amplitude'])
            })
        dominant_freqs.sort(key=lambda x: x['amplitude'], reverse=True)
        
        return {
            'predictions': predictions,
            'confidence': [confidence] * len(predictions),
            'dominant_frequencies': [f['band'] for f in dominant_freqs]
        }
    
    def _project_coefficients(self, model, forecast_days):
        """
        Project Fourier coefficients into the future.
        
        Args:
            model: Asset model dictionary
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary of projected coefficients
        """
        evolution_models = model['evolution_models']
        window_centers = model['fourier_data']['window_centers']
        last_center = window_centers[-1]
        
        # Project to future time points
        future_times = np.arange(last_center, last_center + forecast_days).reshape(-1, 1)
        
        projected = {}
        for band, band_models in evolution_models.items():
            # Project amplitude
            amp_pred = band_models['amplitude'].predict(future_times)
            amp_pred = np.maximum(amp_pred, 0)  # Amplitude must be positive
            
            # Project phase (simplified - use last phase)
            phase_pred = np.full(len(future_times), band_models['last_phase'])
            
            projected[band] = {
                'amplitudes': amp_pred,
                'phases': phase_pred
            }
        
        return projected
    
    def _reconstruct_signal(self, coeffs, forecast_days):
        """
        Reconstruct price signal from Fourier coefficients.
        
        Args:
            coeffs: Projected Fourier coefficients
            forecast_days: Number of days
            
        Returns:
            Array of predicted price changes
        """
        t = np.arange(forecast_days)
        signal = np.zeros(forecast_days)
        
        # Sum contributions from each frequency band
        for band, band_coeffs in coeffs.items():
            freq_min, freq_max = self.frequency_bands[band]
            freq = (freq_min + freq_max) / 2  # Use center frequency
            
            for i in range(len(band_coeffs['amplitudes'])):
                amp = band_coeffs['amplitudes'][i]
                phase = band_coeffs['phases'][i]
                
                # Add sinusoidal component
                signal += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Normalize to reasonable price change range (±5% max)
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * 0.05
        
        return signal
    
    def _apply_cross_asset_adjustments(self, target_asset, base_predictions, forecast_days):
        """
        Apply cross-asset influence based on covariance weights.
        
        Args:
            target_asset: Asset being predicted
            base_predictions: Base predictions from own Fourier analysis
            forecast_days: Number of days
            
        Returns:
            Adjusted predictions
        """
        # Get influence weights for this asset
        weights = self.influence_weights[target_asset]
        
        # Start with self-weighted base predictions
        adjusted = base_predictions * weights[target_asset]
        
        # Add cross-asset influences
        for source_asset in self.asset_names:
            if source_asset == target_asset:
                continue
            
            # Get source asset's predictions
            source_model = self.asset_models[source_asset]
            source_coeffs = self._project_coefficients(source_model, forecast_days)
            source_signal = self._reconstruct_signal(source_coeffs, forecast_days)
            
            # Add weighted influence
            weight = weights[source_asset]
            adjusted += source_signal * weight
        
        return adjusted
    
    def _calculate_confidence(self, model):
        """
        Calculate prediction confidence based on coefficient stability.
        
        Args:
            model: Asset model dictionary
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate R² for amplitude evolution models
        r2_scores = []
        for band, band_models in model['evolution_models'].items():
            amp_model = band_models['amplitude']
            if hasattr(amp_model, 'score'):
                # Get training data
                fourier_data = model['fourier_data']
                X = np.array(fourier_data['window_centers']).reshape(-1, 1)
                y = np.array(fourier_data['amplitudes'][band])
                r2 = amp_model.score(X, y)
                r2_scores.append(max(0, r2))  # Clip negative R²
        
        # Average R² as confidence
        if len(r2_scores) > 0:
            confidence = np.mean(r2_scores)
        else:
            confidence = 0.5
        
        return float(confidence)


def predict_all_assets(historical_data, forecast_days=7):
    """
    Simplified interface to predict all assets for a given forecast period.
    
    Args:
        historical_data: DataFrame with columns [Gold, Bitcoin, Oil, S&P 500] or
                        Dictionary of {asset_name: DataFrame with 'Close' column}
        forecast_days: Number of days to forecast (7 for week, 30 for month, 365 for year)
    
    Returns:
        Dictionary with predictions for each asset:
        {
            'Gold': {'weekly': prices, 'confidence': value},
            'Bitcoin': {...},
            'Oil': {...},
            'S&P 500': {...}
        }
    """
    # Convert DataFrame to dictionary format if needed
    if isinstance(historical_data, pd.DataFrame):
        asset_data = {}
        for asset_name in ['Gold', 'Bitcoin', 'Oil', 'S&P 500']:
            if asset_name in historical_data.columns:
                df = pd.DataFrame({'Close': historical_data[asset_name]})
                asset_data[asset_name] = df
    else:
        asset_data = historical_data
    
    # Initialize and fit Bachata model
    predictor = BachataPredictor(window_months=6, lookback_years=5)
    predictor.fit(asset_data)
    
    # Generate predictions for all assets
    results = {}
    for asset_name in ['Gold', 'Bitcoin', 'Oil', 'S&P 500']:
        if asset_name in asset_data:
            prediction = predictor.predict(asset_name, forecast_days)
            
            # Get current price
            current_price = asset_data[asset_name]['Close'].iloc[-1]
            
            # Convert price changes to absolute prices
            predicted_prices = current_price + np.cumsum(prediction['predictions'])
            
            results[asset_name] = {
                'prices': [float(p) for p in predicted_prices],
                'confidence': float(prediction['confidence'][0]),
                'dominant_frequencies': prediction['dominant_frequencies']
            }
    
    return results
