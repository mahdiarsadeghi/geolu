"""
Bachata - Fourier Series Analysis Algorithm
Copyright (c) 2026 Geolu
Licensed under Proprietary License with Educational Use
See LICENSE file for terms and conditions.

Bachata uses multi-scale Fourier series decomposition to identify cyclical patterns
in market price movements. It tracks coefficient evolution over time and uses
cross-asset dependencies to generate robust predictions.

Strategy:
1. Fourier analysis at multiple time scales (hourly, daily, weekly, monthly)
2. 6-month rolling windows over 5-year history
3. Linear modeling of coefficient evolution
4. Weight adjustment based on estimation quality
5. Multi-asset dependency analysis (Gold/Oil/Bitcoin/S&P 500)
6. Inflation pattern detection across all assets
7. Robust projection using weighted ensemble
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import fft
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


class BachataPredictor:
    """
    Advanced Fourier series-based prediction algorithm with temporal evolution modeling.
    
    The Bachata algorithm performs multi-scale Fourier decomposition on historical data,
    tracks how frequency components evolve over time, and uses cross-asset relationships
    to generate robust market predictions.
    """
    
    def __init__(self, window_months: int = 6, lookback_years: int = 5):
        """
        Initialize the Bachata predictor.
        
        Args:
            window_months: Size of rolling window in months (default: 6)
            lookback_years: Years of historical data to analyze (default: 5)
        """
        self.window_months = window_months
        self.lookback_years = lookback_years
        self.window_days = window_months * 30  # Approximate
        
        # Fourier analysis results
        self.frequency_bands = {
            'hourly': (1/24, 1/12),      # Sub-daily fluctuations
            'daily': (1/7, 1/2),         # Daily to few-day cycles
            'weekly': (1/30, 1/7),       # Weekly cycles
            'monthly': (1/180, 1/30)     # Monthly cycles
        }
        
        # Coefficient evolution
        self.coefficient_models = {}
        self.coefficient_weights = {}
        self.coefficient_history = []
        
        # Multi-asset analysis
        self.asset_correlations = {}
        self.inflation_signal = None
        self.cross_asset_fourier = {}
        self.fourier_covariance = {}  # Covariance of Fourier components between assets
        self.influence_weights = {}   # How much each asset influences others
        
        # Prediction components
        self.last_price = None
        self.trend_component = None
        self.seasonal_components = {}
        
    def fit(self, asset_data: Dict[str, pd.DataFrame]) -> None:
        """
        Fit the Fourier series model to multi-asset historical data.
        
        Args:
            asset_data: Dictionary with asset names as keys and DataFrames with 
                       'Close' prices and DatetimeIndex as values
        """
        # Extract and align data
        self.asset_names = list(asset_data.keys())
        aligned_data = self._align_asset_data(asset_data)
        
        # Perform rolling window Fourier analysis
        self._analyze_rolling_windows(aligned_data)
        
        # Model coefficient evolution over time
        self._model_coefficient_evolution()
        
        # Analyze cross-asset dependencies
        self._analyze_cross_asset_dependencies(aligned_data)
        
        # Calculate Fourier component covariance
        self._calculate_fourier_covariance()
        
        # Determine influence weights
        self._calculate_influence_weights()
        
        # Detect inflation patterns
        self._detect_inflation_patterns(aligned_data)
        
    def predict(self, asset_name: str, forecast_days: int) -> Dict[str, np.ndarray]:
        """
        Generate price predictions using evolved Fourier coefficients.
        
        Args:
            asset_name: Name of asset to predict
            forecast_days: Number of days to forecast ahead
            
        Returns:
            Dictionary with predictions and confidence metrics
        """
        if asset_name not in self.coefficient_models:
            # Fallback if asset not found
            return {
                'predictions': np.zeros(forecast_days),
                'confidence': np.array([0.0]),
                'dominant_frequencies': [],
                'asset_weights': {}
            }
        
        # Project future coefficients for THIS ASSET ONLY
        future_coefficients = self._project_coefficients_for_asset(asset_name, forecast_days)
        
        # Generate base prediction from THIS ASSET'S Fourier series
        base_prediction = self._reconstruct_signal_for_asset(
            asset_name, future_coefficients, forecast_days
        )
        
        # Apply cross-asset adjustments (subtle influence from other assets)
        adjusted_prediction = self._apply_cross_asset_adjustments(
            base_prediction, asset_name, forecast_days
        )
        
        # Apply inflation signal correction
        final_prediction = self._apply_inflation_correction(
            adjusted_prediction, forecast_days
        )
        
        # Calculate confidence intervals
        confidence = self._calculate_confidence_for_asset(asset_name, future_coefficients)
        
        return {
            'predictions': final_prediction,
            'confidence': confidence,
            'dominant_frequencies': self.get_dominant_frequencies_for_asset(asset_name),
            'asset_weights': self._get_asset_contribution_weights(asset_name)
        }
    
    def _align_asset_data(self, asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align all asset data to common date index."""
        aligned = pd.DataFrame()
        
        for asset_name, df in asset_data.items():
            if 'Close' in df.columns:
                aligned[asset_name] = df['Close']
        
        # Forward fill missing values and drop NaN
        aligned = aligned.fillna(method='ffill').dropna()
        
        # Limit to lookback period
        if len(aligned) > self.lookback_years * 252:  # ~252 trading days/year
            aligned = aligned.tail(self.lookback_years * 252)
        
        return aligned
    
    def _analyze_rolling_windows(self, data: pd.DataFrame) -> None:
        """
        Perform Fourier analysis on rolling 6-month windows.
        
        Extracts coefficients for each frequency band and tracks their evolution.
        """
        n_windows = max(1, len(data) // self.window_days)
        
        for asset in data.columns:
            self.coefficient_history.append({
                'asset': asset,
                'windows': [],
                'timestamps': []
            })
            
            # Analyze each rolling window
            for i in range(n_windows):
                start_idx = i * self.window_days
                end_idx = min(start_idx + self.window_days, len(data))
                
                if end_idx - start_idx < 30:  # Skip if too small
                    continue
                
                window_data = data[asset].iloc[start_idx:end_idx].values
                window_time = data.index[end_idx - 1]
                
                # Perform FFT
                coefficients = self._extract_fourier_coefficients(window_data)
                
                # Store with timestamp
                self.coefficient_history[-1]['windows'].append(coefficients)
                self.coefficient_history[-1]['timestamps'].append(window_time)
    
    def _extract_fourier_coefficients(self, signal: np.ndarray) -> Dict[str, complex]:
        """
        Extract Fourier coefficients for different frequency bands.
        
        Args:
            signal: Time series data
            
        Returns:
            Dictionary of frequency band coefficients
        """
        # Detrend signal
        signal_detrended = signal - np.linspace(signal[0], signal[-1], len(signal))
        
        # Perform FFT
        fft_result = fft.rfft(signal_detrended)
        freqs = fft.rfftfreq(len(signal_detrended))
        
        # Extract coefficients by frequency band
        coefficients = {}
        
        for band_name, (freq_min, freq_max) in self.frequency_bands.items():
            # Find indices in this frequency range
            band_mask = (freqs >= freq_min) & (freqs <= freq_max)
            
            if np.any(band_mask):
                # Get dominant coefficient in this band
                band_coeffs = fft_result[band_mask]
                band_freqs = freqs[band_mask]
                
                # Find peak
                amplitudes = np.abs(band_coeffs)
                if len(amplitudes) > 0:
                    peak_idx = np.argmax(amplitudes)
                    coefficients[band_name] = {
                        'coefficient': band_coeffs[peak_idx],
                        'frequency': band_freqs[peak_idx],
                        'amplitude': amplitudes[peak_idx]
                    }
        
        return coefficients
    
    def _model_coefficient_evolution(self) -> None:
        """
        Fit linear models to track how Fourier coefficients change over time.
        
        Creates predictive models for coefficient evolution and calculates
        confidence weights based on fit quality.
        """
        for asset_history in self.coefficient_history:
            asset_name = asset_history['asset']
            windows = asset_history['windows']
            timestamps = asset_history['timestamps']
            
            if len(windows) < 3:  # Need at least 3 windows
                continue
            
            self.coefficient_models[asset_name] = {}
            self.coefficient_weights[asset_name] = {}
            
            # Convert timestamps to numeric (days since first)
            time_numeric = np.array([
                (t - timestamps[0]).total_seconds() / (24 * 3600) 
                for t in timestamps
            ]).reshape(-1, 1)
            
            # Model each frequency band
            for band_name in self.frequency_bands.keys():
                # Extract amplitudes over time
                amplitudes = []
                phases = []
                
                for window in windows:
                    if band_name in window:
                        coeff = window[band_name]['coefficient']
                        amplitudes.append(np.abs(coeff))
                        phases.append(np.angle(coeff))
                    else:
                        amplitudes.append(0)
                        phases.append(0)
                
                amplitudes = np.array(amplitudes)
                phases = np.array(phases)
                
                if len(amplitudes) < 3 or np.all(amplitudes == 0):
                    continue
                
                # Fit linear regression for amplitude evolution
                amp_model = LinearRegression()
                amp_model.fit(time_numeric, amplitudes)
                amp_predictions = amp_model.predict(time_numeric)
                amp_r2 = r2_score(amplitudes, amp_predictions)
                
                # Fit linear regression for phase evolution
                phase_model = LinearRegression()
                phase_model.fit(time_numeric, phases)
                phase_predictions = phase_model.predict(time_numeric)
                phase_r2 = r2_score(phases, phase_predictions)
                
                # Store models
                self.coefficient_models[asset_name][band_name] = {
                    'amplitude_model': amp_model,
                    'phase_model': phase_model,
                    'last_time': time_numeric[-1][0]
                }
                
                # Calculate weight based on fit quality
                # Lower weight if R² is poor
                weight = (amp_r2 + phase_r2) / 2
                weight = max(0.1, min(1.0, weight))  # Clamp between 0.1 and 1.0
                
                self.coefficient_weights[asset_name][band_name] = weight
    
    def _analyze_cross_asset_dependencies(self, data: pd.DataFrame) -> None:
        """
        Analyze correlations and dependencies between assets.
        
        Identifies relationships that can inform predictions.
        """
        # Calculate correlation matrix
        self.asset_correlations = data.corr()
        
        # Perform cross-asset Fourier analysis
        for asset1 in data.columns:
            for asset2 in data.columns:
                if asset1 >= asset2:  # Avoid duplicates
                    continue
                
                # Calculate cross-correlation in frequency domain
                signal1 = data[asset1].values
                signal2 = data[asset2].values
                
                fft1 = fft.rfft(signal1 - signal1.mean())
                fft2 = fft.rfft(signal2 - signal2.mean())
                
                # Cross-power spectrum
                cross_power = fft1 * np.conj(fft2)
                coherence = np.abs(cross_power) / (np.abs(fft1) * np.abs(fft2) + 1e-10)
                
                key = f"{asset1}_{asset2}"
                self.cross_asset_fourier[key] = {
                    'coherence': coherence,
                    'mean_coherence': np.mean(coherence)
                }
    
    def _calculate_fourier_covariance(self) -> None:
        """
        Calculate covariance of Fourier components between assets.
        
        Determines how Fourier patterns in one asset relate to patterns in others.
        """
        # Extract amplitude matrices for each frequency band
        band_amplitudes = {band: {} for band in self.frequency_bands.keys()}
        
        for asset_history in self.coefficient_history:
            asset_name = asset_history['asset']
            
            for band_name in self.frequency_bands.keys():
                # Extract amplitudes across all windows for this band
                amplitudes = []
                for window in asset_history['windows']:
                    if band_name in window:
                        amplitudes.append(window[band_name]['amplitude'])
                    else:
                        amplitudes.append(0)
                
                if len(amplitudes) > 0:
                    band_amplitudes[band_name][asset_name] = np.array(amplitudes)
        
        # Calculate covariance for each frequency band
        for band_name, asset_amps in band_amplitudes.items():
            if len(asset_amps) < 2:  # Need at least 2 assets
                continue
            
            # Create matrix of amplitudes (assets x time_windows)
            asset_names = list(asset_amps.keys())
            min_length = min(len(amps) for amps in asset_amps.values())
            
            amp_matrix = np.array([
                asset_amps[name][:min_length] for name in asset_names
            ])
            
            # Calculate covariance matrix
            if amp_matrix.shape[1] > 1:
                cov_matrix = np.cov(amp_matrix)
                
                # Store covariance relationships
                for i, asset1 in enumerate(asset_names):
                    for j, asset2 in enumerate(asset_names):
                        key = f"{asset1}_{asset2}_{band_name}"
                        self.fourier_covariance[key] = cov_matrix[i, j]
    
    def _calculate_influence_weights(self) -> None:
        """
        Calculate how much each asset influences others based on Fourier covariance.
        
        Creates weights matrix: influence_weights[target_asset][source_asset] = weight
        Higher weight means source asset has more predictive power for target.
        """
        for asset1 in self.asset_names:
            self.influence_weights[asset1] = {}
            
            # Self-weight is always dominant (0.7-0.9)
            total_variance = 0
            cross_variances = {}
            
            for asset2 in self.asset_names:
                # Sum covariance across all frequency bands
                total_cov = 0
                count = 0
                
                for band_name in self.frequency_bands.keys():
                    key = f"{asset1}_{asset2}_{band_name}"
                    if key in self.fourier_covariance:
                        total_cov += abs(self.fourier_covariance[key])
                        count += 1
                
                if count > 0:
                    avg_cov = total_cov / count
                    cross_variances[asset2] = avg_cov
                    total_variance += avg_cov
            
            # Normalize to get weights (self gets at least 70%)
            if total_variance > 0:
                for asset2 in self.asset_names:
                    if asset2 == asset1:
                        # Self-weight: dominant influence
                        self.influence_weights[asset1][asset2] = 0.85
                    else:
                        # Cross-weight: proportional to covariance, scaled to remaining 15%
                        base_weight = cross_variances.get(asset2, 0) / total_variance
                        self.influence_weights[asset1][asset2] = base_weight * 0.15
            else:
                # Fallback: 100% self-weight
                for asset2 in self.asset_names:
                    self.influence_weights[asset1][asset2] = 1.0 if asset2 == asset1 else 0.0
    
    def _detect_inflation_patterns(self, data: pd.DataFrame) -> None:
        """
        Detect inflation patterns by analyzing synchronized movements across all assets.
        
        Rising prices across Gold, Oil, and stocks may indicate inflation.
        """
        # Calculate percentage changes
        pct_changes = data.pct_change().fillna(0)
        
        # Create inflation signal (average of all asset movements)
        self.inflation_signal = pct_changes.mean(axis=1)
        
        # Smooth the signal
        window_size = 30  # 30-day moving average
        if len(self.inflation_signal) >= window_size:
            self.inflation_signal = self.inflation_signal.rolling(
                window=window_size, min_periods=1
            ).mean()
    
    def _project_coefficients_for_asset(self, asset_name: str, forecast_days: int) -> Dict[str, Dict]:
        """
        Project Fourier coefficients for a specific asset into the future.
        
        Args:
            asset_name: Name of asset to project
            forecast_days: Number of days to project ahead
            
        Returns:
            Dictionary of projected coefficients for this asset only
        """
        if asset_name not in self.coefficient_models:
            return {}
        
        projected = {}
        band_models = self.coefficient_models[asset_name]
        
        for band_name, models in band_models.items():
            # Project time forward
            last_time = models['last_time']
            future_time = np.array([[last_time + forecast_days]])
            
            # Project amplitude and phase
            future_amp = models['amplitude_model'].predict(future_time)[0]
            future_phase = models['phase_model'].predict(future_time)[0]
            
            # Reconstruct complex coefficient
            future_coeff = future_amp * np.exp(1j * future_phase)
            
            # Apply weight
            weight = self.coefficient_weights[asset_name].get(band_name, 0.5)
            
            projected[band_name] = {
                'coefficient': future_coeff,
                'weight': weight
            }
        
        return projected
    
    def _reconstruct_signal_for_asset(self, asset_name: str, coefficients: Dict, 
                                     forecast_days: int) -> np.ndarray:
        """
        Reconstruct time series from Fourier coefficients for a specific asset.
        
        Args:
            asset_name: Name of asset
            coefficients: Projected Fourier coefficients for this asset
            forecast_days: Number of days to generate
            
        Returns:
            Reconstructed signal as price changes
        """
        t = np.arange(forecast_days)
        signal = np.zeros(forecast_days)
        
        total_weight = 0
        for band_name, band_data in coefficients.items():
            coeff = band_data['coefficient']
            weight = band_data['weight']
            
            # Get frequency from band center
            freq_min, freq_max = self.frequency_bands[band_name]
            freq = (freq_min + freq_max) / 2
            
            # Add weighted sinusoidal component
            amplitude = np.abs(coeff) * weight
            phtargeted adjustments based on Fourier covariance with other assets.
        
        Uses calculated influence weights to determine how much each asset affects the target.
        
        Args:
            prediction: Base prediction for target asset
            asset_name: Target asset
            forecast_days: Forecast horizon
            
        Returns:
            Adjusted prediction with weighted cross-asset influences
        """
        if asset_name not in self.influence_weights:
            return prediction
        
        adjustment = np.zeros(forecast_days)
        
        # Get influence weights for this asset
        influences = self.influence_weights[asset_name]
        
        # Apply weighted influences from other assets
        for source_asset, weight in influences.items():
            if source_asset == asset_name or weight < 0.01:  # Skip self and negligible weights
                continue
            
            # Get the trend from source asset's recent prediction
            # This represents how source asset's Fourier patterns influence target
            if source_asset in self.coefficient_models:
                source_coeffs = self._project_coefficients_for_asset(source_asset, forecast_days)
                source_signal = self._reconstruct_signal_for_asset(source_asset, source_coeffs, forecast_days)
                
                # Apply weighted influence (weight already incorporates covariance)
                adjustment += source_signal * weight
                # Project time forward
                last_time = models['last_time']
                future_time = np.array([[last_time + forecast_days]])
                
                # Project amplitude and phase
                future_amp = models['amplitude_model'].predict(future_time)[0]
                future_phase = models['phase_model'].predict(future_time)[0]
                
                # Reconstruct complex coefficient
                future_coeff = future_amp * np.exp(1j * future_phase)
                
                # Apply weight
                weight = self.coefficient_weights[asset_name].get(band_name, 0.5)
                
                projected[asset_name][band_name] = {
                    'coefficient': future_coeff,
                    'weight': weight
                }
        
        return projected
    
    def _reconstruct_signal(self, coefficients: Dict, forecast_days: int) -> np.ndarray:
        """
        Reconstruct time series from Fourier coefficients.
        
        Args:
            coefficients: Projected Fourier coefficients
            forecast_days: Number of days to generate
            
        Returns:
            Reconstructed signal as price changes (not absolute prices)
        """
        t = np.arange(forecast_days)
        signal = np.zeros(forecast_days)
        
        # Sum contributions from all frequency bands
        total_weight = 0
        for asset_name, bands in coefficients.items():
            for band_name, band_data in bands.items():
                coeff = band_data['coefficient']
                weight = band_data['weight']
                
                # Get frequency from band center
                freq_min, freq_max = self.frequency_bands[band_name]
                freq = (freq_min + freq_max) / 2
                
                # Add weighted sinusoidal component
                amplitude = np.abs(coeff) * weight
                phase = np.angle(coeff)
                signal += amplitude * np.cos(2 * np.pi * freq * t + phase)
                total_weight += weight
        
        # Normalize by total weight and scale down dramatically
        # These are meant to be small daily/weekly/monthly price changes
        if total_weight > 0:
            signal = signal / total_weight
        
        # Scale to reasonable price change range (e.g., ±5% max over the period)
        max_change_pct = 0.05  # 5% maximum total change
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * max_change_pct * 100  # Convert to dollar amount per $100
        
        return signal
    
    def _apply_cross_asset_adjustments(self, prediction: np.ndarray, 
                                      asset_name: str, forecast_days: int) -> np.ndarray:
        """
        Apply subtle adjustments based on cross-asset relationships.
        
        Args:
            prediction: Base prediction for target asset
            asset_name: Target asset
            forecast_days: Forecast horizon
            
        Returns:
            Adjusted prediction (subtle changes only)
        """
        if asset_name not in self.asset_correlations.index:
            return prediction
        
        adjustment = np.zeros(forecast_days)
        
        # Apply correlated asset influences (very subtle - max 2% influence)
        for other_asset in self.asset_names:
            if other_asset == asset_name:
                continue
            
            if other_asset not in self.asset_correlations.columns:
                continue
            
            # Get correlation strength
            correlation = self.asset_correlations.loc[asset_name, other_asset]
            
            # Apply very small proportional adjustment based on correlation
            # Positive correlation: if we predict increase, others might too
            # Negative correlation: if we predict increase, others might decrease
            trend = prediction[-1] - prediction[0] if len(prediction) > 1 else 0
            adjustment += np.linspace(0, trend * correlation * 0.02, forecast_days)
        
        return prediction + adjustment
    
    def _apply_inflation_correction(self, prediction: np.ndarray, 
                                   forecast_days: int) -> np.ndarray:
        """
        Apply inflation pattern correction to predictions.
        
        Args:
            prediction: Base prediction
            forecast_days: Forecast horizon
            
        Returns:
            Inflation-adjusted prediction
        """
        if self.inflation_signal is None or len(self.inflation_signal) == 0:
            return prediction
        
        # Get recent inflation trend
        recent_inflation = self.inflation_signal.tail(30).mean()
        
        # Project inflation effect forward (linearly decaying influence)
        inflation_effect = recent_inflation * np.linspace(1, 0.5, forecast_days)
        
        # Apply to prediction
        return prediction * (1 + inflation_effect)
    
    def _calculate_confidence_for_asset(self, asset_name: str, 
                                       coefficients: Dict) -> np.ndarray:
        """
        Calculate prediction confidence for a specific asset.
        
        Args:
            asset_name: Asset name
            coefficients: Projected coefficients with weights
            
        Returns:
            Confidence scores (0-1)
        """
        Get contribution weights showing how much each asset influences the target.
        
        Returns both Fourier band weights and cross-asset influence weights.
        """
        weights = {
            'fourier_bands': {},
            'asset_influences': {}
        }
        
        # Fourier band weights (own patterns)
        if asset_name in self.coefficient_weights:
            for band_name, weight in self.coefficient_weights[asset_name].items():
                weights['fourier_bands'][band_name] = weight
        
        # Cross-asset influence weights
        if asset_name in self.influence_weights:
            weights['asset_influences'] = self.influence_weights[asset_name].copy()
        
        # Average weight as overall confidence
        return np.array([np.mean(weights)])
    
    def _calculate_confidence(self, coefficients: Dict) -> np.ndarray:
        """
        Calculate prediction confidence based on coefficient weights.
        
        Args:
            coefficients: Projected coefficients with weights
            
        Returns:
            Confidence scores (0-1)
        """
        weights = []
        
        for asset_bands in coefficients.values():
            for band_data in asset_bands.values():
                weights.append(band_data['weight'])
        
        if len(weights) == 0:
            return np.array([0.5])
        
        # Average weight as overall confidence
        return np.array([np.mean(weights)])
    
    def get_dominant_frequencies_for_asset(self, asset_name: str, 
                                          n_components: int = 5) -> List[Tuple[str, float, float]]:
        """
        Extract dominant frequency components for a specific asset.
        
        Args:
            asset_name: Asset name
            n_components: Number of top frequency components to return
            
        Returns:
            List of (band_name, frequency, amplitude) tuples
        """
        all_components = []
        
        for asset_history in self.coefficient_history:
            if asset_history['asset'] != asset_name:
                continue
                
            if not asset_history['windows']:
                continue
            
            # Get most recent window
            latest_window = asset_history['windows'][-1]
            
            for band_name, band_data in latest_window.items():
                all_components.append((
                    band_name,
                    band_data['frequency'],
                    band_data['amplitude']
                ))
        
        # Sort by amplitude and return top N
        all_components.sort(key=lambda x: x[2], reverse=True)
        return all_components[:n_components]
    
    def get_dominant_frequencies(self, n_components: int = 5) -> List[Tuple[str, float, float]]:
        """
        Extract the dominant frequency components from the analysis.
        
        Args:
            n_components: Number of top frequency components to return
            
        Returns:
            List of (band_name, frequency, amplitude) tuples
        """
        all_components = []
        
        for asset_history in self.coefficient_history:
            if not asset_history['windows']:
                continue
            
            # Get most recent window
            latest_window = asset_history['windows'][-1]
            
            for band_name, band_data in latest_window.items():
                all_components.append((
                    band_name,
                    band_data['frequency'],
                    band_data['amplitude']
                ))
        
        # Sort by amplitude and return top N
        all_components.sort(key=lambda x: x[2], reverse=True)
        return all_components[:n_components]
    
    def _get_asset_contribution_weights(self, asset_name: str) -> Dict[str, float]:
        """Get contribution weights of different assets to the prediction."""
        weights = {}
        
        if asset_name not in self.coefficient_weights:
            return weights
        
        for band_name, weight in self.coefficient_weights[asset_name].items():
            weights[band_name] = weight
        
        return weights


def bachata_predict(asset_data: Dict[str, pd.DataFrame], 
                   target_asset: str,
                   forecast_days: int) -> Dict[str, np.ndarray]:
    """
    Convenience function for Bachata predictions.
    
    Args:
        asset_data: Dictionary of DataFrames with historical data for all assets
        target_asset: Asset to predict (e.g., 'Gold', 'Bitcoin')
        forecast_days: Number of days to forecast
        
    Returns:
        Dictionary containing predictions and metadata
    """
    predictor = BachataPredictor()
    predictor.fit(asset_data)
    return predictor.predict(target_asset, forecast_days)
