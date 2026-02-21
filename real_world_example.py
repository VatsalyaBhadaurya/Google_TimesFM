"""
Real-World Implementation: Electricity Load Forecasting using TimesFM

This example demonstrates how to use TimesFM for predicting electricity load
in power grids, a critical real-world problem for:
- Grid stability management
- Resource optimization
- Cost reduction
- Peak demand planning

The model learns patterns from historical load data and forecasts future demand.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import timesfm
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ElectricityLoadForecaster:
    """
    Real-world electricity load forecasting using TimesFM.
    
    Typical use case:
    - Utility companies predict 24-hour ahead electricity demand
    - Help balance generation and distribution
    - Optimize renewable energy usage
    """
    
    def __init__(self, model_name: str = "google/timesfm-2.5-200m-pytorch"):
        """Initialize forecaster with TimesFM model."""
        print(f"Loading TimesFM model: {model_name}")
        torch.set_float32_matmul_precision("high")
        
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
        
        # Configure for electricity load forecasting
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=1024,  # Use up to 1024 historical points
                max_horizon=24,    # Forecast 24 hours ahead
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        print("✓ Model loaded and configured")
    
    def generate_synthetic_load_data(
        self, 
        num_days: int = 180,
        num_samples: int = 1
    ) -> np.ndarray:
        """
        Generate realistic synthetic electricity load data.
        
        Real-world load patterns:
        - Daily seasonality (peaks during day, dips at night)
        - Weekly seasonality (higher on weekdays)
        - Random fluctuations (weather, events, anomalies)
        
        Returns:
            (num_samples, num_days * 24): Load in MW
        """
        np.random.seed(42)
        hours = num_days * 24
        load_data = np.zeros((num_samples, hours))
        
        for sample in range(num_samples):
            # Base load (average consumption)
            base_load = 5000  # MW
            
            # Daily pattern (sinusoidal - peaks at noon, dips at night)
            hours_array = np.arange(hours)
            daily_pattern = 1500 * np.sin(2 * np.pi * (hours_array % 24) / 24 - np.pi/2)
            
            # Weekly pattern (higher on weekdays 9-5)
            weekly_pattern = np.zeros(hours)
            for h in range(hours):
                day_of_week = (h // 24) % 7
                hour_of_day = h % 24
                if day_of_week < 5 and 9 <= hour_of_day <= 17:  # Weekday business hours
                    weekly_pattern[h] = 800
            
            # Random fluctuations
            noise = np.random.normal(0, 300, hours)
            
            # Weather effect (seasonal changes)
            seasonal = 500 * np.sin(2 * np.pi * hours_array / (365 * 24))
            
            load_data[sample] = base_load + daily_pattern + weekly_pattern + noise + seasonal
            load_data[sample] = np.maximum(load_data[sample], 3000)  # Min threshold
        
        return load_data
    
    def prepare_data(
        self,
        load_data: np.ndarray,
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Returns:
            train_data, val_data, test_data
        """
        total_length = load_data.shape[1]
        train_size = int(total_length * train_ratio)
        val_size = int(total_length * (train_ratio + 0.1))
        
        train_data = load_data[:, :train_size]
        val_data = load_data[:, train_size:val_size]
        test_data = load_data[:, val_size:]
        
        print(f"\n📊 Data Split:")
        print(f"  Training: {train_data.shape[1]} hours ({train_data.shape[1]/24:.1f} days)")
        print(f"  Validation: {val_data.shape[1]} hours ({val_data.shape[1]/24:.1f} days)")
        print(f"  Testing: {test_data.shape[1]} hours ({test_data.shape[1]/24:.1f} days)")
        
        return train_data, val_data, test_data
    
    def forecast(
        self,
        context_data: np.ndarray,
        horizon: int = 24
    ) -> dict:
        """
        Generate 24-hour ahead electricity load forecast.
        
        Args:
            context_data: Historical load data (hours)
            horizon: Hours to forecast (default 24)
            
        Returns:
            Dictionary with point forecast and uncertainty bounds
        """
        # Ensure proper shape
        if context_data.ndim == 1:
            context_data = context_data.reshape(1, -1)
        
        # Make prediction
        point_forecast = self.model.forecast(
            context=context_data,
            prediction_length=horizon,
        )
        
        # Get quantile forecasts for uncertainty
        quantile_forecast = self.model.forecast(
            context=context_data,
            prediction_length=horizon,
        )
        
        return {
            'point_forecast': point_forecast,
            'quantile_forecast': quantile_forecast,
        }
    
    def evaluate_forecast(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> dict:
        """
        Calculate forecast accuracy metrics.
        
        Metrics:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - MAPE: Mean Absolute Percentage Error
        """
        # Flatten if needed
        if actual.ndim > 1:
            actual = actual.flatten()
        if predicted.ndim > 1:
            predicted = predicted.flatten()
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-5))) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'predictions': predicted,
            'actual': actual
        }
    
    def visualize_forecast(
        self,
        historical: np.ndarray,
        actual: np.ndarray,
        forecast: np.ndarray,
        title: str = "24-Hour Electricity Load Forecast"
    ):
        """Visualize forecast results."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical data
        hours_hist = np.arange(len(historical))
        ax.plot(hours_hist, historical, 'b-', linewidth=2, label='Historical Data', alpha=0.7)
        
        # Actual future data
        hours_future = np.arange(len(historical), len(historical) + len(actual))
        ax.plot(hours_future, actual, 'g-', linewidth=2.5, label='Actual Load (Test)', marker='o')
        
        # Forecast
        ax.plot(hours_future, forecast, 'r--', linewidth=2.5, label='TimesFM Forecast', marker='s')
        
        # Formatting
        ax.axvline(x=len(historical) - 0.5, color='gray', linestyle=':', alpha=0.5, label='Forecast Start')
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Electricity Load (MW)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def run_demonstration():
    """Run complete electricity load forecasting demonstration."""
    
    print("=" * 70)
    print("ELECTRICITY LOAD FORECASTING WITH TIMESFM")
    print("=" * 70)
    print("\n🔌 Real-World Use Case:")
    print("  Predicting power grid electricity demand 24 hours in advance")
    print("  for optimal resource allocation and cost management\n")
    
    # Initialize forecaster
    forecaster = ElectricityLoadForecaster()
    
    # Generate synthetic but realistic electricity load data
    print("\n📈 Generating synthetic electricity load data...")
    load_data = forecaster.generate_synthetic_load_data(num_days=180)
    print(f"✓ Generated {load_data.shape[1]} hours of load data")
    print(f"  Load range: {load_data.min():.0f} - {load_data.max():.0f} MW")
    print(f"  Average load: {load_data.mean():.0f} MW")
    
    # Prepare data
    train_data, val_data, test_data = forecaster.prepare_data(load_data)
    
    # Demo: Forecast on test set
    print("\n🔮 Generating forecasts on test set...")
    
    # Use last part of training data as context
    context_hours = 168  # 1 week of historical data
    forecast_horizon = 24  # 24-hour ahead forecast
    
    # Extract a sample from test data
    sample_context = test_data[0, :context_hours]
    sample_actual = test_data[0, context_hours:context_hours + forecast_horizon]
    
    # Make forecast
    print(f"  Context: last {context_hours} hours of data")
    print(f"  Forecast horizon: {forecast_horizon} hours (1 day)")
    
    forecast_result = forecaster.forecast(sample_context, horizon=forecast_horizon)
    forecast_values = forecast_result['point_forecast'][0, :]
    
    # Evaluate
    metrics = forecaster.evaluate_forecast(sample_actual, forecast_values)
    
    print("\n📊 Forecast Performance Metrics:")
    print(f"  Mean Absolute Error (MAE):           {metrics['MAE']:.1f} MW")
    print(f"  Root Mean Squared Error (RMSE):     {metrics['RMSE']:.1f} MW")
    print(f"  Mean Absolute Percentage Error:      {metrics['MAPE']:.2f}%")
    
    print("\n✅ Real-world applications of this forecast:")
    print("  • Power plant dispatch optimization")
    print("  • Renewable energy integration (wind/solar)")
    print("  • Peak demand management")
    print("  • Energy trading and pricing")
    print("  • Demand response program planning")
    print("  • Infrastructure maintenance scheduling")
    
    # Visualization
    print("\n📉 Generating visualization...")
    fig = forecaster.visualize_forecast(
        sample_context,
        sample_actual,
        forecast_values,
        "Electricity Load: 24-Hour Ahead Forecast"
    )
    
    # Save figure
    output_path = "electricity_load_forecast.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()
