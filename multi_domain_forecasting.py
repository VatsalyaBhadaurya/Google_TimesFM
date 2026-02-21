"""
Multiple Real-World TimesFM Use Cases Implementation

This module demonstrates TimesFM applications across different domains:
1. Stock Price Forecasting - Financial markets
2. Sales Forecasting - Retail & E-commerce
3. Server Load Prediction - Cloud infrastructure
4. Air Quality Forecasting - Environmental monitoring
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import timesfm


class TimeSeriesForecasterBase(ABC):
    """Base class for all TimesFM forecasting applications."""
    
    def __init__(self, model_name: str = "google/timesfm-2.5-200m-pytorch"):
        """Initialize with TimesFM model."""
        torch.set_float32_matmul_precision("high")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
    
    @abstractmethod
    def generate_data(self):
        """Generate domain-specific synthetic data."""
        pass
    
    @abstractmethod
    def business_context(self):
        """Return business use case and value proposition."""
        pass


class StockPriceForecast(TimeSeriesForecasterBase):
    """
    Stock Price Forecasting for Financial Markets.
    
    Real-world value:
    - Investment decision support
    - Risk management
    - Trading algorithm inputs
    - Portfolio optimization
    """
    
    def business_context(self):
        return {
            'domain': 'Financial Markets',
            'use_case': 'Stock Price Prediction',
            'horizon': '5 days ahead',
            'value': [
                '📈 Trading signal generation',
                '💰 Portfolio rebalancing',
                '📊 Risk hedging decisions',
                '🎯 Entry/exit point identification',
                '🤖 Algorithmic trading inputs'
            ],
            'accuracy_target': '±2-3% MAPE'
        }
    
    def generate_data(self, num_days=365):
        """Generate realistic stock price data with trend and volatility."""
        np.random.seed(42)
        hours = num_days * 24  # Intraday data
        
        # Initial price
        price = 100
        prices = [price]
        
        for i in range(hours - 1):
            # Trend component (random walk)
            trend = np.random.normal(0.0005, 0.02)
            # Volatility clustering
            volatility = 0.01 + 0.02 * np.random.random()
            # Mean reversion
            mean_reversion = -0.1 * (price - 100)
            
            price = price * (1 + trend + volatility * np.random.normal(0, 1) + mean_reversion * 0.001)
            if price > 0:
                prices.append(price)
            else:
                prices.append(prices[-1])
        
        return np.array([prices])
    
    def analyze(self, prices):
        """Analyze stock price patterns."""
        returns = np.diff(prices[0]) / prices[0][:-1]
        volatility = np.std(returns) * 100
        
        return {
            'avg_price': np.mean(prices[0]),
            'volatility': volatility,
            'min_price': np.min(prices[0]),
            'max_price': np.max(prices[0]),
            'trend': 'Bullish' if np.mean(returns) > 0 else 'Bearish'
        }


class SalesForecast(TimeSeriesForecasterBase):
    """
    Sales Forecasting for Retail & E-commerce.
    
    Real-world value:
    - Inventory optimization
    - Revenue planning
    - Staffing decisions
    - Marketing budget allocation
    """
    
    def business_context(self):
        return {
            'domain': 'Retail & E-commerce',
            'use_case': 'Daily/Weekly Sales Prediction',
            'horizon': '30 days ahead',
            'value': [
                '📦 Inventory planning',
                '💵 Revenue forecasting',
                '👥 Staffing optimization',
                '🎯 Promotional campaign timing',
                '⚙️ Supply chain coordination'
            ],
            'accuracy_target': '±3-5% MAPE'
        }
    
    def generate_data(self, num_days=365):
        """Generate realistic sales data with seasonality and trends."""
        np.random.seed(42)
        daily_sales = []
        
        for day in range(num_days):
            # Base sales
            base = 1000
            
            # Trend (growth)
            trend = 0.5 * day
            
            # Weekly seasonality (weekends higher)
            day_of_week = day % 7
            weekly = 200 * np.sin(2 * np.pi * day_of_week / 7)
            
            # Monthly seasonality (shopping patterns)
            day_of_month = day % 30
            monthly = 300 * np.sin(2 * np.pi * day_of_month / 30)
            
            # Promotions (random events)
            promo = 500 if np.random.random() < 0.1 else 0
            
            # Noise
            noise = np.random.normal(0, 100)
            
            sales = max(base + trend + weekly + monthly + promo + noise, 100)
            daily_sales.append(sales)
        
        return np.array([daily_sales])
    
    def analyze(self, sales):
        """Analyze sales patterns."""
        return {
            'total_revenue': np.sum(sales[0]),
            'avg_daily_sales': np.mean(sales[0]),
            'peak_sales': np.max(sales[0]),
            'volatility': np.std(sales[0]),
            'growth_rate': (sales[0][-1] - sales[0][0]) / sales[0][0] * 100
        }


class ServerLoadForecast(TimeSeriesForecasterBase):
    """
    Server Load Prediction for Cloud Infrastructure.
    
    Real-world value:
    - Auto-scaling triggers
    - Cost optimization
    - SLA compliance
    - Capacity planning
    """
    
    def business_context(self):
        return {
            'domain': 'Cloud & Infrastructure',
            'use_case': 'Server/CPU Load Prediction',
            'horizon': '1-6 hours ahead',
            'value': [
                '⚡ Auto-scaling decisions',
                '💰 Cost reduction',
                '✅ SLA/uptime guarantees',
                '📊 Capacity planning',
                '🔄 Load balancing optimization'
            ],
            'accuracy_target': '±1-2% MAPE'
        }
    
    def generate_data(self, num_days=30):
        """Generate realistic server load data."""
        np.random.seed(42)
        hours = num_days * 24
        cpu_load = []
        
        for hour in range(hours):
            # Base utilization
            base = 30
            
            # Daily pattern (business hours peak)
            hour_of_day = hour % 24
            daily = 40 * np.sin(2 * np.pi * (hour_of_day - 9) / 24) if 9 <= hour_of_day <= 18 else 0
            
            # Weekly pattern (weekdays higher)
            day_of_week = (hour // 24) % 7
            weekly = 15 if day_of_week < 5 else -10
            
            # Traffic spikes (random events)
            spike = 20 if np.random.random() < 0.05 else 0
            
            # Noise
            noise = np.random.normal(0, 5)
            
            load = max(min(base + daily + weekly + spike + noise, 100), 5)
            cpu_load.append(load)
        
        return np.array([cpu_load])
    
    def analyze(self, loads):
        """Analyze server load patterns."""
        return {
            'avg_load': np.mean(loads[0]),
            'peak_load': np.max(loads[0]),
            'min_load': np.min(loads[0]),
            'high_load_hours': np.sum(loads[0] > 70),
            'capacity_utilization': f"{np.mean(loads[0]):.1f}%"
        }


class AirQualityForecast(TimeSeriesForecasterBase):
    """
    Air Quality Forecasting for Environmental Monitoring.
    
    Real-world value:
    - Public health alerts
    - Urban planning
    - Pollution control
    - Environmental compliance
    """
    
    def business_context(self):
        return {
            'domain': 'Environmental & Public Health',
            'use_case': 'Air Quality Index (AQI) Prediction',
            'horizon': '24-48 hours ahead',
            'value': [
                '🏥 Public health alerts',
                '🌆 Urban planning decisions',
                '🚗 Traffic management',
                '♻️ Pollution control activation',
                '📢 Community communication'
            ],
            'accuracy_target': '±5-8% MAPE'
        }
    
    def generate_data(self, num_days=365):
        """Generate realistic AQI data."""
        np.random.seed(42)
        daily_aqi = []
        
        for day in range(num_days):
            # Base AQI
            base = 50
            
            # Seasonal pattern (worse in winter)
            day_of_year = day % 365
            seasonal = 40 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
            
            # Weekly pattern (weekdays higher)
            day_of_week = day % 7
            weekly = 10 if day_of_week < 5 else -5
            
            # Pollution events
            event = 50 if np.random.random() < 0.05 else 0
            
            # Weather impact
            weather = np.random.normal(0, 10)
            
            aqi = max(min(base + seasonal + weekly + event + weather, 500), 0)
            daily_aqi.append(aqi)
        
        return np.array([daily_aqi])
    
    def analyze(self, aqi):
        """Analyze air quality patterns."""
        good = np.sum(aqi[0] < 50)
        moderate = np.sum((aqi[0] >= 50) & (aqi[0] < 100))
        poor = np.sum(aqi[0] >= 100)
        
        return {
            'avg_aqi': np.mean(aqi[0]),
            'peak_aqi': np.max(aqi[0]),
            'good_days': good,
            'moderate_days': moderate,
            'poor_days': poor,
            'health_concern': 'High' if np.mean(aqi[0]) > 100 else 'Moderate' if np.mean(aqi[0]) > 50 else 'Low'
        }


def showcase_all_use_cases():
    """Showcase all TimesFM use cases."""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "TIMESFM: REAL-WORLD USE CASES")
    print("=" * 80)
    
    use_cases = [
        StockPriceForecast(),
        SalesForecast(),
        ServerLoadForecast(),
        AirQualityForecast()
    ]
    
    for i, forecaster in enumerate(use_cases, 1):
        context = forecaster.business_context()
        data = forecaster.generate_data()
        analysis = forecaster.analyze(data)
        
        print(f"\n{'─' * 80}")
        print(f"USE CASE {i}: {context['domain'].upper()}")
        print(f"{'─' * 80}")
        print(f"📌 Application: {context['use_case']}")
        print(f"⏱️  Forecast Horizon: {context['horizon']}")
        print(f"🎯 Target Accuracy: {context['accuracy_target']}")
        
        print(f"\n💼 Business Value:")
        for value in context['value']:
            print(f"   {value}")
        
        print(f"\n📊 Data Analysis:")
        for key, val in analysis.items():
            if isinstance(val, (int, float)):
                print(f"   • {key}: {val:.2f}")
            else:
                print(f"   • {key}: {val}")
    
    print(f"\n{'=' * 80}")
    print(" " * 25 + "✅ END OF SHOWCASES")
    print("=" * 80)


if __name__ == "__main__":
    showcase_all_use_cases()
