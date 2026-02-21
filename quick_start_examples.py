"""
Quick Start Guide: TimesFM for Real-World Forecasting

This script provides quick-start examples for common use cases.
"""

import numpy as np
import torch
import timesfm
from datetime import datetime, timedelta
import json


def example_1_electricity_load():
    """
    Example 1: Electricity Load Forecasting
    Use Case: Power grid demand prediction for optimal generation dispatch
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: ELECTRICITY LOAD FORECASTING")
    print("="*70)
    
    # Initialize model
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=24,
        normalize_inputs=True,
    ))
    
    # Synthetic electricity load data (MW)
    # Real pattern: 24-hour cycling, weekly pattern, noise
    np.random.seed(42)
    hours = 168 * 4  # 4 weeks of hourly data
    load = []
    
    for h in range(hours):
        base = 5000  # Base load
        daily = 1500 * np.sin(2 * np.pi * (h % 24) / 24)  # Daily pattern
        weekly = 500 if (h // 24) % 7 < 5 else -200  # Weekday bonus
        noise = np.random.normal(0, 300)
        load.append(max(base + daily + weekly + noise, 3000))
    
    load = np.array([load])
    
    # Forecast next 24 hours using last week as context
    context = load[:, -168:]  # Last week (168 hours)
    
    forecast = model.forecast(context, prediction_length=24)
    
    print(f"\n📊 Status:")
    print(f"   Historical data: {context.shape[1]} hours")
    print(f"   Forecast horizon: 24 hours")
    print(f"   Average load: {context.mean():.0f} MW")
    
    print(f"\n🔮 Forecast Results:")
    print(f"   Peak load: {forecast.max():.0f} MW at hour {forecast.argmax()}")
    print(f"   Min load: {forecast.min():.0f} MW at hour {forecast.argmin()}")
    print(f"   Average forecast: {forecast.mean():.0f} MW")
    
    print(f"\n💡 Business Actions:")
    print(f"   ✓ Dispatch coal plants for peak hours")
    print(f"   ✓ Schedule preventive maintenance during low-load hours")
    print(f"   ✓ Activate demand response programs if needed")


def example_2_stock_price():
    """
    Example 2: Stock Price Forecasting
    Use Case: Support investment decisions with 5-day forecast
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: STOCK PRICE FORECASTING")
    print("="*70)
    
    # Initialize model
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=512,
        max_horizon=5,
    ))
    
    # Synthetic stock price data
    np.random.seed(42)
    price = 100
    prices = [price]
    
    for i in range(250):  # 250 trading days
        # Random walk with drift
        change = np.random.normal(0.0005, 0.02)
        price = price * (1 + change)
        prices.append(price)
    
    prices = np.array([prices])
    
    # Forecast next 5 trading days
    context = prices[:, -50:]  # Last 50 trading days
    forecast = model.forecast(context, prediction_length=5)
    
    current_price = context[0, -1]
    
    print(f"\n📊 Current Status:")
    print(f"   Current price: ${current_price:.2f}")
    print(f"   52-week range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"   Volatility (σ): {np.std(np.diff(prices[0])/prices[0][:-1]):.4f}")
    
    print(f"\n🔮 5-Day Forecast:")
    for day, pred in enumerate(forecast[0], 1):
        change = (pred - current_price) / current_price * 100
        direction = "↑" if change > 0 else "↓"
        print(f"   Day {day}: ${pred:.2f}  {direction} {change:+.2f}%")
    
    print(f"\n💡 Trading Signals:")
    if forecast[0][-1] > current_price * 1.02:
        print(f"   ✓ BULLISH: Forecast suggests strong upside")
        print(f"   ✓ Consider: BUY or CALL options")
    elif forecast[0][-1] < current_price * 0.98:
        print(f"   ✗ BEARISH: Forecast suggests downside")
        print(f"   ✓ Consider: SELL or PUT options")
    else:
        print(f"   ≈ NEUTRAL: Forecast suggests consolidation")


def example_3_sales():
    """
    Example 3: Retail Sales Forecasting
    Use Case: Inventory planning and staffing optimization
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: RETAIL SALES FORECASTING")
    print("="*70)
    
    # Initialize model
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=365,
        max_horizon=30,
    ))
    
    # Synthetic daily sales data
    np.random.seed(42)
    daily_sales = []
    
    for day in range(365):
        base = 1000
        trend = 0.5 * day  # Growth
        weekly = 300 * np.sin(2 * np.pi * (day % 7) / 7)
        monthly = 200 * np.sin(2 * np.pi * (day % 30) / 30)
        promo = 500 if np.random.random() < 0.08 else 0
        noise = np.random.normal(0, 100)
        
        sales = max(base + trend + weekly + monthly + promo + noise, 300)
        daily_sales.append(sales)
    
    daily_sales = np.array([daily_sales])
    
    # Forecast next 30 days
    context = daily_sales[:, -90:]  # Last quarter for context
    forecast = model.forecast(context, prediction_length=30)
    
    print(f"\n📊 Historical Analysis:")
    print(f"   Historical average: ${context.mean():.0f}")
    print(f"   Peak day: ${context.max():.0f}")
    print(f"   Trend: +{(daily_sales[0][-1] - daily_sales[0][0]) / 365:.1f} per day")
    
    print(f"\n🔮 30-Day Forecast:")
    print(f"   Forecasted average: ${forecast.mean():.0f}")
    print(f"   Forecasted peak: ${forecast.max():.0f}")
    print(f"   Forecasted total: ${forecast.sum():.0f}")
    
    print(f"\n💼 Inventory & Operations:")
    days_high = np.sum(forecast > forecast.mean() * 1.2)
    days_low = np.sum(forecast < forecast.mean() * 0.8)
    
    print(f"   High sales days (>120% avg): {days_high}")
    print(f"   Low sales days (<80% avg): {days_low}")
    print(f"   ✓ Increase inventory for anticipated peak")
    print(f"   ✓ Schedule promotional staff for high-demand days")
    print(f"   ✓ Plan clearance events for low-demand periods")


def example_4_cpu_load():
    """
    Example 4: Server CPU Load Forecasting
    Use Case: Auto-scaling and cost optimization in cloud
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: SERVER CPU LOAD FORECASTING")
    print("="*70)
    
    # Initialize model
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=168,
        max_horizon=6,
    ))
    
    # Synthetic CPU load data (%)
    np.random.seed(42)
    cpu_load = []
    
    for hour in range(168 * 4):  # 4 weeks
        h = hour % 24
        base = 30
        business_hours = 40 * np.sin(2 * np.pi * (h - 9) / 24) if 9 <= h <= 18 else -20
        spike = 25 if np.random.random() < 0.03 else 0
        noise = np.random.normal(0, 5)
        
        load = max(min(base + business_hours + spike + noise, 100), 5)
        cpu_load.append(load)
    
    cpu_load = np.array([cpu_load])
    
    # Forecast next 6 hours
    context = cpu_load[:, -24:]  # Last day
    forecast = model.forecast(context, prediction_length=6)
    
    print(f"\n📊 Current Status:")
    print(f"   Current load: {context[0, -1]:.1f}%")
    print(f"   24h average: {context.mean():.1f}%")
    print(f"   Peak 24h: {context.max():.1f}%")
    
    print(f"\n🔮 6-Hour Forecast:")
    for hour, load in enumerate(forecast[0], 1):
        status = "⚠️ HIGH" if load > 70 else "✓ NORMAL"
        print(f"   Hour {hour}: {load:.1f}% {status}")
    
    print(f"\n⚙️ Auto-scaling Actions:")
    if forecast.max() > 80:
        print(f"   ✓ SCALE UP: Add 2-3 instances for peak hour")
        print(f"   ✓ Cost: ~$5-10 for 6 hours")
    elif forecast.max() < 40:
        print(f"   ✓ SCALE DOWN: Remove 1-2 instances")
        print(f"   ✓ Savings: ~$3-5 per 6 hours")


def example_5_batch_forecasting():
    """
    Example 5: Batch Forecasting (Multiple locations)
    Use Case: Forecast for multiple stores/regions simultaneously
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: BATCH FORECASTING (3 regions)")
    print("="*70)
    
    # Initialize model
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=90,
        max_horizon=7,
    ))
    
    # Synthetic sales data for 3 regions
    np.random.seed(42)
    regions = ["North", "South", "West"]
    data_batch = []
    
    for region_id in range(3):
        daily_sales = []
        base_sales = [1500, 1200, 900][region_id]  # Different base for each region
        
        for day in range(365):
            trend = 0.3 * day * (1 + region_id * 0.1)
            weekly = 300 * np.sin(2 * np.pi * (day % 7) / 7)
            noise = np.random.normal(0, 100)
            
            sales = max(base_sales + trend + weekly + noise, 300)
            daily_sales.append(sales)
        
        data_batch.append(daily_sales)
    
    # Stack all regions: shape (3, 365)
    data_batch = np.array(data_batch)
    
    # Forecast all at once
    context = data_batch[:, -90:]
    forecasts = model.forecast(context, prediction_length=7)
    
    print(f"\n📊 Batch Processing ({len(regions)} regions):")
    for i, region in enumerate(regions):
        print(f"\n   Region: {region}")
        print(f"   - Historical avg: ${context[i].mean():.0f}")
        print(f"   - 7-day forecast: ${forecasts[i].mean():.0f}")
        print(f"   - Growth: {(forecasts[i].mean() - context[i].mean()) / context[i].mean() * 100:+.1f}%")
    
    print(f"\n⚙️ Operations:")
    print(f"   ✓ Process 3 regions in ~300ms (batch)")
    print(f"   ✓ vs ~100ms per region sequentially = faster!")


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 15 + "TIMESFM: QUICK START EXAMPLES" + " " * 24 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    try:
        example_1_electricity_load()
        example_2_stock_price()
        example_3_sales()
        example_4_cpu_load()
        example_5_batch_forecasting()
        
        print("\n" + "█" * 70)
        print("█" + " " * 20 + "✅ ALL EXAMPLES COMPLETED" + " " * 24 + "█")
        print("█" * 70 + "\n")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Note: First run downloads model from HuggingFace (~5-10 minutes)")
        print("   Subsequent runs are much faster due to caching.")
