# TimesFM: Complete Real-World Implementation Guide

## 🎯 Table of Contents
1. [Codebase Overview](#codebase-overview)
2. [Real-World Applications](#real-world-applications)
3. [Implementation Steps](#implementation-steps)
4. [Best Practices](#best-practices)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

---

## Codebase Overview

### What is TimesFM?

**TimesFM** (Time Series Foundation Model) is a **pre-trained decoder-only transformer** developed by Google Research for accurate time-series forecasting across diverse domains.

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Architecture** | Decoder-only Transformer |
| **Latest Version** | TimesFM 2.5 (200M parameters) |
| **Max Context** | 16,384 timesteps (~11 days of hourly data) |
| **Max Horizon** | 1,024 timesteps (~42 days hourly, 1 year daily) |
| **Frameworks** | PyTorch, JAX (Flax) |
| **Pre-training Dataset** | 900+ diverse time-series datasets |
| **License** | Apache 2.0 (open source) |

### Model Versions

```
TimesFM 1.0 (200M)
├─ Context: 512
├─ Horizon: Unlimited
└─ Focus: Point forecasts

TimesFM 2.0 (500M)
├─ Context: 2,048
├─ Horizon: Unlimited
└─ Features: Quantile, external regressors

TimesFM 2.5 (200M) ⭐ Latest
├─ Context: 16,384
├─ Horizon: 1,024
├─ Size: 40% smaller than v2.0
├─ Speed: 2-3x faster inference
└─ Features: Quantile (continuous, up to 1k horizon)
```

---

## Real-World Applications

### 1. Energy & Utilities 🔌
**Challenge**: Power grids need precise demand forecasting
- **Electricity Load Forecasting**: Predict demand for optimal generation dispatch
- **Gas Demand**: Manage pipeline operations
- **Renewable Integration**: Balance wind/solar variability
- **Time Horizon**: 6-24 hours ahead
- **Accuracy Impact**: 1% improvement = $1-10M savings annually per utility

**Implementation Example**:
```python
context_length = 168  # 1 week historical data
forecast_horizon = 24  # 24 hours ahead
normalize = True      # Normalize for stability

forecast = model.forecast(
    context=historical_load,
    prediction_length=forecast_horizon
)
```

### 2. Finance & Trading 📈
**Challenge**: Predict market movements with minimal lag
- **Stock Price Forecasting**: Support investment decisions
- **Volatility Prediction**: Risk management
- **Cryptocurrency**: High-frequency trading signals
- **Commodity Prices**: Hedging decisions
- **Time Horizon**: 5 days to 3 months
- **Accuracy Impact**: 1% improvement = $millions in algorithmic trading

### 3. Retail & E-commerce 🛍️
**Challenge**: Match inventory to unpredictable demand
- **Sales Forecasting**: Daily/weekly product demand
- **Inventory Optimization**: Minimize stockout cost
- **Staffing**: Match floor staff to expected traffic
- **Promotional Planning**: Time campaigns for maximum impact
- **Time Horizon**: 7-30 days
- **Accuracy Impact**: 5% improvement = 2-4% inventory cost reduction

### 4. Cloud & Infrastructure ☁️
**Challenge**: Scale resources efficiently before demand spikes
- **CPU/Memory Forecasting**: Auto-scaling triggers
- **Network Bandwidth**: Capacity planning
- **Cost Optimization**: Reserve instances efficiently
- **SLA Compliance**: Prevent downtime
- **Time Horizon**: 1-6 hours
- **Accuracy Impact**: Prevent 5-10% of alert-worthy incidents

### 5. Transportation 🚗
**Challenge**: Predict traffic, rides, and logistics needs
- **Traffic Flow**: Real-time congestion prediction
- **Ride-hailing Demand**: Uber/Lyft surge pricing
- **Package Delivery**: Route optimization
- **Airport Traffic**: Peak management
- **Time Horizon**: 15-60 minutes
- **Accuracy Impact**: 20-30% reduction in idle wait time

### 6. Healthcare & Epidemiology 🏥
**Challenge**: Prepare for surges in patient demand
- **Hospital Admissions**: Staffing and bed management
- **Disease Spread**: COVID-19, flu predictions
- **Medical Supply**: Blood, drugs, equipment demand
- **Time Horizon**: Daily to weekly
- **Accuracy Impact**: Lives saved through early preparation

### 7. Manufacturing & Supply Chain 🏭
**Challenge**: Optimize production given uncertain demand
- **Product Demand**: Production scheduling
- **Raw Material Demand**: Supplier coordination
- **Equipment Maintenance**: Predictive scheduling
- **Quality Metrics**: Defect rate prediction
- **Time Horizon**: Weekly to monthly
- **Accuracy Impact**: 3-5% reduction in production costs

### 8. Environmental & Climate 🌍
**Challenge**: Monitor and predict environmental hazards
- **Air Quality (AQI)**: Pollution level forecasting
- **Weather**: Temperature, precipitation, wind
- **Water Quality**: Contaminant forecasting
- **Soil Moisture**: Agricultural planning
- **Time Horizon**: 24-72 hours
- **Accuracy Impact**: Public health warnings, agricultural yield optimization

---

## Implementation Steps

### Step 1: Installation

```bash
# Create virtual environment
python -m venv timesfm_env
source timesfm_env/bin/activate  # On Windows: timesfm_env\Scripts\activate

# Install TimesFM with PyTorch backend
pip install "timesfm[torch]>=2.0.0"

# Or with JAX/Flax backend (faster on TPU)
pip install "timesfm[flax]>=2.0.0"

# For covariate support
pip install "timesfm[xreg]>=2.0.0"
```

### Step 2: Load Model

```python
import torch
import timesfm
import numpy as np

# Set optimization level
torch.set_float32_matmul_precision("high")

# Load pre-trained model (automatically downloads from HuggingFace)
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Configure for your use case
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,              # Historical data window
        max_horizon=256,               # Future points to forecast
        normalize_inputs=True,         # Normalize by mean/std
        use_continuous_quantile_head=True,  # Uncertainty bounds
        force_flip_invariance=True,    # Temporal robustness
        infer_is_positive=True,        # Force positive values for certain datasets
        fix_quantile_crossing=True,    # Ensure quantiles don't cross
    )
)
```

### Step 3: Prepare Data

```python
# Your time series (shape: [num_series, timesteps])
historical_data = np.array([[100, 102, 105, 103, ...]])  # 1D array of values

# Ensure shape: [num_series, timesteps]
if historical_data.ndim == 1:
    historical_data = historical_data.reshape(1, -1)

# Recommended: last N timesteps as context
context_window = historical_data[:, -1024:]  # Last 1024 points
```

### Step 4: Generate Forecast

```python
# Single point forecast
point_forecast = model.forecast(
    context=context_window,
    prediction_length=24  # Forecast 24 steps
)
# Returns: shape [num_series, prediction_length]

# With uncertainty estimates
forecast_with_uncertainty = model.forecast(
    context=context_window,
    prediction_length=24,
    quantile_headroom_fraction=0.0,  # Default quantile range
)
```

### Step 5: Evaluate & Deploy

```python
# Calculate accuracy metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

# Deploy forecast
output = {
    'timestamp': pd.Timestamp.now(),
    'forecast': predicted,
    'uncertainty': confidence_interval,
    'accuracy_metrics': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
}
```

---

## Best Practices

### ✅ DO's

**1. Normalize your data**
```python
# Optional but recommended
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data.reshape(-1, 1))
```

**2. Use sufficient context**
- **Minimum**: At least 2-4 weeks of historical data
- **Optimal**: 6-12 months for capturing seasonality
- **Maximum**: Up to 16,384 timesteps

**3. Handle seasonality**
```python
# For strong seasonality, provide seasonal period
# Model will learn from yearly patterns automatically

# If using external regressors for seasonality:
seasonal_features = np.sin(2 * np.pi * np.arange(len(data)) / 365)
```

**4. Monitor forecast accuracy**
```python
def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mape = 100 * np.mean(np.abs((actual - predicted) / actual))
    
    # If MAPE > 15%, investigate:
    # - Data quality
    # - Trend changes
    # - Outliers
    # - Model choice
    return {'mae': mae, 'rmse': rmse, 'mape': mape}
```

**5. Retrain periodically**
```python
# Fine-tune on recent data every 4-12 weeks
# Use only domain-specific data (e.g., electricity load data)
# This improves model for your specific distribution
```

### ❌ DON'Ts

**1. Don't use insufficient data**
- Minimum context should be 2x the forecast horizon
- For 24-hour forecasts, use at least 2-4 weeks history
- For 30-day forecasts, use at least 6 months history

**2. Don't ignore data quality**
- Remove obvious errors/outliers if they're anomalies
- Handle missing values by interpolation
- Detect and separately model events (Black Swan events)

**3. Don't over-optimize on test set**
- Use proper train/val/test splits (70/15/15)
- Implement rolling window validation
- Test on unseen data from different time periods

**4. Don't deploy without backtesting**
```python
def walk_forward_validation(data, context_len=512, forecast_len=24):
    results = []
    
    # Test on sequential windows (realistic deployment)
    for i in range(len(data) - context_len - forecast_len):
        context = data[i:i+context_len]
        truth = data[i+context_len:i+context_len+forecast_len]
        
        pred = model.forecast(context, prediction_length=forecast_len)
        results.append({'pred': pred, 'truth': truth})
    
    return results
```

**5. Don't ignore distribution shifts**
- Monitor actual vs predicted over time
- Retrain if MAPE increases by >20%
- Be prepared for concept drift

---

## Performance Optimization

### Memory Optimization

```python
# For limited GPU/TPU memory:

# 1. Batch processing
batch_size = 32  # Adjust based on available memory
forecasts = []

for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    pred = model.forecast(batch, prediction_length=24)
    forecasts.append(pred)

# 2. Use float32 instead of float64
model = model.to(torch.float32)

# 3. Process shorter sequences if possible
context = data[:, -512:]  # Use shorter context
```

### Speed Optimization

```python
# 1. Compile for faster inference
model.compile(ForecastConfig(...))

# 2. Batch multiple series together
# Processing 64 series at once is faster than 1x64 sequential calls
big_batch = np.stack([series1, series2, ..., series64])
forecasts = model.forecast(big_batch, prediction_length=24)

# 3. Use quantization for deployment
# Post-training quantization reduces model size by 75%
# Trade-off: ~1-2% accuracy loss
```

### Latency Metrics

| Operation | Latency | Hardware |
|-----------|---------|----------|
| Single inference | 50-200ms | GPU (RTX 4090) |
| Batch of 64 | 100-300ms | GPU (RTX 4090) |
| CPU inference | 1-3 seconds | CPU (Intel i7) |

---

## Advanced: Fine-tuning on Your Data

```python
import torch
from torch.utils.data import DataLoader

# Fine-tune on your specific domain data
def finetune_model(model, train_data, val_data, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training loop
        train_loss = 0
        for context, target in train_dataloader:
            # Forward pass
            pred = model(context)
            loss = torch.nn.functional.mse_loss(pred, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        val_loss = evaluate(model, val_dataloader)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
```

---

## Troubleshooting

### Q: Model produces NaN values
**A**: 
- Check for inf/nan in input data: `np.any(~np.isfinite(data))`
- Use `normalize_inputs=True` in config
- Ensure all values are positive: `data[data < 0] = small_value`

### Q: Forecast accuracy is poor
**A**:
- Increase context length (use 2-3x your forecast horizon)
- Check data quality - are there obvious errors?
- Increase training data variety
- Fine-tune on domain-specific data

### Q: Out of memory error
**A**:
- Reduce `batch_size`
- Use shorter `max_context`
- Switch to float32 precision
- Process data in smaller batches

### Q: Model is too slow
**A**:
- Batch multiple series together
- Use GPU instead of CPU
- Reduce context length if possible
- Consider quantization for deployment

---

## Production Deployment Checklist

- [ ] Data collection and storage pipeline ready
- [ ] Data validation and quality checks implemented
- [ ] Model loading and caching configured
- [ ] Forecast generation logic tested
- [ ] Accuracy monitoring and alerting set up
- [ ] Retraining schedule established (weekly/monthly)
- [ ] API/webhook for consuming forecasts ready
- [ ] Fallback to baseline model if something fails
- [ ] Logging and audit trail in place
- [ ] Documentation for maintenance team

---

## Resources

- [TimesFM Paper](https://arxiv.org/abs/2310.10688)
- [GitHub Repository](https://github.com/google-research/timesfm)
- [HuggingFace Hub](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [Google Research Blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)

---

**Created**: February 2026  
**Status**: Production Ready  
**Maintained by**: Google Research
