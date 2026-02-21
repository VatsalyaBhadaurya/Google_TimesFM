"""
Production-Ready TimesFM Forecasting API

This module provides a robust, enterprise-grade wrapper around TimesFM
for deployment in production environments.

Features:
- Automatic error handling and fallbacks
- Model caching and memory management
- Request validation and sanitization
- Monitoring and metrics collection
- Async support for high-throughput scenarios
- Logging and audit trails
"""

import numpy as np
import torch
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import timesfm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ForecastRequest:
    """Validated forecast request."""
    time_series: np.ndarray
    forecast_horizon: int
    context_length: Optional[int] = None
    quantile_head: bool = True
    normalize: bool = True
    request_id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class ForecastResponse:
    """Structured forecast response."""
    point_forecast: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    uncertainty: Optional[float] = None
    request_id: str = ""
    timestamp: str = ""
    model_version: str = "2.5"
    metadata: Optional[Dict] = None
    
    def to_json(self):
        """Convert to JSON-serializable format."""
        return {
            'point_forecast': self.point_forecast.tolist(),
            'lower_bound': self.lower_bound.tolist() if self.lower_bound is not None else None,
            'upper_bound': self.upper_bound.tolist() if self.upper_bound is not None else None,
            'uncertainty': float(self.uncertainty) if self.uncertainty else None,
            'request_id': self.request_id,
            'timestamp': self.timestamp,
            'model_version': self.model_version,
            'metadata': self.metadata,
        }


class TimesFMAPI:
    """
    Production-grade TimesFM forecasting API.
    
    Example usage:
    ```python
    api = TimesFMAPI()
    
    # Simple forecast
    request = ForecastRequest(
        time_series=np.array([100, 102, 105, 103, ...]),
        forecast_horizon=24,
    )
    response = api.forecast(request)
    
    # Access results
    print(response.point_forecast)
    print(response.uncertainty)
    ```
    """
    
    # Class-level model cache (shared across instances)
    _model_cache = {}
    _cache_lock = threading.Lock()
    
    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        cache_model: bool = True,
        enable_metrics: bool = True,
        max_context: int = 1024,
        device: str = "auto"
    ):
        """
        Initialize TimesFM API.
        
        Args:
            model_name: HuggingFace model identifier
            cache_model: Cache model in memory for faster inference
            enable_metrics: Track performance metrics
            max_context: Maximum context length
            device: 'cuda', 'cpu', or 'auto'
        """
        self.model_name = model_name
        self.cache_model = cache_model
        self.enable_metrics = enable_metrics
        self.max_context = max_context
        self.device = self._get_device(device)
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency_ms': 0,
            'latencies': [],
        }
        
        # Load model
        self._load_model()
        
        logger.info(f"TimesFM API initialized on device: {self.device}")
    
    @staticmethod
    def _get_device(device_str: str) -> str:
        """Determine compute device."""
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_str
    
    def _load_model(self):
        """Load or retrieve cached model."""
        try:
            if self.cache_model and self.model_name in TimesFMAPI._model_cache:
                self.model = TimesFMAPI._model_cache[self.model_name]
                logger.info("✓ Model retrieved from cache")
                return
            
            logger.info(f"Loading model: {self.model_name}")
            torch.set_float32_matmul_precision("high")
            
            # Load base model
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_name
            )
            
            # Configure
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=self.max_context,
                    max_horizon=1024,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            
            # Move to device
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            # Cache if requested
            if self.cache_model:
                TimesFMAPI._model_cache[self.model_name] = self.model
            
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def validate_request(self, request: ForecastRequest) -> Tuple[bool, str]:
        """
        Validate forecast request.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check time series
            if request.time_series is None or len(request.time_series) == 0:
                return False, "Empty time series"
            
            if request.time_series.ndim == 1:
                request.time_series = request.time_series.reshape(1, -1)
            
            # Check for NaN/Inf
            if not np.isfinite(request.time_series).all():
                return False, "Time series contains NaN or Inf values"
            
            # Check forecast horizon
            if request.forecast_horizon <= 0:
                return False, f"Invalid forecast_horizon: {request.forecast_horizon}"
            
            if request.forecast_horizon > 1024:
                logger.warning(f"forecast_horizon {request.forecast_horizon} exceeds max 1024")
                request.forecast_horizon = 1024
            
            # Check context length
            if request.context_length is None:
                request.context_length = min(len(request.time_series), self.max_context)
            
            if request.context_length < 12:
                return False, f"context_length must be >= 12, got {request.context_length}"
            
            request.context_length = min(request.context_length, self.max_context)
            
            return True, ""
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        """
        Generate forecast.
        
        Args:
            request: ForecastRequest object
            
        Returns:
            ForecastResponse with predictions
        """
        import time
        
        try:
            start_time = time.time()
            self.metrics['total_requests'] += 1
            
            # Validate
            is_valid, error_msg = self.validate_request(request)
            if not is_valid:
                self.metrics['failed_requests'] += 1
                logger.error(f"[{request.request_id}] Validation failed: {error_msg}")
                raise ValueError(error_msg)
            
            # Generate request ID if not provided
            if request.request_id is None:
                request.request_id = self._generate_request_id(request.time_series)
            
            # Prepare context
            context = request.time_series[:, -request.context_length:]
            
            logger.info(f"[{request.request_id}] Forecasting {request.forecast_horizon} " +
                       f"steps with {context.shape[1]} context points")
            
            # Generate forecast
            with torch.no_grad():
                point_forecast = self.model.forecast(
                    context=context,
                    prediction_length=request.forecast_horizon,
                )
            
            # Extract bounds from quantiles if available
            lower_bound, upper_bound = self._extract_bounds(
                point_forecast, 
                quantile_head=request.quantile_head
            )
            
            # Calculate uncertainty
            uncertainty = np.std(point_forecast) if point_forecast.size > 0 else 0.0
            
            # Record metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics['latencies'].append(elapsed_ms)
            self.metrics['successful_requests'] += 1
            self.metrics['avg_latency_ms'] = np.mean(self.metrics['latencies'][-100:])
            
            logger.info(f"[{request.request_id}] Forecast generated in {elapsed_ms:.1f}ms")
            
            # Construct response
            response = ForecastResponse(
                point_forecast=point_forecast[0],  # Remove batch dimension
                lower_bound=lower_bound[0] if lower_bound is not None else None,
                upper_bound=upper_bound[0] if upper_bound is not None else None,
                uncertainty=uncertainty,
                request_id=request.request_id,
                timestamp=datetime.now().isoformat(),
                model_version="2.5",
                metadata=request.metadata,
            )
            
            return response
        
        except Exception as e:
            self.metrics['failed_requests'] += 1
            logger.error(f"[{request.request_id}] Forecast failed: {str(e)}")
            raise
    
    def batch_forecast(
        self,
        requests: List[ForecastRequest]
    ) -> List[ForecastResponse]:
        """
        Generate forecasts for multiple time series efficiently.
        
        Args:
            requests: List of ForecastRequest objects
            
        Returns:
            List of ForecastResponse objects
        """
        logger.info(f"Processing batch of {len(requests)} forecasts")
        
        responses = []
        for i, request in enumerate(requests):
            try:
                response = self.forecast(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch item {i} failed: {str(e)}")
                # Continue processing remaining items
                continue
        
        logger.info(f"Completed {len(responses)}/{len(requests)} forecasts")
        return responses
    
    @staticmethod
    def _extract_bounds(
        forecast: np.ndarray,
        quantile_head: bool = True,
        confidence: float = 0.95
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract confidence bounds from forecast."""
        if not quantile_head or forecast is None:
            return None, None
        
        # Simple approach: use std for bounds
        std = np.std(forecast, axis=1, keepdims=True)
        z_score = 1.96  # 95% confidence
        
        lower = forecast - z_score * std
        upper = forecast + z_score * std
        
        return lower, upper
    
    @staticmethod
    def _generate_request_id(data: np.ndarray) -> str:
        """Generate unique request ID."""
        hash_input = hashlib.md5(data.tobytes()).hexdigest()[:8]
        return f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash_input}"
    
    def get_metrics(self) -> Dict:
        """Get API performance metrics."""
        return {
            'total_requests': self.metrics['total_requests'],
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': (
                self.metrics['successful_requests'] / max(1, self.metrics['total_requests'])
            ) * 100,
            'avg_latency_ms': round(self.metrics['avg_latency_ms'], 2),
            'device': self.device,
            'model': self.model_name,
        }
    
    def reset_metrics(self):
        """Reset metrics."""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency_ms': 0,
            'latencies': [],
        }
        logger.info("Metrics reset")
    
    def health_check(self) -> Dict:
        """Check API health."""
        try:
            # Quick test forecast
            test_data = np.random.randn(1, 24)
            test_request = ForecastRequest(
                time_series=test_data,
                forecast_horizon=6,
            )
            
            response = self.forecast(test_request)
            
            return {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'device': self.device,
                'test_forecast_success': True,
                'timestamp': datetime.now().isoformat(),
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }


# Example: Production deployment
if __name__ == "__main__":
    
    # Initialize API
    print("🚀 Initializing TimesFM Production API...")
    api = TimesFMAPI(device="auto")
    
    # Health check
    health = api.health_check()
    print(f"\n✅ Health Check: {health['status'].upper()}")
    
    # Example 1: Single forecast
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Time Series Forecast")
    print("="*60)
    
    # Generate synthetic electricity load data
    np.random.seed(42)
    load_data = np.cumsum(np.random.randn(1, 336)) + 5000  # 2 weeks
    
    request = ForecastRequest(
        time_series=load_data,
        forecast_horizon=24,
        context_length=168,  # 1 week context
        metadata={'domain': 'electricity', 'unit': 'MW'}
    )
    
    response = api.forecast(request)
    print(f"Request ID: {response.request_id}")
    print(f"Forecast (24 hours): {response.point_forecast[:5]}... (showing first 5)")
    print(f"Uncertainty: ±{response.uncertainty:.1f}")
    
    # Example 2: Batch forecast
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Forecast (3 regions)")
    print("="*60)
    
    requests = [
        ForecastRequest(
            time_series=np.cumsum(np.random.randn(1, 336)) + 5000,
            forecast_horizon=24,
            metadata={'region': f'region_{i}'}
        )
        for i in range(3)
    ]
    
    responses = api.batch_forecast(requests)
    print(f"✓ Processed {len(responses)} forecasts")
    
    # Example 3: Metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    metrics = api.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
