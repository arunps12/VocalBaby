"""
VocalBaby Prometheus Metrics

Provides Prometheus metrics for monitoring the prediction API.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# Use default registry
REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# Request Metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "vocalbaby_request_total",
    "Total number of prediction requests",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "vocalbaby_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Error Metrics
# ---------------------------------------------------------------------------
PREDICTION_ERRORS = Counter(
    "vocalbaby_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Model Metrics
# ---------------------------------------------------------------------------
MODEL_INFO = Info(
    "vocalbaby_model",
    "Model version and metadata",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Drift Metrics (set by drift detection module)
# ---------------------------------------------------------------------------
DRIFT_SCORE = Gauge(
    "vocalbaby_drift_score",
    "Latest drift detection score (0=no drift, 1=full drift)",
    ["feature_set"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Active Predictions
# ---------------------------------------------------------------------------
ACTIVE_PREDICTIONS = Gauge(
    "vocalbaby_active_predictions",
    "Number of currently in-flight predictions",
    registry=REGISTRY,
)


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest(REGISTRY)


def get_content_type() -> str:
    """Return Prometheus content type header."""
    return CONTENT_TYPE_LATEST


def set_model_info(version: str, model_type: str = "xgboost"):
    """Set the model info metric."""
    MODEL_INFO.info({
        "version": version,
        "model_type": model_type,
        "feature_set": "egemaps",
    })


def record_drift_score(score: float, feature_set: str = "egemaps"):
    """Record a drift score from the drift detection module."""
    DRIFT_SCORE.labels(feature_set=feature_set).set(score)
