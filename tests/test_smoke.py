"""
VocalBaby Smoke Tests

Basic import and structure verification tests.
"""

import pytest


class TestPackageImport:
    """Verify that the vocalbaby package imports correctly."""

    def test_import_vocalbaby(self):
        import vocalbaby
        assert hasattr(vocalbaby, "__version__")
        assert vocalbaby.__version__ == "1.0.0"

    def test_import_exception(self):
        from vocalbaby.exception.exception import VocalBabyException
        assert VocalBabyException is not None

    def test_import_logging(self):
        from vocalbaby.logging.logger import logging
        assert logging is not None

    def test_import_api_app(self):
        from vocalbaby.api.app import app
        assert app is not None
        assert app.title == "VocalBaby API (Prediction Only)"

    def test_import_cli(self):
        from vocalbaby.cli import serve, train
        assert callable(serve)
        assert callable(train)

    def test_import_metrics(self):
        from vocalbaby.monitoring.metrics import (
            REQUEST_COUNT,
            REQUEST_LATENCY,
            PREDICTION_ERRORS,
            get_metrics,
        )
        assert REQUEST_COUNT is not None
        assert REQUEST_LATENCY is not None
        assert PREDICTION_ERRORS is not None
        assert callable(get_metrics)

    def test_import_drift(self):
        from vocalbaby.monitoring.drift import run_drift_report
        assert callable(run_drift_report)

    def test_import_pipeline(self):
        from vocalbaby.pipeline.training_pipeline import TrainingPipeline
        assert TrainingPipeline is not None

    def test_import_prediction_pipeline(self):
        from vocalbaby.pipeline.prediction_pipeline import PredictionPipeline
        assert PredictionPipeline is not None


class TestMetrics:
    """Test Prometheus metrics functionality."""

    def test_get_metrics_returns_bytes(self):
        from vocalbaby.monitoring.metrics import get_metrics
        output = get_metrics()
        assert isinstance(output, bytes)

    def test_set_model_info(self):
        from vocalbaby.monitoring.metrics import set_model_info, get_metrics
        set_model_info(version="1.0.0-test", model_type="xgboost")
        output = get_metrics()
        assert b"vocalbaby_model" in output

    def test_record_drift_score(self):
        from vocalbaby.monitoring.metrics import record_drift_score, get_metrics
        record_drift_score(0.42, feature_set="egemaps")
        output = get_metrics()
        assert b"vocalbaby_drift_score" in output
