"""
VocalBaby Drift Detection Module

Uses Evidently AI to compare reference (training) vs production feature distributions.
Generates HTML/JSON reports and exports drift scores to Prometheus.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def load_config(config_path: str = "configs/drift.yaml") -> dict:
    """Load drift detection configuration."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        return config.get("drift", {})
    except FileNotFoundError:
        logging.warning(f"Drift config not found at {config_path}, using defaults.")
        return {}


def load_features(features_path: str, labels_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load numpy feature arrays and convert to DataFrame for Evidently.
    
    Args:
        features_path: Path to .npy features file
        labels_path: Optional path to .npy labels file
        
    Returns:
        DataFrame with feature columns
    """
    try:
        features = np.load(features_path, allow_pickle=True)
        
        # Create column names
        n_features = features.shape[1] if features.ndim > 1 else 1
        columns = [f"feature_{i}" for i in range(n_features)]
        
        df = pd.DataFrame(features, columns=columns)
        
        if labels_path and os.path.exists(labels_path):
            labels = np.load(labels_path, allow_pickle=True)
            df["label"] = labels
            
        return df
        
    except Exception as e:
        logging.error(f"Failed to load features from {features_path}: {e}")
        raise VocalBabyException(e, sys)


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = "artifacts/drift",
    stattest: str = "wasserstein",
    threshold: float = 0.1,
) -> dict:
    """
    Generate Evidently drift report comparing reference vs current data.
    
    Args:
        reference_df: Training/reference feature DataFrame
        current_df: Production/current feature DataFrame
        output_dir: Directory to save reports
        stattest: Statistical test for drift detection
        threshold: Drift threshold
        
    Returns:
        Dictionary with drift results
    """
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create drift report
        report = Report(metrics=[
            DataDriftPreset(
                stattest=stattest,
                stattest_threshold=threshold,
            ),
        ])
        
        report.run(reference_data=reference_df, current_data=current_df)
        
        # Save HTML report
        html_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")
        report.save_html(html_path)
        logging.info(f"Drift HTML report saved to: {html_path}")
        
        # Extract results as dict
        report_dict = report.as_dict()
        
        # Save JSON report
        json_path = os.path.join(output_dir, f"drift_report_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        logging.info(f"Drift JSON report saved to: {json_path}")
        
        # Extract drift score
        drift_results = _extract_drift_score(report_dict)
        drift_results["html_report"] = html_path
        drift_results["json_report"] = json_path
        drift_results["timestamp"] = timestamp
        
        # Save latest drift summary
        summary_path = os.path.join(output_dir, "latest_drift_summary.yaml")
        with open(summary_path, "w") as f:
            yaml.dump(drift_results, f, default_flow_style=False)
        
        return drift_results
        
    except Exception as e:
        logging.error(f"Failed to generate drift report: {e}")
        raise VocalBabyException(e, sys)


def _extract_drift_score(report_dict: dict) -> dict:
    """Extract drift scores from Evidently report dictionary."""
    try:
        metrics = report_dict.get("metrics", [])
        
        dataset_drift = False
        share_drifted = 0.0
        n_drifted = 0
        n_columns = 0
        
        for metric in metrics:
            result = metric.get("result", {})
            if "share_of_drifted_columns" in result:
                share_drifted = result.get("share_of_drifted_columns", 0.0)
                dataset_drift = result.get("dataset_drift", False)
                n_drifted = result.get("number_of_drifted_columns", 0)
                n_columns = result.get("number_of_columns", 0)
                break
        
        return {
            "dataset_drift": dataset_drift,
            "drift_score": float(share_drifted),
            "n_drifted_columns": int(n_drifted),
            "n_total_columns": int(n_columns),
        }
        
    except Exception:
        return {
            "dataset_drift": False,
            "drift_score": 0.0,
            "n_drifted_columns": 0,
            "n_total_columns": 0,
        }


def export_drift_to_prometheus(drift_score: float, feature_set: str = "egemaps"):
    """Export drift score to Prometheus metrics."""
    try:
        from vocalbaby.monitoring.metrics import record_drift_score
        record_drift_score(drift_score, feature_set=feature_set)
        logging.info(f"Drift score {drift_score:.4f} exported to Prometheus (feature_set={feature_set})")
    except Exception as e:
        logging.warning(f"Could not export drift score to Prometheus: {e}")


def run_drift_report(
    reference_features_path: Optional[str] = None,
    current_features_path: Optional[str] = None,
    reference_labels_path: Optional[str] = None,
    current_labels_path: Optional[str] = None,
    config_path: str = "configs/drift.yaml",
) -> dict:
    """
    Main entry point for drift detection.
    
    Loads reference and current data, generates drift report,
    and exports metrics to Prometheus.
    
    Args:
        reference_features_path: Override path to reference features
        current_features_path: Override path to current features
        reference_labels_path: Override path to reference labels
        current_labels_path: Override path to current labels
        config_path: Path to drift configuration YAML
        
    Returns:
        Dictionary with drift results
    """
    try:
        config = load_config(config_path)
        
        # Resolve paths from config or arguments
        ref_features = reference_features_path or config.get(
            "reference_data_path",
            "artifacts/latest/data_transformation/features/train_compare_features.npy"
        )
        cur_features = current_features_path or config.get(
            "current_data_path",
            "artifacts/latest/data_transformation/features/valid_compare_features.npy"
        )
        ref_labels = reference_labels_path or config.get("reference_labels_path")
        cur_labels = current_labels_path or config.get("current_labels_path")
        
        # Column drift settings
        col_drift = config.get("column_drift", {})
        stattest = col_drift.get("stattest", "wasserstein")
        threshold = col_drift.get("threshold", 0.1)
        
        # Output directory
        report_config = config.get("report", {})
        output_dir = report_config.get("output_dir", "artifacts/drift")
        
        logging.info(f"Loading reference features from: {ref_features}")
        reference_df = load_features(ref_features, ref_labels)
        
        logging.info(f"Loading current features from: {cur_features}")
        current_df = load_features(cur_features, cur_labels)
        
        logging.info(f"Reference shape: {reference_df.shape}, Current shape: {current_df.shape}")
        
        # Generate report
        results = generate_drift_report(
            reference_df=reference_df,
            current_df=current_df,
            output_dir=output_dir,
            stattest=stattest,
            threshold=threshold,
        )
        
        # Export to Prometheus
        prometheus_config = config.get("prometheus", {})
        if prometheus_config.get("enabled", True):
            export_drift_to_prometheus(
                drift_score=results["drift_score"],
                feature_set="egemaps",
            )
        
        logging.info(f"Drift detection complete. Dataset drift: {results['dataset_drift']}, "
                     f"Score: {results['drift_score']:.4f}")
        
        return results
        
    except Exception as e:
        logging.error(f"Drift detection failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    # Run drift detection standalone
    results = run_drift_report()
    print(f"Drift Results: {json.dumps(results, indent=2, default=str)}")
