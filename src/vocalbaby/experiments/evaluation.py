"""
Evaluation utilities for XGBoost feature comparison experiments.

Handles:
- Evaluation on valid and test splits
- Confusion matrix generation (PNG + CSV)
- Metrics computation and saving
- Results aggregation across feature sets
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging
from vocalbaby.utils.ml_utils.metric.classification_metric import get_classification_score


def evaluate_model(
    model: XGBClassifier,
    imputer: SimpleImputer,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    split_name: str,
    feature_set: str,
    output_dir: str,
) -> Dict:
    """
    Evaluate XGBoost model on a data split.
    
    Saves:
    - Confusion matrix (PNG + CSV)
    - Classification report (JSON)
    - Metrics (JSON)
    
    Args:
        model: Trained XGBoost classifier
        imputer: Fitted imputer
        X: Raw features
        y: Encoded labels
        class_names: List of class names
        split_name: "valid" or "test"
        feature_set: Feature set name (e.g., "egemaps", "mfcc")
        output_dir: Output directory for artifacts
        
    Returns:
        metrics_dict: Dictionary with all metrics
    """
    try:
        logging.info("=" * 80)
        logging.info(f"EVALUATING ON {split_name.upper()} SPLIT - {feature_set}")
        logging.info("=" * 80)
        
        # Preprocess features
        X_imp = imputer.transform(X)
        
        # Predict
        y_pred = model.predict(X_imp)
        
        # Compute metrics
        metrics = get_classification_score(y, y_pred)
        
        logging.info(f"UAR: {metrics.uar:.4f}")
        logging.info(f"F1: {metrics.f1_score:.4f}")
        logging.info(f"Precision: {metrics.precision_score:.4f}")
        logging.info(f"Recall: {metrics.recall_score:.4f}")
        
        # Save confusion matrix (PNG)
        cm = confusion_matrix(y, y_pred)
        cm_png_path = os.path.join(output_dir, f"confusion_matrix_{split_name}.png")
        save_confusion_matrix_png(cm, class_names, cm_png_path, 
                                    title=f"{feature_set.upper()} - {split_name.upper()} Set")
        
        # Save confusion matrix (CSV)
        cm_csv_path = os.path.join(output_dir, f"confusion_matrix_{split_name}.csv")
        save_confusion_matrix_csv(cm, class_names, cm_csv_path)
        
        # Save classification report
        report_dict = classification_report(y, y_pred, target_names=class_names, 
                                            output_dict=True, zero_division=0)
        report_path = os.path.join(output_dir, f"classification_report_{split_name}.json")
        save_json(report_dict, report_path)
        
        # Save metrics
        metrics_dict = {
            "uar": float(metrics.uar),
            "f1_score": float(metrics.f1_score),
            "precision": float(metrics.precision_score),
            "recall": float(metrics.recall_score),
        }
        metrics_path = os.path.join(output_dir, f"metrics_{split_name}.json")
        save_json(metrics_dict, metrics_path)
        
        logging.info(f"Evaluation artifacts saved to: {output_dir}")
        logging.info("=" * 80)
        
        return metrics_dict
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def save_confusion_matrix_png(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
):
    """
    Save confusion matrix as PNG.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        output_path: Path to save PNG
        title: Plot title
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logging.info(f"Saved confusion matrix PNG: {output_path}")
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def save_confusion_matrix_csv(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
):
    """
    Save confusion matrix as CSV.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        output_path: Path to save CSV
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(output_path)
        
        logging.info(f"Saved confusion matrix CSV: {output_path}")
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def save_json(data: Dict, output_path: str):
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        output_path: Path to save JSON
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"Saved JSON: {output_path}")
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def aggregate_results(
    feature_sets: List[str],
    base_output_dir: str = "artifacts/results",
) -> pd.DataFrame:
    """
    Aggregate evaluation results across all feature sets.
    
    Reads metrics from:
    - artifacts/eval/<feature_set>/metrics_valid.json
    - artifacts/eval/<feature_set>/metrics_test.json
    
    Creates summary table with all metrics.
    
    Args:
        feature_sets: List of feature set names
        base_output_dir: Base output directory
        
    Returns:
        results_df: DataFrame with aggregated results
    """
    try:
        logging.info("=" * 80)
        logging.info("AGGREGATING RESULTS ACROSS FEATURE SETS")
        logging.info("=" * 80)
        
        results = []
        
        for feature_set in feature_sets:
            eval_dir = os.path.join("artifacts/eval", feature_set)
            
            # Load valid metrics
            valid_metrics_path = os.path.join(eval_dir, "metrics_valid.json")
            if os.path.exists(valid_metrics_path):
                with open(valid_metrics_path, "r") as f:
                    valid_metrics = json.load(f)
            else:
                logging.warning(f"Valid metrics not found: {valid_metrics_path}")
                valid_metrics = {}
            
            # Load test metrics
            test_metrics_path = os.path.join(eval_dir, "metrics_test.json")
            if os.path.exists(test_metrics_path):
                with open(test_metrics_path, "r") as f:
                    test_metrics = json.load(f)
            else:
                logging.warning(f"Test metrics not found: {test_metrics_path}")
                test_metrics = {}
            
            # Build result row
            result_row = {
                "feature_set": feature_set,
                "valid_uar": valid_metrics.get("uar", np.nan),
                "valid_f1": valid_metrics.get("f1_score", np.nan),
                "valid_precision": valid_metrics.get("precision", np.nan),
                "valid_recall": valid_metrics.get("recall", np.nan),
                "test_uar": test_metrics.get("uar", np.nan),
                "test_f1": test_metrics.get("f1_score", np.nan),
                "test_precision": test_metrics.get("precision", np.nan),
                "test_recall": test_metrics.get("recall", np.nan),
            }
            
            results.append(result_row)
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save summary CSV
        os.makedirs(base_output_dir, exist_ok=True)
        summary_path = os.path.join(base_output_dir, "results_summary.csv")
        results_df.to_csv(summary_path, index=False)
        
        logging.info(f"\nResults Summary:")
        logging.info("\n" + results_df.to_string())
        logging.info(f"\nSaved results summary to: {summary_path}")
        logging.info("=" * 80)
        
        return results_df
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def save_class_labels(
    class_names: List[str],
    output_path: str,
):
    """
    Save class label mapping to JSON.
    
    Ensures consistent label order across all evaluations.
    
    Args:
        class_names: List of class names in order
        output_path: Path to save JSON
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        label_mapping = {i: name for i, name in enumerate(class_names)}
        
        with open(output_path, "w") as f:
            json.dump(label_mapping, f, indent=2)
        
        logging.info(f"Saved class labels to: {output_path}")
        
    except Exception as e:
        raise VocalBabyException(e, sys)
