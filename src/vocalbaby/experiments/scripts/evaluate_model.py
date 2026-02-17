"""
Evaluation script for XGBoost comparison experiments.

Evaluates trained models on valid and test splits.
Generates confusion matrices and metrics for both splits.

Usage:
    python -m vocalbaby.experiments.scripts.evaluate_model --feature-set egemaps
    python -m vocalbaby.experiments.scripts.evaluate_model --feature-set all
"""

import os
import sys
import argparse
import json
import numpy as np

from vocalbaby.experiments.data_loader import (
    get_latest_artifact_dir,
    load_labels,
)
from vocalbaby.experiments.training import (
    load_model,
    load_imputer,
    load_label_encoder,
)
from vocalbaby.experiments.evaluation import (
    evaluate_model,
    save_class_labels,
    aggregate_results,
)
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


FEATURE_SETS = ["egemaps", "mfcc", "hubert_ssl", "wav2vec2_ssl"]


def load_features_for_evaluation(feature_set: str):
    """
    Load valid and test features for evaluation.
    
    Args:
        feature_set: Feature set name
        
    Returns:
        X_valid, X_test
    """
    try:
        feature_dir = f"artifacts/features/{feature_set}"
        
        X_valid = np.load(os.path.join(feature_dir, "valid/features.npy"))
        X_test = np.load(os.path.join(feature_dir, "test/features.npy"))
        
        logging.info(f"Loaded {feature_set} features:")
        logging.info(f"  Valid: {X_valid.shape}")
        logging.info(f"  Test: {X_test.shape}")
        
        return X_valid, X_test
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def evaluate_feature_set(
    feature_set: str,
    artifact_dir: str,
):
    """
    Evaluate trained model for a specific feature set.
    
    Args:
        feature_set: Feature set name
        artifact_dir: Path to artifact directory
    """
    try:
        logging.info("=" * 80)
        logging.info(f"EVALUATING MODEL: {feature_set.upper()}")
        logging.info("=" * 80)
        
        # Load model artifacts
        model_dir = f"artifacts/models/{feature_set}"
        
        model_path = os.path.join(model_dir, "xgb_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please train the model first:\n"
                f"  python -m vocalbaby.experiments.scripts.train_model --feature-set {feature_set}"
            )
        
        model = load_model(model_path)
        
        imputer_path = os.path.join(model_dir, "imputer.pkl")
        imputer = load_imputer(imputer_path)
        
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        label_encoder = load_label_encoder(encoder_path)
        class_names = list(label_encoder.classes_)
        
        # Load features
        X_valid, X_test = load_features_for_evaluation(feature_set)
        
        # Load labels
        _, y_valid, y_test, _, _ = load_labels(artifact_dir)
        
        # Evaluate on validation set
        eval_dir = f"artifacts/eval/{feature_set}"
        
        logging.info("\n" + "=" * 80)
        logging.info("VALIDATION SET EVALUATION")
        logging.info("=" * 80)
        
        valid_metrics = evaluate_model(
            model,
            imputer,
            X_valid,
            y_valid,
            class_names,
            split_name="valid",
            feature_set=feature_set,
            output_dir=eval_dir,
        )
        
        # Evaluate on test set
        logging.info("\n" + "=" * 80)
        logging.info("TEST SET EVALUATION")
        logging.info("=" * 80)
        
        test_metrics = evaluate_model(
            model,
            imputer,
            X_test,
            y_test,
            class_names,
            split_name="test",
            feature_set=feature_set,
            output_dir=eval_dir,
        )
        
        # Save class labels
        labels_path = os.path.join(eval_dir, "labels.json")
        save_class_labels(class_names, labels_path)
        
        logging.info("=" * 80)
        logging.info(f"Evaluation completed: {feature_set}")
        logging.info("=" * 80)
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate XGBoost models for comparison experiments"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        required=True,
        choices=FEATURE_SETS + ["all"],
        help="Feature set to evaluate (or 'all' for all sets)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Path to artifact directory (default: latest)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate results across all feature sets",
    )
    
    args = parser.parse_args()
    
    # Get artifact directory
    if args.artifact_dir is None:
        artifact_dir = get_latest_artifact_dir()
    else:
        artifact_dir = args.artifact_dir
    
    logging.info(f"Using artifact directory: {artifact_dir}")
    
    # Evaluate models
    if args.feature_set == "all":
        for feature_set in FEATURE_SETS:
            evaluate_feature_set(feature_set, artifact_dir)
        
        # Aggregate results
        logging.info("\n" + "=" * 80)
        logging.info("AGGREGATING RESULTS")
        logging.info("=" * 80)
        aggregate_results(FEATURE_SETS)
        
    else:
        evaluate_feature_set(args.feature_set, artifact_dir)
        
        if args.aggregate:
            # Check which feature sets have been evaluated
            evaluated_sets = []
            for fs in FEATURE_SETS:
                metrics_path = f"artifacts/eval/{fs}/metrics_valid.json"
                if os.path.exists(metrics_path):
                    evaluated_sets.append(fs)
            
            if evaluated_sets:
                logging.info("\n" + "=" * 80)
                logging.info("AGGREGATING AVAILABLE RESULTS")
                logging.info("=" * 80)
                aggregate_results(evaluated_sets)
    
    logging.info("\n" + "=" * 80)
    logging.info("ALL EVALUATION COMPLETED")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
