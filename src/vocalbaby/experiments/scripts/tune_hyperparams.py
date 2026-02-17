"""
Hyperparameter tuning script for XGBoost comparison experiments.

Runs Optuna tuning for each feature set using the exact workflow from notebook 06.

Usage:
    python -m vocalbaby.experiments.scripts.tune_hyperparams --feature-set egemaps
    python -m vocalbaby.experiments.scripts.tune_hyperparams --feature-set all
"""

import os
import sys
import argparse
import numpy as np

from vocalbaby.experiments.data_loader import (
    get_latest_artifact_dir,
    load_labels,
)
from vocalbaby.experiments.hyperparameter_tuning import (
    run_optuna_tuning,
    save_best_params,
)
from vocalbaby.experiments.training import save_label_encoder
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


FEATURE_SETS = ["egemaps", "mfcc", "hubert_ssl", "wav2vec2_ssl"]


def load_features_for_tuning(feature_set: str, artifact_dir: str):
    """
    Load train and valid features for hyperparameter tuning.
    
    Args:
        feature_set: Feature set name
        artifact_dir: Path to artifact directory
        
    Returns:
        X_train, X_valid
    """
    try:
        feature_dir = os.path.join(artifact_dir, "features", feature_set)
        
        X_train = np.load(os.path.join(feature_dir, "train/features.npy"))
        X_valid = np.load(os.path.join(feature_dir, "valid/features.npy"))
        
        logging.info(f"Loaded {feature_set} features:")
        logging.info(f"  Train: {X_train.shape}")
        logging.info(f"  Valid: {X_valid.shape}")
        
        return X_train, X_valid
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def tune_feature_set(
    feature_set: str,
    artifact_dir: str,
    n_trials: int = 40,
    random_state: int = 42,
    search_space: dict = None,
):
    """
    Run hyperparameter tuning for a specific feature set.
    
    Args:
        feature_set: Feature set name
        artifact_dir: Path to artifact directory
        n_trials: Number of Optuna trials
        random_state: Random seed
        search_space: Search space ranges from params.yaml
    """
    try:
        logging.info("=" * 80)
        logging.info(f"TUNING HYPERPARAMETERS: {feature_set.upper()}")
        logging.info("=" * 80)
        
        # Load features
        X_train, X_valid = load_features_for_tuning(feature_set, artifact_dir)
        
        # Load labels
        y_train, y_valid, y_test, label_encoder, class_names = load_labels(artifact_dir)
        
        # Save label encoder under tuning directory
        tuning_dir = os.path.join(artifact_dir, "tuning", feature_set)
        os.makedirs(tuning_dir, exist_ok=True)
        encoder_path = os.path.join(tuning_dir, "label_encoder.pkl")
        save_label_encoder(label_encoder, encoder_path)
        
        # Run Optuna tuning
        best_params, study = run_optuna_tuning(
            X_train,
            y_train,
            X_valid,
            y_valid,
            n_trials=n_trials,
            random_state=random_state,
            imputer_strategy="median",
            apply_smote=True,
            study_name=f"xgb_{feature_set}_smote_optuna",
            search_space=search_space,
        )
        
        # Save best params
        params_path = os.path.join(tuning_dir, "best_params.json")
        save_best_params(best_params, params_path)
        
        logging.info("=" * 80)
        logging.info(f"Tuning completed: {feature_set}")
        logging.info("=" * 80)
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning for XGBoost comparison experiments"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        required=True,
        choices=FEATURE_SETS + ["all"],
        help="Feature set to tune (or 'all' for all sets)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Path to artifact directory (default: latest)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of Optuna trials (default: 40)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Get artifact directory
    if args.artifact_dir is None:
        artifact_dir = get_latest_artifact_dir()
    else:
        artifact_dir = args.artifact_dir
    
    logging.info(f"Using artifact directory: {artifact_dir}")
    
    # Run tuning
    if args.feature_set == "all":
        for feature_set in FEATURE_SETS:
            tune_feature_set(feature_set, artifact_dir, args.n_trials, args.random_state)
    else:
        tune_feature_set(args.feature_set, artifact_dir, args.n_trials, args.random_state)
    
    logging.info("\n" + "=" * 80)
    logging.info("ALL HYPERPARAMETER TUNING COMPLETED")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
