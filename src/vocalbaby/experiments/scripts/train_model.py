"""
Training script for XGBoost comparison experiments.

Trains XGBoost models with best hyperparameters for each feature set.

Usage:
    python -m vocalbaby.experiments.scripts.train_model --feature-set egemaps
    python -m vocalbaby.experiments.scripts.train_model --feature-set all
"""

import os
import sys
import argparse
import numpy as np

from vocalbaby.experiments.data_loader import (
    get_latest_artifact_dir,
    load_labels,
)
from vocalbaby.experiments.hyperparameter_tuning import load_best_params
from vocalbaby.experiments.training import (
    train_xgboost_with_best_params,
    save_model,
    save_imputer,
    save_label_encoder,
)
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


FEATURE_SETS = ["egemaps", "mfcc", "hubert_ssl", "wav2vec2_ssl"]


def load_features_for_training(feature_set: str, artifact_dir: str):
    """
    Load train features for model training.
    
    Args:
        feature_set: Feature set name
        artifact_dir: Path to artifact directory
        
    Returns:
        X_train
    """
    try:
        feature_dir = os.path.join(artifact_dir, "features", feature_set)
        X_train = np.load(os.path.join(feature_dir, "train/features.npy"))
        
        logging.info(f"Loaded {feature_set} train features: {X_train.shape}")
        return X_train
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def train_feature_set(
    feature_set: str,
    artifact_dir: str,
    random_state: int = 42,
):
    """
    Train XGBoost model for a specific feature set.
    
    Args:
        feature_set: Feature set name
        artifact_dir: Path to artifact directory
        random_state: Random seed
    """
    try:
        logging.info("=" * 80)
        logging.info(f"TRAINING MODEL: {feature_set.upper()}")
        logging.info("=" * 80)
        
        # Load features
        X_train = load_features_for_training(feature_set, artifact_dir)
        
        # Load labels
        y_train, _, _, label_encoder, class_names = load_labels(artifact_dir)
        
        # Load best params from tuning directory
        tuning_dir = os.path.join(artifact_dir, "tuning", feature_set)
        params_path = os.path.join(tuning_dir, "best_params.json")
        
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"Best params not found: {params_path}\n"
                f"Please run hyperparameter tuning first:\n"
                f"  python -m vocalbaby.experiments.scripts.tune_hyperparams --feature-set {feature_set}"
            )
        
        best_params = load_best_params(params_path)
        
        # Train model
        model, imputer = train_xgboost_with_best_params(
            X_train,
            y_train,
            best_params,
            random_state=random_state,
            imputer_strategy="median",
            apply_smote=True,
        )
        
        # Save model artifacts
        model_dir = os.path.join(artifact_dir, "models", feature_set)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "xgb_model.pkl")
        save_model(model, model_path)
        
        imputer_path = os.path.join(model_dir, "imputer.pkl")
        save_imputer(imputer, imputer_path)
        
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        save_label_encoder(label_encoder, encoder_path)
        
        logging.info("=" * 80)
        logging.info(f"Training completed: {feature_set}")
        logging.info(f"Model saved to: {model_path}")
        logging.info("=" * 80)
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for comparison experiments"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        required=True,
        choices=FEATURE_SETS + ["all"],
        help="Feature set to train (or 'all' for all sets)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Path to artifact directory (default: latest)",
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
    
    # Train models
    if args.feature_set == "all":
        for feature_set in FEATURE_SETS:
            train_feature_set(feature_set, artifact_dir, args.random_state)
    else:
        train_feature_set(args.feature_set, artifact_dir, args.random_state)
    
    logging.info("\n" + "=" * 80)
    logging.info("ALL MODEL TRAINING COMPLETED")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
