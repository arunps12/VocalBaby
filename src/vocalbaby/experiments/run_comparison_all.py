"""
Main orchestration script for XGBoost feature comparison experiments.

Runs the complete pipeline:
1. Feature cache generation
2. Hyperparameter tuning
3. Model training
4. Evaluation on valid and test
5. Results aggregation

Usage:
    # Run complete pipeline for all feature sets
    python -m vocalbaby.experiments.run_comparison_all

    # Run for specific feature set
    python -m vocalbaby.experiments.run_comparison_all --feature-set egemaps

    # Skip steps that are already complete
    python -m vocalbaby.experiments.run_comparison_all --skip-features --skip-tuning
"""

import os
import sys
import argparse
from typing import List

from vocalbaby.experiments.data_loader import get_latest_artifact_dir
from vocalbaby.experiments.scripts.generate_features import generate_features_for_set
from vocalbaby.experiments.scripts.tune_hyperparams import tune_feature_set
from vocalbaby.experiments.scripts.train_model import train_feature_set
from vocalbaby.experiments.scripts.evaluate_model import evaluate_feature_set
from vocalbaby.experiments.evaluation import aggregate_results
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


FEATURE_SETS = ["egemaps", "mfcc", "hubert_ssl", "wav2vec2_ssl"]


def check_feature_cache_exists(feature_set: str) -> bool:
    """Check if feature cache already exists."""
    cache_path = f"artifacts/features/{feature_set}/train/features.npy"
    return os.path.exists(cache_path)


def check_best_params_exist(feature_set: str) -> bool:
    """Check if best params already exist."""
    params_path = f"artifacts/models/{feature_set}/best_params.json"
    return os.path.exists(params_path)


def check_model_exists(feature_set: str) -> bool:
    """Check if trained model already exists."""
    model_path = f"artifacts/models/{feature_set}/xgb_model.pkl"
    return os.path.exists(model_path)


def run_pipeline_for_feature_set(
    feature_set: str,
    artifact_dir: str,
    skip_features: bool = False,
    skip_tuning: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False,
    force_regenerate: bool = False,
    n_trials: int = 40,
    random_state: int = 42,
):
    """
    Run complete pipeline for a single feature set.
    
    Args:
        feature_set: Feature set name
        artifact_dir: Path to artifact directory
        skip_features: Skip feature generation if cache exists
        skip_tuning: Skip tuning if best params exist
        skip_training: Skip training if model exists
        skip_evaluation: Skip evaluation
        force_regenerate: Force regeneration of all artifacts
        n_trials: Number of Optuna trials
        random_state: Random seed
    """
    try:
        logging.info("\n" + "=" * 100)
        logging.info(f"RUNNING PIPELINE FOR: {feature_set.upper()}")
        logging.info("=" * 100)
        
        # Step 1: Feature Generation
        if skip_features and not force_regenerate and check_feature_cache_exists(feature_set):
            logging.info(f"\n[SKIP] Feature cache exists for {feature_set}")
        else:
            logging.info(f"\n[STEP 1/4] Generating features for {feature_set}...")
            generate_features_for_set(feature_set, artifact_dir, force_regenerate)
        
        # Step 2: Hyperparameter Tuning
        if skip_tuning and not force_regenerate and check_best_params_exist(feature_set):
            logging.info(f"\n[SKIP] Best params exist for {feature_set}")
        else:
            logging.info(f"\n[STEP 2/4] Tuning hyperparameters for {feature_set}...")
            tune_feature_set(feature_set, artifact_dir, n_trials, random_state)
        
        # Step 3: Training
        if skip_training and not force_regenerate and check_model_exists(feature_set):
            logging.info(f"\n[SKIP] Model exists for {feature_set}")
        else:
            logging.info(f"\n[STEP 3/4] Training model for {feature_set}...")
            train_feature_set(feature_set, artifact_dir, random_state)
        
        # Step 4: Evaluation
        if not skip_evaluation:
            logging.info(f"\n[STEP 4/4] Evaluating model for {feature_set}...")
            evaluate_feature_set(feature_set, artifact_dir)
        else:
            logging.info(f"\n[SKIP] Evaluation skipped for {feature_set}")
        
        logging.info("\n" + "=" * 100)
        logging.info(f"PIPELINE COMPLETED FOR: {feature_set.upper()}")
        logging.info("=" * 100)
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete XGBoost feature comparison pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for all feature sets
  python -m vocalbaby.experiments.run_comparison_all

  # Run for specific feature set only
  python -m vocalbaby.experiments.run_comparison_all --feature-set mfcc

  # Skip steps that are already complete
  python -m vocalbaby.experiments.run_comparison_all --skip-features --skip-tuning

  # Force regeneration of all artifacts
  python -m vocalbaby.experiments.run_comparison_all --force

  # Run with fewer trials for faster testing
  python -m vocalbaby.experiments.run_comparison_all --n-trials 10
        """,
    )
    
    parser.add_argument(
        "--feature-set",
        type=str,
        default="all",
        choices=FEATURE_SETS + ["all"],
        help="Feature set to process (default: all)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Path to artifact directory (default: latest)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature generation if cache exists",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning if best params exist",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training if model exists",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all artifacts",
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
    
    logging.info("\n" + "=" * 100)
    logging.info("XGBOOST FEATURE COMPARISON PIPELINE")
    logging.info("=" * 100)
    logging.info(f"Artifact directory: {artifact_dir}")
    logging.info(f"Feature sets: {args.feature_set}")
    logging.info(f"Optuna trials: {args.n_trials}")
    logging.info(f"Random state: {args.random_state}")
    logging.info("=" * 100)
    
    # Determine which feature sets to process
    if args.feature_set == "all":
        feature_sets_to_process = FEATURE_SETS
    else:
        feature_sets_to_process = [args.feature_set]
    
    # Run pipeline for each feature set
    for feature_set in feature_sets_to_process:
        run_pipeline_for_feature_set(
            feature_set,
            artifact_dir,
            skip_features=args.skip_features,
            skip_tuning=args.skip_tuning,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            force_regenerate=args.force,
            n_trials=args.n_trials,
            random_state=args.random_state,
        )
    
    # Aggregate results if multiple feature sets were processed
    if len(feature_sets_to_process) > 1 and not args.skip_evaluation:
        logging.info("\n" + "=" * 100)
        logging.info("AGGREGATING RESULTS ACROSS ALL FEATURE SETS")
        logging.info("=" * 100)
        
        results_df = aggregate_results(feature_sets_to_process)
        
        logging.info("\n" + "=" * 100)
        logging.info("FINAL RESULTS SUMMARY")
        logging.info("=" * 100)
        logging.info("\n" + results_df.to_string(index=False))
    
    logging.info("\n" + "=" * 100)
    logging.info("✓ ALL PIPELINES COMPLETED SUCCESSFULLY")
    logging.info("=" * 100)
    logging.info("\nGenerated artifacts:")
    logging.info("  - Feature caches: artifacts/features/<feature_set>/")
    logging.info("  - Best params: artifacts/models/<feature_set>/best_params.json")
    logging.info("  - Trained models: artifacts/models/<feature_set>/xgb_model.pkl")
    logging.info("  - Confusion matrices: artifacts/eval/<feature_set>/confusion_matrix_{valid|test}.png")
    logging.info("  - Metrics: artifacts/eval/<feature_set>/metrics_{valid|test}.json")
    logging.info("  - Results summary: artifacts/results/results_summary.csv")
    logging.info("=" * 100)


if __name__ == "__main__":
    main()
