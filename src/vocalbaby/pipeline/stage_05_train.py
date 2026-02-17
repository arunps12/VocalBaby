"""
STAGE 05: MODEL TRAINING

Trains XGBoost models for each feature set using best hyperparameters.
Uses the current run directory from artifacts/latest.

Outputs:
    artifacts/<timestamp>/models/<feature_set>/xgb_model.pkl
    artifacts/<timestamp>/models/<feature_set>/label_encoder.pkl
"""
import sys
import argparse

from vocalbaby.experiments.scripts.train_model import train_feature_set
from vocalbaby.config.schemas import ConfigLoader
from vocalbaby.utils.run_manager import RunManager
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def main():
    """Main entry point for model training stage."""
    parser = argparse.ArgumentParser(description="VocalBaby Pipeline - Stage 05: Model Training")
    parser.add_argument("--feature-sets", nargs="+", help="Feature sets to train (default: all from params.yaml)")
    parser.add_argument("--force", action="store_true", help="Force retraining even if model exists")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    try:
        logging.info("Starting Stage 05: Model Training")

        config = ConfigLoader()
        feature_sets = args.feature_sets or config.get_feature_sets()

        artifact_dir = str(RunManager.get_current_run_dir())
        logging.info(f"Run directory : {artifact_dir}")
        logging.info(f"Feature sets  : {feature_sets}")

        for feature_set in feature_sets:
            logging.info(f"\n{'='*80}")
            logging.info(f"Training model: {feature_set}")
            logging.info(f"{'='*80}")

            train_feature_set(
                feature_set=feature_set,
                artifact_dir=artifact_dir,
            )

        logging.info("Stage 05 completed: Model training")
        return 0

    except Exception as e:
        logging.error(f"Stage 05 failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
