"""
STAGE 04: HYPERPARAMETER TUNING

Runs Optuna hyperparameter optimization for each feature set.
Uses the current run directory from artifacts/latest.

Outputs:
    artifacts/<timestamp>/tuning/<feature_set>/best_params.json
    artifacts/<timestamp>/tuning/<feature_set>/study.pkl
"""
import sys
import argparse

from vocalbaby.experiments.scripts.tune_hyperparams import tune_feature_set
from vocalbaby.config.schemas import ConfigLoader
from vocalbaby.utils.run_manager import RunManager
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def main():
    """Main entry point for hyperparameter tuning stage."""
    parser = argparse.ArgumentParser(description="VocalBaby Pipeline - Stage 04: Hyperparameter Tuning")
    parser.add_argument("--feature-sets", nargs="+", help="Feature sets to tune (default: all from params.yaml)")
    parser.add_argument("--n-trials", type=int, help="Number of Optuna trials (overrides params.yaml)")
    parser.add_argument("--force", action="store_true", help="Force re-tuning even if best_params.json exists")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    try:
        print("\n" + "=" * 70)
        print("  STAGE 04: HYPERPARAMETER TUNING - STARTED")
        print("=" * 70)
        logging.info("Starting Stage 04: Hyperparameter Tuning")

        config = ConfigLoader()
        feature_sets = args.feature_sets or config.get_feature_sets()
        n_trials = args.n_trials or config.get('tuning.n_trials', 40)
        search_space = config.get('tuning.xgb_search_space', None)

        artifact_dir = str(RunManager.get_current_run_dir())
        print(f"  Run directory  : {artifact_dir}")
        print(f"  Feature sets   : {feature_sets}")
        print(f"  Optuna trials  : {n_trials}")
        logging.info(f"Run directory  : {artifact_dir}")
        logging.info(f"Feature sets   : {feature_sets}")
        logging.info(f"Optuna trials  : {n_trials}")

        for i, feature_set in enumerate(feature_sets, 1):
            print(f"\n  [{i}/{len(feature_sets)}] Tuning: {feature_set} ...")
            logging.info(f"\n{'='*80}")
            logging.info(f"Tuning hyperparameters: {feature_set}")
            logging.info(f"{'='*80}")

            tune_feature_set(
                feature_set=feature_set,
                artifact_dir=artifact_dir,
                n_trials=n_trials,
                search_space=search_space,
            )
            print(f"  [{i}/{len(feature_sets)}] Done:   {feature_set}")

        print("\n" + "=" * 70)
        print("  STAGE 04: HYPERPARAMETER TUNING - COMPLETED")
        print("=" * 70 + "\n")
        logging.info("Stage 04 completed: Hyperparameter tuning")
        return 0

    except Exception as e:
        logging.error(f"Stage 04 failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
