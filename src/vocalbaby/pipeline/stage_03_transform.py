"""
STAGE 03: FEATURE EXTRACTION / DATA TRANSFORMATION

Generates features for all configured feature sets.
Uses the current run directory from artifacts/latest.

Outputs:
    artifacts/<timestamp>/features/<feature_set>/<split>/features.npy
    artifacts/<timestamp>/features/<feature_set>/<split>/labels.npy
"""
import sys
import argparse

from vocalbaby.experiments.scripts.generate_features import generate_features_for_set
from vocalbaby.config.schemas import ConfigLoader
from vocalbaby.utils.run_manager import RunManager
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def main():
    """Main entry point for feature extraction stage."""
    parser = argparse.ArgumentParser(description="VocalBaby Pipeline - Stage 03: Feature Extraction")
    parser.add_argument("--feature-sets", nargs="+", help="Feature sets to generate (default: all from params.yaml)")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if features exist")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    try:
        logging.info("Starting Stage 03: Feature Extraction")

        config = ConfigLoader()
        feature_sets = args.feature_sets or config.get_feature_sets()
        artifact_dir = str(RunManager.get_current_run_dir())

        logging.info(f"Run directory : {artifact_dir}")
        logging.info(f"Feature sets  : {feature_sets}")

        for feature_set in feature_sets:
            logging.info(f"\n{'='*80}")
            logging.info(f"Generating features: {feature_set}")
            logging.info(f"{'='*80}")

            generate_features_for_set(
                feature_set=feature_set,
                artifact_dir=artifact_dir,
                force_regenerate=args.force,
            )

        logging.info("Stage 03 completed: Feature extraction")
        return 0

    except Exception as e:
        logging.error(f"Stage 03 failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
