"""
STAGE 03: FEATURE EXTRACTION / DATA TRANSFORMATION

Generates features for all configured feature sets.

Outputs:
    artifacts/features/<feature_set>/<split>/features.npy
    artifacts/features/<feature_set>/<split>/labels.npy
    artifacts/features/<feature_set>/<split>/metadata.parquet
"""
import sys
import os
import argparse

from vocalbaby.experiments.scripts.generate_features import generate_features_for_set
from vocalbaby.config.schemas import ConfigLoader
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
        
        # Load config
        config = ConfigLoader()
        feature_sets = args.feature_sets or config.get_feature_sets()
        
        logging.info(f"Generating features for: {feature_sets}")
        
        # Find latest artifact directory
        artifacts_base = "artifacts"
        artifact_dirs = [d for d in os.listdir(artifacts_base) if d != "latest" and os.path.isdir(os.path.join(artifacts_base, d))]
        if not artifact_dirs:
            raise FileNotFoundError("No artifact directories found. Run ingestion and validation first.")
        
        # Use latest
        latest_dir = sorted(artifact_dirs)[-1]
        artifact_dir = os.path.join(artifacts_base, latest_dir)
        logging.info(f"Using artifact directory: {artifact_dir}")
        
        # Generate features for each set
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
