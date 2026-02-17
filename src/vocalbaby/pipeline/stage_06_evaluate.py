"""
STAGE 06: MODEL EVALUATION

Evaluates trained models on validation and test splits.
Saves confusion matrices, metrics, and classification reports.

Outputs (for each feature set):
    artifacts/eval/<feature_set>/confusion_matrix_valid.{png,csv}
    artifacts/eval/<feature_set>/confusion_matrix_test.{png,csv}
    artifacts/eval/<feature_set>/metrics_valid.json
    artifacts/eval/<feature_set>/metrics_test.json
    artifacts/eval/<feature_set>/classification_report_valid.json
    artifacts/eval/<feature_set>/classification_report_test.json
"""
import sys
import argparse

from vocalbaby.experiments.scripts.evaluate_model import evaluate_model_for_set
from vocalbaby.config.schemas import ConfigLoader
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def main():
    """Main entry point for model evaluation stage."""
    parser = argparse.ArgumentParser(description="VocalBaby Pipeline - Stage 06: Model Evaluation")
    parser.add_argument("--feature-sets", nargs="+", help="Feature sets to evaluate (default: all from params.yaml)")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    try:
        logging.info("Starting Stage 06: Model Evaluation")
        
        # Load config
        config = ConfigLoader()
        feature_sets = args.feature_sets or config.get_feature_sets()
        
        logging.info(f"Evaluating models for: {feature_sets}")
        
        # Evaluate each feature set
        for feature_set in feature_sets:
            logging.info(f"\n{'='*80}")
            logging.info(f"Evaluating model: {feature_set}")
            logging.info(f"{'='*80}")
            
            evaluate_model_for_set(
                feature_set=feature_set,
                force_reeval=args.force,
            )
        
        logging.info("Stage 06 completed: Model evaluation")
        return 0
        
    except Exception as e:
        logging.error(f"Stage 06 failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
