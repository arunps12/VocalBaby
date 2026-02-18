"""
STAGE 06: MODEL EVALUATION

Evaluates trained models on validation and test splits.
Uses the current run directory from artifacts/latest.

Supports flat, hierarchical, or both classification modes
(controlled by params.yaml → classification.mode).

Outputs:
    artifacts/<timestamp>/eval/<feature_set>/metrics_{split}.json               (flat)
    artifacts/<timestamp>/eval/<feature_set>/confusion_matrix_{split}.png       (flat)
    artifacts/<timestamp>/eval/<feature_set>/hierarchical/                      (hierarchical)
        hard/{confusion_matrix_*.png|csv, metrics_*.json, classification_report_*.json}
        soft/{same}
        stage1/{metrics_*.json, confusion_matrix_*.png|csv}
        stage2_emotional/{...}
        stage2_non_emotional/{...}
"""
import sys
import argparse

from vocalbaby.experiments.scripts.evaluate_model import evaluate_feature_set
from vocalbaby.experiments.scripts.evaluate_hierarchical import evaluate_hierarchical_feature_set
from vocalbaby.config.schemas import ConfigLoader
from vocalbaby.utils.run_manager import RunManager
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

        config = ConfigLoader()
        feature_sets = args.feature_sets or config.get_feature_sets()

        # Classification mode: flat | hierarchical | both
        mode = config.get_classification_mode()
        run_flat = mode in ("flat", "both")
        run_hier = mode in ("hierarchical", "both")

        artifact_dir = str(RunManager.get_current_run_dir())
        logging.info(f"Run directory  : {artifact_dir}")
        logging.info(f"Feature sets   : {feature_sets}")
        logging.info(f"Classification : {mode}")

        for feature_set in feature_sets:
            # ── Flat evaluation ──────────────────────────────────────────────
            if run_flat:
                logging.info(f"\n{'='*80}")
                logging.info(f"Evaluating model (flat): {feature_set}")
                logging.info(f"{'='*80}")

                evaluate_feature_set(
                    feature_set=feature_set,
                    artifact_dir=artifact_dir,
                )

            # ── Hierarchical evaluation ──────────────────────────────────────
            if run_hier:
                logging.info(f"\n{'='*80}")
                logging.info(f"Evaluating model (hierarchical): {feature_set}")
                logging.info(f"{'='*80}")

                evaluate_hierarchical_feature_set(
                    feature_set=feature_set,
                    artifact_dir=artifact_dir,
                )

        logging.info("Stage 06 completed: Model evaluation")
        return 0

    except Exception as e:
        logging.error(f"Stage 06 failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
