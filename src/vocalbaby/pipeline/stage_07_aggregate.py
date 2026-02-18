"""
STAGE 07: RESULTS AGGREGATION

Aggregates evaluation results across all feature sets into a comparison table.
Uses the current run directory from artifacts/latest.

Supports flat, hierarchical, or both classification modes
(controlled by params.yaml → classification.mode).

Outputs:
    artifacts/<timestamp>/results/results_summary.csv                 (flat)
    artifacts/<timestamp>/results/comparison_summary.csv              (both/hierarchical)
    artifacts/<timestamp>/results/comparison_summary.json             (both/hierarchical)
    artifacts/<timestamp>/results/per_class_comparison.csv            (both — requires flat + hier)
"""
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

from vocalbaby.config.schemas import ConfigLoader
from vocalbaby.utils.run_manager import RunManager
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def aggregate_results(eval_dir: Path, feature_sets: list, splits: list = ["valid", "test"]) -> pd.DataFrame:
    """
    Aggregate evaluation metrics across feature sets and splits.

    Args:
        eval_dir: Base evaluation directory
        feature_sets: List of feature set names
        splits: List of splits to aggregate

    Returns:
        DataFrame with aggregated results
    """
    records = []

    for feature_set in feature_sets:
        feature_eval_dir = eval_dir / feature_set

        if not feature_eval_dir.exists():
            logging.warning(f"Evaluation directory not found: {feature_eval_dir}")
            continue

        for split in splits:
            metrics_file = feature_eval_dir / f"metrics_{split}.json"

            if not metrics_file.exists():
                logging.warning(f"Metrics file not found: {metrics_file}")
                continue

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            record = {
                'feature_set': feature_set,
                'split': split,
                'uar': metrics.get('uar', 0.0),
                'f1_score': metrics.get('f1_score', 0.0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
            }
            records.append(record)

    if not records:
        raise ValueError("No evaluation results found to aggregate")

    return pd.DataFrame(records)


def main():
    """Main entry point for results aggregation stage."""
    parser = argparse.ArgumentParser(description="VocalBaby Pipeline - Stage 07: Results Aggregation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    try:
        logging.info("Starting Stage 07: Results Aggregation")

        config = ConfigLoader()
        run_dir = RunManager.get_current_run_dir()
        eval_dir = RunManager.eval_dir()
        results_dir = RunManager.results_dir()

        feature_sets = config.get_feature_sets()
        eval_splits = config.get('evaluation.eval_splits', ['valid', 'test'])

        # Classification mode: flat | hierarchical | both
        mode = config.get_classification_mode()
        run_flat = mode in ("flat", "both")
        run_hier = mode in ("hierarchical", "both")

        logging.info(f"Run directory  : {run_dir}")
        logging.info(f"Feature sets   : {feature_sets}")
        logging.info(f"Splits         : {eval_splits}")
        logging.info(f"Classification : {mode}")

        results_dir.mkdir(parents=True, exist_ok=True)
        sort_by = config.get('aggregation.sort_by', 'uar')
        sort_ascending = config.get('aggregation.sort_ascending', False)

        # ── Flat aggregation ─────────────────────────────────────────────────
        if run_flat:
            results_df = aggregate_results(
                eval_dir=eval_dir,
                feature_sets=feature_sets,
                splits=eval_splits,
            )
            results_df = results_df.sort_values(by=sort_by, ascending=sort_ascending)

            output_path = results_dir / "results_summary.csv"
            results_df.to_csv(output_path, index=False)

            logging.info(f"\nFlat Results Summary:")
            logging.info("\n" + results_df.to_string(index=False))
            logging.info(f"\nSaved to: {output_path}")

        # ── Hierarchical comparison aggregation ──────────────────────────────
        if run_hier:
            from vocalbaby.experiments.scripts.aggregate_comparison import (
                aggregate_comparison,
                build_per_class_comparison,
            )

            artifact_dir = str(run_dir)

            comparison_df = aggregate_comparison(
                artifact_dir=artifact_dir,
                feature_sets=feature_sets,
                splits=eval_splits,
            )
            if not comparison_df.empty:
                logging.info(f"\nComparison summary saved to results/")

            # Per-class comparison requires both flat + hier eval data
            if mode == "both":
                per_class_df = build_per_class_comparison(
                    artifact_dir=artifact_dir,
                    feature_sets=feature_sets,
                )
                if not per_class_df.empty:
                    logging.info(f"Per-class comparison saved to results/")

        logging.info("Stage 07 completed: Results aggregation")

        return 0

    except Exception as e:
        logging.error(f"Stage 07 failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
