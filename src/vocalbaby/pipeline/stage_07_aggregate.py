"""
STAGE 07: RESULTS AGGREGATION

Aggregates evaluation results across all feature sets into a comparison table.

Outputs:
    artifacts/results/results_summary.csv
"""
import sys
import os
import json
import argparse
import pandas as pd
from pathlib import Path

from vocalbaby.config.schemas import ConfigLoader, PathConfig
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def aggregate_results(eval_dir: Path, feature_sets: list, splits: list = ["valid", "test"]) -> pd.DataFrame:
    """
    Aggregate evaluation metrics across feature sets and splits.
    
    Args:
        eval_dir: Base evaluation directory
        feature_sets: List of feature set names
        splits: List of splits to aggregate (default: ["valid", "test"])
    
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
            
            # Load metrics
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Create record
            record = {
                'feature_set': feature_set,
                'split': split,
                'n_samples': metrics.get('n_samples', 0),
                'accuracy': metrics.get('accuracy', 0.0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0.0),
                'macro_f1': metrics.get('macro_f1', 0.0),
                'weighted_f1': metrics.get('weighted_f1', 0.0),
                'macro_precision': metrics.get('macro_precision', 0.0),
                'macro_recall': metrics.get('macro_recall', 0.0),
            }
            
            records.append(record)
    
    if not records:
        raise ValueError("No evaluation results found to aggregate")
    
    return pd.DataFrame(records)


def main():
    """Main entry point for results aggregation stage."""
    parser = argparse.ArgumentParser(description="VocalBaby Pipeline - Stage 07: Results Aggregation")
    parser.add_argument("--output", default="artifacts/results/results_summary.csv", help="Output CSV path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    try:
        logging.info("Starting Stage 07: Results Aggregation")
        
        # Load config
        config = ConfigLoader()
        path_config = PathConfig()
        
        feature_sets = config.get_feature_sets()
        eval_splits = config.get('evaluation.eval_splits', ['valid', 'test'])
        
        logging.info(f"Aggregating results for: {feature_sets}")
        logging.info(f"Splits: {eval_splits}")
        
        # Aggregate results
        results_df = aggregate_results(
            eval_dir=path_config.eval_dir,
            feature_sets=feature_sets,
            splits=eval_splits,
        )
        
        # Sort by macro_f1 (test split)
        sort_by = config.get('aggregation.sort_by', 'macro_f1')
        sort_ascending = config.get('aggregation.sort_ascending', False)
        results_df = results_df.sort_values(by=sort_by, ascending=sort_ascending)
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        logging.info(f"\nResults Summary:")
        logging.info("\n" + results_df.to_string(index=False))
        logging.info(f"\nSaved to: {output_path}")
        logging.info("Stage 07 completed: Results aggregation")
        
        return 0
        
    except Exception as e:
        logging.error(f"Stage 07 failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
