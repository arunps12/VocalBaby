"""
Results aggregation for flat + hierarchical comparison.

Reads metrics from both flat and hierarchical evaluation directories
and produces a combined summary CSV + JSON for scientific comparison.

Output:
    artifacts/<run>/results/comparison_summary.csv
    artifacts/<run>/results/comparison_summary.json

Usage:
    python -m vocalbaby.experiments.scripts.aggregate_comparison
    python -m vocalbaby.experiments.scripts.aggregate_comparison --artifact-dir artifacts/02_17_2026_10_52_02
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

from vocalbaby.experiments.data_loader import get_latest_artifact_dir
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


FEATURE_SETS = ["egemaps", "mfcc", "hubert_ssl", "wav2vec2_ssl"]
SPLITS = ["valid", "test"]


def _load_json_safe(path: str) -> Dict:
    """Load JSON or return empty dict."""
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def aggregate_comparison(
    artifact_dir: str,
    feature_sets: List[str] = None,
    splits: List[str] = None,
) -> pd.DataFrame:
    """
    Build a comparison table across feature sets, modes, and splits.

    Columns:
        feature_set | mode | routing | split | uar | f1_score | precision | recall
    """
    try:
        feature_sets = feature_sets or FEATURE_SETS
        splits = splits or SPLITS

        rows = []

        for fs in feature_sets:
            # ── Flat metrics ─────────────────────────────────────────────────
            flat_eval_dir = os.path.join(artifact_dir, "eval", fs)
            for split in splits:
                m = _load_json_safe(os.path.join(flat_eval_dir, f"metrics_{split}.json"))
                if m:
                    rows.append({
                        "feature_set": fs,
                        "mode": "flat",
                        "routing": "-",
                        "split": split,
                        **{k: m.get(k, np.nan) for k in ("uar", "f1_score", "precision", "recall")},
                    })

            # ── Hierarchical metrics (hard + soft) ───────────────────────────
            for routing in ("hard", "soft"):
                hier_route_dir = os.path.join(artifact_dir, "eval", fs, "hierarchical", routing)
                for split in splits:
                    m = _load_json_safe(os.path.join(hier_route_dir, f"metrics_{split}.json"))
                    if m:
                        rows.append({
                            "feature_set": fs,
                            "mode": "hierarchical",
                            "routing": routing,
                            "split": split,
                            **{k: m.get(k, np.nan) for k in ("uar", "f1_score", "precision", "recall")},
                        })

        if not rows:
            logging.warning("No metrics found — nothing to aggregate.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.sort_values(["split", "uar"], ascending=[True, False], inplace=True)

        # ── Save ─────────────────────────────────────────────────────────────
        results_dir = os.path.join(artifact_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "comparison_summary.csv")
        df.to_csv(csv_path, index=False)

        json_path = os.path.join(results_dir, "comparison_summary.json")
        df.to_json(json_path, orient="records", indent=2)

        logging.info(f"\nComparison Summary:\n{df.to_string(index=False)}")
        logging.info(f"\nSaved to: {csv_path}")

        return df

    except Exception as e:
        raise VocalBabyException(e, sys)


def build_per_class_comparison(
    artifact_dir: str,
    feature_sets: List[str] = None,
) -> pd.DataFrame:
    """
    Build a per-class F1 comparison (flat vs hierarchical) for the test split.

    This directly answers the scientific questions:
        1) Does hierarchy improve Laughing recall?
        2) Does hierarchy improve Crying recall?
        3) Which feature set benefits most?
    """
    try:
        feature_sets = feature_sets or FEATURE_SETS
        rows = []

        for fs in feature_sets:
            # Flat classification report
            flat_report_path = os.path.join(
                artifact_dir, "eval", fs, "classification_report_test.json"
            )
            flat_report = _load_json_safe(flat_report_path)

            # Hierarchical (hard routing) classification report
            hier_report_path = os.path.join(
                artifact_dir, "eval", fs, "hierarchical", "hard",
                "classification_report_test.json",
            )
            hier_report = _load_json_safe(hier_report_path)

            if flat_report and hier_report:
                for cls_name in ("Canonical", "Crying", "Junk", "Laughing", "Non-canonical"):
                    flat_cls = flat_report.get(cls_name, {})
                    hier_cls = hier_report.get(cls_name, {})
                    rows.append({
                        "feature_set": fs,
                        "class": cls_name,
                        "flat_precision": flat_cls.get("precision", np.nan),
                        "flat_recall": flat_cls.get("recall", np.nan),
                        "flat_f1": flat_cls.get("f1-score", np.nan),
                        "hier_precision": hier_cls.get("precision", np.nan),
                        "hier_recall": hier_cls.get("recall", np.nan),
                        "hier_f1": hier_cls.get("f1-score", np.nan),
                    })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Compute deltas
        df["delta_f1"] = df["hier_f1"] - df["flat_f1"]
        df["delta_recall"] = df["hier_recall"] - df["flat_recall"]

        results_dir = os.path.join(artifact_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "per_class_comparison.csv")
        df.to_csv(csv_path, index=False)

        logging.info(f"\nPer-class comparison (test):\n{df.to_string(index=False)}")
        logging.info(f"\nSaved to: {csv_path}")

        return df

    except Exception as e:
        raise VocalBabyException(e, sys)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate flat vs hierarchical experiment results"
    )
    parser.add_argument("--artifact-dir", type=str, default=None)

    args = parser.parse_args()
    artifact_dir = args.artifact_dir or get_latest_artifact_dir()

    aggregate_comparison(artifact_dir)
    build_per_class_comparison(artifact_dir)


if __name__ == "__main__":
    main()
