"""
Evaluation script for hierarchical classification.

Loads a trained HierarchicalClassifier, runs inference on valid/test
splits using both hard and soft routing, and saves:
    - 5-class confusion matrices (PNG + CSV)
    - Per-class precision/recall/F1
    - Aggregated metrics (JSON)
    - Per-stage confusion matrices for interpretability

Artifacts are written to:
    artifacts/<run>/eval/<feature_set>/hierarchical/

Usage:
    python -m vocalbaby.experiments.scripts.evaluate_hierarchical --feature-set egemaps
    python -m vocalbaby.experiments.scripts.evaluate_hierarchical --feature-set all
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List

from sklearn.metrics import confusion_matrix, classification_report

from vocalbaby.experiments.data_loader import get_latest_artifact_dir, load_labels
from vocalbaby.experiments.hierarchy import (
    HierarchicalClassifier,
    get_stage1_labels,
    get_stage2_data,
    STAGE1_NAMES,
)
from vocalbaby.experiments.evaluation import (
    save_confusion_matrix_png,
    save_confusion_matrix_csv,
    save_json,
    save_class_labels,
)
from vocalbaby.utils.ml_utils.metric.classification_metric import get_classification_score
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


FEATURE_SETS = ["egemaps", "mfcc", "hubert_ssl", "wav2vec2_ssl"]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_five_class(
    hcls: HierarchicalClassifier,
    X: np.ndarray,
    y_enc: np.ndarray,
    class_names: List[str],
    split_name: str,
    feature_set: str,
    output_dir: str,
    routing: str = "hard",
) -> Dict:
    """
    Full 5-class evaluation through the hierarchical pipeline.

    Saves confusion matrix (PNG+CSV), classification report (JSON),
    and scalar metrics (JSON).
    """
    try:
        tag = f"hierarchical_{routing}"
        logging.info(f"  Evaluating {split_name} ({routing} routing)...")

        y_pred = hcls.predict(X, routing=routing)

        # Scalar metrics
        metrics = get_classification_score(y_enc, y_pred)

        metrics_dict = {
            "uar": float(metrics.uar),
            "f1_score": float(metrics.f1_score),
            "precision": float(metrics.precision_score),
            "recall": float(metrics.recall_score),
            "routing": routing,
        }

        # Per-class report
        report = classification_report(
            y_enc, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        # Confusion matrix
        cm = confusion_matrix(y_enc, y_pred)

        # ── Save artifacts ───────────────────────────────────────────────────
        route_dir = os.path.join(output_dir, routing)
        os.makedirs(route_dir, exist_ok=True)

        save_confusion_matrix_png(
            cm, class_names,
            os.path.join(route_dir, f"confusion_matrix_{split_name}.png"),
            title=f"{feature_set.upper()} Hierarchical ({routing}) — {split_name.upper()}",
        )
        save_confusion_matrix_csv(
            cm, class_names,
            os.path.join(route_dir, f"confusion_matrix_{split_name}.csv"),
        )
        save_json(report, os.path.join(route_dir, f"classification_report_{split_name}.json"))
        save_json(metrics_dict, os.path.join(route_dir, f"metrics_{split_name}.json"))

        logging.info(
            f"    {split_name} ({routing}): UAR={metrics.uar:.4f}  "
            f"F1={metrics.f1_score:.4f}  Prec={metrics.precision_score:.4f}"
        )

        return metrics_dict

    except Exception as e:
        raise VocalBabyException(e, sys)


def _evaluate_stage1(
    hcls: HierarchicalClassifier,
    X: np.ndarray,
    y_enc: np.ndarray,
    class_names: List[str],
    split_name: str,
    feature_set: str,
    output_dir: str,
):
    """Save stage-1 (binary) confusion matrix for interpretability."""
    try:
        X_imp = hcls.imputer.transform(X)
        y_pred_s1 = hcls.model_stage1.predict(X_imp)
        y_true_s1 = get_stage1_labels(y_enc, class_names)

        cm = confusion_matrix(y_true_s1, y_pred_s1)
        s1_dir = os.path.join(output_dir, "stage1")
        os.makedirs(s1_dir, exist_ok=True)

        save_confusion_matrix_png(
            cm, STAGE1_NAMES,
            os.path.join(s1_dir, f"confusion_matrix_{split_name}.png"),
            title=f"{feature_set.upper()} Stage-1 — {split_name.upper()}",
        )
        save_confusion_matrix_csv(
            cm, STAGE1_NAMES,
            os.path.join(s1_dir, f"confusion_matrix_{split_name}.csv"),
        )

        metrics = get_classification_score(y_true_s1, y_pred_s1)
        save_json(
            {
                "uar": float(metrics.uar),
                "f1_score": float(metrics.f1_score),
                "precision": float(metrics.precision_score),
                "recall": float(metrics.recall_score),
            },
            os.path.join(s1_dir, f"metrics_{split_name}.json"),
        )

    except Exception as e:
        raise VocalBabyException(e, sys)


def _evaluate_stage2_branch(
    hcls: HierarchicalClassifier,
    X: np.ndarray,
    y_enc: np.ndarray,
    class_names: List[str],
    branch: str,
    split_name: str,
    feature_set: str,
    output_dir: str,
):
    """Save confusion matrix for one stage-2 branch (on its own subset)."""
    try:
        X_sub, y_sub, le_sub, sub_names = get_stage2_data(
            X, y_enc, class_names, branch,
        )
        X_sub_imp = hcls.imputer.transform(X_sub)

        if branch == "emotional":
            y_pred = hcls.model_stage2_emotional.predict(X_sub_imp)
        else:
            y_pred = hcls.model_stage2_non_emotional.predict(X_sub_imp)

        cm = confusion_matrix(y_sub, y_pred)
        br_dir = os.path.join(output_dir, f"stage2_{branch}")
        os.makedirs(br_dir, exist_ok=True)

        save_confusion_matrix_png(
            cm, sub_names,
            os.path.join(br_dir, f"confusion_matrix_{split_name}.png"),
            title=f"{feature_set.upper()} Stage-2 {branch.title()} — {split_name.upper()}",
        )
        save_confusion_matrix_csv(
            cm, sub_names,
            os.path.join(br_dir, f"confusion_matrix_{split_name}.csv"),
        )

        metrics = get_classification_score(y_sub, y_pred)
        save_json(
            {
                "uar": float(metrics.uar),
                "f1_score": float(metrics.f1_score),
                "precision": float(metrics.precision_score),
                "recall": float(metrics.recall_score),
            },
            os.path.join(br_dir, f"metrics_{split_name}.json"),
        )

    except Exception as e:
        raise VocalBabyException(e, sys)


# ─────────────────────────────────────────────────────────────────────────────
# Feature-set level evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_hierarchical_feature_set(
    feature_set: str,
    artifact_dir: str,
):
    """
    Full hierarchical evaluation for one feature set.

    Produces:
        eval/<feature_set>/hierarchical/
            hard/
                confusion_matrix_valid.png|csv
                confusion_matrix_test.png|csv
                classification_report_valid|test.json
                metrics_valid|test.json
            soft/
                (same layout)
            stage1/
                confusion_matrix_valid|test.png|csv
                metrics_valid|test.json
            stage2_emotional/
                ...
            stage2_non_emotional/
                ...
    """
    try:
        logging.info("=" * 80)
        logging.info(f"HIERARCHICAL EVALUATION: {feature_set.upper()}")
        logging.info("=" * 80)

        # ── Load model ───────────────────────────────────────────────────────
        hier_model_dir = os.path.join(artifact_dir, "models", feature_set, "hierarchical")
        hcls = HierarchicalClassifier.load(hier_model_dir)

        # ── Load features ────────────────────────────────────────────────────
        feat_dir = os.path.join(artifact_dir, "features", feature_set)
        X_valid = np.load(os.path.join(feat_dir, "valid/features.npy"))
        X_test = np.load(os.path.join(feat_dir, "test/features.npy"))

        # ── Load labels ──────────────────────────────────────────────────────
        _, y_valid, y_test, _, class_names = load_labels(artifact_dir)

        eval_dir = os.path.join(artifact_dir, "eval", feature_set, "hierarchical")
        os.makedirs(eval_dir, exist_ok=True)

        # Save class label mapping
        save_class_labels(class_names, os.path.join(eval_dir, "labels.json"))

        # ── 5-class evaluations (hard + soft) ────────────────────────────────
        all_metrics = {}
        for routing in ("hard", "soft"):
            for split_name, X, y in [("valid", X_valid, y_valid), ("test", X_test, y_test)]:
                key = f"{routing}_{split_name}"
                all_metrics[key] = _evaluate_five_class(
                    hcls, X, y, class_names,
                    split_name, feature_set, eval_dir, routing,
                )

        # ── Per-stage evaluations ────────────────────────────────────────────
        for split_name, X, y in [("valid", X_valid, y_valid), ("test", X_test, y_test)]:
            _evaluate_stage1(hcls, X, y, class_names, split_name, feature_set, eval_dir)
            _evaluate_stage2_branch(hcls, X, y, class_names, "emotional", split_name, feature_set, eval_dir)
            _evaluate_stage2_branch(hcls, X, y, class_names, "non_emotional", split_name, feature_set, eval_dir)

        # ── Summary metrics JSON ─────────────────────────────────────────────
        save_json(all_metrics, os.path.join(eval_dir, "metrics_summary.json"))

        logging.info("=" * 80)
        logging.info(f"Hierarchical evaluation completed for {feature_set}")
        logging.info(f"Artifacts saved to: {eval_dir}")
        logging.info("=" * 80)

    except Exception as e:
        raise VocalBabyException(e, sys)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical XGBoost models"
    )
    parser.add_argument(
        "--feature-set", type=str, required=True,
        choices=FEATURE_SETS + ["all"],
    )
    parser.add_argument("--artifact-dir", type=str, default=None)

    args = parser.parse_args()

    artifact_dir = args.artifact_dir or get_latest_artifact_dir()
    logging.info(f"Using artifact directory: {artifact_dir}")

    sets = FEATURE_SETS if args.feature_set == "all" else [args.feature_set]
    for fs in sets:
        evaluate_hierarchical_feature_set(fs, artifact_dir)

    logging.info("\nALL HIERARCHICAL EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
