"""
Training script for hierarchical classification.

Trains three XGBoost models per feature set using best hyperparameters
found by tune_hierarchical.py:
    1. Stage-1:  Emotional vs Non-Emotional
    2. Stage-2A: Crying vs Laughing
    3. Stage-2B: Canonical vs Junk vs Non-canonical

Saves a HierarchicalClassifier bundle under:
    artifacts/<run>/models/<feature_set>/hierarchical/

Usage:
    python -m vocalbaby.experiments.scripts.train_hierarchical --feature-set egemaps
    python -m vocalbaby.experiments.scripts.train_hierarchical --feature-set all
"""

import os
import sys
import argparse
import numpy as np

from vocalbaby.experiments.data_loader import get_latest_artifact_dir, load_labels
from vocalbaby.experiments.hierarchy import (
    get_stage1_labels,
    get_stage2_data,
    train_stage_model,
    HierarchicalClassifier,
    STAGE_NAMES,
)
from vocalbaby.experiments.hyperparameter_tuning import load_best_params
from vocalbaby.experiments.training import save_label_encoder
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


FEATURE_SETS = ["egemaps", "mfcc", "hubert_ssl", "wav2vec2_ssl"]


def train_hierarchical_feature_set(
    feature_set: str,
    artifact_dir: str,
    random_state: int = 42,
):
    """
    Train all three hierarchical stage models for one feature set.
    """
    try:
        logging.info("=" * 80)
        logging.info(f"HIERARCHICAL TRAINING: {feature_set.upper()}")
        logging.info("=" * 80)

        # ── Load features (full training set) ────────────────────────────────
        feat_dir = os.path.join(artifact_dir, "features", feature_set)
        X_train = np.load(os.path.join(feat_dir, "train/features.npy"))
        logging.info(f"Loaded train features: {X_train.shape}")

        # ── Load labels ──────────────────────────────────────────────────────
        y_train, _, _, le_flat, class_names = load_labels(artifact_dir)

        # ── Locate tuning directory ──────────────────────────────────────────
        hier_tune_dir = os.path.join(artifact_dir, "tuning", feature_set, "hierarchical")
        hier_model_dir = os.path.join(artifact_dir, "models", feature_set, "hierarchical")

        # ── Stage 1: Emotional vs Non-Emotional ─────────────────────────────
        print(f"\n  ── Stage 1: Emotional vs Non-Emotional ──", flush=True)
        y_train_s1 = get_stage1_labels(y_train, class_names)

        params_s1 = load_best_params(os.path.join(hier_tune_dir, "stage1", "best_params.json"))
        model_s1, imputer_s1 = train_stage_model(
            X_train, y_train_s1, params_s1,
            n_classes=2, random_state=random_state,
        )

        # ── Stage 2A: Crying vs Laughing ─────────────────────────────────────
        print(f"\n  ── Stage 2A: Crying vs Laughing ──", flush=True)
        X_train_emo, y_train_emo, le_emo, emo_names = get_stage2_data(
            X_train, y_train, class_names, "emotional",
        )

        params_s2e = load_best_params(
            os.path.join(hier_tune_dir, "stage2_emotional", "best_params.json")
        )
        model_s2e, _ = train_stage_model(
            X_train_emo, y_train_emo, params_s2e,
            n_classes=2, random_state=random_state,
        )

        # ── Stage 2B: Canonical vs Junk vs Non-canonical ─────────────────────
        print(f"\n  ── Stage 2B: Canonical vs Junk vs Non-canonical ──", flush=True)
        X_train_non, y_train_non, le_non, non_names = get_stage2_data(
            X_train, y_train, class_names, "non_emotional",
        )

        params_s2n = load_best_params(
            os.path.join(hier_tune_dir, "stage2_non_emotional", "best_params.json")
        )
        model_s2n, _ = train_stage_model(
            X_train_non, y_train_non, params_s2n,
            n_classes=3, random_state=random_state,
        )

        # ── Bundle and save ──────────────────────────────────────────────────
        hcls = HierarchicalClassifier(
            model_stage1=model_s1,
            model_stage2_emotional=model_s2e,
            model_stage2_non_emotional=model_s2n,
            imputer=imputer_s1,           # single imputer for all stages
            flat_class_names=class_names,
            le_stage2_emotional=le_emo,
            le_stage2_non_emotional=le_non,
        )
        hcls.save(hier_model_dir)

        # Also save the flat label encoder alongside
        save_label_encoder(le_flat, os.path.join(hier_model_dir, "label_encoder_flat.pkl"))

        logging.info("=" * 80)
        logging.info(f"Hierarchical training completed for {feature_set}")
        logging.info(f"Models saved to: {hier_model_dir}")
        logging.info("=" * 80)

    except Exception as e:
        raise VocalBabyException(e, sys)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train hierarchical XGBoost models for comparison experiments"
    )
    parser.add_argument(
        "--feature-set", type=str, required=True,
        choices=FEATURE_SETS + ["all"],
    )
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    artifact_dir = args.artifact_dir or get_latest_artifact_dir()
    logging.info(f"Using artifact directory: {artifact_dir}")

    sets = FEATURE_SETS if args.feature_set == "all" else [args.feature_set]
    for fs in sets:
        train_hierarchical_feature_set(fs, artifact_dir, args.random_state)

    logging.info("\nALL HIERARCHICAL TRAINING COMPLETED")


if __name__ == "__main__":
    main()
