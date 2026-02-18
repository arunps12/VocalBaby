"""
Hyperparameter tuning for hierarchical classification.

Tunes three separate XGBoost models per feature set:
    1. Stage-1 (binary: Emotional vs Non-Emotional)
    2. Stage-2A (binary: Crying vs Laughing)
    3. Stage-2B (3-class: Canonical vs Junk vs Non-canonical)

Each model gets its own Optuna study with the same search space,
trial budget, and multi-objective strategy as the flat pipeline.

Usage:
    python -m vocalbaby.experiments.scripts.tune_hierarchical --feature-set egemaps
    python -m vocalbaby.experiments.scripts.tune_hierarchical --feature-set all
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, Tuple
from collections import Counter

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from vocalbaby.experiments.data_loader import get_latest_artifact_dir, load_labels
from vocalbaby.experiments.hierarchy import (
    get_stage1_labels,
    get_stage2_data,
    STAGE_NAMES,
)
from vocalbaby.experiments.hyperparameter_tuning import (
    DEFAULT_SEARCH_SPACE,
    save_best_params,
)
from vocalbaby.experiments.training import save_label_encoder
from vocalbaby.utils.ml_utils.metric.classification_metric import get_classification_score
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


FEATURE_SETS = ["egemaps", "mfcc", "hubert_ssl", "wav2vec2_ssl"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage-specific Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────


def _run_optuna_for_stage(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    n_classes: int,
    n_trials: int,
    random_state: int,
    study_name: str,
    search_space: Dict = None,
    apply_smote: bool = True,
    use_class_weights: bool = True,
) -> Tuple[Dict, optuna.study.Study]:
    """
    Run Optuna tuning for a single stage model.

    Mirrors the flat pipeline's tuning logic but adapts the objective
    function for binary vs multi-class as needed.
    """
    try:
        sp = search_space or DEFAULT_SEARCH_SPACE

        # Preprocessing (same as flat pipeline — fit imputer on train)
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_valid_imp = imputer.transform(X_valid)

        if apply_smote:
            sm = SMOTE(random_state=random_state)
            X_res, y_res = sm.fit_resample(X_train_imp, y_train)
            logging.info(f"    SMOTE distribution: {dict(Counter(y_res))}")
        else:
            X_res, y_res = X_train_imp, y_train

        # Inverse-frequency sample weights
        sw = compute_sample_weight("balanced", y_res) if use_class_weights else None

        if n_classes == 2:
            objective_str = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective_str = "multi:softprob"
            eval_metric = "mlogloss"

        def objective(trial):
            sp_md = sp.get("max_depth", [3, 12])
            sp_lr = sp.get("learning_rate", [0.01, 0.3])
            sp_ne = sp.get("n_estimators", [100, 1500])
            sp_ss = sp.get("subsample", [0.5, 1.0])
            sp_cb = sp.get("colsample_bytree", [0.5, 1.0])
            sp_gm = sp.get("gamma", [0.0, 5.0])
            sp_mc = sp.get("min_child_weight", [1, 10])
            sp_ra = sp.get("reg_alpha", [0.0, 5.0])
            sp_rl = sp.get("reg_lambda", [0.0, 5.0])

            params = {
                "max_depth": trial.suggest_int("max_depth", int(sp_md[0]), int(sp_md[1])),
                "learning_rate": trial.suggest_float("learning_rate", float(sp_lr[0]), float(sp_lr[1]), log=True),
                "n_estimators": trial.suggest_int("n_estimators", int(sp_ne[0]), int(sp_ne[1])),
                "subsample": trial.suggest_float("subsample", float(sp_ss[0]), float(sp_ss[1])),
                "colsample_bytree": trial.suggest_float("colsample_bytree", float(sp_cb[0]), float(sp_cb[1])),
                "gamma": trial.suggest_float("gamma", float(sp_gm[0]), float(sp_gm[1])),
                "min_child_weight": trial.suggest_int("min_child_weight", int(sp_mc[0]), int(sp_mc[1])),
                "reg_lambda": trial.suggest_float("reg_lambda", float(sp_rl[0]), float(sp_rl[1])),
                "reg_alpha": trial.suggest_float("reg_alpha", float(sp_ra[0]), float(sp_ra[1])),
            }

            model = XGBClassifier(
                **params,
                objective=objective_str,
                eval_metric=eval_metric,
                tree_method="hist",
                device="cuda",
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_res, y_res, sample_weight=sw)

            y_pred = model.predict(X_valid_imp)
            metrics = get_classification_score(y_valid, y_pred)

            trial.set_user_attr("UAR", metrics.uar)
            trial.set_user_attr("F1", metrics.f1_score)
            print(f"      [Trial {trial.number:>3d}/{n_trials}] UAR={metrics.uar:.4f}  F1={metrics.f1_score:.4f}", flush=True)

            return metrics.uar, metrics.f1_score

        study = optuna.create_study(
            directions=["maximize", "maximize"],
            study_name=study_name,
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_trial = max(study.best_trials, key=lambda t: t.values[0])
        best_params = best_trial.params

        print(f"      Best trial #{best_trial.number}  UAR={best_trial.values[0]:.4f}  F1={best_trial.values[1]:.4f}", flush=True)
        logging.info(f"    Best params: {best_params}")

        return best_params, study

    except Exception as e:
        raise VocalBabyException(e, sys)


# ─────────────────────────────────────────────────────────────────────────────
# Feature set tuning orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def tune_hierarchical_feature_set(
    feature_set: str,
    artifact_dir: str,
    n_trials: int = 40,
    random_state: int = 42,
    search_space: Dict = None,
    use_class_weights: bool = True,
):
    """
    Tune hierarchical models for one feature set.

    Saves best_params.json for each stage under:
        artifacts/<run>/tuning/<feature_set>/hierarchical/stage1/best_params.json
        artifacts/<run>/tuning/<feature_set>/hierarchical/stage2_emotional/best_params.json
        artifacts/<run>/tuning/<feature_set>/hierarchical/stage2_non_emotional/best_params.json
    """
    try:
        logging.info("=" * 80)
        logging.info(f"HIERARCHICAL TUNING: {feature_set.upper()}")
        logging.info("=" * 80)

        # ── Load features ────────────────────────────────────────────────────
        feat_dir = os.path.join(artifact_dir, "features", feature_set)
        X_train = np.load(os.path.join(feat_dir, "train/features.npy"))
        X_valid = np.load(os.path.join(feat_dir, "valid/features.npy"))

        # ── Load labels (flat 5-class) ───────────────────────────────────────
        y_train, y_valid, _, le_flat, class_names = load_labels(artifact_dir)

        # ── Output base ──────────────────────────────────────────────────────
        hier_tune_dir = os.path.join(artifact_dir, "tuning", feature_set, "hierarchical")

        # ── Stage 1: Emotional vs Non-Emotional ─────────────────────────────
        print(f"\n  ── Stage 1: Emotional vs Non-Emotional ──", flush=True)
        logging.info("Stage 1: Emotional vs Non-Emotional")

        y_train_s1 = get_stage1_labels(y_train, class_names)
        y_valid_s1 = get_stage1_labels(y_valid, class_names)

        logging.info(f"  Stage-1 train distribution: {dict(Counter(y_train_s1))}")
        logging.info(f"  Stage-1 valid distribution: {dict(Counter(y_valid_s1))}")

        best_s1, _ = _run_optuna_for_stage(
            X_train, y_train_s1,
            X_valid, y_valid_s1,
            n_classes=2,
            n_trials=n_trials,
            random_state=random_state,
            study_name=f"hier_{feature_set}_stage1",
            search_space=search_space,
            use_class_weights=use_class_weights,
        )
        s1_dir = os.path.join(hier_tune_dir, "stage1")
        save_best_params(best_s1, os.path.join(s1_dir, "best_params.json"))

        # ── Stage 2A: Crying vs Laughing ─────────────────────────────────────
        print(f"\n  ── Stage 2A: Crying vs Laughing ──", flush=True)
        logging.info("Stage 2A: Crying vs Laughing")

        X_train_emo, y_train_emo, le_emo, emo_names = get_stage2_data(
            X_train, y_train, class_names, "emotional"
        )
        X_valid_emo, y_valid_emo, _, _ = get_stage2_data(
            X_valid, y_valid, class_names, "emotional"
        )

        logging.info(f"  Stage-2A train distribution: {dict(Counter(y_train_emo))}")
        logging.info(f"  Stage-2A valid distribution: {dict(Counter(y_valid_emo))}")

        best_s2e, _ = _run_optuna_for_stage(
            X_train_emo, y_train_emo,
            X_valid_emo, y_valid_emo,
            n_classes=2,
            n_trials=n_trials,
            random_state=random_state,
            study_name=f"hier_{feature_set}_stage2_emotional",
            search_space=search_space,
            use_class_weights=use_class_weights,
        )
        s2e_dir = os.path.join(hier_tune_dir, "stage2_emotional")
        save_best_params(best_s2e, os.path.join(s2e_dir, "best_params.json"))
        save_label_encoder(le_emo, os.path.join(s2e_dir, "label_encoder.pkl"))

        # ── Stage 2B: Canonical vs Junk vs Non-canonical ─────────────────────
        print(f"\n  ── Stage 2B: Canonical vs Junk vs Non-canonical ──", flush=True)
        logging.info("Stage 2B: Canonical vs Junk vs Non-canonical")

        X_train_non, y_train_non, le_non, non_names = get_stage2_data(
            X_train, y_train, class_names, "non_emotional"
        )
        X_valid_non, y_valid_non, _, _ = get_stage2_data(
            X_valid, y_valid, class_names, "non_emotional"
        )

        logging.info(f"  Stage-2B train distribution: {dict(Counter(y_train_non))}")
        logging.info(f"  Stage-2B valid distribution: {dict(Counter(y_valid_non))}")

        best_s2n, _ = _run_optuna_for_stage(
            X_train_non, y_train_non,
            X_valid_non, y_valid_non,
            n_classes=3,
            n_trials=n_trials,
            random_state=random_state,
            study_name=f"hier_{feature_set}_stage2_non_emotional",
            search_space=search_space,
            use_class_weights=use_class_weights,
        )
        s2n_dir = os.path.join(hier_tune_dir, "stage2_non_emotional")
        save_best_params(best_s2n, os.path.join(s2n_dir, "best_params.json"))
        save_label_encoder(le_non, os.path.join(s2n_dir, "label_encoder.pkl"))

        # Save flat label encoder for reference
        save_label_encoder(le_flat, os.path.join(hier_tune_dir, "label_encoder_flat.pkl"))

        logging.info("=" * 80)
        logging.info(f"Hierarchical tuning completed for {feature_set}")
        logging.info("=" * 80)

    except Exception as e:
        raise VocalBabyException(e, sys)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Tune hierarchical XGBoost models for comparison experiments"
    )
    parser.add_argument(
        "--feature-set", type=str, required=True,
        choices=FEATURE_SETS + ["all"],
        help="Feature set to tune (or 'all')",
    )
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    artifact_dir = args.artifact_dir or get_latest_artifact_dir()
    logging.info(f"Using artifact directory: {artifact_dir}")

    sets = FEATURE_SETS if args.feature_set == "all" else [args.feature_set]
    for fs in sets:
        tune_hierarchical_feature_set(
            fs, artifact_dir, args.n_trials, args.random_state,
        )

    logging.info("\nALL HIERARCHICAL TUNING COMPLETED")


if __name__ == "__main__":
    main()
