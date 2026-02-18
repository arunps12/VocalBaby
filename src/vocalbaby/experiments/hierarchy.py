"""
Hierarchical classification for infant vocalization.

Implements a two-stage hierarchy:
    Stage 1 (Binary):  Emotional vs Non-Emotional
    Stage 2A:          Crying vs Laughing         (if Emotional)
    Stage 2B:          Canonical vs Junk vs Non-canonical (if Non-Emotional)

Final prediction combines stage probabilities into a full 5-class vector,
preserving the same label order as the flat LabelEncoder (alphabetical):
    Canonical(0), Crying(1), Junk(2), Laughing(3), Non-canonical(4)
"""

import os
import sys
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy definition (FIXED — do not modify)
# ─────────────────────────────────────────────────────────────────────────────

EMOTIONAL_CLASSES = {"Crying", "Laughing"}
NON_EMOTIONAL_CLASSES = {"Canonical", "Junk", "Non-canonical"}

STAGE1_NAMES = ["Non-Emotional", "Emotional"]  # 0 = Non‑Emotional, 1 = Emotional

STAGE_NAMES = {
    "stage1": "Binary: Emotional vs Non-Emotional",
    "stage2_emotional": "Emotional: Crying vs Laughing",
    "stage2_non_emotional": "Non-Emotional: Canonical vs Junk vs Non-canonical",
}


# ─────────────────────────────────────────────────────────────────────────────
# Label mapping utilities
# ─────────────────────────────────────────────────────────────────────────────


def build_emotional_mask(class_names: List[str]) -> set:
    """Return set of encoded indices that belong to the Emotional branch."""
    return {i for i, name in enumerate(class_names) if name in EMOTIONAL_CLASSES}


def get_stage1_labels(
    y_enc: np.ndarray, class_names: List[str]
) -> np.ndarray:
    """
    Map 5-class encoded labels → binary stage-1 labels.

    Returns:
        y_stage1: 0 = Non-Emotional, 1 = Emotional
    """
    emo_idx = build_emotional_mask(class_names)
    return np.array([1 if yi in emo_idx else 0 for yi in y_enc], dtype=np.int64)


def get_stage2_data(
    X: np.ndarray,
    y_enc: np.ndarray,
    class_names: List[str],
    branch: str,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, List[str]]:
    """
    Extract subset of data belonging to a particular stage-2 branch
    and re-encode labels for that branch.

    Args:
        X:           Feature matrix  (n_samples, n_features)
        y_enc:       Encoded labels   (n_samples,)
        class_names: Full 5-class list from the flat LabelEncoder
        branch:      'emotional' or 'non_emotional'

    Returns:
        X_sub:       Feature subset
        y_sub_enc:   Re-encoded labels for the branch
        le_sub:      Fitted LabelEncoder for the branch
        sub_names:   Class names for the branch (alphabetical within branch)
    """
    if branch == "emotional":
        target_set = EMOTIONAL_CLASSES
    elif branch == "non_emotional":
        target_set = NON_EMOTIONAL_CLASSES
    else:
        raise ValueError(f"Unknown branch: {branch}")

    target_indices = {i for i, name in enumerate(class_names) if name in target_set}
    mask = np.array([yi in target_indices for yi in y_enc])

    X_sub = X[mask]
    y_original_names = np.array([class_names[yi] for yi in y_enc[mask]])

    le_sub = LabelEncoder()
    y_sub_enc = le_sub.fit_transform(y_original_names)
    sub_names = list(le_sub.classes_)

    return X_sub, y_sub_enc, le_sub, sub_names


def compute_inverse_freq_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute per-sample inverse-frequency class weights.

    Returns:
        sample_weights array of shape (n_samples,)
    """
    return compute_sample_weight("balanced", y)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────


def train_stage_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: Dict,
    n_classes: int,
    random_state: int = 42,
    imputer_strategy: str = "median",
    apply_smote: bool = True,
    use_class_weights: bool = True,
) -> Tuple[XGBClassifier, SimpleImputer]:
    """
    Train a single XGBoost model for one stage of the hierarchy.

    Follows the same preprocessing pipeline as the flat trainer:
        1. Median imputation
        2. SMOTE oversampling  (optional)
        3. XGBoost training    (with sample weights if requested)

    Args:
        X_train:           Raw training features
        y_train:           Encoded training labels for this stage
        best_params:       Best hyperparameters
        n_classes:         Number of classes (2 or 3)
        random_state:      Seed
        imputer_strategy:  Imputation strategy
        apply_smote:       Whether to SMOTE
        use_class_weights: Apply inverse-frequency sample weights

    Returns:
        model, imputer
    """
    try:
        logging.info(f"  Train shape: {X_train.shape}, classes: {n_classes}")

        # 1. Imputation
        imputer = SimpleImputer(strategy=imputer_strategy)
        X_imp = imputer.fit_transform(X_train)

        # 2. SMOTE
        if apply_smote:
            sm = SMOTE(random_state=random_state)
            X_res, y_res = sm.fit_resample(X_imp, y_train)
            logging.info(f"  SMOTE distribution: {dict(Counter(y_res))}")
        else:
            X_res, y_res = X_imp, y_train

        # 3. Compute sample weights (inverse-frequency)
        sw = compute_inverse_freq_weights(y_res) if use_class_weights else None

        # 4. Determine objective
        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        model = XGBClassifier(
            **best_params,
            objective=objective,
            eval_metric=eval_metric,
            tree_method="hist",
            device="cuda",
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(X_res, y_res, sample_weight=sw)

        logging.info("  Training completed.")
        return model, imputer

    except Exception as e:
        raise VocalBabyException(e, sys)


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical inference
# ─────────────────────────────────────────────────────────────────────────────


class HierarchicalClassifier:
    """
    Combines three stage models into a single 5-class predictor.

    Supports two routing modes:
        hard  – argmax on stage-1 predictions, route to one branch
        soft  – use stage-1 probabilities as branch weights
    """

    def __init__(
        self,
        model_stage1: XGBClassifier,
        model_stage2_emotional: XGBClassifier,
        model_stage2_non_emotional: XGBClassifier,
        imputer: SimpleImputer,
        flat_class_names: List[str],
        le_stage2_emotional: LabelEncoder,
        le_stage2_non_emotional: LabelEncoder,
    ):
        self.model_stage1 = model_stage1
        self.model_stage2_emotional = model_stage2_emotional
        self.model_stage2_non_emotional = model_stage2_non_emotional
        self.imputer = imputer
        self.flat_class_names = flat_class_names  # alphabetical: Canonical, Crying, Junk, Laughing, Non-canonical

        self.le_emo = le_stage2_emotional
        self.le_non = le_stage2_non_emotional

        # Build index maps from branch-local positions → flat 5-class positions
        self._emo_flat_idx = [
            flat_class_names.index(name) for name in le_stage2_emotional.classes_
        ]
        self._non_flat_idx = [
            flat_class_names.index(name) for name in le_stage2_non_emotional.classes_
        ]

    # ── public API ──────────────────────────────────────────────────────────

    def predict_proba_hard(self, X: np.ndarray) -> np.ndarray:
        """
        Hard routing: argmax on stage-1, then predict via the chosen branch.

        Returns:
            proba: (n_samples, 5) probability matrix in flat class order.
        """
        X_imp = self.imputer.transform(X)
        n = X_imp.shape[0]
        n_classes = len(self.flat_class_names)

        prob_s1 = self.model_stage1.predict_proba(X_imp)  # (n, 2)
        pred_s1 = np.argmax(prob_s1, axis=1)  # 0 = Non-Emotional, 1 = Emotional

        prob_emo = self.model_stage2_emotional.predict_proba(X_imp)  # (n, 2)
        prob_non = self.model_stage2_non_emotional.predict_proba(X_imp)  # (n, 3)

        proba = np.zeros((n, n_classes), dtype=np.float64)

        for i in range(n):
            if pred_s1[i] == 1:  # Emotional
                branch_prob = prob_s1[i, 1]
                for j, flat_j in enumerate(self._emo_flat_idx):
                    proba[i, flat_j] = branch_prob * prob_emo[i, j]
            else:  # Non-Emotional
                branch_prob = prob_s1[i, 0]
                for j, flat_j in enumerate(self._non_flat_idx):
                    proba[i, flat_j] = branch_prob * prob_non[i, j]

        return proba

    def predict_proba_soft(self, X: np.ndarray) -> np.ndarray:
        """
        Soft (probabilistic) routing: use stage-1 probabilities as weights.

        P(class) = P(branch) × P(class | branch)

        Returns:
            proba: (n_samples, 5)
        """
        X_imp = self.imputer.transform(X)
        n = X_imp.shape[0]
        n_classes = len(self.flat_class_names)

        prob_s1 = self.model_stage1.predict_proba(X_imp)  # (n, 2)
        prob_emo = self.model_stage2_emotional.predict_proba(X_imp)  # (n, 2)
        prob_non = self.model_stage2_non_emotional.predict_proba(X_imp)  # (n, 3)

        proba = np.zeros((n, n_classes), dtype=np.float64)

        for j, flat_j in enumerate(self._emo_flat_idx):
            proba[:, flat_j] = prob_s1[:, 1] * prob_emo[:, j]

        for j, flat_j in enumerate(self._non_flat_idx):
            proba[:, flat_j] = prob_s1[:, 0] * prob_non[:, j]

        return proba

    def predict(self, X: np.ndarray, routing: str = "hard") -> np.ndarray:
        """Return predicted class indices in flat 5-class encoding."""
        if routing == "hard":
            proba = self.predict_proba_hard(X)
        elif routing == "soft":
            proba = self.predict_proba_soft(X)
        else:
            raise ValueError(f"Unknown routing mode: {routing}")
        return np.argmax(proba, axis=1)

    def predict_decoded(
        self, X: np.ndarray, routing: str = "hard"
    ) -> np.ndarray:
        """Return predicted class names."""
        preds = self.predict(X, routing=routing)
        return np.array([self.flat_class_names[p] for p in preds])

    # ── serialization ───────────────────────────────────────────────────────

    def save(self, output_dir: str):
        """Persist all sub-models and metadata to *output_dir*."""
        os.makedirs(output_dir, exist_ok=True)

        _save_pkl(self.model_stage1, os.path.join(output_dir, "stage1", "xgb_model.pkl"))
        _save_pkl(self.model_stage2_emotional, os.path.join(output_dir, "stage2_emotional", "xgb_model.pkl"))
        _save_pkl(self.model_stage2_non_emotional, os.path.join(output_dir, "stage2_non_emotional", "xgb_model.pkl"))
        _save_pkl(self.imputer, os.path.join(output_dir, "imputer.pkl"))
        _save_pkl(self.le_emo, os.path.join(output_dir, "stage2_emotional", "label_encoder.pkl"))
        _save_pkl(self.le_non, os.path.join(output_dir, "stage2_non_emotional", "label_encoder.pkl"))

        # Save flat class order for reconstruction
        import json
        meta = {
            "flat_class_names": self.flat_class_names,
            "emo_classes": list(self.le_emo.classes_),
            "non_classes": list(self.le_non.classes_),
        }
        meta_path = os.path.join(output_dir, "hierarchy_meta.json")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logging.info(f"Saved hierarchical classifier to: {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "HierarchicalClassifier":
        """Load a previously saved HierarchicalClassifier."""
        import json

        model_s1 = _load_pkl(os.path.join(model_dir, "stage1", "xgb_model.pkl"))
        model_s2e = _load_pkl(os.path.join(model_dir, "stage2_emotional", "xgb_model.pkl"))
        model_s2n = _load_pkl(os.path.join(model_dir, "stage2_non_emotional", "xgb_model.pkl"))
        imputer = _load_pkl(os.path.join(model_dir, "imputer.pkl"))
        le_emo = _load_pkl(os.path.join(model_dir, "stage2_emotional", "label_encoder.pkl"))
        le_non = _load_pkl(os.path.join(model_dir, "stage2_non_emotional", "label_encoder.pkl"))

        with open(os.path.join(model_dir, "hierarchy_meta.json")) as f:
            meta = json.load(f)

        return cls(
            model_stage1=model_s1,
            model_stage2_emotional=model_s2e,
            model_stage2_non_emotional=model_s2n,
            imputer=imputer,
            flat_class_names=meta["flat_class_names"],
            le_stage2_emotional=le_emo,
            le_stage2_non_emotional=le_non,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Private pickle helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_pkl(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logging.info(f"  Saved: {path}")


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
