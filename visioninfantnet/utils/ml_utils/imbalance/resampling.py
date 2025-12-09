from collections import Counter
import sys

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight

from visioninfantnet.exception.exception import VisionInfantNetException


def resample_data(
    X_train: np.ndarray,
    y_train_enc: np.ndarray,
    method: str,
):
    """
    Apply class imbalance method to training data only.
    Supported:
        - none
        - class_weights
        - smote
        - smote_tomek
        - undersampling
        - smote_enn
    Returns: (X_res, y_res, sample_weight)
    """
    try:
        print(f"\n[Imbalance] Method = {method}")
        print("  Original class distribution:", Counter(y_train_enc))

        X_res, y_res = X_train, y_train_enc
        sample_weight = None

        if method == "none":
            return X_train, y_train_enc, None

        elif method == "class_weights":
            classes = np.unique(y_train_enc)
            cw = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train_enc,
            )
            class_weight_dict = dict(zip(classes, cw))
            sample_weight = np.array([class_weight_dict[c] for c in y_train_enc])

            print("  Using class weights:", class_weight_dict)
            return X_train, y_train_enc, sample_weight

        elif method == "smote":
            try:
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X_train, y_train_enc)
            except ValueError as e:
                print(f"  [WARN] SMOTE failed: {e}")
                print("         Falling back to no resampling.")
                return X_train, y_train_enc, None

        elif method == "smote_tomek":
            try:
                smt = SMOTETomek(random_state=42)
                X_res, y_res = smt.fit_resample(X_train, y_train_enc)
            except ValueError as e:
                print(f"  [WARN] SMOTE-Tomek failed: {e}")
                print("         Falling back to no resampling.")
                return X_train, y_train_enc, None

        elif method == "undersampling":
            rus = RandomUnderSampler(random_state=42)
            X_res, y_res = rus.fit_resample(X_train, y_train_enc)

        elif method == "smote_enn":
            try:
                smote_enn = SMOTEENN(random_state=42)
                X_res, y_res = smote_enn.fit_resample(X_train, y_train_enc)
            except ValueError as e:
                print(f"  [WARN] SMOTEENN failed: {e}")
                print("         Falling back to no resampling.")
                return X_train, y_train_enc, None

        else:
            raise ValueError(f"Unknown imbalance method: {method}")

        print("  New class distribution:", Counter(y_res))
        return X_res, y_res, sample_weight

    except Exception as e:
        raise VisionInfantNetException(e, sys)
