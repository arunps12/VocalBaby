import sys
from typing import Tuple

import numpy as np
from sklearn.impute import SimpleImputer

from vocalbaby.exception.exception import VocalBabyException


def fit_imputer_and_transform(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    strategy: str = "median",
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer]:
    """
    Fit a SimpleImputer on X_train and apply to both X_train and X_valid.
    Returns (X_train_imp, X_valid_imp, imputer).
    """
    try:
        imputer = SimpleImputer(strategy=strategy)
        X_train_imp = imputer.fit_transform(X_train)
        X_valid_imp = imputer.transform(X_valid)
        return X_train_imp, X_valid_imp, imputer
    except Exception as e:
        raise VocalBabyException(e, sys)
