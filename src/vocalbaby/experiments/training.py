"""
XGBoost training utilities for feature comparison experiments.

Handles:
- Training XGBoost with best params
- Applying same preprocessing (imputation, SMOTE) as tuning
- Saving trained models
"""

import os
import sys
import pickle
import numpy as np
from typing import Dict, Tuple
from collections import Counter

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging


def train_xgboost_with_best_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: Dict,
    random_state: int = 42,
    imputer_strategy: str = "median",
    apply_smote: bool = True,
) -> Tuple[XGBClassifier, SimpleImputer]:
    """
    Train XGBoost classifier with best hyperparameters.
    
    Applies same preprocessing as tuning:
    1. Fit imputer on train
    2. Apply SMOTE to train (if enabled)
    3. Train XGBoost
    
    Args:
        X_train: Raw train features
        y_train: Encoded train labels
        best_params: Best hyperparameters from tuning
        random_state: Random seed
        imputer_strategy: Imputation strategy
        apply_smote: Whether to apply SMOTE
        
    Returns:
        model: Trained XGBoost classifier
        imputer: Fitted imputer (needed for test-time preprocessing)
    """
    try:
        logging.info("=" * 80)
        logging.info("TRAINING XGBOOST WITH BEST PARAMS")
        logging.info("=" * 80)
        logging.info(f"Train shape: {X_train.shape}")
        logging.info(f"Best params: {best_params}")
        
        # Step 1: Imputation
        logging.info(f"Fitting imputer (strategy={imputer_strategy})...")
        imputer = SimpleImputer(strategy=imputer_strategy)
        X_train_imp = imputer.fit_transform(X_train)
        
        # Step 2: SMOTE
        if apply_smote:
            logging.info("Applying SMOTE...")
            sm = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = sm.fit_resample(X_train_imp, y_train)
            logging.info(f"After SMOTE distribution: {dict(Counter(y_train_resampled))}")
        else:
            X_train_resampled = X_train_imp
            y_train_resampled = y_train
        
        # Step 3: Build and train model
        logging.info("Building XGBoost model...")
        model = XGBClassifier(
            **best_params,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        
        logging.info("Training model...")
        model.fit(X_train_resampled, y_train_resampled)
        
        logging.info("Training completed successfully!")
        logging.info("=" * 80)
        
        return model, imputer
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def save_model(
    model: XGBClassifier,
    output_path: str,
):
    """
    Save trained XGBoost model.
    
    Args:
        model: Trained XGBoost classifier
        output_path: Path to save model (.pkl)
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(model, f)
        
        logging.info(f"Saved model to: {output_path}")
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def load_model(model_path: str) -> XGBClassifier:
    """
    Load trained XGBoost model.
    
    Args:
        model_path: Path to model file (.pkl)
        
    Returns:
        model: Loaded XGBoost classifier
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        logging.info(f"Loaded model from: {model_path}")
        return model
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def save_imputer(
    imputer: SimpleImputer,
    output_path: str,
):
    """
    Save fitted imputer.
    
    Args:
        imputer: Fitted imputer
        output_path: Path to save imputer (.pkl)
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(imputer, f)
        
        logging.info(f"Saved imputer to: {output_path}")
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def load_imputer(imputer_path: str) -> SimpleImputer:
    """
    Load fitted imputer.
    
    Args:
        imputer_path: Path to imputer file (.pkl)
        
    Returns:
        imputer: Loaded imputer
    """
    try:
        with open(imputer_path, "rb") as f:
            imputer = pickle.load(f)
        
        logging.info(f"Loaded imputer from: {imputer_path}")
        return imputer
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def save_label_encoder(
    label_encoder: LabelEncoder,
    output_path: str,
):
    """
    Save fitted label encoder.
    
    Args:
        label_encoder: Fitted LabelEncoder
        output_path: Path to save encoder (.pkl)
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(label_encoder, f)
        
        logging.info(f"Saved label encoder to: {output_path}")
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def load_label_encoder(encoder_path: str) -> LabelEncoder:
    """
    Load fitted label encoder.
    
    Args:
        encoder_path: Path to encoder file (.pkl)
        
    Returns:
        label_encoder: Loaded LabelEncoder
    """
    try:
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        
        logging.info(f"Loaded label encoder from: {encoder_path}")
        return label_encoder
        
    except Exception as e:
        raise VocalBabyException(e, sys)
