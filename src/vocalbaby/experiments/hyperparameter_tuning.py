"""
Hyperparameter tuning for XGBoost using Optuna.

Replicates the exact workflow from notebook 06:
- Multi-objective optimization: maximize (UAR, F1)
- Train on SMOTE-resampled train set
- Evaluate on validation set
- Select best trial by max UAR
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Tuple
from collections import Counter

import optuna
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging
from vocalbaby.utils.ml_utils.metric.classification_metric import get_classification_score


def run_optuna_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    n_trials: int = 40,
    random_state: int = 42,
    imputer_strategy: str = "median",
    apply_smote: bool = True,
    study_name: str = "xgb_tuning",
) -> Tuple[Dict, optuna.study.Study]:
    """
    Run Optuna hyperparameter tuning for XGBoost.
    
    Exact replication of notebook 06 workflow:
    1. Fit imputer on train, transform train/valid
    2. Apply SMOTE to train (if enabled)
    3. Run Optuna with multi-objective: maximize (UAR, F1)
    4. Select best trial by max UAR
    
    Args:
        X_train: Raw train features
        y_train: Encoded train labels
        X_valid: Raw valid features
        y_valid: Encoded valid labels
        n_trials: Number of Optuna trials
        random_state: Random seed
        imputer_strategy: Imputation strategy ("median", "mean", etc.)
        apply_smote: Whether to apply SMOTE resampling
        study_name: Optuna study name
        
    Returns:
        best_params: Dict of best hyperparameters
        study: Optuna study object
    """
    try:
        logging.info("=" * 80)
        logging.info("STARTING OPTUNA HYPERPARAMETER TUNING")
        logging.info("=" * 80)
        
        # Step 1: Imputation (fit on train only)
        logging.info(f"Fitting imputer (strategy={imputer_strategy}) on train...")
        imputer = SimpleImputer(strategy=imputer_strategy)
        X_train_imp = imputer.fit_transform(X_train)
        X_valid_imp = imputer.transform(X_valid)
        
        logging.info(f"Train shape after imputation: {X_train_imp.shape}")
        logging.info(f"Valid shape after imputation: {X_valid_imp.shape}")
        
        # Step 2: SMOTE (on train only)
        if apply_smote:
            logging.info("Applying SMOTE to training set...")
            sm = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = sm.fit_resample(X_train_imp, y_train)
            logging.info(f"After SMOTE distribution: {dict(Counter(y_train_resampled))}")
        else:
            X_train_resampled = X_train_imp
            y_train_resampled = y_train
            logging.info("SMOTE disabled; using original training data")
        
        # Step 3: Define Optuna objective
        def objective(trial):
            """
            Optuna objective function.
            Returns (UAR, F1) for multi-objective optimization.
            """
            # Hyperparameter search space (matching notebook 06)
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            }
            
            # Build model
            model = XGBClassifier(
                **params,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            )
            
            # Train on resampled train
            model.fit(X_train_resampled, y_train_resampled)
            
            # Predict on validation
            y_valid_pred = model.predict(X_valid_imp)
            
            # Compute metrics
            metrics = get_classification_score(y_valid, y_valid_pred)
            uar = metrics.uar
            f1 = metrics.f1_score
            
            # Store metrics as trial attributes
            trial.set_user_attr("UAR", uar)
            trial.set_user_attr("F1", f1)
            
            logging.info(f"[Trial {trial.number}] UAR={uar:.4f} F1={f1:.4f}")
            
            return uar, f1
        
        # Step 4: Run Optuna study
        logging.info(f"Creating Optuna study: {study_name}")
        study = optuna.create_study(
            directions=["maximize", "maximize"],  # Multi-objective: (UAR, F1)
            study_name=study_name,
        )
        
        logging.info(f"Starting optimization with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials)
        
        logging.info("Optuna tuning completed.")
        logging.info(f"Number of Pareto-optimal trials: {len(study.best_trials)}")
        
        # Step 5: Select best trial (max UAR)
        best_trial = max(study.best_trials, key=lambda t: t.values[0])  # values[0] = UAR
        best_params = best_trial.params
        
        logging.info("=" * 80)
        logging.info("BEST TRIAL SELECTED (by max UAR)")
        logging.info("=" * 80)
        logging.info(f"Trial number: {best_trial.number}")
        logging.info(f"UAR: {best_trial.values[0]:.4f}")
        logging.info(f"F1: {best_trial.values[1]:.4f}")
        logging.info(f"Best params: {best_params}")
        logging.info("=" * 80)
        
        return best_params, study
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def save_best_params(
    best_params: Dict,
    output_path: str,
):
    """
    Save best hyperparameters to JSON file.
    
    Args:
        best_params: Dictionary of hyperparameters
        output_path: Path to save JSON file
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=2)
        
        logging.info(f"Saved best params to: {output_path}")
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def load_best_params(params_path: str) -> Dict:
    """
    Load best hyperparameters from JSON file.
    
    Args:
        params_path: Path to JSON file
        
    Returns:
        Dictionary of hyperparameters
    """
    try:
        with open(params_path, "r") as f:
            params = json.load(f)
        
        logging.info(f"Loaded best params from: {params_path}")
        return params
        
    except Exception as e:
        raise VocalBabyException(e, sys)
