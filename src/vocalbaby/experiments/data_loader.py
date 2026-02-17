"""
Data loading utilities for XGBoost feature comparison experiments.

Provides standardized access to train/valid/test splits and labels,
matching the exact workflow from notebook 06 (eGeMAPS baseline).
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder

from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging


def get_latest_artifact_dir(root_dir: str = "artifacts") -> str:
    """
    Find the most recently created artifact directory.
    
    Args:
        root_dir: Root artifacts directory
        
    Returns:
        Path to latest artifact subdirectory
    """
    try:
        subdirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        if not subdirs:
            raise ValueError(f"No artifact directories found in {root_dir}")
        
        latest = max(subdirs, key=os.path.getmtime)
        logging.info(f"Latest artifact directory: {latest}")
        return latest
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def load_labels(
    artifact_dir: str,
    fit_encoder: bool = True,
    label_col: str = "Answer",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder, List[str]]:
    """
    Load train/valid/test labels from metadata CSVs in artifact directory.
    
    Reads the label column from the metadata CSVs, fits a LabelEncoder
    on the training set, and encodes all splits.
    
    Args:
        artifact_dir: Path to artifact directory (e.g., artifacts/02_17_2026_10_52_02)
        fit_encoder: If True, fit encoder on train; else assumes encoder already fitted
        label_col: Name of the label column in metadata CSV
        
    Returns:
        y_train_enc: Encoded train labels
        y_valid_enc: Encoded valid labels
        y_test_enc: Encoded test labels
        label_encoder: Fitted LabelEncoder
        class_names: List of class names in order
    """
    try:
        # Load metadata CSVs
        train_df, valid_df, test_df = load_metadata_with_audio_paths(artifact_dir)
        
        y_train = train_df[label_col].values
        y_valid = valid_df[label_col].values
        y_test = test_df[label_col].values
        
        label_encoder = LabelEncoder()
        
        if fit_encoder:
            y_train_enc = label_encoder.fit_transform(y_train)
        else:
            y_train_enc = label_encoder.transform(y_train)
            
        y_valid_enc = label_encoder.transform(y_valid)
        y_test_enc = label_encoder.transform(y_test)
        
        class_names = list(label_encoder.classes_)
        
        logging.info("Label encoding mapping:")
        for org, enc in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
            logging.info(f"  {org} → {enc}")
        
        logging.info(f"Train labels: {y_train_enc.shape}")
        logging.info(f"Valid labels: {y_valid_enc.shape}")
        logging.info(f"Test labels: {y_test_enc.shape}")
        
        return y_train_enc, y_valid_enc, y_test_enc, label_encoder, class_names
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def load_metadata_with_audio_paths(artifact_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/valid/test metadata CSVs with audio paths.
    
    Returns metadata DataFrames that can be used for feature extraction.
    
    Args:
        artifact_dir: Path to artifact directory
        
    Returns:
        train_df: Train metadata
        valid_df: Valid metadata
        test_df: Test metadata
    """
    try:
        # Check both possible locations for metadata
        # 1) After data_validation (preferred)
        metadata_root = os.path.join(artifact_dir, "data_validation/validated/metadata")
        if not os.path.exists(metadata_root):
            # 2) After data_ingestion (fallback)
            metadata_root = os.path.join(artifact_dir, "data_ingestion/ingested_metadata")
        
        if not os.path.exists(metadata_root):
            raise FileNotFoundError(
                f"Metadata directory not found. Expected one of:\n"
                f"  - {os.path.join(artifact_dir, 'data_validation/validated/metadata')}\n"
                f"  - {os.path.join(artifact_dir, 'data_ingestion/ingested_metadata')}\n"
                f"\nPlease run the data pipeline first:\n"
                f"  bash scripts/run_ingestion.sh\n"
                f"  bash scripts/run_validation.sh"
            )
        
        # Files are named train.csv, valid.csv, test.csv (not train_metadata.csv)
        train_df = pd.read_csv(os.path.join(metadata_root, "train.csv"))
        valid_df = pd.read_csv(os.path.join(metadata_root, "valid.csv"))
        test_df = pd.read_csv(os.path.join(metadata_root, "test.csv"))
        
        logging.info(f"Loaded metadata from: {metadata_root}")
        logging.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
        
        return train_df, valid_df, test_df
        
    except Exception as e:
        raise VocalBabyException(e, sys)
