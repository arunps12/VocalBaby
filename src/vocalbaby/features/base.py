"""
Base feature extractor interface.
"""
import sys
import numpy as np
import pandas as pd
from typing import Callable, List
from abc import ABC, abstractmethod
from tqdm import tqdm

from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging


class FeatureExtractorBase(ABC):
    """Base class for all feature extractors."""
    
    @abstractmethod
    def extract(self, audio_path: str) -> np.ndarray:
        """
        Extract features from an audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Feature vector as numpy array
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Expected feature dimensionality."""
        pass
    
    @property
    @abstractmethod
    def feature_name(self) -> str:
        """Feature set name."""
        pass


def extract_features_batch(
    audio_paths: List[str],
    extractor_fn: Callable,
    desc: str = "Extracting features",
    **kwargs
) -> np.ndarray:
    """
    Extract features for a batch of audio files.
    
    Args:
        audio_paths: List of audio file paths
        extractor_fn: Feature extraction function
        desc: Progress bar description
        **kwargs: Additional arguments for extractor
    
    Returns:
        Feature array of shape (n_samples, feature_dim)
    """
    try:
        features = []
        
        for audio_path in tqdm(audio_paths, desc=desc):
            feat = extractor_fn(audio_path, **kwargs)
            features.append(feat)
        
        features_array = np.vstack(features)
        logging.info(f"Extracted features: {features_array.shape}")
        
        return features_array
        
    except Exception as e:
        raise VocalBabyException(e, sys)
