"""
eGeMAPS feature extractor (via openSMILE).

Reuses existing implementation - 88-dimensional feature vector.
"""
import sys
import numpy as np
import opensmile

from vocalbaby.features.base import FeatureExtractorBase
from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging


class EGeMAPSExtractor(FeatureExtractorBase):
    """eGeMAPS feature extractor using openSMILE."""
    
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    
    def extract(self, audio_path: str) -> np.ndarray:
        """Extract eGeMAPS features from audio file."""
        try:
            feats = self.smile.process_file(audio_path)
            return feats.values.flatten().astype(np.float32)
        except Exception as e:
            logging.warning(f"eGeMAPS extraction failed for {audio_path}: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    @property
    def feature_dim(self) -> int:
        return 88
    
    @property
    def feature_name(self) -> str:
        return "egemaps"


def extract_egemaps_features(audio_path: str, smile_instance=None) -> np.ndarray:
    """
    Extract eGeMAPS features (functional interface).
    
    Args:
        audio_path: Path to audio file
        smile_instance: Optional pre-initialized Smile instance
    
    Returns:
        88-dim eGeMAPS feature vector
    """
    if smile_instance is None:
        smile_instance = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    
    try:
        feats = smile_instance.process_file(audio_path)
        return feats.values.flatten().astype(np.float32)
    except Exception:
        return np.zeros(88, dtype=np.float32)
