"""
MFCC feature extractor (via librosa).

Extracts MFCC coefficients with temporal pooling.
"""
import sys
import numpy as np
import librosa

from vocalbaby.features.base import FeatureExtractorBase
from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging


class MFCCExtractor(FeatureExtractorBase):
    """MFCC feature extractor using librosa."""
    
    def __init__(self, n_mfcc: int = 20, sr: int = 16000, pooling: str = "mean_std"):
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.pooling = pooling
    
    def extract(self, audio_path: str) -> np.ndarray:
        """Extract MFCC features from audio file."""
        return extract_mfcc_features(
            audio_path,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            pooling=self.pooling,
        )
    
    @property
    def feature_dim(self) -> int:
        return 2 * self.n_mfcc if self.pooling == "mean_std" else self.n_mfcc
    
    @property
    def feature_name(self) -> str:
        return "mfcc"


def extract_mfcc_features(
    audio_path: str,
    sr: int = 16000,
    n_mfcc: int = 20,
    pooling: str = "mean_std",
) -> np.ndarray:
    """
    Extract MFCC features with temporal pooling.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        n_mfcc: Number of MFCC coefficients
        pooling: "mean", "mean_std", or "mean+std"
    
    Returns:
        Fixed-size MFCC feature vector
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Extract MFCC time series
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Temporal pooling
        pooling = pooling.replace("+", "_")  # Normalize naming
        if pooling == "mean":
            features = np.mean(mfcc, axis=1)
        elif pooling in ("mean_std", "mean+std"):
            mean_feats = np.mean(mfcc, axis=1)
            std_feats = np.std(mfcc, axis=1)
            features = np.concatenate([mean_feats, std_feats])
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        return features.astype(np.float32)
        
    except Exception as e:
        dim = n_mfcc if pooling == "mean" else 2 * n_mfcc
        logging.warning(f"MFCC extraction failed for {audio_path}: {e}")
        return np.zeros(dim, dtype=np.float32)
