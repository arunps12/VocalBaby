"""
Feature extraction utilities for XGBoost comparison experiments.

Implements extractors for:
- eGeMAPS (reuses existing implementation)
- MFCC (pooled fixed-size vectors)
- HuBERT embeddings (SSL model from HuggingFace)
- Wav2Vec2 embeddings (SSL model from HuggingFace)

All extractors produce fixed-size feature vectors suitable for XGBoost.
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import torch
from typing import Tuple, Optional, Callable
from tqdm import tqdm

from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Feature extraction device: {DEVICE}")


# ============================================================================
# eGeMAPS EXTRACTOR (Reuse existing)
# ============================================================================

def extract_egemaps_features(
    audio_path: str,
    smile_instance=None,
) -> np.ndarray:
    """
    Extract eGeMAPS features using opensmile (88-dim).
    
    Reuses existing implementation from data_transformation.py
    
    Args:
        audio_path: Path to audio file
        smile_instance: Pre-initialized opensmile.Smile instance
        
    Returns:
        88-dim eGeMAPS feature vector (float32)
    """
    import opensmile
    
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


# ============================================================================
# MFCC EXTRACTOR (Pooled)
# ============================================================================

def extract_mfcc_features(
    audio_path: str,
    sr: int = 16000,
    n_mfcc: int = 20,
    pooling: str = "mean",
) -> np.ndarray:
    """
    Extract MFCC features with temporal pooling.
    
    Default: 20 MFCCs pooled with mean → 20-dim vector
    Optional: mean+std → 40-dim vector
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        n_mfcc: Number of MFCC coefficients
        pooling: "mean" or "mean+std"
        
    Returns:
        Fixed-size MFCC feature vector (float32)
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Extract MFCC time series: shape (n_mfcc, time_frames)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Temporal pooling
        if pooling == "mean":
            features = np.mean(mfcc, axis=1)  # (n_mfcc,)
        elif pooling == "mean+std":
            mean_feats = np.mean(mfcc, axis=1)  # (n_mfcc,)
            std_feats = np.std(mfcc, axis=1)    # (n_mfcc,)
            features = np.concatenate([mean_feats, std_feats])  # (2*n_mfcc,)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
            
        return features.astype(np.float32)
        
    except Exception as e:
        dim = n_mfcc if pooling == "mean" else 2 * n_mfcc
        logging.warning(f"MFCC extraction failed for {audio_path}: {e}")
        return np.zeros(dim, dtype=np.float32)


# ============================================================================
# HUBERT EMBEDDING EXTRACTOR (SSL)
# ============================================================================

def load_hubert_model(model_name: str = "arunps/hubert-home-hindibabynet-ssl"):
    """
    Load HuBERT model from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        processor, model
    """
    try:
        from transformers import Wav2Vec2Processor, HubertModel
        
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = HubertModel.from_pretrained(model_name)
        model.to(DEVICE)
        model.eval()
        
        logging.info(f"Loaded HuBERT model: {model_name}")
        return processor, model
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def extract_hubert_embedding(
    audio_path: str,
    processor,
    model,
    sr: int = 16000,
    pooling: str = "mean",
) -> np.ndarray:
    """
    Extract HuBERT embedding with temporal pooling.
    
    HuBERT outputs hidden states: (batch, time_steps, hidden_dim)
    We pool over time_steps to get fixed-size vector.
    
    Args:
        audio_path: Path to audio file
        processor: HuBERT processor
        model: HuBERT model
        sr: Target sample rate
        pooling: "mean" or "mean+std"
        
    Returns:
        Fixed-size HuBERT embedding (float32)
    """
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Process audio
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(DEVICE)
        
        # Extract features
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state  # (1, time_steps, hidden_dim)
        
        # Pool over time
        hidden_states = hidden_states.squeeze(0)  # (time_steps, hidden_dim)
        
        if pooling == "mean":
            embedding = torch.mean(hidden_states, dim=0)  # (hidden_dim,)
        elif pooling == "mean+std":
            mean_emb = torch.mean(hidden_states, dim=0)  # (hidden_dim,)
            std_emb = torch.std(hidden_states, dim=0)    # (hidden_dim,)
            embedding = torch.cat([mean_emb, std_emb])   # (2*hidden_dim,)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
            
        return embedding.cpu().numpy().astype(np.float32)
        
    except Exception as e:
        # Estimate hidden_dim (HuBERT-base: 768, HuBERT-large: 1024)
        # Default to 768 for base model
        dim = 768 if pooling == "mean" else 1536
        logging.warning(f"HuBERT extraction failed for {audio_path}: {e}")
        return np.zeros(dim, dtype=np.float32)


# ============================================================================
# WAV2VEC2 EMBEDDING EXTRACTOR (SSL)
# ============================================================================

def load_wav2vec2_model(model_name: str = "arunps/wav2vec2-home-hindibabynet-ssl"):
    """
    Load Wav2Vec2 model from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        processor, model
    """
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.to(DEVICE)
        model.eval()
        
        logging.info(f"Loaded Wav2Vec2 model: {model_name}")
        return processor, model
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def extract_wav2vec2_embedding(
    audio_path: str,
    processor,
    model,
    sr: int = 16000,
    pooling: str = "mean",
) -> np.ndarray:
    """
    Extract Wav2Vec2 embedding with temporal pooling.
    
    Wav2Vec2 outputs hidden states: (batch, time_steps, hidden_dim)
    We pool over time_steps to get fixed-size vector.
    
    Args:
        audio_path: Path to audio file
        processor: Wav2Vec2 processor
        model: Wav2Vec2 model
        sr: Target sample rate
        pooling: "mean" or "mean+std"
        
    Returns:
        Fixed-size Wav2Vec2 embedding (float32)
    """
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Process audio
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(DEVICE)
        
        # Extract features
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state  # (1, time_steps, hidden_dim)
        
        # Pool over time
        hidden_states = hidden_states.squeeze(0)  # (time_steps, hidden_dim)
        
        if pooling == "mean":
            embedding = torch.mean(hidden_states, dim=0)  # (hidden_dim,)
        elif pooling == "mean+std":
            mean_emb = torch.mean(hidden_states, dim=0)  # (hidden_dim,)
            std_emb = torch.std(hidden_states, dim=0)    # (hidden_dim,)
            embedding = torch.cat([mean_emb, std_emb])   # (2*hidden_dim,)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
            
        return embedding.cpu().numpy().astype(np.float32)
        
    except Exception as e:
        # Estimate hidden_dim (Wav2Vec2-base: 768, Wav2Vec2-large: 1024)
        # Default to 768 for base model
        dim = 768 if pooling == "mean" else 1536
        logging.warning(f"Wav2Vec2 extraction failed for {audio_path}: {e}")
        return np.zeros(dim, dtype=np.float32)


# ============================================================================
# BATCH FEATURE EXTRACTION
# ============================================================================

def extract_features_from_metadata(
    metadata_df: pd.DataFrame,
    extractor_fn: Callable,
    audio_path_col: str = "path",  # Column name from existing data pipeline
    desc: str = "Extracting features",
    **extractor_kwargs,
) -> np.ndarray:
    """
    Extract features for all audio files in metadata DataFrame.
    
    Args:
        metadata_df: DataFrame with audio paths
        extractor_fn: Feature extraction function
        audio_path_col: Column name for audio paths
        desc: Progress bar description
        **extractor_kwargs: Additional arguments for extractor
        
    Returns:
        Feature array of shape (n_samples, feature_dim)
    """
    try:
        features = []
        
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc=desc):
            audio_path = row[audio_path_col]
            feat = extractor_fn(audio_path, **extractor_kwargs)
            features.append(feat)
        
        features_array = np.vstack(features)
        logging.info(f"Extracted features: {features_array.shape}")
        
        return features_array
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def save_features(
    features: np.ndarray,
    output_path: str,
):
    """
    Save feature array to .npy file.
    
    Args:
        features: Feature array
        output_path: Path to save .npy file
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        logging.info(f"Saved features to {output_path}: {features.shape}")
        
    except Exception as e:
        raise VocalBabyException(e, sys)
