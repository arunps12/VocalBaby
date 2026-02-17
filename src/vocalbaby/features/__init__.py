"""
VocalBaby Feature Extraction Module

Provides extractors for all supported feature sets:
- eGeMAPS (openSMILE)
- MFCC (librosa)
- HuBERT SSL embeddings (transformers)
- Wav2Vec2 SSL embeddings (transformers)
"""

from vocalbaby.features.egemaps import extract_egemaps_features, EGeMAPSExtractor
from vocalbaby.features.mfcc import extract_mfcc_features, MFCCExtractor
from vocalbaby.features.hubert import extract_hubert_features, HuBERTExtractor
from vocalbaby.features.wav2vec2 import extract_wav2vec2_features, Wav2Vec2Extractor
from vocalbaby.features.base import FeatureExtractorBase, extract_features_batch

__all__ = [
    'extract_egemaps_features',
    'extract_mfcc_features',
    'extract_hubert_features',
    'extract_wav2vec2_features',
    'EGeMAPSExtractor',
    'MFCCExtractor',
    'HuBERTExtractor',
    'Wav2Vec2Extractor',
    'FeatureExtractorBase',
    'extract_features_batch',
]
