"""
Wav2Vec2 SSL embeddings extractor.

For now, imports from experiments. TODO: Consolidate later.
"""
import numpy as np

from vocalbaby.experiments.feature_extractors import (
    load_wav2vec2_model,
    extract_wav2vec2_embedding,
)
from vocalbaby.features.base import FeatureExtractorBase


class Wav2Vec2Extractor(FeatureExtractorBase):
    """Wav2Vec2 SSL embeddings extractor."""
    
    def __init__(self, model_name: str = "arunps/wav2vec2-home-hindibabynet-ssl", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model, self.processor = load_wav2vec2_model(model_name, device)
    
    def extract(self, audio_path: str) -> np.ndarray:
        """Extract Wav2Vec2 embeddings."""
        return extract_wav2vec2_embedding(
            audio_path,
            model=self.model,
            processor=self.processor,
            device=self.device,
        )
    
    @property
    def feature_dim(self) -> int:
        return 768  # Wav2Vec2-base hidden size
    
    @property
    def feature_name(self) -> str:
        return "wav2vec2_ssl"


# Convenience aliases
extract_wav2vec2_features = extract_wav2vec2_embedding

__all__ = ['Wav2Vec2Extractor', 'load_wav2vec2_model', 'extract_wav2vec2_embedding', 'extract_wav2vec2_features']
