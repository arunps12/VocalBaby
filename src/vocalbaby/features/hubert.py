"""
HuBERT SSL embeddings extractor.

For now, imports from experiments. TODO: Consolidate later.
"""
import numpy as np

from vocalbaby.experiments.feature_extractors import (
    load_hubert_model,
    extract_hubert_embedding,
)
from vocalbaby.features.base import FeatureExtractorBase


class HuBERTExtractor(FeatureExtractorBase):
    """HuBERT SSL embeddings extractor."""
    
    def __init__(self, model_name: str = "arunps/hubert-home-hindibabynet-ssl", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model, self.processor = load_hubert_model(model_name, device)
    
    def extract(self, audio_path: str) -> np.ndarray:
        """Extract HuBERT embeddings."""
        return extract_hubert_embedding(
            audio_path,
            model=self.model,
            processor=self.processor,
            device=self.device,
        )
    
    @property
    def feature_dim(self) -> int:
        return 768  # HuBERT-base hidden size
    
    @property
    def feature_name(self) -> str:
        return "hubert_ssl"


# Convenience aliases
extract_hubert_features = extract_hubert_embedding

__all__ = ['HuBERTExtractor', 'load_hubert_model', 'extract_hubert_embedding', 'extract_hubert_features']
