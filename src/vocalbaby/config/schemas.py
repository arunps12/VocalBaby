"""
Configuration schemas and path definitions for VocalBaby pipeline.
"""
import os
from pathlib import Path
from typing import Dict, Any, List
import yaml

from vocalbaby.constant.training_pipeline import (
    ARTIFACT_DIR,
    CHILD_ID_COLUMN,
    AUDIO_ID_COLUMN,
    TARGET_COLUMN,
    AUDIO_PATH_COLUMN,
)


class PathConfig:
    """Centralized path configuration for the pipeline."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.params = self._load_params()
        
    def _load_params(self) -> Dict[str, Any]:
        """Load params.yaml."""
        params_path = self.base_dir / "params.yaml"
        if params_path.exists():
            with open(params_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    @property
    def artifacts_dir(self) -> Path:
        """Base artifacts directory."""
        return self.base_dir / self.params.get('artifacts', {}).get('base_dir', ARTIFACT_DIR)
    
    @property
    def data_dir(self) -> Path:
        """Data artifacts directory."""
        return self.artifacts_dir / "data"
    
    @property
    def features_dir(self) -> Path:
        """Features artifacts directory."""
        return self.artifacts_dir / "features"
    
    @property
    def models_dir(self) -> Path:
        """Models artifacts directory."""
        return self.artifacts_dir / "models"
    
    @property
    def eval_dir(self) -> Path:
        """Evaluation artifacts directory."""
        return self.artifacts_dir / "eval"
    
    @property
    def results_dir(self) -> Path:
        """Results artifacts directory."""
        return self.artifacts_dir / "results"
    
    def feature_set_dir(self, feature_set: str) -> Path:
        """Get feature set directory."""
        return self.features_dir / feature_set
    
    def model_dir(self, feature_set: str) -> Path:
        """Get model directory for a feature set."""
        return self.models_dir / feature_set
    
    def eval_set_dir(self, feature_set: str) -> Path:
        """Get evaluation directory for a feature set."""
        return self.eval_dir / feature_set


class ConfigLoader:
    """Load and validate configuration from params.yaml."""
    
    def __init__(self, params_path: str = "params.yaml"):
        self.params_path = params_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load params.yaml."""
        if not os.path.exists(self.params_path):
            raise FileNotFoundError(f"Configuration file not found: {self.params_path}")
        
        with open(self.params_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example:
            config.get('features.mfcc.n_mfcc') -> 20
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def get_feature_sets(self) -> List[str]:
        """Get list of enabled feature sets."""
        return self.get('features.sets', ['egemaps'])
    
    def get_feature_config(self, feature_set: str) -> Dict[str, Any]:
        """Get configuration for a specific feature set."""
        return self.get(f'features.{feature_set}', {})
    
    def get_tuning_config(self) -> Dict[str, Any]:
        """Get hyperparameter tuning configuration."""
        return self.get('tuning', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get model training configuration."""
        return self.get('training', {})
    
    def get_eval_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get('evaluation', {})


# Column name constants (for compatibility with existing code)
CHILD_ID_COL = CHILD_ID_COLUMN
AUDIO_ID_COL = AUDIO_ID_COLUMN
TARGET_COL = TARGET_COLUMN
AUDIO_PATH_COL = AUDIO_PATH_COLUMN
