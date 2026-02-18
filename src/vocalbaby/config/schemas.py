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
    """
    Centralized path configuration for the pipeline.

    All artifact paths resolve through artifacts/latest (symlink)
    so every stage in a run shares the same timestamped directory.
    """

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
        """Current run directory (resolved via artifacts/latest)."""
        from vocalbaby.utils.run_manager import RunManager
        try:
            return RunManager.get_current_run_dir()
        except FileNotFoundError:
            # Fallback before any run exists
            return self.base_dir / ARTIFACT_DIR

    @property
    def data_ingestion_dir(self) -> Path:
        return self.artifacts_dir / "data_ingestion"

    @property
    def data_validation_dir(self) -> Path:
        return self.artifacts_dir / "data_validation"

    @property
    def features_dir(self) -> Path:
        return self.artifacts_dir / "features"

    @property
    def tuning_dir(self) -> Path:
        return self.artifacts_dir / "tuning"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def eval_dir(self) -> Path:
        return self.artifacts_dir / "eval"

    @property
    def results_dir(self) -> Path:
        return self.artifacts_dir / "results"

    def feature_set_dir(self, feature_set: str) -> Path:
        return self.features_dir / feature_set

    def model_dir(self, feature_set: str) -> Path:
        return self.models_dir / feature_set

    def eval_set_dir(self, feature_set: str) -> Path:
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

    def get_classification_mode(self) -> str:
        """
        Get classification mode from params.yaml.

        Returns:
            One of 'flat', 'hierarchical', or 'both'.
            Defaults to 'flat' if not set.
        """
        mode = self.get('classification.mode', 'flat')
        if mode not in ('flat', 'hierarchical', 'both'):
            raise ValueError(
                f"Invalid classification.mode='{mode}'. "
                "Must be one of: flat, hierarchical, both"
            )
        return mode

    def get_classification_routing(self) -> str:
        """Get hierarchical routing mode ('hard' or 'soft')."""
        return self.get('classification.routing', 'hard')

    def get_classification_use_class_weights(self) -> bool:
        """Whether to apply inverse-frequency class weights per stage."""
        return self.get('classification.use_class_weights', True)


# Column name constants (for compatibility with existing code)
CHILD_ID_COL = CHILD_ID_COLUMN
AUDIO_ID_COL = AUDIO_ID_COLUMN
TARGET_COL = TARGET_COLUMN
AUDIO_PATH_COL = AUDIO_PATH_COLUMN
