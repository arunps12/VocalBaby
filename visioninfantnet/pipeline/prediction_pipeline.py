import os
import sys
from typing import List, Union, Tuple

import numpy as np

from visioninfantnet.exception.exception import VisionInfantNetException
from visioninfantnet.logging.logger import logging
from visioninfantnet.utils.main_utils.utils import load_object


from visioninfantnet.components.data_transformation import extract_egemaps


class PredictionPipeline:
    """
    Prediction pipeline for audio segments (wav) using:
      - eGeMAPS features (opensmile)
      - Preprocessing object (preprocessing.pkl)
      - Trained XGBoost model (xgb_egemaps_smote_optuna.pkl)
      - LabelEncoder (label_encoder.pkl)

    It supports:
      - Single wav path (str)
      - List of wav paths
      - Directory containing multiple .wav files
    """

    def __init__(self, model_trainer_dir: str):
        """
        Parameters
        ----------
        model_trainer_dir : str
            Path to the `model_trainer` directory inside artifacts
        """
        try:
            logging.info(f"Initializing PredictionPipeline with dir: {model_trainer_dir}")

            self.model_trainer_dir = model_trainer_dir

            
            self.model_path = os.path.join(
                self.model_trainer_dir,
                "xgb_egemaps_smote_optuna.pkl",
            )
            self.preprocessing_path = os.path.join(
                self.model_trainer_dir,
                "preprocessing.pkl",
            )
            
            self.label_encoder_path = os.path.join(
                self.model_trainer_dir,
                "label_encoder.pkl",
            )

            # Load objects
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessing_path)
            self.label_encoder = load_object(self.label_encoder_path)

            logging.info("PredictionPipeline initialized successfully.")

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # resolve audio paths from string / list / directory
    # ------------------------------------------------------------------
    def _resolve_audio_paths(
        self, source: Union[str, List[str]]
    ) -> List[str]:
        """
        Accepts:
          - a single file path
          - a directory path (all .wav inside will be used)
          - a list of file paths

        Returns:
          List of valid .wav paths
        """
        try:
            if isinstance(source, list):
                paths = [p for p in source if os.path.isfile(p)]
                if not paths:
                    raise ValueError("No valid files found in the provided list.")
                return paths

            if isinstance(source, str):
                if os.path.isdir(source):
                    # List all .wav files in the directory
                    wavs = [
                        os.path.join(source, f)
                        for f in os.listdir(source)
                        if f.lower().endswith(".wav")
                    ]
                    if not wavs:
                        raise ValueError(f"No .wav files found in directory: {source}")
                    return sorted(wavs)
                else:
                   
                    if not os.path.isfile(source):
                        raise ValueError(f"File does not exist: {source}")
                    return [source]

            raise ValueError(
                "source must be: str (file or dir) or List[str] of file paths."
            )

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # extract eGeMAPS for a list of audio paths
    # ------------------------------------------------------------------
    def _extract_egemaps_batch(self, audio_paths: List[str]) -> np.ndarray:
        """
        Extract eGeMAPS features for each audio file.

        Returns:
          X_egemaps: np.ndarray of shape (N, 88)
        """
        try:
            logging.info(f"Extracting eGeMAPS for {len(audio_paths)} files...")

            feats_list = []
            for p in audio_paths:
                vec = extract_egemaps(p)  # uses opensmile, returns (88,)
                feats_list.append(vec)

            X = np.stack(feats_list, axis=0).astype(np.float32)
            logging.info(f"eGeMAPS feature matrix shape: {X.shape}")
            return X

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    #  predict from audio
    # ------------------------------------------------------------------
    def predict_from_audio(
        self, source: Union[str, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        High-level prediction function.

        Parameters
        ----------
        source : str or List[str]
            - Single wav file path
            - Directory containing wavs
            - List of wav paths

        Returns
        -------
        y_pred_encoded : np.ndarray
            Encoded labels (0..4)
        y_pred_decoded : np.ndarray
            Decoded class names (e.g. "Canonical", "Cry", ...)
        audio_paths : List[str]
            The list of audio paths in the same order as predictions.
        """
        try:
            logging.info("Starting prediction for audio source.")

            #  Resolve paths
            audio_paths = self._resolve_audio_paths(source)
            logging.info(f"Resolved {len(audio_paths)} audio files for prediction.")

            #  Extract eGeMAPS features
            X_egemaps = self._extract_egemaps_batch(audio_paths)

            #  Preprocess (imputer/scaler)
            logging.info("Applying preprocessing to eGeMAPS features...")
            X_proc = self.preprocessor.transform(X_egemaps)

            #  Model prediction
            logging.info("Running XGBoost model predictions...")
            y_pred_enc = self.model.predict(X_proc)

            #  Decode labels
            logging.info("Decoding predictions with LabelEncoder...")
            y_pred_dec = self.label_encoder.inverse_transform(y_pred_enc)

            logging.info("Prediction completed successfully.")
            return y_pred_enc, y_pred_dec, audio_paths

        except Exception as e:
            raise VisionInfantNetException(e, sys)
