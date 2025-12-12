import os
import sys
from typing import List, Union, Tuple

import numpy as np
import librosa

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

    Supports:
      - Single wav file path
      - List of wav file paths
      - Directory containing wav files

    Optionally:
      - Chunk long audio into 400 ms windows and aggregate predictions.
    """

    def __init__(self, model_trainer_dir: str):
        try:
            logging.info(f"Initializing PredictionPipeline with dir: {model_trainer_dir}")

            self.model_trainer_dir = model_trainer_dir

            self.model_path = os.path.join(
                self.model_trainer_dir, "xgb_egemaps_smote_optuna.pkl"
            )
            self.preprocessing_path = os.path.join(
                self.model_trainer_dir, "preprocessing.pkl"
            )
            self.label_encoder_path = os.path.join(
                self.model_trainer_dir, "label_encoder.pkl"
            )

            # Load objects
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessing_path)
            self.label_encoder = load_object(self.label_encoder_path)

            logging.info("PredictionPipeline initialized successfully.")

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------
    # Resolve audio paths
    # ------------------------------------------------------------
    def _resolve_audio_paths(
        self, source: Union[str, List[str]]
    ) -> List[str]:
        try:
            if isinstance(source, list):
                paths = [p for p in source if os.path.isfile(p)]
                if not paths:
                    raise ValueError("No valid files found in the provided list.")
                return paths

            if isinstance(source, str):
                if os.path.isdir(source):
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

    # ------------------------------------------------------------
    # eGeMAPS on whole file 
    # ------------------------------------------------------------
    def _extract_egemaps_batch(self, audio_paths: List[str]) -> np.ndarray:
        try:
            logging.info(f"Extracting eGeMAPS for {len(audio_paths)} files...")

            feats_list = []
            for p in audio_paths:
                vec = extract_egemaps(p)  # (88,)
                feats_list.append(vec)

            X = np.stack(feats_list, axis=0).astype(np.float32)
            logging.info(f"eGeMAPS feature matrix shape: {X.shape}")
            return X

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------
    # Chunk-based eGeMAPS for a single file
    # ------------------------------------------------------------
    def _extract_egemaps_chunked(
        self,
        audio_path: str,
        chunk_duration: float = 0.4,
        min_chunk_duration: float = 0.2,
    ) -> np.ndarray:
        """
        Split audio into ~chunk_duration (s) windows and extract eGeMAPS per chunk.

        Returns:
            X_chunks: np.ndarray of shape (num_chunks, 88)
        """
        try:
            logging.info(f"Chunking + extracting eGeMAPS for: {audio_path}")

            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            total_duration = len(y) / sr

            if total_duration <= chunk_duration:
                # Fallback: just one vector on whole file
                logging.info("File shorter than chunk_duration -> using whole file.")
                vec = extract_egemaps(audio_path)
                return np.expand_dims(vec, axis=0).astype(np.float32)

            chunk_samples = int(chunk_duration * sr)
            min_samples = int(min_chunk_duration * sr)

            feats_list = []

            # Non-overlapping windows 
            for start in range(0, len(y), chunk_samples):
                end = start + chunk_samples
                if end > len(y):
                    # Last partial chunk
                    if len(y) - start < min_samples:
                        # too small -> skip
                        break
                    end = len(y)

                # Extract temp chunk
                chunk = y[start:end]

                # Save to temp file for opensmile
                import tempfile
                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    sf.write(tmp.name, chunk, sr)
                    vec = extract_egemaps(tmp.name)

                feats_list.append(vec)

            if not feats_list:
                raise ValueError(f"No valid chunks extracted for {audio_path}")

            X_chunks = np.stack(feats_list, axis=0).astype(np.float32)
            logging.info(
                f"Extracted {X_chunks.shape[0]} chunks for {audio_path}, "
                f"shape={X_chunks.shape}"
            )
            return X_chunks

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------
    #predict from audio (file / dir / list), with optional chunking
    # ------------------------------------------------------------
    def predict_from_audio(
        self,
        source: Union[str, List[str]],
        use_chunking: bool = True,
        chunk_duration: float = 0.4,
        min_chunk_duration: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        High-level prediction function.

        If `use_chunking=True`, each file is internally split into ~400 ms chunks
        and we aggregate chunk predictions to get one label per file.

        Returns
        -------
        y_pred_encoded : np.ndarray   (N_files,)
        y_pred_decoded : np.ndarray   (N_files,)
        audio_paths    : List[str]
        """
        try:
            logging.info("Starting prediction for audio source.")
            audio_paths = self._resolve_audio_paths(source)
            logging.info(f"Resolved {len(audio_paths)} audio files for prediction.")

            file_level_y_enc = []
            file_level_y_dec = []

            for path in audio_paths:
                if use_chunking:
                    # 1) extract per-chunk eGeMAPS
                    X_chunks = self._extract_egemaps_chunked(
                        path,
                        chunk_duration=chunk_duration,
                        min_chunk_duration=min_chunk_duration,
                    )
                else:
                    # original behaviour: one vector per file
                    X_chunks = self._extract_egemaps_batch([path])

                # 2) preprocess
                logging.info(f"Applying preprocessing for file: {path}")
                X_proc = self.preprocessor.transform(X_chunks)

                # 3) model predictions (probabilities)
                if hasattr(self.model, "predict_proba"):
                    logging.info("Running model.predict_proba on chunks...")
                    proba = self.model.predict_proba(X_proc)  # (num_chunks, n_classes)
                    mean_proba = np.mean(proba, axis=0, keepdims=True)  # (1, n_classes)
                    y_enc = np.argmax(mean_proba, axis=1)  # (1,)
                else:
                    # fallback: majority vote on hard predictions
                    logging.info("Model has no predict_proba -> using majority vote.")
                    chunk_preds = self.model.predict(X_proc)
                    # majority vote
                    vals, counts = np.unique(chunk_preds, return_counts=True)
                    y_enc = np.array([vals[np.argmax(counts)]])

                y_dec = self.label_encoder.inverse_transform(y_enc)

                file_level_y_enc.append(y_enc[0])
                file_level_y_dec.append(y_dec[0])

            y_pred_enc = np.array(file_level_y_enc)
            y_pred_dec = np.array(file_level_y_dec)

            logging.info("Prediction completed successfully.")
            return y_pred_enc, y_pred_dec, audio_paths

        except Exception as e:
            raise VisionInfantNetException(e, sys)
