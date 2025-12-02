import os
import sys
import numpy as np
import pandas as pd
import librosa
from PIL import Image  

from visioninfantnet.entity.config_entity import DataTransformationConfig
from visioninfantnet.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from visioninfantnet.constant.training_pipeline import (
    AUDIO_PATH_COLUMN,
    TARGET_COLUMN,
    TARGET_SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
)
from visioninfantnet.exception.exception import VisionInfantNetException
from visioninfantnet.logging.logger import logging
from visioninfantnet.utils.main_utils.utils import save_numpy_array_data


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.config = data_transformation_config
            self.validation_artifact = data_validation_artifact
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ------------ helpers ------------

    def _load_valid_metadata(self):
        """
        Load validated metadata CSVs from DataValidationArtifact.
        """
        try:
            train_df = pd.read_csv(
                self.validation_artifact.validated_train_metadata_path
            )
            valid_df = pd.read_csv(
                self.validation_artifact.validated_validation_metadata_path
            )
            test_df = pd.read_csv(
                self.validation_artifact.validated_test_metadata_path
            )

            return {"train": train_df, "validation": valid_df, "test": test_df}
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    def _compute_melspectrogram(self, audio_path: str) -> np.ndarray:
        """
        Load audio, resample to TARGET_SAMPLE_RATE, compute Mel-spectrogram in dB.
        Output shape: (n_mels, time_frames)
        """
        try:
            y, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
            S = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            return S_db
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    def _spec_to_image_224(self, spec: np.ndarray) -> np.ndarray:
        """
        Convert a mel-spectrogram (n_mels, T) to a 224x224 RGB uint8 image.

        Steps:
        - Normalize to [0, 1]
        - Scale to [0, 255] uint8
        - Resize to 224x224
        - Stack to 3 channels (RGB)
        """
        try:
            spec_min = spec.min()
            spec_max = spec.max()
            if spec_max - spec_min < 1e-6:
                
                spec_norm = np.zeros_like(spec, dtype=np.float32)
            else:
                spec_norm = (spec - spec_min) / (spec_max - spec_min)

            spec_uint8 = (spec_norm * 255).astype(np.uint8)  # (H, W)

            # Resize to 224x224 using Pillow
            img = Image.fromarray(spec_uint8)  # grayscale
            img = img.resize((224, 224), Image.BICUBIC)

            # Convert to RGB (H, W, 3)
            img_rgb = img.convert("RGB")
            img_array = np.array(img_rgb)  # (224, 224, 3), dtype=uint8

            return img_array
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    def _save_spectrogram_image(self, img_array: np.ndarray, save_path: str):
        """
        Save a 224x224 RGB image (numpy array) as PNG.
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img = Image.fromarray(img_array)  # expects uint8 HxWx3
            img.save(save_path)
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    def _transform_split(
        self,
        df: pd.DataFrame,
        spectrogram_dir: str,
    ):
        """
        For a given split (train/valid/test):
        - compute Mel-spectrogram for each audio file
        - convert to 224x224 RGB image
        - stack into numpy array [num_samples, 224, 224, 3]
        - keep labels as array
        - save PNG spectrograms
        """
        images = []
        labels = []

        os.makedirs(spectrogram_dir, exist_ok=True)

        for idx, row in df.iterrows():
            audio_path = row[AUDIO_PATH_COLUMN]
            label = row[TARGET_COLUMN]

            # 1) mel-spectrogram
            spec = self._compute_melspectrogram(audio_path)

            # 2) convert to 224x224 RGB image
            img_array = self._spec_to_image_224(spec)

            images.append(img_array)
            labels.append(label)

            # 3) save PNG for image models
            spec_file = os.path.join(spectrogram_dir, f"{idx}.png")
            self._save_spectrogram_image(img_array, spec_file)

        images_array = np.stack(images, axis=0)  # [N, 224, 224, 3]
        labels_array = np.array(labels)

        return images_array, labels_array

    # ------------ main entry ------------

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("===== Data Transformation Started =====")

            metadata_splits = self._load_valid_metadata()

            # Transform each split
            train_images, train_labels = self._transform_split(
                metadata_splits["train"], self.config.train_spectrogram_dir
            )
            valid_images, valid_labels = self._transform_split(
                metadata_splits["validation"], self.config.valid_spectrogram_dir
            )
            test_images, test_labels = self._transform_split(
                metadata_splits["test"], self.config.test_spectrogram_dir
            )

            # Save numpy arrays (224x224x3 images + labels)
            os.makedirs(self.config.feature_dir, exist_ok=True)

            save_numpy_array_data(self.config.train_feature_file_path, train_images)
            save_numpy_array_data(self.config.valid_feature_file_path, valid_images)
            save_numpy_array_data(self.config.test_feature_file_path, test_images)

            save_numpy_array_data(self.config.train_label_file_path, train_labels)
            save_numpy_array_data(self.config.valid_label_file_path, valid_labels)
            save_numpy_array_data(self.config.test_label_file_path, test_labels)

            logging.info("===== Data Transformation Completed =====")

            return DataTransformationArtifact(
                train_feature_file_path=self.config.train_feature_file_path,
                valid_feature_file_path=self.config.valid_feature_file_path,
                test_feature_file_path=self.config.test_feature_file_path,
                train_label_file_path=self.config.train_label_file_path,
                valid_label_file_path=self.config.valid_label_file_path,
                test_label_file_path=self.config.test_label_file_path,
                train_spectrogram_dir=self.config.train_spectrogram_dir,
                valid_spectrogram_dir=self.config.valid_spectrogram_dir,
                test_spectrogram_dir=self.config.test_spectrogram_dir,
            )

        except Exception as e:
            raise VisionInfantNetException(e, sys)
