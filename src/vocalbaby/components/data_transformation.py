import os
import sys
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import opensmile
from panns_inference import AudioTagging
# from torch_vggish_yamnet import YAMNet, preprocess_audio
import timm

from vocalbaby.entity.config_entity import DataTransformationConfig
from vocalbaby.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from vocalbaby.constant.training_pipeline import (
    AUDIO_PATH_COLUMN,
    TARGET_COLUMN,
    TARGET_SAMPLE_RATE,
)
from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging
from vocalbaby.utils.main_utils.utils import save_numpy_array_data


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def load_audio(path, sr=16000):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y
    except Exception:
        return np.zeros(sr)  # 1-second silence placeholder


def compute_melspec(y, sr=16000, n_mels=64):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)


def save_png(img_array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_array).save(path)


def mel_to_png_image(melspec):
    m_min, m_max = melspec.min(), melspec.max()
    m_norm = (melspec - m_min) / (m_max - m_min + 1e-8)
    img = (m_norm * 255).astype(np.uint8)
    img = Image.fromarray(img).resize((224, 224), Image.BICUBIC)
    return np.array(img.convert("RGB"))


# ============================================================================
# MFCC LLD EXTRACTION (for BoAW + FV)
# ============================================================================

def extract_mfcc_llds(y, sr=16000, n_mfcc=20):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T   # shape: (frames, 20)


# ============================================================================
# OPENSMLIE eGeMAPSv02 FEATURES (88-dim)
# ============================================================================

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_egemaps(path):
    try:
        feats = smile.process_file(path)
        return feats.values.flatten().astype(np.float32)
    except Exception:
        return np.zeros(88, dtype=np.float32)


# ============================================================================
# GLOBAL DEVICE
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# RESNET50 IMAGE EMBEDDINGS (timm)
# ============================================================================

def load_resnet50_embedding_model():
    """
    Create a ResNet50 backbone that outputs a 2048-d embedding.
    We use timm with num_classes=0 and global_pool='avg' to remove classifier.
    """
    model = timm.create_model(
        "resnet50",
        pretrained=True,
        num_classes=0,       # no classification head
        global_pool="avg",   # global average pooling -> (N, 2048)
    )
    model.to(DEVICE)
    model.eval()
    return model


def extract_resnet_embedding(img_array: np.ndarray, model) -> np.ndarray:
    """
    img_array: (224, 224, 3) uint8
    Returns: (2048,) float32 embedding.
    """
    try:
        # Convert to float tensor, normalize with ImageNet stats
        img = torch.from_numpy(img_array).float() / 255.0  # (H, W, 3)
        img = img.permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        img = (img - mean) / (std + 1e-8)

        with torch.no_grad():
            emb = model(img)  # (1, 2048)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        return np.zeros(2048, dtype=np.float32)


# ============================================================================
# PANNs CNN14 EMBEDDINGS (panns-inference)
# ============================================================================

def load_panns_model():
    """
    Returns a PANNs AudioTagging model for clip-level embeddings (2048-d).
    """
    model = AudioTagging(checkpoint_path=None, device=DEVICE)
    return model


def extract_panns_embedding(panns_model, audio_path: str) -> np.ndarray:
    """
    Uses panns-inference AudioTagging to get a 2048-d clip embedding.
    """
    try:
        (clip_embedding, _) = panns_model.inference(audio_path)
        clip_embedding = clip_embedding.squeeze(0)  # (2048,)
        return clip_embedding.cpu().numpy().astype(np.float32)
    except Exception:
        return np.zeros(2048, dtype=np.float32)


# ============================================================================
# YAMNET EMBEDDINGS (torch_vggish_yamnet)
# ============================================================================

def load_yamnet_model():
    """
    Placeholder YAMNet loader.
    Currently returns None; the pipeline will fill 1024-d zeros.
    """
    return None


def extract_yamnet_embedding(yamnet_model, audio_path: str) -> np.ndarray:
    """
    Placeholder YAMNet embedding extractor.
    Returns a 1024-d zero vector.
    """
    return np.zeros(1024, dtype=np.float32)


# ============================================================================
# BoAW ENCODER (KMeans over MFCC LLDs)
# ============================================================================

class BoAWEncoder:
    """
    Bag-of-Audio-Words using sklearn KMeans over MFCC LLDs.

    - Fit on TRAIN MFCC LLDs only.
    - Transform any MFCC matrix (frames, dim) into a histogram (codebook_size,).
    """

    def __init__(self, codebook_size: int = 1000, random_state: int = 42):
        self.codebook_size = codebook_size
        self.random_state = random_state
        self.kmeans: KMeans | None = None

    def fit(self, mfcc_list: list[np.ndarray]):
        """
        mfcc_list: list of arrays, each shape (frames, mfcc_dim)
        Fit KMeans on stacked MFCC frames.
        """
        try:
            all_frames = np.vstack(mfcc_list)  # (total_frames, mfcc_dim)
            self.kmeans = KMeans(
                n_clusters=self.codebook_size,
                random_state=self.random_state,
                n_init=10,
            )
            self.kmeans.fit(all_frames)
        except Exception as e:
            raise VocalBabyException(e, sys)

    def transform_one(self, mfcc: np.ndarray) -> np.ndarray:
        """
        mfcc: (frames, mfcc_dim)
        Returns: (codebook_size,) histogram vector.
        """
        if self.kmeans is None or mfcc.size == 0:
            return np.zeros(self.codebook_size, dtype=np.float32)

        try:
            labels = self.kmeans.predict(mfcc)  # (frames,)
            hist, _ = np.histogram(
                labels,
                bins=np.arange(self.codebook_size + 1),
                density=False,
            )
            if hist.sum() > 0:
                hist = hist / hist.sum()
            return hist.astype(np.float32)
        except Exception:
            return np.zeros(self.codebook_size, dtype=np.float32)


# ============================================================================
# FISHER VECTOR ENCODER (GMM over MFCC LLDs)
# ============================================================================

class FisherVectorEncoder:
    """
    Fisher Vector encoder using sklearn GaussianMixture over MFCC LLDs.

    - Fit GMM on TRAIN MFCC LLDs only.
    - Transform MFCC sequences into FV vectors (2 * K * D).
    """

    def __init__(self, n_components: int = 16, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm: GaussianMixture | None = None
        self.eps = 1e-8

    def fit(self, mfcc_list: list[np.ndarray]):
        """
        mfcc_list: list of arrays, each shape (frames, mfcc_dim)
        """
        try:
            all_frames = np.vstack(mfcc_list)  # (total_frames, mfcc_dim)
            self.gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                random_state=self.random_state,
            )
            self.gmm.fit(all_frames)
        except Exception as e:
            raise VocalBabyException(e, sys)

    def transform_one(self, mfcc: np.ndarray) -> np.ndarray:
        """
        mfcc: (frames, mfcc_dim)
        Returns: Fisher Vector of shape (2 * K * D,)
        """
        if self.gmm is None or mfcc.size == 0:
            # Fallback vector of expected dimension (2*K*D = 640)
            return np.zeros(2 * self.n_components * 20, dtype=np.float32)

        try:
            X = mfcc  # (T, D)
            T_frames, D = X.shape
            K = self.n_components

            # Responsibilities
            q = self.gmm.predict_proba(X)  # (T, K)
            # Mixture params
            w = self.gmm.weights_          # (K,)
            mu = self.gmm.means_           # (K, D)
            sigma = self.gmm.covariances_  # (K, D)

            # Avoid division by zero
            sigma = sigma + self.eps

            # Compute posterior sums
            N_k = q.sum(axis=0)[:, np.newaxis]  # (K, 1)

            # Centered features
            X_expanded = X[np.newaxis, :, :]              # (1, T, D)
            mu_expanded = mu[:, np.newaxis, :]            # (K, 1, D)
            sigma_expanded = sigma[:, np.newaxis, :]      # (K, 1, D)

            x_mu = (X_expanded - mu_expanded) / np.sqrt(sigma_expanded)

            # Compute d_mu
            q_expanded = q[:, :, np.newaxis]              # (T, K, 1)
            x_mu_weighted = q_expanded * x_mu.transpose(1, 0, 2)  # (T, K, D)
            d_mu = x_mu_weighted.sum(axis=0)              # (K, D)

            # Normalize by sqrt(weights) and T
            d_mu = d_mu / (np.sqrt(w[:, np.newaxis] + self.eps))
            d_mu = d_mu / (T_frames + self.eps)

            # Compute d_sigma
            x_sigma = ((X_expanded - mu_expanded) ** 2) / sigma_expanded - 1
            x_sigma_weighted = q_expanded * x_sigma.transpose(1, 0, 2)  # (T, K, D)
            d_sigma = x_sigma_weighted.sum(axis=0)                      # (K, D)

            d_sigma = d_sigma / (np.sqrt(2 * w[:, np.newaxis] + self.eps))
            d_sigma = d_sigma / (T_frames + self.eps)

            fv = np.concatenate([d_mu, d_sigma], axis=0).flatten()  # (2*K*D,)

            # Power normalization (signed square root)
            fv = np.sign(fv) * np.sqrt(np.abs(fv) + self.eps)

            # L2 normalization
            fv = fv / (np.linalg.norm(fv) + self.eps)


            return fv.astype(np.float32)
        except Exception:
            # Fallback vector of expected dimension (2*K*D = 640)
            return np.zeros(2 * self.n_components * 20, dtype=np.float32)


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.config = data_transformation_config
            self.validation_artifact = data_validation_artifact

            
            # Models for embeddings
            # logging.info("Loading ResNet50 model for image embeddings...")
            # self.resnet_model = load_resnet50_embedding_model()

            # logging.info("Loading PANNs CNN14 model for audio embeddings...")
            # self.panns_model = load_panns_model()

            # logging.info("Loading YAMNet model for audio embeddings...")
            # self.yamnet_model = load_yamnet_model()

            # Encoders for BoAW and Fisher Vector
            # logging.info("Initializing BoAW and Fisher Vector encoders...")
            # self.boaw_encoder = BoAWEncoder(codebook_size=1000, random_state=42)
            # self.fv_encoder = FisherVectorEncoder(n_components=16, random_state=42)

        except Exception as e:
            raise VocalBabyException(e, sys)

    # ----------------------------------------------------------------------
    # Load validated metadata
    # ----------------------------------------------------------------------
    def _load_valid_metadata(self):
        """
        Load validated train/valid/test metadata CSVs from DataValidationArtifact.
        """
        try:
            train_df = pd.read_csv(self.validation_artifact.validated_train_metadata_path)
            valid_df = pd.read_csv(self.validation_artifact.validated_validation_metadata_path)
            test_df = pd.read_csv(self.validation_artifact.validated_test_metadata_path)

            logging.info(
                f"Loaded validated metadata: "
                f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}"
            )

            return {
                "train": train_df,
                "validation": valid_df,
                "test": test_df,
            }

        except Exception as e:
            raise VocalBabyException(e, sys)

    # ----------------------------------------------------------------------
    # Helper: pad variable-length mel spectrograms
    # ----------------------------------------------------------------------
    @staticmethod
    def _pad_melspec_list(melspec_list: list[np.ndarray]) -> np.ndarray:
        """
        melspec_list: list of arrays, each (n_mels, T_i)
        Returns: (N, n_mels, max_T) padded with zeros on the right.
        NOTE: Not used in eGeMAPS-only run, but kept for future use.
        """
        if len(melspec_list) == 0:
            return np.zeros((0, 64, 0), dtype=np.float32)

        n_mels = melspec_list[0].shape[0]
        max_T = max(m.shape[1] for m in melspec_list)
        N = len(melspec_list)

        padded = np.zeros((N, n_mels, max_T), dtype=np.float32)
        for i, m in enumerate(melspec_list):
            T = m.shape[1]
            padded[i, :, :T] = m

        return padded

    # ----------------------------------------------------------------------
    # Extract all base features for a given split
    # ----------------------------------------------------------------------
    def _extract_split_base_features(
        self,
        df: pd.DataFrame,
        spectrogram_image_dir: str,
    ):
        """
        Original version computed:
        - Mel spectrograms, PNG images
        - ResNet embeddings
        - PANNs & YAMNet embeddings
        - eGeMAPS
        - MFCC LLDs

        For **eGeMAPS-only** we ONLY compute:
        - eGeMAPS (88)
        - labels


        """

        try:
            # os.makedirs(spectrogram_image_dir, exist_ok=True)

            # image_arrays = []
            # mel_specs = []
            # resnet_embeds = []
            # panns_embeds = []
            # yamnet_embeds = []
            egemaps_feats = []
            labels = []
            # mfcc_lld_list = []

            logging.info(f"Extracting eGeMAPS-only base features for split with {len(df)} files...")

            for idx, row in tqdm(df.iterrows(), total=len(df)):
                audio_path = row[AUDIO_PATH_COLUMN]
                label = row[TARGET_COLUMN]

                # -----------------------
                # Load audio 
                # -----------------------
                # y = load_audio(audio_path, sr=TARGET_SAMPLE_RATE)

                # -----------------------
                # Mel spectrogram (64 mels) 
                # -----------------------
                # melspec = compute_melspec(y, sr=TARGET_SAMPLE_RATE, n_mels=64)
                # mel_specs.append(melspec)

                # -----------------------
                # PNG spectrogram image + image array 
                # -----------------------
                # img_rgb = mel_to_png_image(melspec)            # (224, 224, 3)
                # image_arrays.append(img_rgb)

                # png_path = os.path.join(spectrogram_image_dir, f"{idx}.png")
                # save_png(img_rgb, png_path)

                # -----------------------
                # ResNet50 embedding from image - 
                # -----------------------
                # resnet_emb = extract_resnet_embedding(img_rgb, self.resnet_model)
                # resnet_embeds.append(resnet_emb)

                # -----------------------
                # PANNs embedding (2048) - 
                # -----------------------
                # panns_emb = extract_panns_embedding(self.panns_model, audio_path)
                # panns_embeds.append(panns_emb)

                # -----------------------
                # YAMNet embedding (1024) - 
                # -----------------------
                # yam_emb = extract_yamnet_embedding(self.yamnet_model, audio_path)
                # yamnet_embeds.append(yam_emb)

                # -----------------------
                # eGeMAPS (88) - 
                # -----------------------
                egemaps_vec = extract_egemaps(audio_path)
                egemaps_feats.append(egemaps_vec)

                # -----------------------
                # MFCC LLDs (frames, 20) for BoAW and FV - 
                # -----------------------
                # mfcc_lld = extract_mfcc_llds(y, sr=TARGET_SAMPLE_RATE, n_mfcc=20)
                # mfcc_lld_list.append(mfcc_lld)

                # -----------------------
                # Labels 
                # -----------------------
                labels.append(label)

            # Convert lists to numpy arrays
            # image_arrays = np.stack(image_arrays, axis=0).astype(np.uint8)          # (N, 224, 224, 3)
            # mel_specs_padded = self._pad_melspec_list(mel_specs).astype(np.float32) # (N, 64, T_max)
            # resnet_embeds = np.stack(resnet_embeds, axis=0).astype(np.float32)      # (N, 2048)
            # panns_embeds = np.stack(panns_embeds, axis=0).astype(np.float32)        # (N, 2048)
            # yamnet_embeds = np.stack(yamnet_embeds, axis=0).astype(np.float32)      # (N, 1024)
            egemaps_feats = np.stack(egemaps_feats, axis=0).astype(np.float32)      # (N, 88)
            labels = np.array(labels)

            return {
                # "image_arrays": image_arrays,
                # "mel_specs": mel_specs_padded,
                # "resnet_embeddings": resnet_embeds,
                # "panns_embeddings": panns_embeds,
                # "yamnet_embeddings": yamnet_embeds,
                "egemaps": egemaps_feats,
                "labels": labels,
                # "mfcc_lld_list": mfcc_lld_list,
            }

        except Exception as e:
            raise VocalBabyException(e, sys)


    # ----------------------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------------------
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("===== Data Transformation Started  =====")

            # 1. Load validated metadata
            metadata_splits = self._load_valid_metadata()

            train_df = metadata_splits["train"]
            valid_df = metadata_splits["validation"]
            test_df = metadata_splits["test"]

            # 2. Extract base features for each split
            logging.info("Extracting base features for TRAIN split...")
            train_base = self._extract_split_base_features(
                train_df, ""  # self.config.train_spectrogram_image_dir 
            )

            logging.info("Extracting base features for VALIDATION split...")
            valid_base = self._extract_split_base_features(
                valid_df, ""  # self.config.valid_spectrogram_image_dir 
            )

            logging.info("Extracting base features for TEST split...")
            test_base = self._extract_split_base_features(
                test_df, ""  # self.config.test_spectrogram_image_dir 
            )

            # ------------------------------------------------------------------
            # 3. Fit BoAW & FV encoders on TRAIN MFCC LLDs only - 
            # ------------------------------------------------------------------
            # logging.info("Fitting BoAW encoder (KMeans) on TRAIN MFCC LLDs...")
            # self.boaw_encoder.fit(train_base["mfcc_lld_list"])

            # logging.info("Fitting Fisher Vector encoder (GMM) on TRAIN MFCC LLDs...")
            # self.fv_encoder.fit(train_base["mfcc_lld_list"])

            # ------------------------------------------------------------------
            # 4. Compute BoAW & FV features for each split -
            # ------------------------------------------------------------------
            # def compute_boaw_and_fv(mfcc_lld_list: list[np.ndarray]):
            #     boaw_list = []
            #     fv_list = []
            #     for mfcc in mfcc_lld_list:
            #         boaw_feat = self.boaw_encoder.transform_one(mfcc)
            #         fv_feat = self.fv_encoder.transform_one(mfcc)
            #         boaw_list.append(boaw_feat)
            #         fv_list.append(fv_feat)
            #     return (
            #         np.stack(boaw_list, axis=0).astype(np.float32),
            #         np.stack(fv_list, axis=0).astype(np.float32),
            #     )

            # logging.info("Computing BoAW & FV for TRAIN...")
            # train_boaw, train_fv = compute_boaw_and_fv(train_base["mfcc_lld_list"])

            # logging.info("Computing BoAW & FV for VALIDATION...")
            # valid_boaw, valid_fv = compute_boaw_and_fv(valid_base["mfcc_lld_list"])

            # logging.info("Computing BoAW & FV for TEST...")
            # test_boaw, test_fv = compute_boaw_and_fv(test_base["mfcc_lld_list"])

            # ------------------------------------------------------------------
            # 5. AUDEEP placeholders (N, 0) - 
            # ------------------------------------------------------------------
            # train_N = len(train_base["labels"])
            # valid_N = len(valid_base["labels"])
            # test_N = len(test_base["labels"])

            # train_audeep = np.zeros((train_N, 0), dtype=np.float32)
            # valid_audeep = np.zeros((valid_N, 0), dtype=np.float32)
            # test_audeep = np.zeros((test_N, 0), dtype=np.float32)

            # ------------------------------------------------------------------
            # 6. Save ALL features to disk using config paths
            # ------------------------------------------------------------------
            logging.info("Creating feature directory if needed...")
            os.makedirs(self.config.feature_dir, exist_ok=True)

            # ========== IMAGE FEATURE ARRAYS (224x224x3) -  ==========
            # save_numpy_array_data(
            #     self.config.train_image_feature_file_path, train_base["image_arrays"]
            # )
            # save_numpy_array_data(
            #     self.config.valid_image_feature_file_path, valid_base["image_arrays"]
            # )
            # save_numpy_array_data(
            #     self.config.test_image_feature_file_path, test_base["image_arrays"]
            # )

            # ========== MEL-SPECTROGRAM ARRAYS (64 x T_max) -  ==========
            # save_numpy_array_data(
            #     self.config.train_spectrogram_feature_file_path, train_base["mel_specs"]
            # )
            # save_numpy_array_data(
            #     self.config.valid_spectrogram_feature_file_path, valid_base["mel_specs"]
            # )
            # save_numpy_array_data(
            #     self.config.test_spectrogram_feature_file_path, test_base["mel_specs"]
            # )

            # ========== CLASSICAL FEATURES: eGeMAPS (ComParE-like) -  ==========
            save_numpy_array_data(
                self.config.train_compare_feature_file_path, train_base["egemaps"]
            )
            save_numpy_array_data(
                self.config.valid_compare_feature_file_path, valid_base["egemaps"]
            )
            save_numpy_array_data(
                self.config.test_compare_feature_file_path, test_base["egemaps"]
            )

            # ========== BoAW FEATURES (1000-d histograms) -  ==========
            # save_numpy_array_data(
            #     self.config.train_boaw_feature_file_path, train_boaw
            # )
            # save_numpy_array_data(
            #     self.config.valid_boaw_feature_file_path, valid_boaw
            # )
            # save_numpy_array_data(
            #     self.config.test_boaw_feature_file_path, test_boaw
            # )

            # ========== AUDEEP PLACEHOLDER (N, 0) -  ==========
            # save_numpy_array_data(
            #     self.config.train_audeep_feature_file_path, train_audeep
            # )
            # save_numpy_array_data(
            #     self.config.valid_audeep_feature_file_path, valid_audeep
            # )
            # save_numpy_array_data(
            #     self.config.test_audeep_feature_file_path, test_audeep
            # )

            # ========== FISHER VECTOR FEATURES (640-d) -  ==========
            # save_numpy_array_data(
            #     self.config.train_fv_feature_file_path, train_fv
            # )
            # save_numpy_array_data(
            #     self.config.valid_fv_feature_file_path, valid_fv
            # )
            # save_numpy_array_data(
            #     self.config.test_fv_feature_file_path, test_fv
            # )

            # ========== DEEP AUDIO EMBEDDINGS: PANNs (2048-d) -  ==========
            # save_numpy_array_data(
            #     self.config.train_panns_feature_file_path, train_base["panns_embeddings"]
            # )
            # save_numpy_array_data(
            #     self.config.valid_panns_feature_file_path, valid_base["panns_embeddings"]
            # )
            # save_numpy_array_data(
            #     self.config.test_panns_feature_file_path, test_base["panns_embeddings"]
            # )

            # ========== DEEP AUDIO EMBEDDINGS: YAMNet (1024-d) -  ==========
            # save_numpy_array_data(
            #     self.config.train_yamnet_feature_file_path, train_base["yamnet_embeddings"]
            # )
            # save_numpy_array_data(
            #     self.config.valid_yamnet_feature_file_path, valid_base["yamnet_embeddings"]
            # )
            # save_numpy_array_data(
            #     self.config.test_yamnet_feature_file_path, test_base["yamnet_embeddings"]
            # )

            # ========== IMAGE EMBEDDINGS: ResNet50 (2048-d) -  ==========
            # save_numpy_array_data(
            #     self.config.train_image_embedding_file_path, train_base["resnet_embeddings"]
            # )
            # save_numpy_array_data(
            #     self.config.valid_image_embedding_file_path, valid_base["resnet_embeddings"]
            # )
            # save_numpy_array_data(
            #     self.config.test_image_embedding_file_path, test_base["resnet_embeddings"]
            # )

            # ========== LABELS -  ==========
            save_numpy_array_data(
                self.config.train_label_file_path, train_base["labels"]
            )
            save_numpy_array_data(
                self.config.valid_label_file_path, valid_base["labels"]
            )
            save_numpy_array_data(
                self.config.test_label_file_path, test_base["labels"]
            )

            logging.info("===== Data Transformation Completed Successfully (eGeMAPS-only) =====")

            # ------------------------------------------------------------------
            # 7. Return DataTransformationArtifact
            # ------------------------------------------------------------------
            data_transformation_artifact = DataTransformationArtifact(
                # CLASSICAL FEATURES (eGeMAPS)
                train_compare_feature_file_path=self.config.train_compare_feature_file_path,
                valid_compare_feature_file_path=self.config.valid_compare_feature_file_path,
                test_compare_feature_file_path=self.config.test_compare_feature_file_path,

                # LABEL FILES
                train_label_file_path=self.config.train_label_file_path,
                valid_label_file_path=self.config.valid_label_file_path,
                test_label_file_path=self.config.test_label_file_path,

                
                # IMAGE FEATURE ARRAYS
                # train_image_feature_file_path=self.config.train_image_feature_file_path,
                # valid_image_feature_file_path=self.config.valid_image_feature_file_path,
                # test_image_feature_file_path=self.config.test_image_feature_file_path,

                # MEL-SPECTROGRAM ARRAYS
                # train_spectrogram_feature_file_path=self.config.train_spectrogram_feature_file_path,
                # valid_spectrogram_feature_file_path=self.config.valid_spectrogram_feature_file_path,
                # test_spectrogram_feature_file_path=self.config.test_spectrogram_feature_file_path,

                # BoAW
                # train_boaw_feature_file_path=self.config.train_boaw_feature_file_path,
                # valid_boaw_feature_file_path=self.config.valid_boaw_feature_file_path,
                # test_boaw_feature_file_path=self.config.test_boaw_feature_file_path,

                # AUDEEP (placeholder)
                # train_audeep_feature_file_path=self.config.train_audeep_feature_file_path,
                # valid_audeep_feature_file_path=self.config.valid_audeep_feature_file_path,
                # test_audeep_feature_file_path=self.config.test_audeep_feature_file_path,

                # Fisher Vectors
                # train_fv_feature_file_path=self.config.train_fv_feature_file_path,
                # valid_fv_feature_file_path=self.config.valid_fv_feature_file_path,
                # test_fv_feature_file_path=self.config.test_fv_feature_file_path,

                # DEEP AUDIO EMBEDDINGS: PANNs
                # train_panns_feature_file_path=self.config.train_panns_feature_file_path,
                # valid_panns_feature_file_path=self.config.valid_panns_feature_file_path,
                # test_panns_feature_file_path=self.config.test_panns_feature_file_path,

                # DEEP AUDIO EMBEDDINGS: YAMNet
                # train_yamnet_feature_file_path=self.config.train_yamnet_feature_file_path,
                # valid_yamnet_feature_file_path=self.config.valid_yamnet_feature_file_path,
                # test_yamnet_feature_file_path=self.config.test_yamnet_feature_file_path,

                # IMAGE EMBEDDINGS: ResNet50
                # train_image_embedding_file_path=self.config.train_image_embedding_file_path,
                # valid_image_embedding_file_path=self.config.valid_image_embedding_file_path,
                # test_image_embedding_file_path=self.config.test_image_embedding_file_path,

                # PNG SPECTROGRAM DIRECTORIES
                # train_spectrogram_image_dir=self.config.train_spectrogram_image_dir,
                # valid_spectrogram_image_dir=self.config.valid_spectrogram_image_dir,
                # test_spectrogram_image_dir=self.config.test_spectrogram_image_dir,
            )

            return data_transformation_artifact

        except Exception as e:
            raise VocalBabyException(e, sys)
