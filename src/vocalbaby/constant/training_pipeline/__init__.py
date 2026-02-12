import os

"""
defining common constant variable for training pipeline
"""

PIPELINE_NAME: str = "VisionInfantNet"
ARTIFACT_DIR: str = "artifacts"

# Metadata file name 
METADATA_FILE_NAME: str = "private_metadata.csv"

# Important metadata column names 
CHILD_ID_COLUMN: str = "child_ID"
AUDIO_ID_COLUMN: str = "clip_ID"      # wav file name
TARGET_COLUMN: str = "Answer"         # label column
AGE_COLUMN: str = "age_mo_round"
GENDER_COLUMN: str = "child_gender"
CORPUS_COLUMN: str = "corpus"

# Column will generate during ingestion (full wav path)
AUDIO_PATH_COLUMN: str = "path"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

# Name of the artifacts folder created for this stage
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Raw input locations (relative to project root)
DATA_INGESTION_RAW_AUDIO_DIR: str = "data/audio/raw"
DATA_INGESTION_RAW_METADATA_FILE: str = "data/metadata/private_metadata.csv"

# Output folders where ingested data will be saved inside artifacts/
DATA_INGESTION_INGESTED_AUDIO_DIR: str = "ingested_audio"
DATA_INGESTION_INGESTED_METADATA_DIR: str = "ingested_metadata"

# Full metadata filename to save after ingestion (with AUDIO_PATH_COLUMN added)
DATA_INGESTION_ARTIFACT_FULL_METADATA_FILE: str = "full_metadata.csv"

# Train/validation/test metadata filenames to save after ingestion; split is child-disjoint, do NOT need split ratios
DATA_INGESTION_TRAIN_METADATA_FILE: str = "train.csv"
DATA_INGESTION_VALID_METADATA_FILE: str = "valid.csv"
DATA_INGESTION_TEST_METADATA_FILE: str = "test.csv"

# Audio subfolders for child-disjoint split output
DATA_INGESTION_TRAIN_AUDIO_DIR: str = "train"
DATA_INGESTION_VALID_AUDIO_DIR: str = "valid"
DATA_INGESTION_TEST_AUDIO_DIR: str = "test"
# ------------------------------------------------------------------
# Child-disjoint split counts according to Babble Corpus
# ------------------------------------------------------------------
SPLIT_COUNTS = {
    "train": {
        "Crying": 243,
        "Laughing": 46,
        "Canonical": 444,
        "Non-canonical": 1437,
        "Junk": 1826
    },
    "valid": {
        "Crying": 163,
        "Laughing": 41,
        "Canonical": 378,
        "Non-canonical": 1678,
        "Junk": 1357
    },
    "test": {
        "Crying": 263,
        "Laughing": 62,
        "Canonical": 604,
        "Non-canonical": 1370,
        "Junk": 1392
    }
}


"""
Data Validation related constants start with DATA_VALIDATION_ VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"

DATA_VALIDATION_REPORT_FILE: str = "validation_report.yaml"

DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"

"""
Data Transformation related constants start with DATA_TRANSFORMATION_ VAR NAME
"""

# Root dir for all data transformation artifacts
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

# =============================================================================
# Numpy feature arrays (per split)
# =============================================================================

# All .npy feature files live under this directory
DATA_TRANSFORMATION_FEATURE_DIR: str = "features"

# ============================================================
# 1) IMAGE FEATURES 
# ============================================================
# DATA_TRANSFORMATION_TRAIN_IMAGE_FEATURE_FILE: str = "train_image_features.npy"
# DATA_TRANSFORMATION_VALID_IMAGE_FEATURE_FILE: str = "valid_image_features.npy"
# DATA_TRANSFORMATION_TEST_IMAGE_FEATURE_FILE: str = "test_image_features.npy"

# ============================================================
# 2) MEL-SPECTROGRAM FEATURES 
# ============================================================
# DATA_TRANSFORMATION_TRAIN_SPECTROGRAM_FEATURE_FILE: str = "train_melspectrogram_features.npy"
# DATA_TRANSFORMATION_VALID_SPECTROGRAM_FEATURE_FILE: str = "valid_melspectrogram_features.npy"
# DATA_TRANSFORMATION_TEST_SPECTROGRAM_FEATURE_FILE: str = "test_melspectrogram_features.npy"

# ============================================================
# 3) CLASSICAL / GLOBAL AUDIO FEATURES
# ============================================================

# ---- 3.1 ComParE / eGeMAPS ----
DATA_TRANSFORMATION_TRAIN_COMPARE_FEATURE_FILE: str = "train_compare_features.npy"
DATA_TRANSFORMATION_VALID_COMPARE_FEATURE_FILE: str = "valid_compare_features.npy"
DATA_TRANSFORMATION_TEST_COMPARE_FEATURE_FILE: str = "test_compare_features.npy"

# ---- 3.2 Bag-of-Audio-Words  ----
# DATA_TRANSFORMATION_TRAIN_BOAW_FEATURE_FILE: str = "train_boaw_features.npy"
# DATA_TRANSFORMATION_VALID_BOAW_FEATURE_FILE: str = "valid_boaw_features.npy"
# DATA_TRANSFORMATION_TEST_BOAW_FEATURE_FILE: str = "test_boaw_features.npy"

# ---- 3.3 AUDEEP embeddings  ----
# DATA_TRANSFORMATION_TRAIN_AUDEEP_FEATURE_FILE: str = "train_audeep_features.npy"
# DATA_TRANSFORMATION_VALID_AUDEEP_FEATURE_FILE: str = "valid_audeep_features.npy"
# DATA_TRANSFORMATION_TEST_AUDEEP_FEATURE_FILE: str = "test_audeep_features.npy"

# ---- 3.4 Fisher Vectors  ----
# DATA_TRANSFORMATION_TRAIN_FV_FEATURE_FILE: str = "train_fv_features.npy"
# DATA_TRANSFORMATION_VALID_FV_FEATURE_FILE: str = "valid_fv_features.npy"
# DATA_TRANSFORMATION_TEST_FV_FEATURE_FILE: str = "test_fv_features.npy"

# ============================================================
# 4) DEEP AUDIO EMBEDDINGS 
# ============================================================

# PANNs embeddings
# DATA_TRANSFORMATION_TRAIN_PANNS_FEATURE_FILE = "train_panns_features.npy"
# DATA_TRANSFORMATION_VALID_PANNS_FEATURE_FILE = "valid_panns_features.npy"
# DATA_TRANSFORMATION_TEST_PANNS_FEATURE_FILE = "test_panns_features.npy"

# YAMNet embeddings
# DATA_TRANSFORMATION_TRAIN_YAMNET_FEATURE_FILE = "train_yamnet_features.npy"
# DATA_TRANSFORMATION_VALID_YAMNET_FEATURE_FILE = "valid_yamnet_features.npy"
# DATA_TRANSFORMATION_TEST_YAMNET_FEATURE_FILE = "test_yamnet_features.npy"

# ============================================================
# 5) IMAGE EMBEDDINGS 
# ============================================================
# DATA_TRANSFORMATION_TRAIN_IMAGE_EMB_FEATURE_FILE = "train_image_embeddings.npy"
# DATA_TRANSFORMATION_VALID_IMAGE_EMB_FEATURE_FILE = "valid_image_embeddings.npy"
# DATA_TRANSFORMATION_TEST_IMAGE_EMB_FEATURE_FILE = "test_image_embeddings.npy"

# ============================================================
# 6) LABELS 
# ============================================================
DATA_TRANSFORMATION_TRAIN_LABEL_FILE: str = "train_labels.npy"
DATA_TRANSFORMATION_VALID_LABEL_FILE: str = "valid_labels.npy"
DATA_TRANSFORMATION_TEST_LABEL_FILE: str = "test_labels.npy"

# ============================================================
# Spectrogram PNG Directories 
# ============================================================
#DATA_TRANSFORMATION_SPECTROGRAM_IMAGE_DIR: str = "spectrogram_images"
#DATA_TRANSFORMATION_TRAIN_SPECTROGRAM_IMAGE_SUBDIR: str = "train"
#DATA_TRANSFORMATION_VALID_SPECTROGRAM_IMAGE_SUBDIR: str = "valid"
#DATA_TRANSFORMATION_TEST_SPECTROGRAM_IMAGE_SUBDIR: str = "test"

# ============================================================
# Audio parameters (still required)
# ============================================================
TARGET_SAMPLE_RATE: int = 16000
N_FFT: int = 512
HOP_LENGTH: int = 160
N_MELS: int = 64


"""
Training pipeline constants.

Model trainer related constants start with MODEL_TRAINER_*
"""

# Root directory inside artifacts for everything related to model training
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

# Subdirectory where the final trained model will be stored
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

# File name of the final trained model (XGBoost best model on eGeMAPS+SMOTE+Optuna)
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "xgb_egemaps_smote_optuna.pkl"

# Directory (inside model_trainer) where preprocessing objects are stored
MODEL_TRAINER_PREPROCESSING_DIR: str = "preprocessing"

# File name for the preprocessing object (e.g., SimpleImputer, scaler, encoder, etc.)
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

# File name for the label encoder object
LABEL_ENCODER_OBJECT_FILE_NAME: str = "label_encoder.pkl"

# CONFUSION MATRIX FILES
MODEL_TRAINER_CONFUSION_MATRIX_DIR: str = "confusion_matrices"
MODEL_TRAINER_TRAIN_CM_FILE_NAME: str = "train_cm.png"
MODEL_TRAINER_VALID_CM_FILE_NAME: str = "valid_cm.png"
MODEL_TRAINER_TEST_CM_FILE_NAME: str = "test_cm.png"

#  remote bucket (S3) for model artifacts
TRAINING_BUCKET_NAME: str = "visioninfantnet"
