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

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

# numpy feature arrays store path (per split)
DATA_TRANSFORMATION_FEATURE_DIR: str = "features"

DATA_TRANSFORMATION_TRAIN_FEATURE_FILE: str = "train_features.npy"
DATA_TRANSFORMATION_VALID_FEATURE_FILE: str = "valid_features.npy"
DATA_TRANSFORMATION_TEST_FEATURE_FILE: str = "test_features.npy"

DATA_TRANSFORMATION_TRAIN_LABEL_FILE: str = "train_labels.npy"
DATA_TRANSFORMATION_VALID_LABEL_FILE: str = "valid_labels.npy"
DATA_TRANSFORMATION_TEST_LABEL_FILE: str = "test_labels.npy"

# spectrogram image directories 
DATA_TRANSFORMATION_SPECTROGRAM_DIR: str = "spectrograms"
DATA_TRANSFORMATION_TRAIN_SPECTROGRAM_DIR: str = "train"
DATA_TRANSFORMATION_VALID_SPECTROGRAM_DIR: str = "valid"
DATA_TRANSFORMATION_TEST_SPECTROGRAM_DIR: str = "test"

# Audio / spectrogram hyperparameters
TARGET_SAMPLE_RATE: int = 16000
N_FFT: int = 512
HOP_LENGTH: int = 160
N_MELS: int = 64
