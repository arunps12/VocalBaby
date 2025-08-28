import os

# Define the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data/raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data/processed")
PROCESSED_TRAIN_DATA_DIR = os.path.join(ROOT_DIR, "data/train_processed")

METADATA_PATH = os.path.join(ROOT_DIR, "data/metadata.csv")
PROCESSED_METADATA_PATH = os.path.join(ROOT_DIR, "data/processed_metadata.csv")
TRAIN_METADATA_PATH = os.path.join(ROOT_DIR, "data/train_voc.csv")
PROCESSED_TRAIN_METADATA_PATH = os.path.join(ROOT_DIR, "data/processed_train_voc.csv")
BALANCED_TRAIN_METADATA_PATH = os.path.join(ROOT_DIR, "data/balanced_train_voc.csv")


TEST_METADATA_PATH = os.path.join(ROOT_DIR, "data/test_voc.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOCAL_WAV2VEC2_MODEL_PATH = os.path.join(ROOT_DIR, "models", "local_wav2ve2")
# Input and output directories
INPUT_test_audio_DIR = r"\\hypatia.uio.no\lh-hf-iln-sociocognitivelab\Research\Ecology_early_language\Data\Collected\WAV\11 months"
