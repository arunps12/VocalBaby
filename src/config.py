import os

# Define the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data/raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data/processed")
METADATA_PATH = os.path.join(ROOT_DIR, "data/metadata.csv")
PROCESSED_METADATA_PATH = os.path.join(ROOT_DIR, "data/processed_metadata.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
# Input and output directories
INPUT_test_audio_DIR = r"your/path/to/audio_dir"
