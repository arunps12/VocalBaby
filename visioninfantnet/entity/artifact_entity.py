from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_metadata_path: str
    valid_metadata_path: str
    test_metadata_path: str
    train_audio_dir: str
    valid_audio_dir: str
    test_audio_dir: str