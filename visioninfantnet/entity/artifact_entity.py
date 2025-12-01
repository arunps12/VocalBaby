from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_metadata_path: str
    valid_metadata_path: str
    test_metadata_path: str
    train_audio_dir: str
    valid_audio_dir: str
    test_audio_dir: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    report_file_path: str
    drift_report_file_path: str
    validated_train_metadata_path: str
    validated_validation_metadata_path: str
    validated_test_metadata_path: str
    invalid_train_metadata_path: str
    invalid_validation_metadata_path: str
    invalid_test_metadata_path: str
    validated_train_audio_dir: str
    validated_validation_audio_dir: str
    validated_test_audio_dir: str
