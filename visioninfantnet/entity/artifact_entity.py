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


@dataclass
class DataTransformationArtifact:
    train_feature_file_path: str
    valid_feature_file_path: str
    test_feature_file_path: str
    train_label_file_path: str
    valid_label_file_path: str
    test_label_file_path: str

    train_spectrogram_dir: str
    valid_spectrogram_dir: str
    test_spectrogram_dir: str


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    uar: float # Unweighted Average Recall
