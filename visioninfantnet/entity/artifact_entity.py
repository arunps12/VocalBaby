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

    # ======================= IMAGE FEATURE ARRAYS ============================
    train_image_feature_file_path: str
    valid_image_feature_file_path: str
    test_image_feature_file_path: str

    # ======================= MEL-SPECTROGRAM ARRAYS =========================
    train_spectrogram_feature_file_path: str
    valid_spectrogram_feature_file_path: str
    test_spectrogram_feature_file_path: str

    # ======================= CLASSICAL FEATURES =============================
    # ComParE
    train_compare_feature_file_path: str
    valid_compare_feature_file_path: str
    test_compare_feature_file_path: str

    # BoAW
    train_boaw_feature_file_path: str
    valid_boaw_feature_file_path: str
    test_boaw_feature_file_path: str

    # AUDEEP 
    train_audeep_feature_file_path: str
    valid_audeep_feature_file_path: str
    test_audeep_feature_file_path: str

    # Fisher Vectors
    train_fv_feature_file_path: str
    valid_fv_feature_file_path: str
    test_fv_feature_file_path: str

    # ======================= DEEP AUDIO EMBEDDINGS ==========================
    # PANNs embeddings
    train_panns_feature_file_path: str
    valid_panns_feature_file_path: str
    test_panns_feature_file_path: str

    # YAMNet embeddings
    train_yamnet_feature_file_path: str
    valid_yamnet_feature_file_path: str
    test_yamnet_feature_file_path: str

    # ======================= IMAGE EMBEDDINGS ================================
    train_image_embedding_file_path: str
    valid_image_embedding_file_path: str
    test_image_embedding_file_path: str

    # ======================= LABEL FILES ====================================
    train_label_file_path: str
    valid_label_file_path: str
    test_label_file_path: str

    # ======================= PNG SPECTROGRAM DIRECTORIES ====================
    train_spectrogram_image_dir: str
    valid_spectrogram_image_dir: str
    test_spectrogram_image_dir: str



@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    uar: float # Unweighted Average Recall


@dataclass
class ModelTrainerArtifact:
    

    trained_model_file_path: str
    preprocessing_object_file_path: str

    train_metric_artifact: ClassificationMetricArtifact
    valid_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact

    train_confusion_matrix_path: str
    valid_confusion_matrix_path: str
    test_confusion_matrix_path: str