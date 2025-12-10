from datetime import datetime
import os
from visioninfantnet.constant import training_pipeline


class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_name: str = training_pipeline.ARTIFACT_DIR
        self.artifact_dir: str = os.path.join(self.artifact_name, timestamp)
        self.model_dir: str = os.path.join("final_model")
        self.timestamp: str = timestamp


class DataIngestionConfig:
    """
    Builds paths needed for data ingestion:
    - Raw audio + raw metadata input
    - Ingested audio + metadata output inside artifacts/
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )

        
        self.raw_audio_dir: str = training_pipeline.DATA_INGESTION_RAW_AUDIO_DIR
        self.raw_metadata_file: str = training_pipeline.DATA_INGESTION_RAW_METADATA_FILE

        self.full_metadata_file: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_ARTIFACT_FULL_METADATA_FILE,
        )

        
        self.ingested_audio_dir: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_AUDIO_DIR,
        )

        self.train_audio_dir: str = os.path.join(
            self.ingested_audio_dir,
            training_pipeline.DATA_INGESTION_TRAIN_AUDIO_DIR,
        )
        self.valid_audio_dir: str = os.path.join(
            self.ingested_audio_dir,
            training_pipeline.DATA_INGESTION_VALID_AUDIO_DIR,
        )
        self.test_audio_dir: str = os.path.join(
            self.ingested_audio_dir,
            training_pipeline.DATA_INGESTION_TEST_AUDIO_DIR,
        )

       
        self.ingested_metadata_dir: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_METADATA_DIR,
        )

        
        self.train_metadata_file: str = os.path.join(
            self.ingested_metadata_dir,
            training_pipeline.DATA_INGESTION_TRAIN_METADATA_FILE,
        )

        self.valid_metadata_file: str = os.path.join(
            self.ingested_metadata_dir,
            training_pipeline.DATA_INGESTION_VALID_METADATA_FILE,
        )

        self.test_metadata_file: str = os.path.join(
            self.ingested_metadata_dir,
            training_pipeline.DATA_INGESTION_TEST_METADATA_FILE,
        )

        self.split_counts: dict = training_pipeline.SPLIT_COUNTS


class DataValidationConfig:
    """
    Builds paths needed for data validation:
    - Validated metadata directory (no audio copied)
    - Invalid metadata directory
    - Validation + drift report paths
    - Schema path
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME,
        )

        self.report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_REPORT_FILE,
        )

    
        self.drift_report_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
        )
        self.drift_report_file_path: str = os.path.join(
            self.drift_report_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )

        

        self.validated_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR,
        )

        self.validated_metadata_dir: str = os.path.join(
            self.validated_dir,
            "metadata",
        )

        self.validated_train_metadata_path: str = os.path.join(
            self.validated_metadata_dir,
            training_pipeline.DATA_INGESTION_TRAIN_METADATA_FILE,
        )

        self.validated_validation_metadata_path: str = os.path.join(
            self.validated_metadata_dir,
            training_pipeline.DATA_INGESTION_VALID_METADATA_FILE,
        )

        self.validated_test_metadata_path: str = os.path.join(
            self.validated_metadata_dir,
            training_pipeline.DATA_INGESTION_TEST_METADATA_FILE,
        )

        

        self.invalid_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR, 
        )

        self.invalid_metadata_dir: str = os.path.join(
            self.invalid_dir,
            "metadata",
        )

        self.invalid_train_metadata_path: str = os.path.join(
            self.invalid_metadata_dir,
            "invalid_train_metadata.csv",
        )

        self.invalid_validation_metadata_path: str = os.path.join(
            self.invalid_metadata_dir,
            "invalid_valid_metadata.csv",
        )

        self.invalid_test_metadata_path: str = os.path.join(
            self.invalid_metadata_dir,
            "invalid_test_metadata.csv",
        )

        

        self.schema_file_path: str = training_pipeline.SCHEMA_FILE_PATH




class DataTransformationConfig:
    """
    Configuration for saving transformed features.
    Now ONLY ComParE (eGeMAPS) features are enabled.
    All other feature types are commented out.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # =========================================================================
        # Root directory for all transformation artifacts
        # =========================================================================
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
        )

        # =========================================================================
        # Feature directory (all .npy matrices saved here)
        # =========================================================================
        self.feature_dir = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_FEATURE_DIR,
        )

        # =====================================================================
        # 1) IMAGE FEATURES 
        # =====================================================================
        # self.train_image_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TRAIN_IMAGE_FEATURE_FILE,
        # )
        # self.valid_image_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_VALID_IMAGE_FEATURE_FILE,
        # )
        # self.test_image_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TEST_IMAGE_FEATURE_FILE,
        # )

        # =====================================================================
        # 2) MEL-SPECTROGRAM FEATURES 
        # =====================================================================
        # self.train_spectrogram_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TRAIN_SPECTROGRAM_FEATURE_FILE,
        # )
        # self.valid_spectrogram_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_VALID_SPECTROGRAM_FEATURE_FILE,
        # )
        # self.test_spectrogram_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TEST_SPECTROGRAM_FEATURE_FILE,
        # )

        # =====================================================================
        # 3) CLASSICAL FEATURES 
        # =====================================================================

        # ---- ComParE (openSMILE eGeMAPS) ----
        self.train_compare_feature_file_path = os.path.join(
            self.feature_dir,
            training_pipeline.DATA_TRANSFORMATION_TRAIN_COMPARE_FEATURE_FILE,
        )
        self.valid_compare_feature_file_path = os.path.join(
            self.feature_dir,
            training_pipeline.DATA_TRANSFORMATION_VALID_COMPARE_FEATURE_FILE,
        )
        self.test_compare_feature_file_path = os.path.join(
            self.feature_dir,
            training_pipeline.DATA_TRANSFORMATION_TEST_COMPARE_FEATURE_FILE,
        )

        # ---- BoAW  ----
        # self.train_boaw_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TRAIN_BOAW_FEATURE_FILE,
        # )
        # self.valid_boaw_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_VALID_BOAW_FEATURE_FILE,
        # )
        # self.test_boaw_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TEST_BOAW_FEATURE_FILE,
        # )

        # ---- AUDEEP  ----
        # self.train_audeep_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TRAIN_AUDEEP_FEATURE_FILE,
        # )
        # self.valid_audeep_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_VALID_AUDEEP_FEATURE_FILE,
        # )
        # self.test_audeep_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TEST_AUDEEP_FEATURE_FILE,
        # )

        # ---- Fisher Vectors  ----
        # self.train_fv_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TRAIN_FV_FEATURE_FILE,
        # )
        # self.valid_fv_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_VALID_FV_FEATURE_FILE,
        # )
        # self.test_fv_feature_file_path = os.path.join(
        #     self.feature_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TEST_FV_FEATURE_FILE,
        # )

        # =========================================================================
        # 4) LABEL FILES 
        # =========================================================================
        self.train_label_file_path = os.path.join(
            self.feature_dir,
            training_pipeline.DATA_TRANSFORMATION_TRAIN_LABEL_FILE,
        )
        self.valid_label_file_path = os.path.join(
            self.feature_dir,
            training_pipeline.DATA_TRANSFORMATION_VALID_LABEL_FILE,
        )
        self.test_label_file_path = os.path.join(
            self.feature_dir,
            training_pipeline.DATA_TRANSFORMATION_TEST_LABEL_FILE,
        )

        # =========================================================================
        # 5) SPECTROGRAM IMAGE DIRECTORIES 
        # =========================================================================
        # self.spectrogram_image_base_dir = os.path.join(
        #     self.data_transformation_dir,
        #     training_pipeline.DATA_TRANSFORMATION_SPECTROGRAM_IMAGE_DIR,
        # )
        # self.train_spectrogram_image_dir = os.path.join(
        #     self.spectrogram_image_base_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TRAIN_SPECTROGRAM_IMAGE_SUBDIR,
        # )
        # self.valid_spectrogram_image_dir = os.path.join(
        #     self.spectrogram_image_base_dir,
        #     training_pipeline.DATA_TRANSFORMATION_VALID_SPECTROGRAM_IMAGE_SUBDIR,
        # )
        # self.test_spectrogram_image_dir = os.path.join(
        #     self.spectrogram_image_base_dir,
        #     training_pipeline.DATA_TRANSFORMATION_TEST_SPECTROGRAM_IMAGE_SUBDIR,
        # )

        # =========================================================================
        # 6) PANNs and YAMNet embeddings 
        # =========================================================================
        # self.train_panns_feature_file_path = ...
        # self.valid_panns_feature_file_path = ...
        # self.test_panns_feature_file_path = ...
        #
        # self.train_yamnet_feature_file_path = ...
        # self.valid_yamnet_feature_file_path = ...
        # self.test_yamnet_feature_file_path = ...

        # =========================================================================
        # 7) IMAGE EMBEDDINGS
        # =========================================================================
        # self.train_image_embedding_file_path = ...
        # self.valid_image_embedding_file_path = ...
        # self.test_image_embedding_file_path = ...


class ModelTrainerConfig:
    """
    Configuration for Model Trainer Component:
    - Where to save trained model
    - Where to save preprocessing object (SimpleImputer)
    - Where model_trainer directory lives inside artifacts/
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # model_trainer/
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )

        # model_trainer/trained_model/model.pkl
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME
        )

        # model_trainer/preprocessing/preprocessing.pkl
        self.preprocessing_object_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_PREPROCESSING_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )

        # model_trainer/preprocessing/label_encoder.pkl
        self.label_encoder_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_PREPROCESSING_DIR,
            training_pipeline.LABEL_ENCODER_OBJECT_FILE_NAME
        )

        # confusion matrix paths
        self.confusion_matrix_dir: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_CONFUSION_MATRIX_DIR,
        )

        self.train_confusion_matrix_path: str = os.path.join(
            self.confusion_matrix_dir,
            training_pipeline.MODEL_TRAINER_TRAIN_CM_FILE_NAME,
        )

        self.valid_confusion_matrix_path: str = os.path.join(
            self.confusion_matrix_dir,
            training_pipeline.MODEL_TRAINER_VALID_CM_FILE_NAME,
        )

        self.test_confusion_matrix_path: str = os.path.join(
            self.confusion_matrix_dir,
            training_pipeline.MODEL_TRAINER_TEST_CM_FILE_NAME,
        )

        # bucket name (for pushing model to cloud storage)
        self.bucket_name: str = training_pipeline.TRAINING_BUCKET_NAME
