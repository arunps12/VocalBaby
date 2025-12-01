from datetime import datetime
import os
from visioninfantnet.constant import training_pipeline


class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_name: str = training_pipeline.ARTIFACT_DIR
        self.artifact_dir: str = os.path.join(self.artifact_name, timestamp)
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
