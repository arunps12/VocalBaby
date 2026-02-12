import sys

from vocalbaby.components.data_ingestion import DataIngestion
from vocalbaby.components.data_validation import DataValidation
from vocalbaby.components.data_transformation import DataTransformation
from vocalbaby.components.model_trainer import ModelTrainer

from vocalbaby.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging


if __name__ == "__main__":
    try:
        # ============================================================
        # 1) DATA INGESTION
        # ============================================================
        logging.info("=== VisionInfantNet: Data Ingestion Pipeline Started ===")

        training_pipeline_config = TrainingPipelineConfig()

        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("Initiating data ingestion...")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully.")
        logging.info(f"DataIngestionArtifact: {data_ingestion_artifact}")
        print("\nData Ingestion Artifact:")
        print(data_ingestion_artifact)

        logging.info("=== VisionInfantNet: Data Ingestion Pipeline Finished ===")

        # ============================================================
        # 2) DATA VALIDATION
        # ============================================================
        logging.info("=== VisionInfantNet: Data Validation Started ===")

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact,
        )

        logging.info("Initiating data validation...")
        data_validation_artifact = data_validation.initiate_data_validation()

        logging.info("Data validation completed successfully.")
        logging.info(f"DataValidationArtifact: {data_validation_artifact}")
        print("\nData Validation Artifact:")
        print(data_validation_artifact)

        logging.info("=== VisionInfantNet: Data Validation Finished ===")

        # ============================================================
        # 3) DATA TRANSFORMATION
        # ============================================================
        logging.info("=== VisionInfantNet: Data Transformation Started ===")

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            data_validation_artifact=data_validation_artifact,
        )

        logging.info("Initiating data transformation...")
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        logging.info("Data transformation completed successfully.")
        logging.info(f"DataTransformationArtifact: {data_transformation_artifact}")
        print("\nData Transformation Artifact:")
        print(data_transformation_artifact)

        logging.info("=== VisionInfantNet: Data Transformation Finished ===")

        # ============================================================
        # 4) MODEL TRAINER (XGBoost + eGeMAPS + SMOTE)
        # ============================================================
        logging.info("=== VisionInfantNet: Model Training Started ===")

        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact,
        )

        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logging.info("Model training completed successfully.")
        logging.info(f"ModelTrainerArtifact: {model_trainer_artifact}")
        print("\nModel Trainer Artifact:")
        print(model_trainer_artifact)

        logging.info("=== VisionInfantNet: Pipeline Finished Successfully ===")

    except Exception as e:
        logging.exception("Error occurred in VisionInfantNet pipeline.")
        raise VocalBabyException(e, sys)
