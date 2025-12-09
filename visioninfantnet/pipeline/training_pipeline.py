import sys

from visioninfantnet.exception.exception import VisionInfantNetException
from visioninfantnet.logging.logger import logging

from visioninfantnet.components.data_ingestion import DataIngestion
from visioninfantnet.components.data_validation import DataValidation
from visioninfantnet.components.data_transformation import DataTransformation
from visioninfantnet.components.model_trainer import ModelTrainer

from visioninfantnet.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from visioninfantnet.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)


class TrainingPipeline:
    """
    Orchestrates the full VisionInfantNet pipeline:

        1. Data Ingestion
        2. Data Validation
        3. Data Transformation
        4. Model Training (XGBoost on eGeMAPS + SMOTE)

    Usage:
        tp = TrainingPipeline()
        model_trainer_artifact = tp.run_pipeline()
    """

    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # 1) DATA INGESTION
    # ------------------------------------------------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("=== [TrainingPipeline] Data Ingestion: Started ===")

            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_ingestion = DataIngestion(
                data_ingestion_config=data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(
                f"=== [TrainingPipeline] Data Ingestion: Completed. "
                f"Artifact: {data_ingestion_artifact} ==="
            )
            return data_ingestion_artifact

        except Exception as e:
            logging.exception("Error in start_data_ingestion")
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # 2) DATA VALIDATION
    # ------------------------------------------------------------------
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> DataValidationArtifact:
        try:
            logging.info("=== [TrainingPipeline] Data Validation: Started ===")

            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_validation_config=data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info(
                f"=== [TrainingPipeline] Data Validation: Completed. "
                f"Artifact: {data_validation_artifact} ==="
            )
            return data_validation_artifact

        except Exception as e:
            logging.exception("Error in start_data_validation")
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # 3) DATA TRANSFORMATION
    # ------------------------------------------------------------------
    def start_data_transformation(
        self,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            logging.info("=== [TrainingPipeline] Data Transformation: Started ===")

            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_transformation_config=data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logging.info(
                f"=== [TrainingPipeline] Data Transformation: Completed. "
                f"Artifact: {data_transformation_artifact} ==="
            )
            return data_transformation_artifact

        except Exception as e:
            logging.exception("Error in start_data_transformation")
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # 4) MODEL TRAINER
    # ------------------------------------------------------------------
    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        try:
            logging.info("=== [TrainingPipeline] Model Training: Started ===")

            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info(
                f"=== [TrainingPipeline] Model Training: Completed. "
                f"Artifact: {model_trainer_artifact} ==="
            )
            return model_trainer_artifact

        except Exception as e:
            logging.exception("Error in start_model_trainer")
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # RUN FULL PIPELINE
    # ------------------------------------------------------------------
    def run_pipeline(self) -> ModelTrainerArtifact:
        """
        Run the full VisionInfantNet pipeline:
        ingestion → validation → transformation → model training.
        """
        try:
            logging.info("=== VisionInfantNet: Full Training Pipeline Started ===")

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            logging.info(
                "=== VisionInfantNet: Full Training Pipeline Finished Successfully ==="
            )
            return model_trainer_artifact

        except Exception as e:
            logging.exception("Error in run_pipeline")
            raise VisionInfantNetException(e, sys)
