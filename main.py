import sys

from visioninfantnet.components.data_ingestion import DataIngestion
from visioninfantnet.components.data_validation import DataValidation
from visioninfantnet.components.data_transformation import DataTransformation
from visioninfantnet.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
)
from visioninfantnet.exception.exception import VisionInfantNetException
from visioninfantnet.logging.logger import logging


if __name__ == "__main__":
    try:
        logging.info("=== VisionInfantNet: Data Ingestion Pipeline Started ===")

        # Create training pipeline config
        training_pipeline_config = TrainingPipelineConfig()

        #  Create data ingestion config
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)

        # Create data ingestion component
        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("Initiating data ingestion...")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully.")
        logging.info(f"DataIngestionArtifact: {data_ingestion_artifact}")
        print("\nData Ingestion Artifact:")
        print(data_ingestion_artifact)

        logging.info("=== VisionInfantNet: Data Ingestion Pipeline Finished ===")

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


        logging.info("=== VisionInfantNet: Data Transforamtion Started ===")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            data_validation_artifact=data_validation_artifact,
        )
        logging.info("Initiating data transforamtion...")
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        logging.info("Data transformation completed successfully.")
        logging.info(f"DataTransformationArtifact: {data_transformation_artifact}")
        print("\nData Transformation Artifact:")
        print(data_transformation_artifact)
        logging.info("=== VisionInfantNet: Data Transforamtion Finished ===")

        #logging.info("=== VisionInfantNet: Pipeline Finished Successfully ===")
       

    except Exception as e:
        logging.exception("Error occurred in VisionInfantNet pipeline.")
        raise VisionInfantNetException(e, sys)


        