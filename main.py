import sys

from visioninfantnet.components.data_ingestion import DataIngestion
from visioninfantnet.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
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

        print(data_ingestion_artifact)

        logging.info("=== VisionInfantNet: Data Ingestion Pipeline Finished ===")

    except Exception as e:
        logging.exception("Error occurred in data ingestion pipeline.")
        raise VisionInfantNetException(e, sys)
