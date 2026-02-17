"""
Data ingestion module - wrapper around existing DataIngestion component.

This module provides a functional interface to the existing data ingestion pipeline
while maintaining backwards compatibility.
"""
import sys
from typing import Tuple
import pandas as pd

from vocalbaby.components.data_ingestion import DataIngestion
from vocalbaby.entity.config_entity import DataIngestionConfig
from vocalbaby.entity.artifact_entity import DataIngestionArtifact
from vocalbaby.utils.run_manager import RunManager
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def run_data_ingestion() -> DataIngestionArtifact:
    """
    Run data ingestion stage.

    Creates a NEW run (new timestamp + artifacts/latest symlink), then
    ingests raw audio + metadata into artifacts/<timestamp>/data_ingestion/.

    Returns:
        DataIngestionArtifact containing paths to ingested data
    """
    try:
        logging.info("=" * 80)
        logging.info("STAGE 01: DATA INGESTION")
        logging.info("=" * 80)

        # Create a fresh run  (new timestamp, updates artifacts/latest)
        pipeline_config = RunManager.get_pipeline_config(new_run=True)
        ingestion_config = DataIngestionConfig(pipeline_config)

        logging.info(f"Run directory: {pipeline_config.artifact_dir}")

        # Run ingestion
        ingestion = DataIngestion(ingestion_config)
        artifact = ingestion.initiate_data_ingestion()

        logging.info("Data ingestion completed successfully")
        logging.info(f"  Train metadata : {artifact.train_metadata_path}")
        logging.info(f"  Valid metadata  : {artifact.valid_metadata_path}")
        logging.info(f"  Test metadata   : {artifact.test_metadata_path}")
        logging.info("=" * 80)

        return artifact

    except Exception as e:
        raise VocalBabyException(e, sys)


def load_ingested_data(artifact: DataIngestionArtifact) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load ingested metadata from artifact.

    Args:
        artifact: DataIngestionArtifact from run_data_ingestion()

    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    try:
        train_df = pd.read_csv(artifact.train_metadata_path)
        valid_df = pd.read_csv(artifact.valid_metadata_path)
        test_df = pd.read_csv(artifact.test_metadata_path)

        logging.info(f"Loaded metadata: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

        return train_df, valid_df, test_df

    except Exception as e:
        raise VocalBabyException(e, sys)
