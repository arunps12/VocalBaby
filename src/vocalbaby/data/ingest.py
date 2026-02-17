"""
Data ingestion module - wrapper around existing DataIngestion component.

This module provides a functional interface to the existing data ingestion pipeline
while maintaining backwards compatibility.
"""
import sys
from typing import Tuple
import pandas as pd

from vocalbaby.components.data_ingestion import DataIngestion
from vocalbaby.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
)
from vocalbaby.entity.artifact_entity import DataIngestionArtifact
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def run_data_ingestion() -> DataIngestionArtifact:
    """
    Run data ingestion stage.
    
    This function:
    1. Loads raw audio files and metadata
    2. Creates child-disjoint train/valid/test splits
    3. Saves split metadata and organizes audio files
    
    Returns:
        DataIngestionArtifact containing paths to ingested data
    
    Raises:
        VocalBabyException: If data ingestion fails
    """
    try:
        logging.info("=" * 80)
        logging.info("STAGE 01: DATA INGESTION")
        logging.info("=" * 80)
        
        # Initialize configs
        pipeline_config = TrainingPipelineConfig()
        ingestion_config = DataIngestionConfig(pipeline_config)
        
        # Run ingestion
        logging.info("Initializing data ingestion component...")
        ingestion = DataIngestion(ingestion_config)
        
        logging.info("Starting data ingestion...")
        artifact = ingestion.initiate_data_ingestion()
        
        logging.info(f"Data ingestion completed successfully")
        logging.info(f"Train metadata: {artifact.train_metadata_path}")
        logging.info(f"Valid metadata: {artifact.valid_metadata_path}")
        logging.info(f"Test metadata: {artifact.test_metadata_path}")
        logging.info(f"Train audio dir: {artifact.train_audio_dir}")
        logging.info(f"Valid audio dir: {artifact.valid_audio_dir}")
        logging.info(f"Test audio dir: {artifact.test_audio_dir}")
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
