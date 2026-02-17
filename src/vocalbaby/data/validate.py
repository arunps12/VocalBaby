"""
Data validation module - wrapper around existing DataValidation component.

This module provides a functional interface to the existing data validation pipeline.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

from vocalbaby.components.data_validation import DataValidation
from vocalbaby.entity.config_entity import (
    TrainingPipelineConfig,
    DataValidationConfig,
)
from vocalbaby.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def _get_latest_timestamp() -> datetime:
    """
    Resolve artifacts/latest symlink to recover the run timestamp,
    so validation writes into the same artifact directory as ingestion.
    """
    latest = Path("artifacts/latest")
    if latest.is_symlink():
        ts_str = os.readlink(str(latest))  # e.g. "02_17_2026_10_29_46"
        return datetime.strptime(ts_str, "%m_%d_%Y_%H_%M_%S")
    # Fallback: current time
    return datetime.now()


def run_data_validation(ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
    """
    Run data validation stage.
    
    This function:
    1. Validates schema compliance
    2. Checks for data quality issues
    3. Detects data drift
    4. Saves validated data
    
    Args:
        ingestion_artifact: Artifact from data ingestion stage
    
    Returns:
        DataValidationArtifact containing validation results
    
    Raises:
        VocalBabyException: If data validation fails
    """
    try:
        logging.info("=" * 80)
        logging.info("STAGE 02: DATA VALIDATION")
        logging.info("=" * 80)
        
        # Initialize configs - reuse the ingestion run's timestamp
        pipeline_config = TrainingPipelineConfig(timestamp=_get_latest_timestamp())
        validation_config = DataValidationConfig(pipeline_config)
        
        # Run validation
        logging.info("Initializing data validation component...")
        validation = DataValidation(validation_config, ingestion_artifact)
        
        logging.info("Starting data validation...")
        artifact = validation.initiate_data_validation()
        
        logging.info(f"Data validation completed successfully")
        logging.info(f"Validation report: {artifact.report_file_path}")
        logging.info(f"Drift report: {artifact.drift_report_file_path}")
        logging.info(f"Valid status: {artifact.validation_status}")
        logging.info("=" * 80)
        
        if not artifact.validation_status:
            raise VocalBabyException("Data validation failed - check validation report", sys)
        
        return artifact
        
    except Exception as e:
        raise VocalBabyException(e, sys)
