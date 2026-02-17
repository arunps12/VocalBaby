"""
Data validation module - wrapper around existing DataValidation component.

This module provides a functional interface to the existing data validation pipeline.
"""
import sys

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
        
        # Initialize configs
        pipeline_config = TrainingPipelineConfig()
        validation_config = DataValidationConfig(pipeline_config)
        
        # Run validation
        logging.info("Initializing data validation component...")
        validation = DataValidation(ingestion_artifact, validation_config)
        
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
