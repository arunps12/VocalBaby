"""
Data validation module - wrapper around existing DataValidation component.

This module provides a functional interface to the existing data validation pipeline.
"""
import sys

from vocalbaby.components.data_validation import DataValidation
from vocalbaby.entity.config_entity import DataValidationConfig
from vocalbaby.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from vocalbaby.utils.run_manager import RunManager
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def run_data_validation(ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
    """
    Run data validation stage.

    Reuses the current run's timestamp (from artifacts/latest) so all
    validation outputs land in the same artifacts/<timestamp>/ directory.

    Args:
        ingestion_artifact: Artifact from data ingestion stage

    Returns:
        DataValidationArtifact containing validation results
    """
    try:
        logging.info("=" * 80)
        logging.info("STAGE 02: DATA VALIDATION")
        logging.info("=" * 80)

        # Reuse the current run (same timestamp as ingestion)
        pipeline_config = RunManager.get_pipeline_config(new_run=False)
        validation_config = DataValidationConfig(pipeline_config)

        logging.info(f"Run directory: {pipeline_config.artifact_dir}")

        # Run validation
        validation = DataValidation(validation_config, ingestion_artifact)
        artifact = validation.initiate_data_validation()

        logging.info("Data validation completed successfully")
        logging.info(f"  Report : {artifact.report_file_path}")
        logging.info(f"  Drift  : {artifact.drift_report_file_path}")
        logging.info(f"  Status : {artifact.validation_status}")
        logging.info("=" * 80)

        if not artifact.validation_status:
            raise VocalBabyException("Data validation failed - check validation report", sys)

        return artifact

    except Exception as e:
        raise VocalBabyException(e, sys)
