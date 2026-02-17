"""
STAGE 02: DATA VALIDATION

Validates schema, checks data quality, detects drift.
"""
import sys
import argparse
import os

from vocalbaby.data.ingest import run_data_ingestion
from vocalbaby.data.validate import run_data_validation
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def main():
    """Main entry point for data validation stage."""
    parser = argparse.ArgumentParser(description="VocalBaby Pipeline - Stage 02: Data Validation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    try:
        logging.info("Starting Stage 02: Data Validation")
        
        # Check if ingestion artifacts exist, otherwise run ingestion first
        latest_link = "artifacts/latest"
        if not os.path.exists(latest_link):
            logging.info("No ingestion artifacts found - running ingestion first...")
            ingestion_artifact = run_data_ingestion()
        else:
            # Load existing ingestion artifact
            from vocalbaby.entity.artifact_entity import DataIngestionArtifact
            ingestion_artifact = run_data_ingestion()  # Re-run to get artifact object
        
        # Run validation
        validation_artifact = run_data_validation(ingestion_artifact)
        logging.info(f"Stage 02 completed: {validation_artifact.artifact_dir}")
        return 0
        
    except Exception as e:
        logging.error(f"Stage 02 failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
