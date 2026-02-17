"""
STAGE 01: DATA INGESTION

Loads raw audio files and metadata, creates child-disjoint train/valid/test splits.
"""
import sys
import argparse

from vocalbaby.data.ingest import run_data_ingestion
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def main():
    """Main entry point for data ingestion stage."""
    parser = argparse.ArgumentParser(description="VocalBaby Pipeline - Stage 01: Data Ingestion")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    try:
        logging.info("Starting Stage  1: Data Ingestion")
        artifact = run_data_ingestion()
        logging.info(f"Stage 01 completed: {artifact.artifact_dir}")
        return 0
        
    except Exception as e:
        logging.error(f"Stage 01 failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
