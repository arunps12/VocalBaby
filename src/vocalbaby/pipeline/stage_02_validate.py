"""
STAGE 02: DATA VALIDATION

Validates schema, checks data quality, detects drift.
Reuses the current run's timestamp from artifacts/latest.
"""
import sys
import os
import argparse

from vocalbaby.data.validate import run_data_validation
from vocalbaby.entity.artifact_entity import DataIngestionArtifact
from vocalbaby.utils.run_manager import RunManager
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException


def _load_ingestion_artifact() -> DataIngestionArtifact:
    """
    Reconstruct DataIngestionArtifact from the current run's data_ingestion dir.
    """
    base = RunManager.data_ingestion_dir()
    meta_dir = base / "ingested_metadata"
    audio_dir = base / "ingested_audio"

    return DataIngestionArtifact(
        train_metadata_path=str(meta_dir / "train.csv"),
        valid_metadata_path=str(meta_dir / "valid.csv"),
        test_metadata_path=str(meta_dir / "test.csv"),
        train_audio_dir=str(audio_dir / "train"),
        valid_audio_dir=str(audio_dir / "valid"),
        test_audio_dir=str(audio_dir / "test"),
    )


def main():
    """Main entry point for data validation stage."""
    parser = argparse.ArgumentParser(description="VocalBaby Pipeline - Stage 02: Data Validation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    try:
        logging.info("Starting Stage 02: Data Validation")
        logging.info(f"Run directory: {RunManager.get_current_run_dir()}")

        ingestion_artifact = _load_ingestion_artifact()
        validation_artifact = run_data_validation(ingestion_artifact)

        logging.info(f"Stage 02 completed: report={validation_artifact.report_file_path}")
        return 0

    except Exception as e:
        logging.error(f"Stage 02 failed: {e}")
        raise VocalBabyException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
