"""
Centralized run management for the VocalBaby pipeline.

All pipeline stages share a single timestamped artifact directory per run.
The `artifacts/latest` symlink always points to the current run's directory.

Usage:
    # Stage 01 (ingest): creates a NEW run
    run_dir = RunManager.create_new_run()

    # Stages 02-07: reuse the CURRENT run
    run_dir = RunManager.get_current_run_dir()

    # Any stage: get a TrainingPipelineConfig for the current run
    pipeline_config = RunManager.get_pipeline_config()
"""
import os
import sys
from datetime import datetime
from pathlib import Path

from vocalbaby.constant.training_pipeline import ARTIFACT_DIR
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException

TIMESTAMP_FMT = "%m_%d_%Y_%H_%M_%S"


class RunManager:
    """Manages timestamped artifact directories and the `latest` symlink."""

    _artifacts_root = Path(ARTIFACT_DIR)  # artifacts/
    _latest_link = _artifacts_root / "latest"  # artifacts/latest

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def create_new_run(cls) -> Path:
        """
        Start a brand-new pipeline run.

        Creates  artifacts/<MM_DD_YYYY_HH_MM_SS>/
        Updates  artifacts/latest  ->  <MM_DD_YYYY_HH_MM_SS>

        Returns the absolute Path to the new run directory.
        """
        ts = datetime.now().strftime(TIMESTAMP_FMT)
        run_dir = cls._artifacts_root / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        cls._update_symlink(ts)
        logging.info(f"[RunManager] New run: {run_dir}")
        return run_dir

    @classmethod
    def get_current_run_dir(cls) -> Path:
        """
        Return the directory of the current (latest) run.

        Reads  artifacts/latest  symlink to find the timestamped directory.
        Raises if no run exists yet.
        """
        if not cls._latest_link.exists():
            raise FileNotFoundError(
                "No pipeline run found.  Run stage 01 (ingest) first to "
                "create a new run under artifacts/."
            )
        # Resolve symlink to get the real path
        return cls._latest_link.resolve()

    @classmethod
    def get_current_timestamp(cls) -> str:
        """Return the timestamp string (e.g. '02_17_2026_10_29_46') of the current run."""
        if cls._latest_link.is_symlink():
            return os.readlink(str(cls._latest_link))
        # Fallback: resolve and take dir name
        return cls.get_current_run_dir().name

    @classmethod
    def get_pipeline_config(cls, *, new_run: bool = False):
        """
        Return a TrainingPipelineConfig tied to the current run.

        Args:
            new_run: If True, create a fresh run first (stage 01 only).
                     If False, reuse the existing latest run.
        """
        # Import here to avoid circular imports
        from vocalbaby.entity.config_entity import TrainingPipelineConfig

        if new_run:
            cls.create_new_run()

        ts_str = cls.get_current_timestamp()
        ts_dt = datetime.strptime(ts_str, TIMESTAMP_FMT)
        return TrainingPipelineConfig(timestamp=ts_dt)

    # ------------------------------------------------------------------
    # Path helpers  (all resolve through latest)
    # ------------------------------------------------------------------

    @classmethod
    def data_ingestion_dir(cls) -> Path:
        return cls.get_current_run_dir() / "data_ingestion"

    @classmethod
    def data_validation_dir(cls) -> Path:
        return cls.get_current_run_dir() / "data_validation"

    @classmethod
    def features_dir(cls) -> Path:
        return cls.get_current_run_dir() / "features"

    @classmethod
    def tuning_dir(cls) -> Path:
        return cls.get_current_run_dir() / "tuning"

    @classmethod
    def models_dir(cls) -> Path:
        return cls.get_current_run_dir() / "models"

    @classmethod
    def eval_dir(cls) -> Path:
        return cls.get_current_run_dir() / "eval"

    @classmethod
    def results_dir(cls) -> Path:
        return cls.get_current_run_dir() / "results"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @classmethod
    def _update_symlink(cls, target_name: str) -> None:
        """Create / update  artifacts/latest -> <target_name>."""
        if cls._latest_link.is_symlink() or cls._latest_link.exists():
            cls._latest_link.unlink()
        cls._latest_link.symlink_to(target_name)
        logging.info(f"[RunManager] Symlink: {cls._latest_link} -> {target_name}")
