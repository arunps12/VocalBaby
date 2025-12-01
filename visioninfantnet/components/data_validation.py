import os
import sys
import librosa
import numpy as np
import pandas as pd

from visioninfantnet.entity.config_entity import DataValidationConfig
from visioninfantnet.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from visioninfantnet.utils.main_utils.utils import read_yaml_file, write_yaml_file
from visioninfantnet.constant.training_pipeline import (
    CHILD_ID_COLUMN,
    TARGET_COLUMN,
    AUDIO_PATH_COLUMN,
    AGE_COLUMN,
)
from visioninfantnet.exception.exception import VisionInfantNetException
from visioninfantnet.logging.logger import logging


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            self.config = data_validation_config
            self.ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # --------------------------- helpers ---------------------------

    def _load_schema(self) -> dict:
        """
        Load schema.yaml (types, required_columns, allowed_labels, audio_validation).
        """
        try:
            return read_yaml_file(self.config.schema_file_path)
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    def _load_metadata_splits(self):
        """
        Load train / validation / test metadata CSVs from ingestion artifacts.
        """
        try:
            train_df = pd.read_csv(self.ingestion_artifact.train_metadata_path)
            valid_df = pd.read_csv(self.ingestion_artifact.valid_metadata_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_metadata_path)
            return {"train": train_df, "validation": valid_df, "test": test_df}
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # -------------------- schema & label & dtype validation --------------------

    def _validate_schema_and_labels(self, metadata_dict, schema):
        """
        - Check required columns present / extras
        - Check dtypes against schema['columns']
        - Check labels in TARGET_COLUMN are within allowed_labels
        """
        report = {"schema": {}, "labels": {}, "dtypes": {}}
        overall_ok = True

        required_columns = schema.get("required_columns", [])
        allowed_labels = schema.get("allowed_labels", [])
        expected_types = schema.get("columns", {})

        for split, df in metadata_dict.items():
            # ---------- column presence ----------
            missing = [c for c in required_columns if c not in df.columns]
            extra = [c for c in df.columns if c not in required_columns]

            if missing:
                overall_ok = False

            # ---------- dtype checks ----------
            dtype_issues = {}
            for col, expected_type in expected_types.items():
                if col not in df.columns:
                    dtype_issues[col] = {
                        "expected": expected_type,
                        "actual": None,
                        "status": "missing_column",
                    }
                    continue

                actual_dtype = str(df[col].dtype)

                
                status = "ok"
                if expected_type.startswith("int"):
                    if "int" not in actual_dtype:
                        status = "mismatch"
                elif expected_type.startswith("float"):
                    if "float" not in actual_dtype:
                        status = "mismatch"
                elif expected_type == "object":
                    if actual_dtype != "object":
                        status = "mismatch"

                if status != "ok":
                    overall_ok = False

                dtype_issues[col] = {
                    "expected": expected_type,
                    "actual": actual_dtype,
                    "status": status,
                }

            # ---------- label checks ----------
            if TARGET_COLUMN in df.columns:
                unique_labels = set(df[TARGET_COLUMN].unique())
            else:
                unique_labels = set()
                overall_ok = False

            invalid_labels = [l for l in unique_labels if l not in allowed_labels]
            if invalid_labels:
                overall_ok = False

            # ---------- build report ----------
            report["schema"][split] = {
                "missing_columns": missing,
                "extra_columns": extra,
            }
            report["labels"][split] = {
                "unique_labels": sorted(list(unique_labels)),
                "invalid_labels": invalid_labels,
            }
            report["dtypes"][split] = dtype_issues

        return overall_ok, report

    # -------------------- child-disjoint validation --------------------

    def _validate_child_disjoint(self, metadata_dict, report):
        """
        Ensure no child_ID appears in more than one split
        (train / validation / test).
        Uses CHILD_ID_COLUMN from training_pipeline constants.
        """
        try:
            train_ids = set(metadata_dict["train"][CHILD_ID_COLUMN])
            valid_ids = set(metadata_dict["validation"][CHILD_ID_COLUMN])
            test_ids = set(metadata_dict["test"][CHILD_ID_COLUMN])

            overlap_train_valid = train_ids & valid_ids
            overlap_train_test = train_ids & test_ids
            overlap_valid_test = valid_ids & test_ids

            child_report = {
                "train_valid_overlap": list(overlap_train_valid),
                "train_test_overlap": list(overlap_train_test),
                "valid_test_overlap": list(overlap_valid_test),
            }

            report["child_disjoint"] = child_report

            has_overlap = any(
                [overlap_train_valid, overlap_train_test, overlap_valid_test]
            )

            return not has_overlap, report
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # -------------------- audio validation --------------------

    def _validate_audio(self, metadata_dict, schema, report):
        """
        Validate:
        - audio file existence
        - duration in [min_duration_seconds, max_duration_seconds]
        Using AUDIO_PATH_COLUMN from constants and audio_validation in schema.
        """
        audio_report = {}
        overall_ok = True

        audio_cfg = schema.get("audio_validation", {})
        min_dur = float(audio_cfg.get("min_duration_seconds", 0.0))
        max_dur = float(audio_cfg.get("max_duration_seconds", float("inf")))

        for split, df in metadata_dict.items():
            split_info = {
                "total_files": len(df),
                "missing_files": [],
                "duration_out_of_range": 0,
            }

            durations = []

            for _, row in df.iterrows():
                path = row[AUDIO_PATH_COLUMN]

                if not os.path.exists(path):
                    split_info["missing_files"].append(path)
                    overall_ok = False
                    continue

                try:
                    audio, sr = librosa.load(path, sr=None)
                    duration = len(audio) / sr
                    durations.append(duration)
                except Exception:
                    split_info["missing_files"].append(path)
                    overall_ok = False
                    continue

                if not (min_dur <= duration <= max_dur):
                    split_info["duration_out_of_range"] += 1

            if durations:
                split_info["duration_stats"] = {
                    "mean": float(np.mean(durations)),
                    "std": float(np.std(durations)),
                    "min": float(np.min(durations)),
                    "max": float(np.max(durations)),
                }
            else:
                split_info["duration_stats"] = None

            audio_report[split] = split_info

        report["audio"] = audio_report
        return overall_ok, report

    # -------------------- simple drift detection --------------------

    def _compute_drift(self, metadata_dict, audio_report):
        """
        Simple drift analysis:
        - class distribution drift (TARGET_COLUMN)
        - age distribution drift (AGE_COLUMN)
        - duration distribution drift (from audio_report)
        """
        drift = {}

        train_df = metadata_dict["train"]

        # Class distribution
        baseline_class = train_df[TARGET_COLUMN].value_counts(normalize=True)

        class_drift = {}
        for split in ["validation", "test"]:
            df = metadata_dict[split]
            dist = df[TARGET_COLUMN].value_counts(normalize=True)
            combined_index = sorted(set(baseline_class.index) | set(dist.index))

            split_info = {}
            for label in combined_index:
                base_p = float(baseline_class.get(label, 0.0))
                new_p = float(dist.get(label, 0.0))
                diff = new_p - base_p
                split_info[label] = {
                    "baseline": base_p,
                    "current": new_p,
                    "difference": diff,
                }
            class_drift[split] = split_info

        drift["class_distribution"] = class_drift

        # Age distribution (mean age)
        baseline_age_mean = float(train_df[AGE_COLUMN].mean())
        age_drift = {}
        for split in ["validation", "test"]:
            df = metadata_dict[split]
            age_mean = float(df[AGE_COLUMN].mean())
            age_drift[split] = {
                "baseline_mean_age": baseline_age_mean,
                "current_mean_age": age_mean,
                "difference": age_mean - baseline_age_mean,
            }
        drift["age_distribution"] = age_drift

        # Duration drift (mean duration)
        duration_drift = {}
        train_stats = audio_report["train"]["duration_stats"]
        if train_stats is not None:
            base_mean = train_stats["mean"]
            for split in ["validation", "test"]:
                stats = audio_report[split]["duration_stats"]
                if stats is None:
                    duration_drift[split] = None
                    continue
                duration_drift[split] = {
                    "baseline_mean_duration": base_mean,
                    "current_mean_duration": stats["mean"],
                    "difference": stats["mean"] - base_mean,
                }

        drift["duration_distribution"] = duration_drift

        return drift

    # -------------------- save reports & validated CSVs --------------------

    def _write_yaml(self, obj, path):
        write_yaml_file(path, obj, replace=True)

    def _copy_valid_metadata(self, metadata_dict):
        os.makedirs(self.config.validated_metadata_dir, exist_ok=True)

        metadata_dict["train"].to_csv(
            self.config.validated_train_metadata_path, index=False
        )
        metadata_dict["validation"].to_csv(
            self.config.validated_validation_metadata_path, index=False
        )
        metadata_dict["test"].to_csv(
            self.config.validated_test_metadata_path, index=False
        )

    # -------------------- main entrypoint --------------------

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("===== Data Validation Started =====")

            schema = self._load_schema()
            metadata_dict = self._load_metadata_splits()
            report = {}

            # 1) Schema + labels + dtypes
            schema_ok, report_part = self._validate_schema_and_labels(
                metadata_dict, schema
            )
            report.update(report_part)
            logging.info(f"Schema/label/dtype validation OK? {schema_ok}")

            # 2) Child-disjoint
            child_ok, report = self._validate_child_disjoint(metadata_dict, report)
            logging.info(f"Child-disjoint validation OK? {child_ok}")

            # 3) Audio checks (existence + duration)
            audio_ok, report = self._validate_audio(metadata_dict, schema, report)
            logging.info(f"Audio validation OK? {audio_ok}")

            # 4) Drift calculation
            drift = self._compute_drift(metadata_dict, report["audio"])

            # Write reports
            report["overall_status"] = bool(schema_ok and child_ok and audio_ok)
            self._write_yaml(report, self.config.report_file_path)
            self._write_yaml(drift, self.config.drift_report_file_path)

            # Copy validated CSVs (audio dirs remain from ingestion)
            self._copy_valid_metadata(metadata_dict)

            logging.info("===== Data Validation Completed =====")

            return DataValidationArtifact(
            validation_status=report["overall_status"],
            report_file_path=self.config.report_file_path,
            drift_report_file_path=self.config.drift_report_file_path,

            # Valid metadata
            validated_train_metadata_path=self.config.validated_train_metadata_path,
            validated_validation_metadata_path=self.config.validated_validation_metadata_path,
            validated_test_metadata_path=self.config.validated_test_metadata_path,

            # Invalid metadata
            invalid_train_metadata_path=self.config.invalid_train_metadata_path,
            invalid_validation_metadata_path=self.config.invalid_validation_metadata_path,
            invalid_test_metadata_path=self.config.invalid_test_metadata_path,

            # Audio dirs
            validated_train_audio_dir=self.ingestion_artifact.train_audio_dir,
            validated_validation_audio_dir=self.ingestion_artifact.valid_audio_dir,
            validated_test_audio_dir=self.ingestion_artifact.test_audio_dir,
        )


        except Exception as e:
            raise VisionInfantNetException(e, sys)
