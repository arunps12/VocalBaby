import sys

from visioninfantnet.exception.exception import VisionInfantNetException
from visioninfantnet.logging.logger import logging

from visioninfantnet.components.data_ingestion import DataIngestion
from visioninfantnet.components.data_validation import DataValidation
from visioninfantnet.components.data_transformation import DataTransformation
from visioninfantnet.components.model_trainer import ModelTrainer

from visioninfantnet.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from visioninfantnet.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

import os
import shutil

from visioninfantnet.utils.main_utils.utils import write_yaml_file
from visioninfantnet.constant.training_pipeline import TRAINING_BUCKET_NAME
from visioninfantnet.cloud.s3_syncer import S3Sync
class TrainingPipeline:
    """
    Orchestrates the full VisionInfantNet pipeline:

        1. Data Ingestion
        2. Data Validation
        3. Data Transformation
        4. Model Training (XGBoost on eGeMAPS + SMOTE)

    Usage:
        tp = TrainingPipeline()
        model_trainer_artifact = tp.run_pipeline()
    """

    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
            self.s3_sync = S3Sync()
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # 1) DATA INGESTION
    # ------------------------------------------------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("=== [TrainingPipeline] Data Ingestion: Started ===")

            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_ingestion = DataIngestion(
                data_ingestion_config=data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(
                f"=== [TrainingPipeline] Data Ingestion: Completed. "
                f"Artifact: {data_ingestion_artifact} ==="
            )
            return data_ingestion_artifact

        except Exception as e:
            logging.exception("Error in start_data_ingestion")
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # 2) DATA VALIDATION
    # ------------------------------------------------------------------
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> DataValidationArtifact:
        try:
            logging.info("=== [TrainingPipeline] Data Validation: Started ===")

            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_validation_config=data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info(
                f"=== [TrainingPipeline] Data Validation: Completed. "
                f"Artifact: {data_validation_artifact} ==="
            )
            return data_validation_artifact

        except Exception as e:
            logging.exception("Error in start_data_validation")
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # 3) DATA TRANSFORMATION
    # ------------------------------------------------------------------
    def start_data_transformation(
        self,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            logging.info("=== [TrainingPipeline] Data Transformation: Started ===")

            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_transformation_config=data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logging.info(
                f"=== [TrainingPipeline] Data Transformation: Completed. "
                f"Artifact: {data_transformation_artifact} ==="
            )
            return data_transformation_artifact

        except Exception as e:
            logging.exception("Error in start_data_transformation")
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # 4) MODEL TRAINER
    # ------------------------------------------------------------------
    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        try:
            logging.info("=== [TrainingPipeline] Model Training: Started ===")

            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info(
                f"=== [TrainingPipeline] Model Training: Completed. "
                f"Artifact: {model_trainer_artifact} ==="
            )
            return model_trainer_artifact

        except Exception as e:
            logging.exception("Error in start_model_trainer")
            raise VisionInfantNetException(e, sys)
        

    ## local artifact is going to s3 bucket    
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise VisionInfantNetException(e,sys)
        
    ## local final model is going to s3 bucket 
        
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.model_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise VisionInfantNetException(e,sys)

    # ------------------------------------------------------------------
    # RUN FULL PIPELINE
    # ------------------------------------------------------------------
    def run_pipeline(self) -> ModelTrainerArtifact:
        """
        Run the full VisionInfantNet pipeline:
        ingestion → validation → transformation → model training.
        """
        try:
            logging.info("=== VisionInfantNet: Full Training Pipeline Started ===")

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            # Copy final model to final_model_dir
            final_model_dir = self.training_pipeline_config.model_dir
            os.makedirs(final_model_dir, exist_ok=True)
            # -----------------------------------------------------------
            # 5) COPY model, preprocessor, label_encoder to final_model/
            # -----------------------------------------------------------
            
            src_model_path = model_trainer_artifact.trained_model_file_path
            src_preproc_path = model_trainer_artifact.preprocessing_object_file_path
            src_label_encoder_path = model_trainer_artifact.label_encoder_file_path

            dst_model_path = os.path.join(final_model_dir, os.path.basename(src_model_path))
            dst_preproc_path = os.path.join(final_model_dir, os.path.basename(src_preproc_path))
            dst_label_encoder_path = os.path.join(
                final_model_dir, os.path.basename(src_label_encoder_path)
            )

            # Copy files
            shutil.copy2(src_model_path, dst_model_path)
            shutil.copy2(src_preproc_path, dst_preproc_path)
            shutil.copy2(src_label_encoder_path, dst_label_encoder_path)

            logging.info(
                f"Copied final model artifacts to `{final_model_dir}`:\n"
                f"  - model: {dst_model_path}\n"
                f"  - preprocessing: {dst_preproc_path}\n"
                f"  - label_encoder: {dst_label_encoder_path}"
            )

            # ============================================================
            # 6) WRITE model_info.yaml in final_model/
            # ============================================================
            model_info = {
                "pipeline": {
                    "name": self.training_pipeline_config.pipeline_name,
                    "timestamp": self.training_pipeline_config.timestamp,
                    "artifact_dir": self.training_pipeline_config.artifact_dir,
                },
                "artifacts": {
                    "trained_model_file_path": src_model_path,
                    "preprocessing_object_file_path": src_preproc_path,
                    "label_encoder_file_path": src_label_encoder_path,
                },
                "final_model": {
                    "model_dir": final_model_dir,
                    "model_file": os.path.basename(dst_model_path),
                    "preprocessing_file": os.path.basename(dst_preproc_path),
                    "label_encoder_file": os.path.basename(dst_label_encoder_path),
                },
                "metrics": {
                    "train": {
                        "f1": model_trainer_artifact.train_metric_artifact.f1_score,
                        "precision": model_trainer_artifact.train_metric_artifact.precision_score,
                        "recall": model_trainer_artifact.train_metric_artifact.recall_score,
                        "uar": model_trainer_artifact.train_metric_artifact.uar,
                    },
                    "valid": {
                        "f1": model_trainer_artifact.valid_metric_artifact.f1_score,
                        "precision": model_trainer_artifact.valid_metric_artifact.precision_score,
                        "recall": model_trainer_artifact.valid_metric_artifact.recall_score,
                        "uar": model_trainer_artifact.valid_metric_artifact.uar,
                    },
                    "test": {
                        "f1": model_trainer_artifact.test_metric_artifact.f1_score,
                        "precision": model_trainer_artifact.test_metric_artifact.precision_score,
                        "recall": model_trainer_artifact.test_metric_artifact.recall_score,
                        "uar": model_trainer_artifact.test_metric_artifact.uar,
                    },
                },
            }
            
            model_info_path = os.path.join(final_model_dir, "model_info.yaml")
            write_yaml_file(file_path=model_info_path, content=model_info, replace=True)

            logging.info(f"Written model_info.yaml at: {model_info_path}")

            logging.info("Syncing artifact_dir to S3...")
            self.sync_artifact_dir_to_s3()

            logging.info("Syncing final_model dir to S3...")
            self.sync_saved_model_dir_to_s3()
            
            logging.info(
                "=== VisionInfantNet: Full Training Pipeline Finished Successfully ==="
            )
            return model_trainer_artifact

        except Exception as e:
            logging.exception("Error in run_pipeline")
            raise VisionInfantNetException(e, sys)

if __name__ == "__main__":
    try:
        logging.info("=== VisionInfantNet: Full Training Pipeline Started (training_pipeline.py) ===")

        pipeline = TrainingPipeline()
        model_trainer_artifact = pipeline.run_pipeline()

        logging.info(f"ModelTrainerArtifact from pipeline: {model_trainer_artifact}")
        print("\nModel Trainer Artifact:")
        print(model_trainer_artifact)

        logging.info("=== VisionInfantNet: Full Training Pipeline Finished Successfully ===")

    except Exception as e:
        logging.exception("Error occurred in VisionInfantNet training pipeline.")
        raise VisionInfantNetException(e, sys)
