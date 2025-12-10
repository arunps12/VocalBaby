import os
import sys
import numpy as np

from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import dagshub
from urllib.parse import urlparse

from visioninfantnet.exception.exception import VisionInfantNetException
from visioninfantnet.logging.logger import logging

from visioninfantnet.entity.config_entity import ModelTrainerConfig
from visioninfantnet.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from visioninfantnet.utils.main_utils.utils import (
    load_numpy_array_data,
    save_object,
)
from visioninfantnet.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)
from visioninfantnet.utils.ml_utils.preprocessing.imputation import (
    fit_imputer_and_transform,
)
from visioninfantnet.utils.ml_utils.imbalance.resampling import (
    resample_data,
)
from visioninfantnet.utils.ml_utils.plotting.confusion_matrix_utils import (
    plot_and_save_confusion_matrix,
)

from visioninfantnet.utils.ml_utils.model_selection.evaluate import (
    evaluate_splits,
)

from visioninfantnet.utils.ml_utils.preprocessing.label_encoding import (
    encode_labels,
)

# Fixed best hyperparameters from previous Optuna tuning
# Experiment: 06_xgboost_egemaps_smote_optuna_experiment.ipynb
BEST_XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.01310361913002795,
    "n_estimators": 737,
    "subsample": 0.6185127156068264,
    "colsample_bytree": 0.6972153039737041,
    "gamma": 1.6324639148961462,
    "min_child_weight": 1,
    "reg_lambda": 0.057931609965269915,
    "reg_alpha": 3.631533545994319,
}

# Initialize DagsHub + MLflow tracking
dagshub.init(repo_owner="arunps12", repo_name="VisionInfantNet", mlflow=True)
print("MLflow + DagsHub initialized.")

class ModelTrainer:
    """
    Model Trainer for XGBoost on eGeMAPS features with SMOTE.
    Fixed best hyperparameters from previous Optuna tuning
    Experiment: 06_xgboost_egemaps_smote_optuna_experiment.ipynb
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            # Fixed best params from Optuna (Trial 24)
            self.best_params = {
                "max_depth": 4,
                "learning_rate": 0.01310361913002795,
                "n_estimators": 737,
                "subsample": 0.6185127156068264,
                "colsample_bytree": 0.6972153039737041,
                "gamma": 1.6324639148961462,
                "min_child_weight": 1,
                "reg_lambda": 0.057931609965269915,
                "reg_alpha": 3.631533545994319,
            }

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ------------------------------------------------------------------
    # Helper: load eGeMAPS features + labels from artifact paths
    # ------------------------------------------------------------------
    def _load_data(self):
        """
        Load eGeMAPS features and labels using DataTransformationArtifact.

        path of 'compare' feature files as eGeMAPS:
          - train_compare_feature_file_path
          - valid_compare_feature_file_path
          - test_compare_feature_file_path

        And the label files:
          - train_label_file_path
          - valid_label_file_path
          - test_label_file_path
        """
        try:
            logging.info("Loading eGeMAPS (compare) features and labels from artifacts...")

            # ============= eGeMAPS / ComParE FEATURES =================
            X_train = load_numpy_array_data(
                self.data_transformation_artifact.train_compare_feature_file_path
            )
            X_valid = load_numpy_array_data(
                self.data_transformation_artifact.valid_compare_feature_file_path
            )
            X_test = load_numpy_array_data(
                self.data_transformation_artifact.test_compare_feature_file_path
            )

            # ======================== LABELS ==========================
            y_train = load_numpy_array_data(
                self.data_transformation_artifact.train_label_file_path
            )
            y_valid = load_numpy_array_data(
                self.data_transformation_artifact.valid_label_file_path
            )
            y_test = load_numpy_array_data(
                self.data_transformation_artifact.test_label_file_path
            )

            logging.info(
                f"Shapes - X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}"
            )
            logging.info(
                f"Shapes - y_train: {y_train.shape}, y_valid: {y_valid.shape}, y_test: {y_test.shape}"
            )

            return X_train, y_train, X_valid, y_valid, X_test, y_test

        except Exception as e:
            raise VisionInfantNetException(e, sys)
    # ------------------------------------------------------------------
    # Helper: build XGBoost model
    # ------------------------------------------------------------------
    def _build_model(self) -> XGBClassifier:
        """
        Build XGBClassifier with fixed best parameters from Optuna.
        """
        params = self.best_params

        model = XGBClassifier(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            gamma=params["gamma"],
            min_child_weight=params["min_child_weight"],
            reg_lambda=params["reg_lambda"],
            reg_alpha=params["reg_alpha"],
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        return model

        # ------------------------------------------------------------------
    # MLflow / DagsHub tracking
    # ------------------------------------------------------------------
    def _log_to_mlflow(
        self,
        model,
        train_metrics: ClassificationMetricArtifact,
        valid_metrics: ClassificationMetricArtifact,
        test_metrics: ClassificationMetricArtifact,
    ):
        """
        Log params, metrics, confusion matrices, and model to MLflow (DagsHub).
        """
        # set registry URI 
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        run_name = "xgb_egemaps_smote_optuna_best"

        with mlflow.start_run(run_name=run_name):

            # ---- Log hyperparameters  ----
            for k, v in self.best_params.items():
                mlflow.log_param(k, v)

            # ---- Log metrics (valid + test like) ----
         
            # Valid
            mlflow.log_metric("valid_uar", valid_metrics.uar)
            mlflow.log_metric("valid_f1", valid_metrics.f1_score)
            mlflow.log_metric("valid_precision", valid_metrics.precision_score)
            mlflow.log_metric("valid_recall", valid_metrics.recall_score)

            # Test
            mlflow.log_metric("test_uar", test_metrics.uar)
            mlflow.log_metric("test_f1", test_metrics.f1_score)
            mlflow.log_metric("test_precision", test_metrics.precision_score)
            mlflow.log_metric("test_recall", test_metrics.recall_score)

            # ---- Log confusion matrix images as artifacts ----
            mlflow.log_artifact(self.model_trainer_config.train_confusion_matrix_path)
            mlflow.log_artifact(self.model_trainer_config.valid_confusion_matrix_path)
            mlflow.log_artifact(self.model_trainer_config.test_confusion_matrix_path)
            # ---- Log preprocessing + label encoder objects ----
            mlflow.log_artifact(self.model_trainer_config.preprocessing_object_file_path)
            mlflow.log_artifact(self.model_trainer_config.label_encoder_file_path)
            # ---- Log model ----
            mlflow.sklearn.log_model(model, "model")

            
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                     model,
                     "model",
                     registered_model_name="xgb_egemaps_smote_optuna_best",
                 )

        print("MLflow logging completed.")

    # ------------------------------------------------------------------
    #  training + evaluation
    # ------------------------------------------------------------------
    def train_and_evaluate(self) -> ModelTrainerArtifact:
        try:
            #  Load raw features + raw labels (strings)
            X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, X_test_raw, y_test_raw = (
                self._load_data()
            )

            #  Label encoding (strings → integers)
            logging.info("Encoding string labels with LabelEncoder...")
            (
                y_train_enc,
                y_valid_enc,
                y_test_enc,
                class_names,
                label_encoder,
            ) = encode_labels(y_train_raw, y_valid_raw, y_test_raw)

            #  Imputation on features (fit on train, apply to valid + test)
            logging.info("Fitting SimpleImputer on training data and transforming...")
            X_train_imp, X_valid_imp, imputer = fit_imputer_and_transform(
                X_train_raw, X_valid_raw, strategy="median"
            )
            X_test_imp = imputer.transform(X_test_raw)

            #  SMOTE on training data only (to handle class imbalance)
            logging.info("Applying SMOTE on training data...")
            X_train_res, y_train_res, _ = resample_data(
                X_train_imp, y_train_enc, method="smote"
            )

            logging.info(
                f"Resampled train shape: X_train_res={X_train_res.shape}, y_train_res={y_train_res.shape}"
            )

            #  Build model (XGB with fixed best params)
            model = self._build_model()

            #  Train model
            logging.info("Training XGBClassifier on resampled training data...")
            model.fit(
                X_train_res,
                y_train_res,
                eval_set=[(X_valid_imp, y_valid_enc)],
                verbose=False,
            )

            #  Predictions on train/valid/test
            logging.info("Generating predictions for train/valid/test...")
            y_train_pred = model.predict(X_train_imp)
            y_valid_pred = model.predict(X_valid_imp)
            y_test_pred = model.predict(X_test_imp)

            #  Metrics via evaluate_splits (encoded labels)
            logging.info("Computing classification metrics (F1, Precision, Recall, UAR)...")
            train_metrics, valid_metrics, test_metrics = evaluate_splits(
                y_train=y_train_enc,
                y_train_pred=y_train_pred,
                y_valid=y_valid_enc,
                y_valid_pred=y_valid_pred,
                y_test=y_test_enc,
                y_test_pred=y_test_pred,
            )

            logging.info(
                f"Train metrics: F1={train_metrics.f1_score:.4f}, "
                f"Precision={train_metrics.precision_score:.4f}, "
                f"Recall={train_metrics.recall_score:.4f}, "
                f"UAR={train_metrics.uar:.4f}"
            )
            logging.info(
                f"Valid metrics: F1={valid_metrics.f1_score:.4f}, "
                f"Precision={valid_metrics.precision_score:.4f}, "
                f"Recall={valid_metrics.recall_score:.4f}, "
                f"UAR={valid_metrics.uar:.4f}"
            )
            logging.info(
                f"Test  metrics: F1={test_metrics.f1_score:.4f}, "
                f"Precision={test_metrics.precision_score:.4f}, "
                f"Recall={test_metrics.recall_score:.4f}, "
                f"UAR={test_metrics.uar:.4f}"
            )

            #  Save preprocessing object (imputer)
            logging.info(
                f"Saving preprocessing object (imputer) to {self.model_trainer_config.preprocessing_object_file_path}"
            )
            save_object(
                file_path=self.model_trainer_config.preprocessing_object_file_path,
                obj=imputer,
            )

            # save the label_encoder
            logging.info(f"Saving LabelEncoder to {self.model_trainer_config.label_encoder_file_path}")
            save_object(
                    file_path=self.model_trainer_config.label_encoder_file_path,
                    obj=label_encoder
            )

            #  Save trained model
            logging.info(
                f"Saving trained XGB model to {self.model_trainer_config.trained_model_file_path}"
            )
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True,
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
            )

            #  Confusion matrices
            logging.info("Saving confusion matrices for train/valid/test...")

            # Here we use *string* class_names for axes:
            # ['Canonical', 'Crying', 'Junk', 'Laughing', 'Non-canonical']
            os.makedirs(self.model_trainer_config.confusion_matrix_dir, exist_ok=True)

            # Train CM
            plot_and_save_confusion_matrix(
                y_true=y_train_enc,
                y_pred=y_train_pred,
                class_names=class_names,
                file_path=self.model_trainer_config.train_confusion_matrix_path,
                title="Train Confusion Matrix — XGB eGeMAPS SMOTE",
            )

            # Valid CM
            plot_and_save_confusion_matrix(
                y_true=y_valid_enc,
                y_pred=y_valid_pred,
                class_names=class_names,
                file_path=self.model_trainer_config.valid_confusion_matrix_path,
                title="Valid Confusion Matrix — XGB eGeMAPS SMOTE",
            )

            # Test CM
            plot_and_save_confusion_matrix(
                y_true=y_test_enc,
                y_pred=y_test_pred,
                class_names=class_names,
                file_path=self.model_trainer_config.test_confusion_matrix_path,
                title="Test Confusion Matrix — XGB eGeMAPS SMOTE",
            )

            #  Build and return ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                preprocessing_object_file_path=self.model_trainer_config.preprocessing_object_file_path,
                label_encoder_file_path=self.model_trainer_config.label_encoder_file_path,
                train_metric_artifact=train_metrics,
                valid_metric_artifact=valid_metrics,
                test_metric_artifact=test_metrics,
                train_confusion_matrix_path=self.model_trainer_config.train_confusion_matrix_path,
                valid_confusion_matrix_path=self.model_trainer_config.valid_confusion_matrix_path,
                test_confusion_matrix_path=self.model_trainer_config.test_confusion_matrix_path,
            )

            #  Log to MLflow / DagsHub
            self._log_to_mlflow(
                model=model,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                test_metrics=test_metrics,
            )

            logging.info(f"ModelTrainerArtifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise VisionInfantNetException(e, sys)


    # ------------------------------------------------------------------
    # entrypoint for pipeline
    # ------------------------------------------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestrate the full training + evaluation routine and
        return ModelTrainerArtifact.
        """
        try:
            logging.info("==== Starting Model Trainer for XGB + eGeMAPS + SMOTE ====")
            artifact = self.train_and_evaluate()
            logging.info("==== Model Trainer completed successfully ====")
            return artifact
        except Exception as e:
            raise VisionInfantNetException(e, sys)
