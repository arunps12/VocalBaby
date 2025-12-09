import os
import sys
import shutil
from typing import List
import zipfile
import tempfile

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

from visioninfantnet.exception.exception import VisionInfantNetException
from visioninfantnet.logging.logger import logging

from visioninfantnet.pipeline.training_pipeline import TrainingPipeline
from visioninfantnet.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI(
    title="VisionInfantNet API",
    description="Training + Prediction API for VisionInfantNet (XGBoost on eGeMAPS)",
    version="1.0.0",
)

# ----------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# find latest model_trainer directory under artifacts/
# ----------------------------------------------------------------------
def _get_latest_model_trainer_dir(artifacts_root: str = "artifacts") -> str:
    """
    Finds the latest timestamped run inside `artifacts/` and returns its
    model_trainer directory.
    """
    try:
        if not os.path.isdir(artifacts_root):
            raise FileNotFoundError(f"Artifacts root not found: {artifacts_root}")

        # List timestamped subdirectories
        subdirs = [
            d for d in os.listdir(artifacts_root)
            if os.path.isdir(os.path.join(artifacts_root, d))
        ]
        if not subdirs:
            raise FileNotFoundError("No timestamped artifact directories found.")

        # Sort lexicographically and pick the latest
        subdirs.sort()
        latest_run = subdirs[-1]

        model_trainer_dir = os.path.join(artifacts_root, latest_run, "model_trainer")
        if not os.path.isdir(model_trainer_dir):
            raise FileNotFoundError(
                f"model_trainer directory not found at: {model_trainer_dir}"
            )

        logging.info(f"Using latest model_trainer dir: {model_trainer_dir}")
        return model_trainer_dir

    except Exception as e:
        raise VisionInfantNetException(e, sys)


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/", tags=["root"])
async def index():
    """Redirect to Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["training"])
async def train_route():
    """
    Run the full training pipeline:
      1. Data Ingestion
      2. Data Validation
      3. Data Transformation
      4. Model Training (XGBoost + eGeMAPS + SMOTE + MLflow logging)
    """
    try:
        logging.info("=== /train endpoint called: starting TrainingPipeline ===")
        pipeline = TrainingPipeline()
        model_trainer_artifact = pipeline.run_pipeline()
        logging.info("TrainingPipeline finished successfully.")

        return {
            "message": "Training completed successfully.",
            "model_trainer_artifact": str(model_trainer_artifact),
        }

    except Exception as e:
        logging.exception("Error occurred in /train endpoint.")
        raise VisionInfantNetException(e, sys)


# ----------------------------------------------------------------------
# /predict : multiple .wav files upload
# ----------------------------------------------------------------------
@app.post("/predict", tags=["prediction"])
async def predict_route(
    request: Request,
    files: List[UploadFile] = File(..., description="One or more .wav audio segments"),
):
    """
    Predict labels for uploaded audio segment(s).

    Input:
      - One or more .wav files (UploadFile)

    Steps:
      1. Save uploaded files to a local directory.
      2. Load latest trained model via PredictionPipeline.
      3. Extract eGeMAPS, preprocess, predict with XGBoost.
      4. Return JSON with file names and predicted labels.
    """
    try:
        logging.info("=== /predict endpoint called ===")

        # --------------------------------------------------------------
        # Save uploads to a directory
        # --------------------------------------------------------------
        upload_dir = "uploaded_audio"
        os.makedirs(upload_dir, exist_ok=True)

        saved_paths = []
        for file in files:
            filename = os.path.basename(file.filename)
            dest_path = os.path.join(upload_dir, filename)

            with open(dest_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            saved_paths.append(dest_path)

        logging.info(f"Saved {len(saved_paths)} uploaded audio files to {upload_dir}.")

        # --------------------------------------------------------------
        # Find latest model_trainer dir and create PredictionPipeline
        # --------------------------------------------------------------
        model_trainer_dir = _get_latest_model_trainer_dir(artifacts_root="artifacts")
        predictor = PredictionPipeline(model_trainer_dir=model_trainer_dir)

        # --------------------------------------------------------------
        # Run prediction
        # --------------------------------------------------------------
        y_pred_enc, y_pred_dec, audio_paths = predictor.predict_from_audio(saved_paths)

        # --------------------------------------------------------------
        # Build response 
        # --------------------------------------------------------------
        results = []
        for p, enc, dec in zip(audio_paths, y_pred_enc, y_pred_dec):
            results.append(
                {
                    "file_path": p,
                    "file_name": os.path.basename(p),
                    "predicted_label_encoded": int(enc),
                    "predicted_label": str(dec),
                }
            )

        logging.info("Prediction completed for all uploaded files.")
        return {"results": results}

    except Exception as e:
        logging.exception("Error occurred in /predict endpoint.")
        raise VisionInfantNetException(e, sys)


# ----------------------------------------------------------------------
# /predict_zip : a single ZIP containing many .wav files
# ----------------------------------------------------------------------
@app.post("/predict_zip", tags=["prediction"])
async def predict_zip_route(
    file: UploadFile = File(..., description="ZIP file containing .wav audio segments"),
):
    """
    Predict labels for a ZIP archive of audio segments.

    Steps:
      1. Save uploaded ZIP file.
      2. Extract it to a temporary directory.
      3. Collect all .wav files recursively.
      4. Use PredictionPipeline to predict.
      5. Return the same JSON structure as /predict.
    """
    try:
        logging.info("=== /predict_zip endpoint called ===")

        # --------------------------------------------------------------
        # Save ZIP file
        # --------------------------------------------------------------
        zip_upload_dir = "uploaded_zips"
        os.makedirs(zip_upload_dir, exist_ok=True)

        zip_path = os.path.join(zip_upload_dir, os.path.basename(file.filename))
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logging.info(f"Saved uploaded ZIP to {zip_path}.")

        # --------------------------------------------------------------
        # Extract ZIP to a temporary directory
        # --------------------------------------------------------------
        extract_dir = tempfile.mkdtemp(prefix="segments_")
        logging.info(f"Extracting ZIP into {extract_dir}...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # --------------------------------------------------------------
        # Collect all .wav files (recursive)
        # --------------------------------------------------------------
        audio_paths = []
        for root, _, filenames in os.walk(extract_dir):
            for fname in filenames:
                if fname.lower().endswith(".wav"):
                    audio_paths.append(os.path.join(root, fname))

        if not audio_paths:
            raise VisionInfantNetException(
                f"No .wav files found in extracted ZIP: {extract_dir}", sys
            )

        logging.info(f"Found {len(audio_paths)} .wav files in extracted ZIP.")

        # --------------------------------------------------------------
        # Find latest model_trainer dir and create PredictionPipeline
        # --------------------------------------------------------------
        model_trainer_dir = _get_latest_model_trainer_dir(artifacts_root="artifacts")
        predictor = PredictionPipeline(model_trainer_dir=model_trainer_dir)

        # --------------------------------------------------------------
        # Run prediction
        # --------------------------------------------------------------
        y_pred_enc, y_pred_dec, audio_paths = predictor.predict_from_audio(audio_paths)

        # --------------------------------------------------------------
        # Build response 
        # --------------------------------------------------------------
        results = []
        for p, enc, dec in zip(audio_paths, y_pred_enc, y_pred_dec):
            results.append(
                {
                    "file_path": p,
                    "file_name": os.path.basename(p),
                    "predicted_label_encoded": int(enc),
                    "predicted_label": str(dec),
                }
            )

        logging.info("Prediction completed for all files in ZIP.")
        return {"results": results}

    except Exception as e:
        logging.exception("Error occurred in /predict_zip endpoint.")
        raise VisionInfantNetException(e, sys)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8080)
