import os
import sys
import shutil
import tempfile
import zipfile
from typing import List, Optional

import yaml
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging

# Prediction 
from vocalbaby.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI(
    title="VisionInfantNet API (Prediction Only)",
    description="Prediction API for VisionInfantNet (XGBoost on eGeMAPS). Training disabled.",
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
# Globals: final_model + cached predictor
# ----------------------------------------------------------------------
FINAL_MODEL_DIR = "final_model"
MODEL_INFO_PATH = os.path.join(FINAL_MODEL_DIR, "model_info.yaml")

_predictor: Optional[PredictionPipeline] = None


def _get_final_model_dir_from_model_info() -> str:
    """
    Reads final_model/model_info.yaml to locate the model directory + filenames.

    Expected model_info.yaml structure:

    final_model:
      model_dir: final_model
      model_file: xgb_egemaps_smote_optuna.pkl
      preprocessing_file: preprocessing.pkl
      label_encoder_file: label_encoder.pkl
    """
    try:
        if not os.path.exists(MODEL_INFO_PATH):
            raise FileNotFoundError(
                f"model_info.yaml not found at {MODEL_INFO_PATH}. "
                "Ensure `final_model/` is included in the Docker image."
            )

        with open(MODEL_INFO_PATH, "r") as f:
            info = yaml.safe_load(f) or {}

        final_info = info.get("final_model", {})
        model_dir = final_info.get("model_dir")
        model_file = final_info.get("model_file")
        preprocessing_file = final_info.get("preprocessing_file")
        label_encoder_file = final_info.get("label_encoder_file")

        if not model_dir:
            raise ValueError("model_info.yaml: final_model.model_dir is missing.")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Final model directory not found: {model_dir}")

        # Sanity-check that all required files exist (no hard-coded names)
        required_files = [model_file, preprocessing_file, label_encoder_file]
        missing_keys = [k for k in ["model_file", "preprocessing_file", "label_encoder_file"] if not final_info.get(k)]
        if missing_keys:
            raise ValueError(f"model_info.yaml: missing keys in final_model: {missing_keys}")

        missing_files = [
            f for f in required_files
            if not os.path.exists(os.path.join(model_dir, f))
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Missing files in {model_dir}: {missing_files}. "
                "Make sure final_model/ is present inside the container."
            )

        logging.info(f"Using final model directory: {model_dir}")
        return model_dir

    except Exception as e:
        raise VocalBabyException(e, sys)


def _get_prediction_pipeline() -> PredictionPipeline:
    """Construct (and cache) PredictionPipeline from model_info.yaml."""
    global _predictor

    if _predictor is not None:
        return _predictor

    model_dir = _get_final_model_dir_from_model_info()
    _predictor = PredictionPipeline(model_trainer_dir=model_dir)
    return _predictor


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/", tags=["root"])
async def index():
    """Redirect to Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["health"])
async def health():
    """
    Lightweight health check.
    Confirms that final_model + model_info + required files exist.
    """
    try:
        model_dir = _get_final_model_dir_from_model_info()
        return {"status": "ok", "model_dir": model_dir}
    except Exception as e:
        # Return 503 instead of crashing
        logging.exception("Health check failed.")
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.post("/predict", tags=["prediction"])
async def predict_route(
    request: Request,
    files: List[UploadFile] = File(..., description="One or more .wav audio segments"),
):
    """
    Predict labels for uploaded audio segment(s).

    Steps:
      1) Save uploaded files to uploaded_audio/
      2) Load PredictionPipeline from final_model/model_info.yaml
      3) Predict labels
    """
    try:
        logging.info("=== /predict endpoint called ===")

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

        predictor = _get_prediction_pipeline()
        y_pred_enc, y_pred_dec, audio_paths = predictor.predict_from_audio(saved_paths)

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
        raise VocalBabyException(e, sys)


@app.post("/predict_zip", tags=["prediction"])
async def predict_zip_route(
    file: UploadFile = File(..., description="ZIP file containing .wav audio segments"),
):
    """
    Predict labels for a ZIP archive of audio segments.
    """
    try:
        logging.info("=== /predict_zip endpoint called ===")

        zip_upload_dir = "uploaded_zips"
        os.makedirs(zip_upload_dir, exist_ok=True)

        zip_path = os.path.join(zip_upload_dir, os.path.basename(file.filename))
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logging.info(f"Saved uploaded ZIP to {zip_path}.")

        # Extract ZIP to a temporary directory
        extract_dir = tempfile.mkdtemp(prefix="segments_")
        logging.info(f"Extracting ZIP into {extract_dir}...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Collect all .wav files recursively
        audio_paths = []
        for root, _, filenames in os.walk(extract_dir):
            for fname in filenames:
                if fname.lower().endswith(".wav"):
                    audio_paths.append(os.path.join(root, fname))

        if not audio_paths:
            raise VocalBabyException(
                f"No .wav files found in extracted ZIP: {extract_dir}", sys
            )

        logging.info(f"Found {len(audio_paths)} .wav files in extracted ZIP.")

        predictor = _get_prediction_pipeline()
        y_pred_enc, y_pred_dec, audio_paths = predictor.predict_from_audio(audio_paths)

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
        raise VocalBabyException(e, sys)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: eager-load predictor on startup so container fails fast if model missing
    try:
        _ = _get_prediction_pipeline()
        logging.info("Model loaded successfully at startup.")
    except Exception:
        logging.exception("Failed to load model at startup. Container will still start, but /predict will fail.")

    app_run(app, host="0.0.0.0", port=8080)
