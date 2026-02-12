#!/usr/bin/env bash
# scripts/run_inference.sh - Run inference / prediction
set -euo pipefail

AUDIO_FILE="${1:?Usage: $0 <audio_file_path>}"

echo "=== VocalBaby: Inference ==="
uv run python -c "
import sys
from vocalbaby.pipeline.prediction_pipeline import PredictionPipeline
from vocalbaby.logging.logger import logging

audio_path = '${AUDIO_FILE}'
logging.info(f'Running inference on: {audio_path}')

pipeline = PredictionPipeline()
result = pipeline.predict(audio_path)
print(f'Prediction result: {result}')
"
echo "=== Inference Complete ==="
