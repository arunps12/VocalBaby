#!/usr/bin/env bash
# scripts/run_training.sh - Run full training pipeline
set -euo pipefail

echo "=== VocalBaby: Full Training Pipeline ==="
uv run python -c "
from vocalbaby.pipeline.training_pipeline import TrainingPipeline
from vocalbaby.logging.logger import logging

logging.info('Starting full training pipeline...')
pipeline = TrainingPipeline()
pipeline.run_pipeline()
logging.info('Full training pipeline completed.')
print('Training Pipeline completed successfully.')
"
echo "=== Training Pipeline Complete ==="
