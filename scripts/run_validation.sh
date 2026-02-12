#!/usr/bin/env bash
# scripts/run_validation.sh - Run Data Validation stage
set -euo pipefail

echo "=== VocalBaby: Data Validation ==="
uv run python -c "
from vocalbaby.pipeline.training_pipeline import TrainingPipeline
from vocalbaby.logging.logger import logging

logging.info('Starting data validation...')
pipeline = TrainingPipeline()

# Run ingestion first to get artifacts
ingestion_artifact = pipeline.start_data_ingestion()

# Then run validation
validation_artifact = pipeline.start_data_validation(ingestion_artifact)
logging.info(f'Data validation complete: {validation_artifact}')
print('Data Validation completed successfully.')
"
echo "=== Data Validation Complete ==="
