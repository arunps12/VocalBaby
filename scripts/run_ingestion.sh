#!/usr/bin/env bash
# scripts/run_ingestion.sh - Run Data Ingestion stage
set -euo pipefail

echo "=== VocalBaby: Data Ingestion ==="
uv run python -c "
from vocalbaby.components.data_ingestion import DataIngestion
from vocalbaby.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from vocalbaby.logging.logger import logging

logging.info('Starting data ingestion...')
pipeline_config = TrainingPipelineConfig()
ingestion_config = DataIngestionConfig(pipeline_config)
ingestion = DataIngestion(ingestion_config)
artifact = ingestion.initiate_data_ingestion()
logging.info(f'Data ingestion complete: {artifact}')
print('Data Ingestion completed successfully.')
"
echo "=== Data Ingestion Complete ==="
