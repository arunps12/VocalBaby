#!/usr/bin/env bash
# scripts/01_ingest.sh - Run Data Ingestion stage
set -euo pipefail

echo "======================================================================"
echo "VocalBaby Pipeline - Stage 01: Data Ingestion"
echo "======================================================================"

python -m vocalbaby.pipeline.stage_01_ingest "$@"

echo "======================================================================"
echo "Stage 01 Complete"
echo "======================================================================"
