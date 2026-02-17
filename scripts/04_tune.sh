#!/usr/bin/env bash
# scripts/04_tune.sh - Run Hyperparameter Tuning stage
set -euo pipefail

export PYTHONUNBUFFERED=1

echo "======================================================================"
echo "VocalBaby Pipeline - Stage 04: Hyperparameter Tuning"
echo "======================================================================"

python -m vocalbaby.pipeline.stage_04_tune "$@"

echo "======================================================================"
echo "Stage 04 Complete"
echo "======================================================================"
