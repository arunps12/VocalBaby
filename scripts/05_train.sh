#!/usr/bin/env bash
# scripts/05_train.sh - Run Model Training stage
set -euo pipefail

echo "======================================================================"
echo "VocalBaby Pipeline - Stage 05: Model Training"
echo "======================================================================"

python -m vocalbaby.pipeline.stage_05_train "$@"

echo "======================================================================"
echo "Stage 05 Complete"
echo "======================================================================"
