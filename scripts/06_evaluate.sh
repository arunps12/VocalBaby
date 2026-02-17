#!/usr/bin/env bash
# scripts/06_evaluate.sh - Run Model Evaluation stage
set -euo pipefail

echo "======================================================================"
echo "VocalBaby Pipeline - Stage 06: Model Evaluation"
echo "======================================================================"

python -m vocalbaby.pipeline.stage_06_evaluate "$@"

echo "======================================================================"
echo "Stage 06 Complete"
echo "======================================================================"
