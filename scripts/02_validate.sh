#!/usr/bin/env bash
# scripts/02_validate.sh - Run Data Validation stage
set -euo pipefail

echo "======================================================================"
echo "VocalBaby Pipeline - Stage 02: Data Validation"
echo "======================================================================"

python -m vocalbaby.pipeline.stage_02_validate "$@"

echo "======================================================================"
echo "Stage 02 Complete"
echo "======================================================================"
