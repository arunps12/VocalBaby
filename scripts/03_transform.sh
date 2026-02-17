#!/usr/bin/env bash
# scripts/03_transform.sh - Run Feature Extraction stage
set -euo pipefail

echo "======================================================================"
echo "VocalBaby Pipeline - Stage 03: Feature Extraction"
echo "======================================================================"

python -m vocalbaby.pipeline.stage_03_transform "$@"

echo "======================================================================"
echo "Stage 03 Complete"
echo "======================================================================"
