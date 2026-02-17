#!/usr/bin/env bash
# scripts/07_aggregate.sh - Run Results Aggregation stage
set -euo pipefail

echo "======================================================================"
echo "VocalBaby Pipeline - Stage 07: Results Aggregation"
echo "======================================================================"

python -m vocalbaby.pipeline.stage_07_aggregate "$@"

echo "======================================================================"
echo "Stage 07 Complete"
echo "======================================================================"
