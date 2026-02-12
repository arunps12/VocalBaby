#!/usr/bin/env bash
# scripts/run_drift.sh - Run Evidently drift detection
set -euo pipefail

echo "=== VocalBaby: Drift Detection ==="
uv run python -c "
from vocalbaby.monitoring.drift import run_drift_report
from vocalbaby.logging.logger import logging

logging.info('Starting drift detection...')
report = run_drift_report()
logging.info('Drift detection completed.')
print('Drift Detection completed successfully.')
"
echo "=== Drift Detection Complete ==="
