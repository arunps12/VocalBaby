"""
VocalBaby Data Module

Handles data ingestion, validation, and split management.
"""

from vocalbaby.data.ingest import run_data_ingestion
from vocalbaby.data.validate import run_data_validation

__all__ = [
    'run_data_ingestion',
    'run_data_validation',
]
