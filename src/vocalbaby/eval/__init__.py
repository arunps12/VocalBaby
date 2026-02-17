"""
VocalBaby Evaluation Module

Metrics computation, confusion matrices, and evaluation utilities.
"""
from vocalbaby.experiments.evaluation import (
    evaluate_model,
    compute_metrics,
    save_confusion_matrix,
    save_classification_report,
)

__all__ = [
    'evaluate_model',
    'compute_metrics',
    'save_confusion_matrix',
    'save_classification_report',
]
