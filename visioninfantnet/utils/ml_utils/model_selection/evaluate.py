import sys
from typing import Tuple

from visioninfantnet.exception.exception import VisionInfantNetException
from visioninfantnet.entity.artifact_entity import ClassificationMetricArtifact
from visioninfantnet.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)


def evaluate_splits(
    y_train,
    y_train_pred,
    y_valid,
    y_valid_pred,
    y_test,
    y_test_pred,
) -> Tuple[
    ClassificationMetricArtifact,
    ClassificationMetricArtifact,
    ClassificationMetricArtifact,
]:
    """
    Compute classification metrics (F1, Precision, Recall, UAR)
    for train, valid, and test splits.

    Returns:
        (train_metrics, valid_metrics, test_metrics)
        each as ClassificationMetricArtifact.
    """
    try:
        train_metrics = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        valid_metrics = get_classification_score(y_true=y_valid, y_pred=y_valid_pred)
        test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        return train_metrics, valid_metrics, test_metrics

    except Exception as e:
        raise VisionInfantNetException(e, sys)
