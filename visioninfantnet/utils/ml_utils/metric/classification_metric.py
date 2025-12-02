import sys
from typing import Optional

from sklearn.metrics import f1_score, precision_score, recall_score
from visioninfantnet.entity.artifact_entity import ClassificationMetricArtifact
from visioninfantnet.exception.exception import VisionInfantNetException


def get_classification_score(
    y_true,
    y_pred,
    average: str = "weighted",
    zero_division: int = 0,
) -> ClassificationMetricArtifact:
    
    try:
        # Main metrics --- weighted average by default
        model_f1_score = f1_score(
            y_true, y_pred, average=average, zero_division=zero_division
        )
        model_recall_score = recall_score(
            y_true, y_pred, average=average, zero_division=zero_division
        )
        model_precision_score = precision_score(
            y_true, y_pred, average=average, zero_division=zero_division
        )

        # UAR: Unweighted Average Recall (macro recall)
        model_uar = recall_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        )

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            uar=model_uar,
        )

        return classification_metric

    except Exception as e:
        raise VisionInfantNetException(e, sys)
