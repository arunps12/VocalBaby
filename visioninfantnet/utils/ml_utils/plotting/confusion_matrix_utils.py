import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from visioninfantnet.exception.exception import VisionInfantNetException


def plot_and_save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    file_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Compute confusion matrix from y_true / y_pred,
    plot it with seaborn, and save as PNG.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

    except Exception as e:
        raise VisionInfantNetException(e, sys)
