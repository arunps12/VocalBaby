import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder

from vocalbaby.exception.exception import VocalBabyException
from vocalbaby.logging.logger import logging


def encode_labels(y_train, y_valid, y_test):
    """
    Fit a LabelEncoder on y_train (string labels), transform train/valid/test,
    and return encoded labels + class names + encoder.

    Returns:
        y_train_enc, y_valid_enc, y_test_enc, class_names, label_encoder
    """
    try:
        # Ensure 1D
        y_train = np.ravel(y_train)
        y_valid = np.ravel(y_valid)
        y_test = np.ravel(y_test)

        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        y_valid_enc = label_encoder.transform(y_valid)
        y_test_enc = label_encoder.transform(y_test)

        class_names = list(label_encoder.classes_)

        # Log mapping 
        logging.info("Label encoding mapping:")
        for org, enc in zip(
            label_encoder.classes_, label_encoder.transform(label_encoder.classes_)
        ):
            logging.info(f"{org} → {enc}")

        
        print("Label encoding mapping:")
        for org, enc in zip(
            label_encoder.classes_, label_encoder.transform(label_encoder.classes_)
        ):
            print(f"{org} → {enc}")

        return y_train_enc, y_valid_enc, y_test_enc, class_names, label_encoder

    except Exception as e:
        raise VocalBabyException(e, sys)
