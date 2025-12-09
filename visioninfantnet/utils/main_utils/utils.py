import yaml
import os
import sys
import numpy as np
import pickle
import subprocess

from visioninfantnet.exception.exception import VisionInfantNetException


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise VisionInfantNetException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.safe_dump(content, file)
    except Exception as e:
        raise VisionInfantNetException(e, sys)

import os
import sys
import numpy as np
from visioninfantnet.exception.exception import VisionInfantNetException


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise VisionInfantNetException(e, sys)

def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array data from file_path using np.load.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file does not exist: {file_path}")

        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise VisionInfantNetException(e, sys)
    

def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object (model, imputer, encoder, etc.) using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise VisionInfantNetException(e, sys)

def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file does not exist: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise VisionInfantNetException(e, sys)


def get_latest_artifact_dir(root_dir: str) -> str:
    """
    Return the latest (by mtime) subdirectory inside artifacts root.
    """
    try:
        subdirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        if not subdirs:
            raise Exception(f"No subdirectories found in {root_dir}")
        return max(subdirs, key=os.path.getmtime)
    except Exception as e:
        raise VisionInfantNetException(e, sys)
