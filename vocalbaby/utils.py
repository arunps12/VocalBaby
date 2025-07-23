import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
from collections import Counter
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from vocalbaby.labels import LABEL2ID, ID2LABEL

# Audio augmentation for upsampling
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
    PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5)
])

import os

def load_metadata(path=None, audio_root=None):
            
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if path is None:
        path = os.path.join(project_root, "data", "metadata.csv")
    if audio_root is None:
        audio_root = os.path.join(project_root, "data", "raw")

    df = pd.read_csv(path)

    # Rename standard column names
    df = df.rename(columns={
        "clip_ID": "path",
        "classes": "label",
        "child_ID": "child_id"
    })

    # Expand and join full audio paths
    df["path"] = df["path"].apply(lambda x: os.path.join(audio_root, x))
    required_cols = {'path', 'label', 'child_id'}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"
    return df

def create_child_disjunct_dataset(df, train_size=0.7, dev_size=0.15, test_size=0.15, seed=42):
    child_ids = df['child_id'].unique()
    train_ids, temp_ids = train_test_split(child_ids, train_size=train_size, random_state=seed)
    dev_ids, test_ids = train_test_split(temp_ids, test_size=test_size / (dev_size + test_size), random_state=seed)

    return DatasetDict({
        "train": Dataset.from_pandas(df[df['child_id'].isin(train_ids)].reset_index(drop=True)),
        "validation": Dataset.from_pandas(df[df['child_id'].isin(dev_ids)].reset_index(drop=True)),
        "test": Dataset.from_pandas(df[df['child_id'].isin(test_ids)].reset_index(drop=True)),
    })

def balance_dataset(df, label_col='label', max_ratio=3):
    counts = Counter(df[label_col])
    max_class = max(counts, key=counts.get)
    max_count = counts[max_class]

    augmented_rows = []
    for label, count in counts.items():
        subset = df[df[label_col] == label]
        if count < max_count:
            repeat_count = min(max_count, max_ratio * count) - count
            sampled = subset.sample(n=repeat_count, replace=True, random_state=42)
            sampled = sampled.copy()
            sampled['augmented'] = True
            augmented_rows.append(sampled)

    if augmented_rows:
        df = pd.concat([df] + augmented_rows)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def encode_label(label_str):
    return LABEL2ID[label_str]

def decode_label(label_id):
    return ID2LABEL[label_id]

def encode_labels_column(df, column='label'):
    df[column] = df[column].map(LABEL2ID)
    return df

def compute_class_weights(labels, num_classes=5):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)
