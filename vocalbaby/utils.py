import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
from collections import Counter
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, AddBackgroundNoise
from vocalbaby.labels import LABEL2ID, ID2LABEL

# Augmentation strategies per class
CLASS_AUGMENTATIONS = {
    "Four": Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        PitchShift(min_semitones=-1, max_semitones=2, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        #Reverb(p=0.5)
    ]),
    "Five": Compose([
        PitchShift(min_semitones=-1, max_semitones=2, p=0.5),
        TimeStretch(min_rate=0.95, max_rate=1.1, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        #Reverb(p=0.5)
    ]),
    "Three": Compose([
        PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
        Gain(min_gain_db=-3, max_gain_db=3, p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5)
    ]),
    "One": Compose([
        TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        #Reverb(p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5)
    ]),
    "Two": Compose([
        AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.02, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        #Reverb(p=0.5)
    ])
}

TARGET_COUNTS = {
    "Four": 9830,
    "Five": 3491,
    "Three": 9762,
    "One": 9766,
    "Two": 9838
}

SPLIT_COUNTS = {
    "train": {
        "Four": 243,
        "Five": 46,
        "Three": 444,
        "One": 1437,
        "Two": 1826
    },
    "validation": {
        "Four": 163,
        "Five": 41,
        "Three": 378,
        "One": 1678,
        "Two": 1357
    },
    "test": {
        "Four": 263,
        "Five": 62,
        "Three": 604,
        "One": 1370,
        "Two": 1392
    }
}

def load_metadata(path=None, audio_root=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if path is None:
        path = os.path.join(project_root, "data", "metadata.csv")
    if audio_root is None:
        audio_root = os.path.join(project_root, "data", "raw")
    df = pd.read_csv(path)
    df = df.rename(columns={"clip_ID": "path", "classes": "label", "child_ID": "child_id"})
    df["path"] = df["path"].apply(lambda x: os.path.join(audio_root, x))
    required_cols = {'path', 'label', 'child_id'}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"
    return df

def create_child_disjunct_dataset(df, split_counts, seed=42):
    # Step 1: Get unique child IDs
    unique_ids = df['child_id'].unique()
    train_ids, temp_ids = train_test_split(unique_ids, train_size=0.34, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed)

    # Step 2: Split by child ID
    id_split_map = {
        'train': train_ids,
        'validation': val_ids,
        'test': test_ids
    }

    # Step 3: Subsample by label per split
    df_split = {}
    for split, ids in id_split_map.items():
        df_subset = df[df['child_id'].isin(ids)]
        class_samples = []
        for label, count in split_counts[split].items():
            class_data = df_subset[df_subset["label"] == label]
            sampled = class_data.sample(n=min(len(class_data), count), random_state=seed)
            class_samples.append(sampled)
        df_split[split] = pd.concat(class_samples).reset_index(drop=True)

    return DatasetDict({
        k: Dataset.from_pandas(v) for k, v in df_split.items()
    })
def balance_dataset(df):
    new_rows = []
    for label in TARGET_COUNTS:
        class_df = df[df["label"] == label]
        current_count = len(class_df)
        target_count = TARGET_COUNTS[label]

        if current_count == 0:
            print(f"[WARNING] No samples found for class '{label}'. Skipping augmentation.")
            continue

        needed = target_count - current_count
        if needed > 0:
            augment = CLASS_AUGMENTATIONS[label]
            for i in range(needed):
                sample = class_df.sample(n=1, replace=True).copy()
                sample["augmented"] = True
                sample["augment_type"] = str(i)  
                new_rows.append(sample)

    if new_rows:
        df = pd.concat([df] + new_rows, ignore_index=True)
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
