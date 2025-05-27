import os

import numpy as np
import pandas as pd

from config import PROCESSED_TRAIN_DATA_DIR, TRAIN_METADATA_PATH, PROCESSED_TRAIN_METADATA_PATH, BALANCED_TRAIN_METADATA_PATH


# Load original train metadata
original_df = pd.read_csv(TRAIN_METADATA_PATH)

# Create a mapping from clip_ID to full row (excluding clip_ID)
metadata_map = original_df.set_index("clip_ID").to_dict(orient="index")

# List to hold new metadata
new_rows = []

# Process all .wav files in the processed directory
for filename in os.listdir(PROCESSED_TRAIN_DATA_DIR):
    if not filename.endswith(".wav"):
        continue

    # Identify base clip and augmentation type
    if "_" in filename:
        parts = filename.split("_")
        base_id = parts[0] + ".wav"
        aug_type = parts[1]
    else:
        base_id = filename
        aug_type = "original"

    # Copy metadata from base
    if base_id in metadata_map:
        new_row = metadata_map[base_id].copy()
        new_row["clip_ID"] = filename
        new_row["AugmentationType"] = aug_type
        new_rows.append(new_row)
    else:
        print(f"Metadata missing for base: {base_id}")

# Create new DataFrame
augmented_df = pd.DataFrame(new_rows)

# Save it
augmented_df.to_csv(PROCESSED_TRAIN_METADATA_PATH, index=False)
print(f"Saved combined metadata to: {PROCESSED_TRAIN_METADATA_PATH}")

# Load updated metadata
df = pd.read_csv(PROCESSED_TRAIN_METADATA_PATH)

# Group other classes completely
final_df = df[df["VocLabel"] != "NonCanonical"]

# Handle NonCanonical (majority class)
noncanon_aug = df[(df["VocLabel"] == "NonCanonical") & (df["AugmentationType"] != "original")]
noncanon_orig = df[(df["VocLabel"] == "NonCanonical") & (df["AugmentationType"] == "original")]

# How many augmented are available
num_aug = len(noncanon_aug)
num_needed_total = 4485
num_orig_needed = num_needed_total - num_aug

# Sample from original
noncanon_orig_sampled = noncanon_orig.sample(n=num_orig_needed, random_state=42)
# Combine
# Combine
final_df = pd.concat([final_df, noncanon_aug, noncanon_orig_sampled], ignore_index=True)

# Save the final metadata CSV
final_df.to_csv(BALANCED_TRAIN_METADATA_PATH, index=False)
print("Saved: final_balanced_metadata.csv")

