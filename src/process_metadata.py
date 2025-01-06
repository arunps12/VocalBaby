import pandas as pd
import os
from config import METADATA_PATH, PROCESSED_METADATA_PATH

# Paths
metadata_path = METADATA_PATH
output_metadata_path = PROCESSED_METADATA_PATH

# Load metadata
metadata = pd.read_csv(metadata_path)

# Merge classes into binary labels
class_mapping = {
    "One": "A",
    "Three": "A",
    "Four": "A",
    "Five": "A",
    "Two": "B"
}
metadata["BinaryClass"] = metadata["classes"].map(class_mapping)

# Save processed metadata
metadata.to_csv(output_metadata_path, index=False)
print(f"Processed metadata saved to {output_metadata_path}")
