import os
import sys
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

from visioninfantnet.entity.config_entity import DataIngestionConfig
from visioninfantnet.entity.artifact_entity import DataIngestionArtifact
from visioninfantnet.logging.logger import logging
from visioninfantnet.exception.exception import VisionInfantNetException
from visioninfantnet.constant.training_pipeline import (
    CHILD_ID_COLUMN,
    AUDIO_ID_COLUMN,
    AUDIO_PATH_COLUMN,
    TARGET_COLUMN,
)

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ----------------------------------------------------------------------
    # Load Metadata
    # ----------------------------------------------------------------------
    def load_metadata(self):
        try:
            df = pd.read_csv(self.config.raw_metadata_file)

            # Add full raw audio path
            df[AUDIO_PATH_COLUMN] = df[AUDIO_ID_COLUMN].apply(
                lambda x: os.path.join(self.config.raw_audio_dir, x)
            )

            # Save a snapshot of full metadata (raw paths)
            os.makedirs(self.config.data_ingestion_dir, exist_ok=True)
            df.to_csv(self.config.full_metadata_file, index=False)

            logging.info(
                f"Loaded metadata ({len(df)} rows) and saved snapshot to: {self.config.full_metadata_file}"
            )

            return df

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ----------------------------------------------------------------------
    # Child-disjoint splitting
    # ----------------------------------------------------------------------
    def create_child_disjoint_split(self, df):
        try:
            logging.info("Starting child-disjoint splitting...")

            unique_children = df[CHILD_ID_COLUMN].unique()
            train_ids, temp_ids = train_test_split(unique_children, train_size=0.34, random_state=42)
            valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

            split_children = {
                "train": train_ids,
                "valid": valid_ids,
                "test": test_ids,
            }

            # allocate results
            df_split = {}

            for split, children in split_children.items():
                df_sub = df[df[CHILD_ID_COLUMN].isin(children)]
                samples = []

                for label, count in self.config.split_counts[split].items():
                    class_data = df_sub[df_sub[TARGET_COLUMN] == label]

                    if len(class_data) == 0:
                        logging.warning(f"No samples for label '{label}' in split '{split}'.")
                        continue

                    sampled = class_data.sample(
                        n=min(len(class_data), count), random_state=42
                    )
                    samples.append(sampled)

                df_split[split] = pd.concat(samples).reset_index(drop=True)
                logging.info(f"{split} split created with {len(df_split[split])} samples.")

            return df_split

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ----------------------------------------------------------------------
    # Copy audio to ingested folders
    # ----------------------------------------------------------------------
    def copy_audio_files(self, df, dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)

            for _, row in df.iterrows():
                src = row[AUDIO_PATH_COLUMN]
                filename = os.path.basename(src)
                dest = os.path.join(dest_dir, filename)

                if not os.path.exists(src):
                    logging.warning(f"Missing audio file: {src}")
                    continue

                shutil.copy2(src, dest)

            logging.info(f"Copied {len(df)} audio files to: {dest_dir}")

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ----------------------------------------------------------------------
    # Write metadata CSVs with updated audio paths (artifact paths)
    # ----------------------------------------------------------------------
    def write_split_metadata(self, df, csv_path, audio_dir):
        try:
            updated_df = df.copy()

            updated_df[AUDIO_PATH_COLUMN] = updated_df[AUDIO_ID_COLUMN].apply(
                lambda fname: os.path.join(audio_dir, fname)
            )

            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            updated_df.to_csv(csv_path, index=False)

            logging.info(f"Saved metadata: {csv_path} ({len(updated_df)} rows)")

        except Exception as e:
            raise VisionInfantNetException(e, sys)

    # ----------------------------------------------------------------------
    # Main ingestion
    # ----------------------------------------------------------------------
    def initiate_data_ingestion(self):
        try:
            logging.info("===== Data Ingestion Started =====")

            # 1. Load full metadata and save snapshot
            df = self.load_metadata()

            # 2. Child-disjoint splitting
            splits = self.create_child_disjoint_split(df)

            # 3. Copy audio files to ingested folders
            self.copy_audio_files(splits["train"], self.config.train_audio_dir)
            self.copy_audio_files(splits["valid"], self.config.valid_audio_dir)
            self.copy_audio_files(splits["test"], self.config.test_audio_dir)

            # 4. Save train/valid/test metadata with updated (artifact) audio paths
            self.write_split_metadata(splits["train"], self.config.train_metadata_file, self.config.train_audio_dir)
            self.write_split_metadata(splits["valid"], self.config.valid_metadata_file, self.config.valid_audio_dir)
            self.write_split_metadata(splits["test"], self.config.test_metadata_file, self.config.test_audio_dir)

            logging.info("===== Data Ingestion Completed Successfully =====")

            dataingestionartifact = DataIngestionArtifact(
                train_metadata_path=self.config.train_metadata_file,
                valid_metadata_path=self.config.valid_metadata_file,
                test_metadata_path=self.config.test_metadata_file,
                train_audio_dir=self.config.train_audio_dir,
                valid_audio_dir=self.config.valid_audio_dir,
                test_audio_dir=self.config.test_audio_dir,
            )

            return dataingestionartifact

        except Exception as e:
            raise VisionInfantNetException(e, sys)
