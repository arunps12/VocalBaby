"""
Feature cache generation script for XGBoost comparison experiments.

Extracts and caches features for all four feature sets:
- eGeMAPS
- MFCC
- HuBERT SSL embeddings
- Wav2Vec2 SSL embeddings

Usage:
    python -m vocalbaby.experiments.scripts.generate_features --feature-set egemaps
    python -m vocalbaby.experiments.scripts.generate_features --feature-set all
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

from vocalbaby.experiments.data_loader import (
    get_latest_artifact_dir,
    load_metadata_with_audio_paths,
)
from vocalbaby.experiments.feature_extractors import (
    extract_egemaps_features,
    extract_mfcc_features,
    extract_hubert_embedding,
    extract_wav2vec2_embedding,
    load_hubert_model,
    load_wav2vec2_model,
    extract_features_from_metadata,
    save_features,
)
from vocalbaby.logging.logger import logging
from vocalbaby.exception.exception import VocalBabyException

import opensmile


FEATURE_CONFIGS = {
    "egemaps": {
        "extractor": "egemaps",
        "output_dir": "artifacts/features/egemaps",
        "model_required": False,
    },
    "mfcc": {
        "extractor": "mfcc",
        "output_dir": "artifacts/features/mfcc",
        "model_required": False,
        "n_mfcc": 20,
        "pooling": "mean",
    },
    "hubert_ssl": {
        "extractor": "hubert",
        "output_dir": "artifacts/features/hubert_ssl",
        "model_required": True,
        "model_name": "arunps/hubert-home-hindibabynet-ssl",
        "pooling": "mean",
    },
    "wav2vec2_ssl": {
        "extractor": "wav2vec2",
        "output_dir": "artifacts/features/wav2vec2_ssl",
        "model_required": True,
        "model_name": "arunps/wav2vec2-home-hindibabynet-ssl",
        "pooling": "mean",
    },
}


def generate_features_for_set(
    feature_set: str,
    artifact_dir: str,
    force_regenerate: bool = False,
):
    """
    Generate and cache features for a specific feature set.
    
    Args:
        feature_set: Feature set name (egemaps, mfcc, hubert_ssl, wav2vec2_ssl)
        artifact_dir: Path to artifact directory
        force_regenerate: If True, regenerate even if cache exists
    """
    try:
        if feature_set not in FEATURE_CONFIGS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        
        config = FEATURE_CONFIGS[feature_set]
        # Build output_dir relative to artifact_dir instead of using hardcoded path
        output_dir = os.path.join(artifact_dir, "features", feature_set)
        
        logging.info("=" * 80)
        logging.info(f"GENERATING FEATURES: {feature_set.upper()}")
        logging.info("=" * 80)
        
        # Check if features already exist
        train_cache = os.path.join(output_dir, "train", "features.npy")
        if os.path.exists(train_cache) and not force_regenerate:
            logging.info(f"Features already cached at {train_cache}")
            logging.info("Use --force to regenerate")
            return
        
        # Load metadata
        train_df, valid_df, test_df = load_metadata_with_audio_paths(artifact_dir)
        
        # Prepare extractor
        if config["extractor"] == "egemaps":
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            extractor_fn = lambda path: extract_egemaps_features(path, smile)
            extractor_kwargs = {}
            
        elif config["extractor"] == "mfcc":
            extractor_fn = extract_mfcc_features
            extractor_kwargs = {
                "n_mfcc": config["n_mfcc"],
                "pooling": config["pooling"],
            }
            
        elif config["extractor"] == "hubert":
            logging.info(f"Loading HuBERT model: {config['model_name']}")
            processor, model = load_hubert_model(config["model_name"])
            extractor_fn = extract_hubert_embedding
            extractor_kwargs = {
                "processor": processor,
                "model": model,
                "pooling": config["pooling"],
            }
            
        elif config["extractor"] == "wav2vec2":
            logging.info(f"Loading Wav2Vec2 model: {config['model_name']}")
            processor, model = load_wav2vec2_model(config["model_name"])
            extractor_fn = extract_wav2vec2_embedding
            extractor_kwargs = {
                "processor": processor,
                "model": model,
                "pooling": config["pooling"],
            }
        else:
            raise ValueError(f"Unknown extractor: {config['extractor']}")
        
        # Extract features for each split
        for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            logging.info(f"\nProcessing {split_name} split ({len(split_df)} samples)...")
            
            features = extract_features_from_metadata(
                split_df,
                extractor_fn,
                audio_path_col="path",  # Column name from existing data pipeline
                desc=f"Extracting {feature_set} - {split_name}",
                **extractor_kwargs,
            )
            
            # Save features
            split_output_dir = os.path.join(output_dir, split_name)
            output_path = os.path.join(split_output_dir, "features.npy")
            save_features(features, output_path)
        
        logging.info("=" * 80)
        logging.info(f"Feature generation completed: {feature_set}")
        logging.info("=" * 80)
        
    except Exception as e:
        raise VocalBabyException(e, sys)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and cache features for XGBoost comparison experiments"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        required=True,
        choices=["egemaps", "mfcc", "hubert_ssl", "wav2vec2_ssl", "all"],
        help="Feature set to generate (or 'all' for all sets)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Path to artifact directory (default: latest)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists",
    )
    
    args = parser.parse_args()
    
    # Get artifact directory
    if args.artifact_dir is None:
        artifact_dir = get_latest_artifact_dir()
    else:
        artifact_dir = args.artifact_dir
    
    logging.info(f"Using artifact directory: {artifact_dir}")
    
    # Generate features
    if args.feature_set == "all":
        for feature_set in ["egemaps", "mfcc", "hubert_ssl", "wav2vec2_ssl"]:
            generate_features_for_set(feature_set, artifact_dir, args.force)
    else:
        generate_features_for_set(args.feature_set, artifact_dir, args.force)
    
    logging.info("\n" + "=" * 80)
    logging.info("ALL FEATURE GENERATION COMPLETED")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
