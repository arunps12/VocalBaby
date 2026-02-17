"""
Sanity check script to verify all imports and dependencies are available.

Runs lightweight checks without triggering full training/tuning.

Usage:
    python -m vocalbaby.experiments.sanity_check
"""

import sys
from vocalbaby.logging.logger import logging


def check_imports():
    """Verify all required imports are available."""
    logging.info("=" * 80)
    logging.info("CHECKING IMPORTS")
    logging.info("=" * 80)
    
    try:
        # Core ML libraries
        import numpy
        logging.info("✓ numpy")
        
        import pandas
        logging.info("✓ pandas")
        
        import sklearn
        logging.info("✓ scikit-learn")
        
        import xgboost
        logging.info("✓ xgboost")
        
        import optuna
        logging.info("✓ optuna")
        
        import imblearn
        logging.info("✓ imbalanced-learn")
        
        # Audio processing
        import librosa
        logging.info("✓ librosa")
        
        import opensmile
        logging.info("✓ opensmile")
        
        # Deep learning
        import torch
        logging.info("✓ torch")
        
        import transformers
        logging.info("✓ transformers")
        
        # Visualization
        import matplotlib
        logging.info("✓ matplotlib")
        
        import seaborn
        logging.info("✓ seaborn")
        
        logging.info("\n✓ All imports successful!")
        return True
        
    except ImportError as e:
        logging.error(f"\n✗ Import failed: {e}")
        logging.error("\nPlease install missing dependencies:")
        logging.error("  pip install -r requirements.txt")
        return False


def check_modules():
    """Verify experiment modules can be imported."""
    logging.info("\n" + "=" * 80)
    logging.info("CHECKING EXPERIMENT MODULES")
    logging.info("=" * 80)
    
    try:
        from vocalbaby.experiments import data_loader
        logging.info("✓ data_loader")
        
        from vocalbaby.experiments import feature_extractors
        logging.info("✓ feature_extractors")
        
        from vocalbaby.experiments import hyperparameter_tuning
        logging.info("✓ hyperparameter_tuning")
        
        from vocalbaby.experiments import training
        logging.info("✓ training")
        
        from vocalbaby.experiments import evaluation
        logging.info("✓ evaluation")
        
        from vocalbaby.experiments.scripts import generate_features
        logging.info("✓ scripts.generate_features")
        
        from vocalbaby.experiments.scripts import tune_hyperparams
        logging.info("✓ scripts.tune_hyperparams")
        
        from vocalbaby.experiments.scripts import train_model
        logging.info("✓ scripts.train_model")
        
        from vocalbaby.experiments.scripts import evaluate_model
        logging.info("✓ scripts.evaluate_model")
        
        from vocalbaby.experiments import run_comparison_all
        logging.info("✓ run_comparison_all")
        
        logging.info("\n✓ All experiment modules loaded successfully!")
        return True
        
    except ImportError as e:
        logging.error(f"\n✗ Module import failed: {e}")
        return False


def check_gpu():
    """Check GPU availability for SSL models."""
    logging.info("\n" + "=" * 80)
    logging.info("CHECKING GPU AVAILABILITY")
    logging.info("=" * 80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logging.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            logging.info(f"  CUDA version: {torch.version.cuda}")
            logging.info(f"  GPU count: {torch.cuda.device_count()}")
            return True
        else:
            logging.info("⚠ No GPU available - will use CPU")
            logging.info("  SSL models (HuBERT, Wav2Vec2) will run slower on CPU")
            logging.info("  Consider using GPU for faster feature extraction")
            return False
            
    except Exception as e:
        logging.error(f"✗ GPU check failed: {e}")
        return False


def check_artifact_dir():
    """Check if artifact directory exists."""
    logging.info("\n" + "=" * 80)
    logging.info("CHECKING ARTIFACT DIRECTORY")
    logging.info("=" * 80)
    
    try:
        from vocalbaby.experiments.data_loader import get_latest_artifact_dir
        import os
        
        if os.path.exists("artifacts"):
            artifact_dir = get_latest_artifact_dir()
            logging.info(f"✓ Artifact directory found: {artifact_dir}")
            
            # Check for required subdirectories
            data_transform_dir = os.path.join(artifact_dir, "data_transformation/features")
            if os.path.exists(data_transform_dir):
                logging.info(f"  ✓ Features directory exists")
                
                # Check for label files
                train_labels = os.path.join(data_transform_dir, "train_labels.npy")
                if os.path.exists(train_labels):
                    logging.info(f"  ✓ Label files found")
                else:
                    logging.warning(f"  ⚠ Label files not found")
                    logging.warning("    Run data pipeline first:")
                    logging.warning("      python -m vocalbaby.pipeline.training_pipeline")
            else:
                logging.warning(f"  ⚠ Features directory not found")
                logging.warning("    Run data pipeline first:")
                logging.warning("      python -m vocalbaby.pipeline.training_pipeline")
            
            return True
        else:
            logging.warning("⚠ No artifacts directory found")
            logging.warning("  Run data pipeline first:")
            logging.warning("    python -m vocalbaby.pipeline.training_pipeline")
            return False
            
    except Exception as e:
        logging.error(f"✗ Artifact check failed: {e}")
        return False


def main():
    """Run all sanity checks."""
    logging.info("\n" + "=" * 100)
    logging.info("VOCALBABY EXPERIMENTS - SANITY CHECK")
    logging.info("=" * 100)
    
    results = {}
    
    # Run checks
    results["imports"] = check_imports()
    results["modules"] = check_modules()
    results["gpu"] = check_gpu()
    results["artifacts"] = check_artifact_dir()
    
    # Summary
    logging.info("\n" + "=" * 100)
    logging.info("SANITY CHECK SUMMARY")
    logging.info("=" * 100)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logging.info(f"{status:10s} - {check}")
    
    logging.info("=" * 100)
    
    if all_passed:
        logging.info("\n✓ ALL CHECKS PASSED - Ready to run experiments!")
        logging.info("\nNext steps:")
        logging.info("  1. Generate features:")
        logging.info("       python -m vocalbaby.experiments.scripts.generate_features --feature-set all")
        logging.info("  2. Run complete pipeline:")
        logging.info("       python -m vocalbaby.experiments.run_comparison_all")
        logging.info("\nOr use the main orchestration script:")
        logging.info("  python -m vocalbaby.experiments.run_comparison_all")
        return 0
    else:
        logging.error("\n✗ SOME CHECKS FAILED")
        logging.error("\nPlease fix the issues above before running experiments.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
