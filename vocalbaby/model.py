import os
import torch
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2Config,
    AutoConfig
)

# === 1. Inference Loader ===

def load_model_for_inference(repo_id_or_path):
    """Load Wav2Vec2 model and processor for CLI/inference."""
    processor = Wav2Vec2Processor.from_pretrained(repo_id_or_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(repo_id_or_path)
    model.eval()
    return model, processor


# === 2. Training Loader ===

def load_model_for_training(base_model_path, num_labels=5):
    """
    Load a Wav2Vec2 model and processor from a Hugging Face repo or a local path for training.
    Automatically detects local folder or remote repo.
    """
    if os.path.isdir(base_model_path):
        config = AutoConfig.from_pretrained(base_model_path)
        config.num_labels = num_labels
        config.problem_type = "single_label_classification"
        model = Wav2Vec2ForSequenceClassification.from_pretrained(base_model_path, config=config)
    else:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
    processor = Wav2Vec2Processor.from_pretrained(base_model_path)
    return model, processor


