import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

def load_model_for_inference(repo_id_or_path):
    """Load Wav2Vec2 model and processor for CLI/inference."""
    processor = Wav2Vec2Processor.from_pretrained(repo_id_or_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(repo_id_or_path)
    model.eval()
    return model, processor

def load_model_for_training(base_model_path, num_labels=5):
    """Load model and processor for fine-tuning."""
    processor = Wav2Vec2Processor.from_pretrained(base_model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    return model, processor
