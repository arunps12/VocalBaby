import os
import torch
import pandas as pd
import numpy as np
import librosa
import torchaudio
from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from datasets import Dataset, Audio
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
import sys
sys.path.append(os.path.abspath("src"))
from config import TEST_METADATA_PATH, MODELS_DIR, RAW_DATA_DIR

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration 
MODEL_DIR = MODELS_DIR
TEST_METADATA_PATH = TEST_METADATA_PATH  #  test metadata CSV path
TEST_AUDIO_DIR = RAW_DATA_DIR  #  test audio path


# Load model and feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# Label mapping
label_mapping = {"One": 0, "Two": 1, "Three": 2, "Four": 3, "Five": 4}
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# Load metadata
metadata = pd.read_csv(TEST_METADATA_PATH)
metadata["audio_path"] = metadata["clip_ID"].apply(lambda x: os.path.join(TEST_AUDIO_DIR, x))

# Preprocessing
def preprocess(waveform, sr, sample_rate=16000, duration=1):
    if sr != sample_rate:
        waveform = Resample(sr, sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_len = sample_rate * duration
    if waveform.shape[1] < target_len:
        repeat_factor = (target_len // waveform.shape[1]) + 1
        waveform = waveform.repeat(1, repeat_factor)[:, :target_len]
    else:
        waveform = waveform[:, :target_len]
    return waveform

# Augmentations
def pitch_shift(waveform_np, sr):
    return librosa.effects.pitch_shift(waveform_np, sr = sr, n_steps=np.random.uniform(-2, 2))

def add_noise(waveform_np):
    noise = np.random.randn(len(waveform_np)) * np.random.uniform(0.002, 0.006)
    return waveform_np + noise

# Prediction with augmentations
def predict_augmented(filepath, sample_rate=16000, duration=1):
    waveform, sr = torchaudio.load(filepath)
    waveform = preprocess(waveform, sr, sample_rate=sample_rate, duration=duration)
    base_np = waveform.squeeze().numpy()

    preds = []

    # Original
    inputs = feature_extractor(base_np, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    preds.append(torch.argmax(logits, dim=-1).item())

    # Pitch augmentation
    aug_pitch = pitch_shift(base_np, sample_rate)
    inputs = feature_extractor(aug_pitch, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    preds.append(torch.argmax(logits, dim=-1).item())

    # Noise augmentation
    aug_noise = add_noise(base_np)
    inputs = feature_extractor(aug_noise, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    preds.append(torch.argmax(logits, dim=-1).item())

    # Pitch + Noise augmentation
    aug_both = add_noise(pitch_shift(base_np, sample_rate))
    inputs = feature_extractor(aug_both, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    preds.append(torch.argmax(logits, dim=-1).item())

    # Majority voting
    majority_vote = Counter(preds).most_common(1)[0][0]
    return majority_vote


# Run predictions
predictions = []
for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    pred = predict_augmented(row["audio_path"])
    predictions.append(pred)

# Save results
metadata["predicted_int"] = predictions
metadata["predicted_label"] = metadata["predicted_int"].map(inv_label_mapping)
metadata.to_csv("test_predictions_augmented.csv", index=False)
print("Saved predictions to test_predictions_augmented.csv")
