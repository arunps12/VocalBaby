from pathlib import Path
import os
import pytest
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from collections import Counter
from vocalbaby.utils import load_metadata, create_child_disjunct_dataset, balance_dataset, CLASS_AUGMENTATIONS, SPLIT_COUNTS
from vocalbaby.preprocess import load_and_preprocess_audio, apply_center_padding
from vocalbaby.feature import extract_prosodic_features, prosody_to_sinusoid
from vocalbaby.pipeline import preprocess_example
from vocalbaby.labels import LABEL2ID, ID2LABEL
from transformers import Wav2Vec2Processor

@pytest.fixture(scope="module")
def example_and_processor():
    df = load_metadata()
    print("\nMetadata loaded with", len(df), "rows")
    assert not df.empty, "Metadata should not be empty"
    splits = create_child_disjunct_dataset(df, split_counts=SPLIT_COUNTS)

    print("\n Class distribution (before balancing):")
    for split in ["train", "validation", "test"]:
        labels = splits[split]["label"]
        counts = Counter(labels)
        #df_split = splits[split].to_pandas()
        #counts = Counter(df_split["label"])
        print(f"{split.capitalize():<12}:", dict(counts))

    train_df_balanced = balance_dataset(splits["train"].to_pandas())
    print("\n Class distribution (after balancing):")
    counts = Counter(train_df_balanced["label"])
    print(f"{'Train':<12}:", dict(counts))

    example = train_df_balanced.sample(n=1).iloc[0]
    print("\n Example path:", example["path"])
    print("\n Example label ID:", LABEL2ID[example["label"]])
    print("\n Example label:", example["label"])

    waveform = load_and_preprocess_audio(example["path"])
    sf.write("tests/raw.wav", waveform, samplerate=16000)

    waveform_padded, _ = apply_center_padding(waveform, 16000)
    sf.write("tests/waveform_padded.wav", waveform_padded, samplerate=16000)

    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    print("\n Pitch shape:", pitch.shape)
    print("Energy shape:", energy.shape)
    print("Pitch stats:", pitch.min(), pitch.max(), pitch.mean())
    print("Energy stats:", energy.min(), energy.max(), energy.mean())
    print("Pitch:", pitch)    
    print("Energy:", energy)
    prosody_signal = prosody_to_sinusoid(pitch, energy)
    sf.write("tests/prosody.wav", prosody_signal, samplerate=16000)
    prosody_signal_padded, _ = apply_center_padding(prosody_signal, 16000)
    sf.write("tests/prosody_padded.wav", prosody_signal_padded, samplerate=16000)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    return example, processor

def test_load_metadata_structure():
    df = load_metadata()
    assert not df.empty, "Metadata should not be empty"
    assert "path" in df.columns
    assert "label" in df.columns
    assert "child_id" in df.columns

def test_create_child_disjunct_dataset():
    df = load_metadata()
    splits = create_child_disjunct_dataset(df, split_counts=SPLIT_COUNTS)
    assert "train" in splits and "validation" in splits and "test" in splits
    train_ids = set(splits["train"]["child_id"])
    val_ids = set(splits["validation"]["child_id"])
    test_ids = set(splits["test"]["child_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

def test_label_valid(example_and_processor):
    example, _ = example_and_processor
    assert example["label"] in LABEL2ID

def test_audio_loading_and_padding(example_and_processor):
    example, _ = example_and_processor
    waveform = load_and_preprocess_audio(example["path"])
    print("\nWaveform shape:", waveform.shape)
    print("Waveform stats:", waveform.min(), waveform.max(), waveform.mean())
    print("Waveform:", waveform)
    waveform_padded, _ = apply_center_padding(waveform, 16000)
    assert isinstance(waveform_padded, np.ndarray)
    assert waveform_padded.ndim == 1
    assert len(waveform_padded) == 16000

def test_prosodic_feature_extraction(example_and_processor):
    example, _ = example_and_processor
    waveform = load_and_preprocess_audio(example["path"])
    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    assert isinstance(pitch, np.ndarray)
    assert isinstance(energy, np.ndarray)

def test_prosody_to_sinusoid(example_and_processor):
    example, _ = example_and_processor
    waveform = load_and_preprocess_audio(example["path"])
    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    signal = prosody_to_sinusoid(pitch, energy)
    print("\nProsody signal shape:", signal.shape)
    print("Prosody signal stats:", signal.min(), signal.max(), signal.mean())
    print("Prosody signal:", signal)
    signal, _ = apply_center_padding(signal, 16000)
    
    assert isinstance(signal, np.ndarray)
    assert signal.shape == (16000,)

def test_audio_augmentation_runs():
    sample_rate = 16000
    original = np.random.randn(sample_rate).astype(np.float32)

    for label, augment in CLASS_AUGMENTATIONS.items():
        augmented = augment(samples=original, sample_rate=sample_rate)
        assert isinstance(augmented, np.ndarray), f"{label} augmentation did not return ndarray"
        assert augmented.shape == original.shape, f"{label} augmentation changed shape"
        assert not np.isnan(augmented).any(), f"{label} augmentation introduced NaNs"

def test_preprocess_example_output(example_and_processor):
    from vocalbaby.pipeline import preprocess_example
    example, processor = example_and_processor
    out = preprocess_example(example, processor, max_length=16000)
    print("\nPreprocessed example output:", out)
    assert isinstance(out, dict)

    assert "input_values" in out
    assert out["input_values"].shape[0] == 16000

    assert "attention_mask" in out
    assert out["attention_mask"].shape[0] == 16000

    assert "prosody_signal" in out
    assert isinstance(out["prosody_signal"], np.ndarray)
    assert out["prosody_signal"].shape == (16000,)
    assert out["prosody_signal"].dtype == np.float32

    assert "labels" in out
    assert isinstance(out["labels"], int)

def test_processor_output(example_and_processor):
    example, processor = example_and_processor
    waveform = load_and_preprocess_audio(example["path"])
    waveform_padded,_ = apply_center_padding(waveform, 16000)
    inputs = processor(waveform_padded, sampling_rate=16000, return_tensors="pt", padding=True)

    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    prosody_signal = prosody_to_sinusoid(pitch, energy)
    prosody_signal, _ = apply_center_padding(prosody_signal, 16000)

    assert "input_values" in inputs
    assert inputs["input_values"].shape[1] == 16000
    assert inputs["input_values"].dtype == torch.float32
    assert isinstance(prosody_signal, np.ndarray)
    assert prosody_signal.shape == (16000,)
    assert prosody_signal.dtype == np.float32