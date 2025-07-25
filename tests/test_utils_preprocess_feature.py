from pathlib import Path
import os
import pytest
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from collections import Counter
from vocalbaby.utils import load_metadata, create_child_disjunct_dataset, balance_dataset, augment
from vocalbaby.preprocess import load_and_preprocess_audio, apply_center_padding
from vocalbaby.feature import extract_prosodic_features, prosody_to_sinusoid
from vocalbaby.labels import LABEL2ID
from transformers import Wav2Vec2Processor

@pytest.fixture(scope="module")
def example_and_processor():
    df = load_metadata()
    splits = create_child_disjunct_dataset(df)

    print("\\n Class distribution (before balancing):")
    for split in ["train", "validation", "test"]:
        labels = splits[split]["label"]
        counts = Counter(labels)
        print(f"{split.capitalize():<12}:", dict(counts))

    train_df_balanced = balance_dataset(splits["train"].to_pandas())
    print("\\n Class distribution (after balancing):")
    counts = Counter(train_df_balanced["label"])
    print(f"{'Train':<12}:", dict(counts))

    example = train_df_balanced.sample(n=1).iloc[0]
    print("\\n Example path:", example["path"])
    print("\\n Example label ID:", LABEL2ID[example["label"]])
    print("\\n Example label:", example["label"])

    # Save waveforms
    waveform = load_and_preprocess_audio(example["path"])
    sf.write("tests/raw.wav", waveform, samplerate=16000)
    

    # Save prosody sinusoid
    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    prosody_signal = prosody_to_sinusoid(pitch, energy)
    sf.write("tests/prosody.wav", prosody_signal, samplerate=16000)

    #padded = apply_center_padding(prosody_signal, target_len=16000)
    #sf.write("tests/padded.wav", padded, samplerate=16000)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    return example, processor

def test_load_metadata_structure():
    from vocalbaby.utils import load_metadata
    df = load_metadata()
    assert not df.empty, "Metadata should not be empty"
    assert "path" in df.columns
    assert "label" in df.columns
    assert "child_id" in df.columns

def test_create_child_disjunct_dataset():
    from vocalbaby.utils import load_metadata, create_child_disjunct_dataset
    df = load_metadata()
    splits = create_child_disjunct_dataset(df)

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
    assert isinstance(waveform, np.ndarray)
    assert waveform.ndim == 1
    assert len(waveform) == 16000  #  1 second of audio

def test_prosodic_feature_extraction(example_and_processor):
    example, _ = example_and_processor
    waveform = load_and_preprocess_audio(example["path"])
    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    assert isinstance(pitch, np.ndarray)
    assert isinstance(energy, np.ndarray)
    #assert pitch.shape == (1,)
    #assert energy.shape == (1,)
    

def test_prosody_to_sinusoid(example_and_processor):
    example, _ = example_and_processor
    waveform = load_and_preprocess_audio(example["path"])
    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    signal = prosody_to_sinusoid(pitch, energy)
    #padded = apply_center_padding(signal, target_len=16000)    
    assert isinstance(signal, np.ndarray)
    assert signal.shape == (16000,)
    #assert len(padded) == 16000
    #assert isinstance(padded, np.ndarray)
    #assert padded.shape == (16000,)

def test_audio_augmentation_runs():
    from vocalbaby.utils import augment
    import numpy as np

    original = np.random.randn(16000).astype(np.float32)
    augmented = augment(samples=original, sample_rate=16000)

    assert isinstance(augmented, np.ndarray)
    assert augmented.shape == original.shape
    assert not np.isnan(augmented).any(), "Augmented audio has NaNs"


def test_processor_output(example_and_processor):
    example, processor = example_and_processor
    waveform = load_and_preprocess_audio(example["path"])
    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    signal = prosody_to_sinusoid(pitch, energy)
    #padded = apply_center_padding(signal, target_len=16000) 
    inputs = processor(signal, sampling_rate=16000, return_tensors="pt", padding=False)
    assert "input_values" in inputs
    assert inputs["input_values"].shape[1] == 16000
