import os
import random
import torchaudio
import torch
import numpy as np
import pandas as pd
from pydub import AudioSegment
from torchaudio.transforms import Resample
import librosa

from config import RAW_DATA_DIR, PROCESSED_TRAIN_DATA_DIR, TRAIN_METADATA_PATH


# Paths
raw_data_dir = RAW_DATA_DIR
processed_data_dir = PROCESSED_TRAIN_DATA_DIR
os.makedirs(processed_data_dir, exist_ok=True)

# Load metadata with 'clip_ID' and 'VocLabel' columns
metadata = pd.read_csv(TRAIN_METADATA_PATH)

# Count number of samples per class
class_counts = metadata['VocLabel'].value_counts().to_dict()
max_class_count = max(class_counts.values())

# Define how many samples to augment
def get_augmentation_plan(label, current_count):
    augment_plan = {'pitch': 0, 'noise': 0, 'both': 0}
    if current_count < max_class_count:
        needed = max_class_count - current_count
        augment_plan['pitch'] = needed // 3
        augment_plan['noise'] = needed // 3
        augment_plan['both'] = needed - augment_plan['pitch'] - augment_plan['noise']
    elif current_count == max_class_count:
        augment_plan['pitch'] = current_count // 6
        augment_plan['noise'] = current_count // 6
        augment_plan['both'] = (current_count // 2) - (augment_plan['pitch'] + augment_plan['noise'])
    return augment_plan

# Augmentation functions with random values
def apply_pitch_shift(waveform, sample_rate):
    n_steps = random.uniform(-2.5, 2.5)
    y = waveform.squeeze().numpy()
    y_shifted = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=n_steps)
    return torch.tensor(y_shifted).unsqueeze(0)

def add_noise(waveform):
    noise_level = random.uniform(0.002, 0.007)
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())
    waveform = torch.tensor(samples).unsqueeze(0).float()
    return waveform, sr

def preemphasis_filter(waveform, alpha=0.95):
    emphasized = torch.cat((waveform[:, 0:1], waveform[:, 1:] - alpha * waveform[:, :-1]), dim=1)
    return emphasized

def preprocess_and_export(waveform, sr, output_path, sample_rate=16000, duration_sec=1):
    if sr != sample_rate:
        resampler = Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = preemphasis_filter(waveform)

    target_samples = sample_rate * duration_sec
    if waveform.size(1) < target_samples:
        loops = (target_samples // waveform.size(1)) + 1
        waveform = waveform.repeat(1, loops)[:, :target_samples]
    else:
        waveform = waveform[:, :target_samples]

    samples = waveform.squeeze().numpy().astype(np.int16)
    audio = AudioSegment(samples.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)
    audio.export(output_path, format="wav")

def preprocess_all():
    augment_counter = {}

    for _, row in metadata.iterrows():
        file = row['clip_ID']
        label = row['VocLabel']
        input_path = os.path.join(raw_data_dir, file)
        base_name = os.path.splitext(file)[0]

        output_path = os.path.join(processed_data_dir, file)
        waveform, sr = load_audio(input_path)
        preprocess_and_export(waveform, sr, output_path)

        if label not in augment_counter:
            augment_counter[label] = {'pitch': 0, 'noise': 0, 'both': 0}
        plan = get_augmentation_plan(label, class_counts[label])

        for aug_type in ['pitch', 'noise', 'both']:
            while augment_counter[label][aug_type] < plan[aug_type]:
                augmented_waveform = waveform.clone()
                if aug_type == 'pitch':
                    augmented_waveform = apply_pitch_shift(augmented_waveform, sr)
                elif aug_type == 'noise':
                    augmented_waveform = add_noise(augmented_waveform)
                elif aug_type == 'both':
                    augmented_waveform = apply_pitch_shift(augmented_waveform, sr)
                    augmented_waveform = add_noise(augmented_waveform)

                aug_filename = f"{base_name}_{aug_type}_{augment_counter[label][aug_type]}.wav"
                aug_path = os.path.join(processed_data_dir, aug_filename)
                preprocess_and_export(augmented_waveform, sr, aug_path)
                augment_counter[label][aug_type] += 1

if __name__ == "__main__":
    preprocess_all()
