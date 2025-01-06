import os
import torchaudio
import torch
import numpy as np
from pydub import AudioSegment
from torchaudio.transforms import Resample
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Paths
raw_data_dir = RAW_DATA_DIR
processed_data_dir = PROCESSED_DATA_DIR
os.makedirs(processed_data_dir, exist_ok=True)

def load_audio(file_path):
    """Load audio using pydub and return waveform and sample rate."""
    audio = AudioSegment.from_file(file_path)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())
    waveform = torch.tensor(samples).unsqueeze(0).float()
    return waveform, sr

def preemphasis_filter(waveform, alpha=0.95):
    """Apply a pre-emphasis filter to audio waveform."""
    emphasized = torch.cat(
        (waveform[:, 0:1], waveform[:, 1:] - alpha * waveform[:, :-1]), dim=1
    )
    return emphasized

def preprocess_audio(file_path, output_path, sample_rate=16000, target_duration_seconds=1):
    """Preprocess audio file to ensure 1-second duration with Wav2Vec 2.0 requirements."""
    waveform, sr = load_audio(file_path)

    # Resample if necessary
    if sr != sample_rate:
        resampler = Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if necessary
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Apply pre-emphasis filter
    waveform = preemphasis_filter(waveform)

    # Adjust duration
    target_samples = sample_rate * target_duration_seconds
    current_samples = waveform.size(1)

    if current_samples < target_samples:
        # Loop audio to make it at least 1 second
        loops = (target_samples // current_samples) + 1
        waveform = waveform.repeat(1, loops)[:, :target_samples]
    else:
        # Trim to exactly 1 second
        waveform = waveform[:, :target_samples]

    print(f"Processed duration: {waveform.size(1) / sample_rate:.2f} seconds")

    # Convert the waveform back to numpy for exporting
    processed_samples = waveform.squeeze().numpy().astype(np.int16)

    # Create a pydub AudioSegment
    processed_audio = AudioSegment(
        processed_samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  
        channels=1
    )

    # Export processed audio using pydub
    processed_audio.export(output_path, format="wav")
    print(f"Processed: {output_path}")

def preprocess_all():
    """Preprocess all raw audio files."""
    for file in os.listdir(raw_data_dir):
        if file.endswith(".wav"):
            input_file_path = os.path.join(raw_data_dir, file)
            output_file_path = os.path.join(processed_data_dir, file)
            preprocess_audio(input_file_path, output_file_path)

if __name__ == "__main__":
    preprocess_all()
