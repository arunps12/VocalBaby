import os
import csv
import textgrid
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
from config import MODELS_DIR, INPUT_test_audio_DIR

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Wav2Vec2 model and feature extractor
MODEL_PATH = MODELS_DIR
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH).to(device)


# Preprocessing Functions
def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    return waveform, sr


def preemphasis_filter(waveform, alpha=0.95):
    emphasized = torch.cat((waveform[:, 0:1], waveform[:, 1:] - alpha * waveform[:, :-1]), dim=1)
    return emphasized


def preprocess_audio(waveform, sr, sample_rate=16000):
    if sr != sample_rate:
        resampler = Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = preemphasis_filter(waveform)
    return waveform, sample_rate


def bandpass_filter(audio, lowcut, highcut, sr):
    """
    Apply a bandpass filter to the audio signal.
    """
    # Convert NumPy array to torch.Tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).to(device)

    # Apply high-pass and low-pass filters
    highpassed = torchaudio.functional.highpass_biquad(audio_tensor, sr, lowcut)
    bandpassed = torchaudio.functional.lowpass_biquad(highpassed, sr, highcut)

    return bandpassed.cpu().numpy()  # Convert back to NumPy array for librosa operations


def detect_voiced_segments_with_enhanced_noise_reduction(audio, sr, frame_length=1024, hop_length=256, energy_threshold=0.0001, lowcut=80, highcut=5000):
    """
    Detect voiced segments in an audio signal using RMS energy after noise reduction.
    """
    # Step 1: Apply Bandpass Filter
    filtered_audio = bandpass_filter(audio, lowcut, highcut, sr)

    # Step 2: Compute RMS Energy
    rms_energy = librosa.feature.rms(y=filtered_audio, frame_length=frame_length, hop_length=hop_length)[0]
    normalized_energy = rms_energy / np.max(rms_energy)

    # Step 3: Detect Voiced Frames
    voiced_frames = normalized_energy > energy_threshold

    # Step 4: Convert Voiced Frames to Time Ranges
    voiced_segments = []
    start_frame = None

    for i, is_voiced in enumerate(voiced_frames):
        if is_voiced and start_frame is None:
            start_frame = i
        elif not is_voiced and start_frame is not None:
            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            voiced_segments.append((start_time, end_time))
            start_frame = None

    if start_frame is not None:
        start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        end_time = librosa.frames_to_time(len(voiced_frames), sr=sr, hop_length=hop_length)
        voiced_segments.append((start_time, end_time))

    return voiced_segments


def extract_audio_segment(waveform, sr, start_time, end_time):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    return waveform[:, start_sample:end_sample]


def create_chunks_from_segment(segment, sr, chunk_duration=0.5):
    """
    Split a segment into 500 ms non-overlapping chunks.
    """
    chunk_samples = int(chunk_duration * sr)
    num_chunks = segment.size(1) // chunk_samples

    chunks = [segment[:, i * chunk_samples:(i + 1) * chunk_samples] for i in range(num_chunks)]
    return chunks


def get_chunk_intervals(segment_start_time, num_chunks, chunk_duration=0.5):
    """
    Get start and end times for each 500 ms chunk within a voiced segment.
    """
    chunk_intervals = []
    for i in range(num_chunks):
        start_time = segment_start_time + i * chunk_duration  # Start of the chunk
        end_time = start_time + chunk_duration  # End of the chunk
        chunk_intervals.append((start_time, end_time))
    return chunk_intervals


def classify_chunks(chunks, sr=16000):
    labels = []
    probabilities = []

    for chunk in chunks:
        chunk = preprocess_chunk(chunk, sample_rate=sr)
        inputs = feature_extractor(chunk.squeeze().numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()
            predicted_id = torch.argmax(probs).item()
            labels.append("Infant" if predicted_id == 0 else "Other")
            probabilities.append(probs.tolist())

    return labels, probabilities


def preprocess_chunk(chunk, sample_rate=16000, target_duration_seconds=1):
    target_samples = sample_rate * target_duration_seconds
    current_samples = chunk.size(1)

    if current_samples < target_samples:
        loops = (target_samples // current_samples) + 1
        chunk = chunk.repeat(1, loops)[:, :target_samples]
    else:
        chunk = chunk[:, :target_samples]

    return chunk


def save_textgrid(intervals, labels, probabilities, file_path, tier_name):
    new_tg = textgrid.TextGrid()
    new_tier = textgrid.IntervalTier(name=tier_name)

    last_end_time = 0.0

    for (start, end), label, prob in zip(intervals, labels, probabilities):
        adjusted_start = max(last_end_time, start)
        if end <= adjusted_start:
            continue

        annotation = f"{label} ({prob[0]:.2f}, {prob[1]:.2f})"
        new_tier.add(adjusted_start, end, annotation)
        last_end_time = end

    new_tg.append(new_tier)

    tg_file_path = f"{file_path}.TextGrid"
    with open(tg_file_path, "w", encoding="utf-8") as tg_file:
        new_tg.write(tg_file)

    print(f"TextGrid saved to {tg_file_path}")


def save_classification_summary(results, output_csv_path):
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["audio_file_name", "infant_count", "other_count", "total_chunks"])

        for result in results:
            writer.writerow([result["audio_file_name"], result["infant_count"], result["other_count"], result["total_chunks"]])

    print(f"Classification summary saved to {output_csv_path}")


def process_audio_pipeline(input_dir, output_csv_path):
    classification_results = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing : {file_path}")
                file_name = os.path.splitext(file)[0]
                tier_name = "Voiced Segments"

                waveform, sr = load_audio(file_path)
                waveform, sr = preprocess_audio(waveform, sr, sample_rate=16000)

                # Detect voiced segments
                voiced_segments_times = detect_voiced_segments_with_enhanced_noise_reduction(waveform.numpy().squeeze(), sr)

                if not voiced_segments_times:
                    print(f"No voiced segments detected in {file}. Skipping...")
                    continue

                labels, probabilities, all_intervals = [], [], []

                for start_time, end_time in voiced_segments_times:
                    segment = extract_audio_segment(waveform, sr, start_time, end_time)
                    chunks = create_chunks_from_segment(segment, sr)
                    if not chunks:
                        continue

                    segment_labels, segment_probs = classify_chunks(chunks, sr)

                    num_chunks = len(segment_labels)
                    chunk_intervals = get_chunk_intervals(start_time, num_chunks)

                    labels.extend(segment_labels)
                    probabilities.extend(segment_probs)
                    all_intervals.extend(chunk_intervals)

                infant_count = labels.count("Infant")
                other_count = labels.count("Other")
                total_chunks = len(labels)

                classification_results.append({
                    "audio_file_name": file,
                    "infant_count": infant_count,
                    "other_count": other_count,
                    "total_chunks": total_chunks
                })

                save_textgrid(all_intervals, labels, probabilities, os.path.join(root, file_name), tier_name)

    save_classification_summary(classification_results, output_csv_path)


# Input directory and output CSV path
INPUT_DIR = INPUT_test_audio_DIR
OUTPUT_CSV_PATH = "classification_summary.csv"

if __name__ == "__main__":
    process_audio_pipeline(INPUT_DIR, OUTPUT_CSV_PATH)
