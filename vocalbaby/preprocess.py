import numpy as np
import librosa

def load_and_preprocess_audio(path, preemphasis=0.95, sample_rate=16000):
    waveform, sr = librosa.load(path, sr=None, mono=True)

    if sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)

    # Apply preemphasis
    #waveform = np.append(waveform[0], waveform[1:] - preemphasis * waveform[:-1])

    # Repeat waveform to ensure length of 1 second (16000 samples)
    #if len(waveform) < sample_rate:
        #num_repeats = int(np.ceil(sample_rate / len(waveform)))
        #waveform = np.tile(waveform, num_repeats)

    # Truncate to exactly 1 second
    #waveform = waveform[:sample_rate]

    return waveform


def compute_padding_length(waveforms):
    return max(len(wf) for wf in waveforms)

def apply_center_padding(waveform, target_len):
    L = len(waveform)
    pad_total = target_len - L
    if pad_total <= 0:
        return waveform[:target_len]
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded = np.pad(waveform, (pad_left, pad_right), mode='constant')

    # Create attention mask
    attention_mask = np.concatenate([
        np.zeros(pad_left),
        np.ones(L),
        np.zeros(pad_right)
    ])

    return padded, attention_mask