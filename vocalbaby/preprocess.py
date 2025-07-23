import numpy as np
import librosa

def load_and_preprocess_audio(path, preemphasis=0.95, sample_rate=16000):
    waveform, sr = librosa.load(path, sr=None, mono=True)
    if sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
    waveform = np.append(waveform[0], waveform[1:] - preemphasis * waveform[:-1])
    return waveform

def compute_padding_length(waveforms):
    return max(len(wf) for wf in waveforms)

def apply_center_padding(waveform, target_len):
    pad_total = target_len - len(waveform)
    if pad_total <= 0:
        return waveform[:target_len]
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(waveform, (pad_left, pad_right), mode='constant')
