import numpy as np
import librosa

def extract_prosodic_features(waveform, sr):
    energy = librosa.feature.rms(y=waveform)[0]
    pitch, _ = librosa.piptrack(y=waveform, sr=sr)
    pitch = pitch[pitch > 0]
    pitch = np.mean(pitch) if pitch.size > 0 else 0
    energy = np.mean(energy) if energy.size > 0 else 0
    return np.array([pitch, energy])

def prosody_to_sinusoid(prosody_features, duration=1.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration))
    pitch, energy = prosody_features
    signal = energy * np.sin(2 * np.pi * pitch * t)
    return signal.astype(np.float32)
