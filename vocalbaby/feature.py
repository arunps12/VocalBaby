import numpy as np
import librosa

def extract_prosodic_features(waveform, sr=16000, hop_length=256, n_fft=512):
    """
    Extract framewise pitch and energy features from a 1-second waveform.
    Assumes waveform is already 16000 samples (1 second).
    """
    # Framewise RMS Energy
    energy = librosa.feature.rms(y=waveform, hop_length=hop_length, frame_length=n_fft)[0]

    # Framewise Pitch using piptrack
    pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr, hop_length=hop_length, n_fft=n_fft)
    pitch = np.zeros(pitches.shape[1])
    for i in range(pitches.shape[1]):
        idx = magnitudes[:, i].argmax()
        pitch[i] = pitches[idx, i]

    return pitch, energy

def prosody_to_sinusoid(pitch, energy, sr=16000, hop_length=256, frame_length=512):
    num_frames = len(pitch)
    total_len = num_frames * hop_length
    signal = np.zeros(total_len + frame_length)  # overlap-add length

    for i in range(num_frames):
        f = pitch[i]
        a = energy[i]
        t = np.linspace(0, frame_length / sr, frame_length, endpoint=False)
        frame_signal = a * np.sin(2 * np.pi * f * t)
        start = i * hop_length
        signal[start:start + frame_length] += frame_signal  # Overlap-add

    # Final trimming/padding to match 1 second (16000 samples)
    signal = signal[:16000] 
    return signal.astype(np.float32)