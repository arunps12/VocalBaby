import os
from pydub import AudioSegment
import numpy as np
import wave
from config import PROCESSED_DATA_DIR


# Paths
processed_data_dir = PROCESSED_DATA_DIR

def validate_audio(file_path):
    """Validate a single audio file's properties."""
    try:
        # Open the audio file
        with wave.open(file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            duration = wav_file.getnframes() / sample_rate

            # Ensure the file is mono, 16 kHz, and ~1-15 seconds
            assert sample_rate == 16000, f"Sample rate is {sample_rate}, expected 16000"
            assert channels == 1, f"Channels are {channels}, expected 1"
            #assert 1 <= duration <= 15, f"Duration is {duration:.2f}s, expected 1-15s"
            assert duration == 1, f"Duration is {duration:.2f}s, expected 1s"

        # Check amplitude normalization
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples())
        max_amplitude = max(abs(samples))
        assert max_amplitude <= (2 ** (audio.sample_width * 8 - 1)), "Amplitude not normalized"

        print(f"Validated: {file_path}")
        return True

    except AssertionError as e:
        print(f"Validation failed for {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def validate_pipeline():
    """Validate all processed audio files."""
    all_valid = True
    for file in os.listdir(processed_data_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(processed_data_dir, file)
            if not validate_audio(file_path):
                all_valid = False

    if all_valid:
        print("All files passed validation!")
    else:
        print("Some files failed validation. Check logs above.")

if __name__ == "__main__":
    validate_pipeline()
