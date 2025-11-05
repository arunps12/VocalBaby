import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torchaudio
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import argparse

def preemphasis_filter(waveform, coeff=0.97):
    return waveform - coeff * np.roll(waveform, 1)

def MelSpectrogram(sample_rate, n_mels=128, n_fft=1024, hop_length=512):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

def train_test_melspectrograms_imgs(voc_dir, metadata_df_file_path, train_dir, test_dir, test_size=0.20, sample_rate=16000):
    # Create output directory if it doesn't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    df = pd.read_csv(metadata_df_file_path)
    df = df[(df['label'] != 'Unknown') & (df['label'] != 'ADS') & (df['label'] != 'IDS')]
    # Split files into train and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Iterate through train_df
    for audio_file in train_df['file_path']:
        try:
            # Load the audio
            audio_path = os.path.join(voc_dir, audio_file)
            waveform, sr = torchaudio.load(audio_path)

            # Downsample the audio to 16000 Hz
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            # Convert the audio to monochannel
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Apply pre-emphasis filter
            waveform = preemphasis_filter(waveform)

            # Adjust audio length
            duration = waveform.size(1) / sample_rate
            target_length = sample_rate  # Target length is 1 second

            if duration < 1:
                # Loop audio to make it 1 second
                loops = int(target_length / duration) + 1
                waveform = waveform.repeat(1, loops)[:, :target_length]
                # Compute the Mel spectrogram
                mel_spec = MelSpectrogram(sample_rate)(waveform)

                # Convert the Mel spectrogram to a numpy array
                mel_spec = mel_spec.squeeze().numpy()
                # Apply the 'jet' colormap
                mel_spec_jet = cm.jet(mel_spec)

                # Convert the NumPy array to a PIL image
                mel_spec_jet_image = Image.fromarray((mel_spec_jet * 255).astype(np.uint8))

                # Resize the image to 224x224 resolution
                mel_spec_jet_image = mel_spec_jet_image.resize((224, 224), Image.LANCZOS)

                # Save the image
                image_file = os.path.join(train_dir, audio_file.split('.wav')[0] + '.png')
                mel_spec_jet_image.save(image_file)

            elif duration > 1:
                # Split audio into 1-second segments
                for i in range(int(duration)):
                    segment = waveform[:, i * sample_rate : (i + 1) * sample_rate]
                    # Compute the Mel spectrogram
                    mel_spec = MelSpectrogram(sample_rate)(segment)

                    # Convert the Mel spectrogram to a numpy array
                    mel_spec = mel_spec.squeeze().numpy()
                    # Apply the 'jet' colormap
                    mel_spec_jet = cm.jet(mel_spec)

                    # Convert the NumPy array to a PIL image
                    mel_spec_jet_image = Image.fromarray((mel_spec_jet * 255).astype(np.uint8))

                    # Resize the image to 224x224 resolution
                    mel_spec_jet_image = mel_spec_jet_image.resize((224, 224), Image.LANCZOS)

                    # Save the image
                    image_file = os.path.join(train_dir, str(i) + '_' + audio_file.split('.wav')[0] + '.png')
                    mel_spec_jet_image.save(image_file)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

    # Iterate through test_df
    for audio_file in test_df['file_path']:
        try:
            # Load the audio
            audio_path = os.path.join(voc_dir, audio_file)
            waveform, sr = torchaudio.load(audio_path)

            # Downsample the audio to 16000 Hz
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            # Convert the audio to monochannel
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Apply pre-emphasis filter
            waveform = preemphasis_filter(waveform)

            # Adjust audio length
            duration = waveform.size(1) / sample_rate
            target_length = sample_rate  # Target length is 1 second

            if duration < 1:
                # Loop audio to make it 1 second
                loops = int(target_length / duration) + 1
                waveform = waveform.repeat(1, loops)[:, :target_length]
                # Compute the Mel spectrogram
                mel_spec = MelSpectrogram(sample_rate)(waveform)

                # Convert the Mel spectrogram to a numpy array
                mel_spec = mel_spec.squeeze().numpy()
                # Apply the 'jet' colormap
                mel_spec_jet = cm.jet(mel_spec)

                # Convert the NumPy array to a PIL image
                mel_spec_jet_image = Image.fromarray((mel_spec_jet * 255).astype(np.uint8))

                # Resize the image to 224x224 resolution
                mel_spec_jet_image = mel_spec_jet_image.resize((224, 224), Image.LANCZOS)

                # Save the image
                image_file = os.path.join(test_dir, audio_file.split('.wav')[0] + '.png')
                mel_spec_jet_image.save(image_file)

            elif duration > 1:
                # Split audio into 1-second segments
                for i in range(int(duration)):
                    segment = waveform[:, i * sample_rate : (i + 1) * sample_rate]
                    # Compute the Mel spectrogram
                    mel_spec = MelSpectrogram(sample_rate)(segment)

                    # Convert the Mel spectrogram to a numpy array
                    mel_spec = mel_spec.squeeze().numpy()
                    # Apply the 'jet' colormap
                    mel_spec_jet = cm.jet(mel_spec)

                    # Convert the NumPy array to a PIL image
                    mel_spec_jet_image = Image.fromarray((mel_spec_jet * 255).astype(np.uint8))

                    # Resize the image to 224x224 resolution
                    mel_spec_jet_image = mel_spec_jet_image.resize((224, 224), Image.LANCZOS)

                    # Save the image
                    image_file = os.path.join(test_dir, str(i) + '_' + audio_file.split('.wav')[0] + '.png')
                    mel_spec_jet_image.save(image_file)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process train and test audio files and save spec images.')
    parser.add_argument('voc_dir', type=str, help='The directory containing vocalization audio clips')
    parser.add_argument('metadata_df_file_path', type=str, help='The file path of csv file have all metadata')
    parser.add_argument('train_dir', type=str, help='The output directory containing train images')
    parser.add_argument('test_dir', type=str, help='The output directory containing test images')
    args = parser.parse_args()
    
    # Call the processing function with the provided arguments
    train_test_melspectrograms_imgs(args.voc_dir, args.metadata_df_file_path, args.train_dir, args.test_dir)

if __name__ == "__main__":
    main()
