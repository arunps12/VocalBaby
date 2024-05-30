# VisionInfantNet

## Prerequisites

Ensure you have the following installed:
- Python 3.10.5

## Requirements

First, to install the required Python packages, use the provided `install_requirements.py` script.

### Installing Requirements

1. Ensure you have `pip` installed.
2. Run the following command to install the required packages:
   ```sh
   python install_requirements.py
   ```
Then you need to run the script named `extracting_voc_chunks_metadata.py` designed to extract vocalization chunks from raw audio recordings and generate a metadata CSV file. The raw audio data is sourced from the [Infant and Adult Vocalizations in Home Audio Recordings dataset](https://figshare.com/articles/dataset/Infant_and_adult_vocalizations_in_home_audio_recordings/6108392), which is open source and provides both the raw audio and the timings of each vocalization chunk in corresponding text files.

The `extracting_voc_chunks_metadata.py` script performs the following tasks:
1. Reads the raw audio files and their corresponding timing text files.
2. Extracts vocalization chunks from the raw audio based on the provided timings.
3. Saves the extracted chunks into a specified directory.
4. Generates a metadata CSV file with the file names of the extracted chunks and their labels.
To run the script, run it from the command line with three string inputs specifying the directories for the raw audio, the output directory for vocalization chunks, and the metadata directory.

```sh
python extracting_voc_chunks_metadata.py <raw_audio_dir> <voc_chunks_dir> <metadata_dir>
```
## Vocalization Chunks Extraction and Mel-Spectrogram Image Generation
To split the vocalization data into train and test sets and generate mel spectrogram images for both, run the script from the command line with four inputs: the directory for the vocalization audio, the metadata CSV file path, and the directories to save the train and test images. The script also preprocesses the data by applying a preemphasis filter, converting the audio to mono, and downsampling it to 16000Hz. Also, the script ensures that all audio segments are exactly 1 second long: if an audio segment is less than 1 second, it repeats the segment until it reaches 1 second; if it is longer than 1 second, it splits the segment into multiple 1-second chunks.
 
```sh
python train_test_imgs.py <voc_dir> <metadata_df_file_path> <train_dir> <test_dir>
```
### Arguments
<voc_dir>: Directory containing the vocalization audio clips and their corresponding timing text files.
<metadata_df_file_path>: File path of the CSV file containing metadata.
<train_dir>: Directory where the training mel-spectrogram images will be saved.
<test_dir>: Directory where the testing mel-spectrogram images will be saved.
