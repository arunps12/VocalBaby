import os

import pydub
from pydub import AudioSegment

import numpy as np
import pandas as pd

import argparse

def audio_split_and_metadata(raw_audio_dir, voc_chunks_dir, metadata_dir):
    # Create the dir to store extracted vocalization chunks
    if not os.path.exists(voc_chunks_dir):
        os.makedirs(voc_chunks_dir)
    
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    
    for file in os.listdir(raw_audio_dir):
        if file.endswith('.txt'): # reading only txt files
            print(file)
            file_path = os.path.join(raw_audio_dir, file) # Raw audio file path
            df = pd.read_csv(file_path, sep = '\t') # loading txt file info in DataFrame
            df.fillna('', inplace=True)
            df['classes'] = df.iloc[:, -2] + df.iloc[: , -1] # Voc Class name
            # Create a DICT to map voc classes
            Class_name = {'T': 'IDS', 'N': 'ADS', 'U': 'Unknown', 'C': 'Canonical', 'X': 'NonCanonical', 'R': 'LaughCry', 'V': 'Other'}
            df['classes'] = df['classes'].map(Class_name)
            df['classes'].fillna('Other', inplace=True)
            for index, _ in df.iterrows():
                #print(index)
                t1 = df.loc[index, 'Adjusted Begin Time - ss.msec'] # Strart time of voc clip
                t2 = df.loc[index, 'Adjusted End Time - ss.msec'] # end time of voc clip
                lable = df.loc[index, 'classes']
     
                t1 = t1 * 1000
                t2 = t2 * 1000
                raw_audio_file_path = in_path = os.path.join(file_path.split('/')[0], file_path.split('.txt')[0].split('/')[-1] + '.wav')
                new_audio = AudioSegment.from_wav(raw_audio_file_path) # create a audio object
                new_audio = new_audio[t1:t2] # extracting voc clips
                voc_chunks_file_path = os.path.join(voc_chunks_dir, file_path.split('\\')[-1].split('.txt')[0] + '_' + str(index) + '_' + lable + '.wav')
                new_audio.export(voc_chunks_file_path, format = 'wav') #Exports to a voc clip audio file in the out path.
            #create metadata df
            vocaud = []
            classes = []
            for file in os.listdir(voc_chunks_dir):
                vocaud.append(file)
                label = file.split('.wav')[0].split('_')[-1]
                #if (label == 'ADS') | (label == 'IDS'):
                    #classes.append('Adult')
                if label == 'LaughCry':
                    classes.append('Other')
                else:
                    classes.append(label)
            voc_df = pd.DataFrame(vocaud, columns=['file_path'])
            voc_df['label'] = classes
            metadata_path = os.path.join(metadata_dir, 'metadata.csv')
            voc_df.to_csv(metadata_path)
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process raw audio files and extract voc clips audio files.')
    parser.add_argument('raw_audio_dir', type=str, help='The directory containing raw audio files')
    parser.add_argument('voc_chunks_dir', type=str, help='The directory containing output voc chunks audio files')
    parser.add_argument('metadata_dir', type=str, help='The output directory containing csv file have all metadata')
    
    args = parser.parse_args()
    
    # Call the processing function with the provided arguments
    audio_split_and_metadata(args.raw_audio_dir, args.voc_chunks_dir, args.metadata_dir)

if __name__ == "__main__":
    main()