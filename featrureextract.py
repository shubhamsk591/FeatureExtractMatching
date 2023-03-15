import os
import numpy as np
import librosa
import pandas as pd
def mean_normalized(a):
    a = (a - a.min()) / (a.max() - a.min())
    a=np.mean(a)
    return a
#Defining the feature extraction function
def extract_features(file,file_name):
    # Load the audio file
    audio, sample_rate = librosa.load(file_name)

    # Extract beats per minutes
    tempo,beat = librosa.beat.beat_track(y=audio,sr=sample_rate)
    
    mfcc=librosa.feature.mfcc(y=audio, sr=sample_rate)
    mfcc=mean_normalized(mfcc)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_centroid=mean_normalized(spectral_centroid)
    chroma_frequencies = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_frequencies=mean_normalized(chroma_frequencies)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spectral_contrast=mean_normalized(spectral_contrast)
    tonal_centroid_features = librosa.feature.tonnetz(y=audio, sr=sample_rate)
    tonal_centroid_features=mean_normalized(tonal_centroid_features)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_rate=mean_normalized(zero_crossing_rate)
    final_value=np.array([mfcc,spectral_centroid,chroma_frequencies,spectral_contrast,tonal_centroid_features,zero_crossing_rate])
    final_value=mean_normalized(final_value)
    # Store the extracted features into a database
    t={'name':file,'path':file_name,'mfcc':mfcc,'spectral_centroid':spectral_centroid,'chroma_frequencies':chroma_frequencies,
       'spectral_contrast':spectral_contrast,'tonal_centroid_features':tonal_centroid_features,'zero_crossing_rate':zero_crossing_rate,
       'no_beat':tempo,'final_value':final_value}
    features = pd.DataFrame(data = t, index=[0])
    return features


# Get the list of all the song files in folder
song_files = os.listdir('Data/Test_song/')

# Initialize the database
features_db = pd.DataFrame()

# Iterate over all the songs and extract the features
for file in song_files:
   file_name="Data/Test_song/"+file
   features = extract_features(file,file_name)
   features_db = features_db.append(features, ignore_index = True)
# Store the database
features_db.to_csv('features_db.csv')