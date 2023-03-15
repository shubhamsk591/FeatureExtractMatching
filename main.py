import os
import math
import numpy as np
import pandas as pd
import csv
import librosa


def mean_normalized(a):
    a = (a - a.min()) / (a.max() - a.min())
    a=np.mean(a)
    return a
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
    return tempo,final_value

#calculate euclidean distance
def euclidean_distance(x1,x2,y1,y2):
    x1=float(x1)
    y1=float(y1)
    x2=float(x2)
    y2=float(y2)
    n=max(x1,x2)
    x1=x1/n
    x2=x2/n
    return math.sqrt((x2 - x1)**2 + (y2 - y1) ** 2)
def covariance_distance(x1,x2,y1,y2):
    x1=float(x1)
    y1=float(y1)
    x2=float(x2)
    y2=float(y2)
    n=max(x1,x2)
    mean1=(x1+y1)/2
    mean2=(x2+y2)/2
    s1=((x1-mean1)*(x2-mean2))
    s2=((y1-mean1)*(y2-mean2))
    return s1+s2

def cosine_similarity(x1,x2,y1,y2):
    x1=float(x1)
    y1=float(y1)
    x2=float(x2)
    y2=float(y2)
    n=max(x1,x2)
    x1=x1/n
    x2=x2/n
    dot=(x1*x2)+(y1*y2)
    xa=math.sqrt((x1**2)+(y1**2))
    xb=math.sqrt((x2**2)+(y2**2))
    return dot/(xa*xb)

def get_value_from_index(index): 
	# open the csv file 
	with open('features_db.csv', 'r') as csvfile: 
		# read the csv file 
		csvreader = csv.reader(csvfile) 
		# get all the rows in the csv file 
		rows = list(csvreader) 
		# return the value from the specified index 
		return rows[index][1] 

Vector = [] 
Vector1 = [] 

# open the csv file
with open('features_db.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    
    # iterate over the rows in the csv
    for row in reader:
        Vector.append(row[9]) 
        Vector1.append(row[10])



song_files = os.listdir('Test/')
for file in song_files:
    file_name="Test/"+file
    a,b= extract_features(file,file_name)
    print(file)
    print("On basis of euclidean")
    distances = []
    for i in range(1,len(Vector1)):
         distances.append(euclidean_distance(a,Vector[i],b,Vector1[i]))
    output=[]
    for j in range(len(distances)):
        if(distances[j]<0.6):
             output.append([j+1,distances[j]])
        
    output.sort(key = lambda x : x[1])
    print(len(output))
    for i in range(len(output)):
         print(get_value_from_index(output[i][0]))
    print("\n")
    print("On basis of cosine")
    distances = []
    for i in range(1,len(Vector1)):
         distances.append(cosine_similarity(a,Vector[i],b,Vector1[i]))
   
    output=[]
    for j in range(len(distances)):
        if(distances[j]>0.9):
             output.append([j+1,distances[j]])
        
    output.sort(key = lambda x : x[1],reverse=True)
   
    for i in range(len(output)):
         print(get_value_from_index(output[i][0]))
    print("\n")
    print("On basis of covariance")
    distances = []
    for i in range(1,len(Vector1)):
         distances.append(covariance_distance(a,Vector[i],b,Vector1[i]))
    print(distances)
    output=[]
    for j in range(len(distances)):
        if(distances[j]>0.9):
             output.append([j+1,distances[j]])
        
    output.sort(key = lambda x : x[1])
    print(len(output))
    for i in range(len(output)):
         print(get_value_from_index(output[i][0]))