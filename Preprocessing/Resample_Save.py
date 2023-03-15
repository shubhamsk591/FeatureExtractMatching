import os
import Resample
song_files = os.listdir('Test_song/')
for file in song_files:
    file_name="Test_song/"+file
    Resample.resample(file_name)
