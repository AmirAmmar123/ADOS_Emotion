import Files
import pandas as pd
from scipy.io.wavfile import read,write
import numpy as np
from video2audioSplit import convert_video_to_audio_moviepy

csv_file = Files.get_file('select csv speech file')
audio_file = Files.get_file('selet audio speech file')
if audio_file.split('.')[-1]=='mp4':
    audio_file = convert_video_to_audio_moviepy(audio_file,output_folder=None)
rate,data = read(audio_file)
speech_df = None
if csv_file.split('.')[-1]=='csv':
    speech_df = pd.read_csv(csv_file)
if csv_file.split('.')[-1]=='xlsx':
    speech_df = pd.read_excel(csv_file,engine='openpyxl')
if speech_df is None:
    print('speech file error')
    exit(1)
print(speech_df)
data = np.rot90(data)
new_data_l = []
diffs = 0
for index,row in speech_df.iterrows():
    print(row)
    if (row['Type']=='BOTH_TALKING'):
        continue
    start = int((int(row['Start'])/1000)*rate)
    end = int((int(row['End'])/1000)*rate)
    new_data_l.append(data[0,start:end])   
    diffs += int(row['End'])-int(row['Start'])
new_data = np.concatenate(new_data_l,axis=0)
write("testFais.wav",rate,new_data)
print(diffs/1000)