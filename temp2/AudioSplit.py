from Files import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from moviepy import *
from moviepy.editor import *
from pathlib import Path
from math import ceil
import subprocess
import os
import sys
import shutil
audio_file=get_file()
clip = AudioFileClip(audio_file)
MAX_DURATION = 60*1-1
# with open("times.txt") as f:
#   times = f.readlines()

# times = [x.strip() for x in times] 
times = list([(0,30),(30,60)])
# path = Path(required_video_file)
duration = clip.duration
num_of_parts = ceil(duration/MAX_DURATION)
for i in range(num_of_parts):
    times.append((i*MAX_DURATION,(i+1)*MAX_DURATION))
dir = get_folder()
for time in times:
  starttime = time[0]
  endtime = min((time[1],duration))
  output=dir+"\\"+str(times.index(time)+1)
  print("hey")
  split_sound=clip.subclip(starttime, endtime)
  split_sound.write_audiofile(filename=output+".wav")
os.startfile(dir)