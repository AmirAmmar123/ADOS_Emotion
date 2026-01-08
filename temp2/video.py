import tkinter as tk
from tkinter import filedialog
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_resize
from moviepy.editor import *
from pathlib import Path
from math import ceil
import os
def get_file():
    root = tk.Tk()
    root.withdraw()
    xl_file_path = filedialog.askopenfilename(title= "Please select a file:")
    return xl_file_path
def get_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title= "Please select a Directory:")
    return folder_path   
# Replace the filename below.
required_video_file = get_file()
clip = VideoFileClip(required_video_file)
duration = clip.duration
MAX_DURATION = 60*120-1
# with open("times.txt") as f:
#   times = f.readlines()

# times = [x.strip() for x in times] 
times = list()
path = Path(required_video_file)
num_of_parts = ceil(duration/MAX_DURATION)
for i in range(num_of_parts):
    times.append((i*MAX_DURATION,(i+1)*MAX_DURATION))
filename=path.name.split(".")[0]
dir = str(path.parent)+"\\"+filename
drive = dir.split(':')[0]
#if it's not on the computer, then choose the folder
if drive!='C':
    dir = get_folder()
if not os.path.exists(dir):
    os.mkdir(dir)
# times = [((1*60+15-(9651-3302)/1000),(38*60+26)-(9651-3302)/1000)]
# times = [((37*60+10 +(17932-2234365)/1000),(38*60+44+(17932-2234365)/1000))]
times = [((2*60+21)+(2840-9504)/1000 ,(3*60+13)+(2840-9504)/1000)]
# times = [(0,1*60)]
# times = [(((21546-(9651-3302))/1000),(((36546-(9651-3302))/1000)+0.5))]
# times = [(((21546-(9651-19015))/1000),(((36546-(9651-19015))/1000)+0.5))]

for time in times:
  starttime = time[0]
  endtime = time[1]
  output=dir+"\\"+filename+"_"+str(times.index(time)+1)+str(path.suffix)
  print(output)
  videoclip = VideoFileClip(required_video_file)
  videoclip = videoclip.subclip(starttime,endtime)
  videoclip.write_videofile(output)
#   ffmpeg_resize(required_video_file,output,(0.9,0.9))
os.startfile(dir)
