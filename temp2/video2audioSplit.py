import tkinter as tk
from tkinter import filedialog
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
from Files import *


def convert_video_to_audio_moviepy(video_file, output_ext="wav",output_folder=""):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    if output_folder is None:
        output_folder = str(Path(video_file).parent)
    filename1, ext = os.path.splitext(video_file)
    filename = video_file.split("/")[-1].split(".")[0]
    original = video_file
    target = output_folder+"/"+video_file.split("/")[-1]
    # shutil.copyfile(original,target)
    clip = VideoFileClip(video_file)
    output_file = output_folder+"/"+filename+"."+output_ext
    clip.audio.write_audiofile(output_file)
    # os.remove(target)
    return output_file

# def get_file():
#     root = tk.Tk()
#     root.withdraw()
#     xl_file_path = filedialog.askopenfilename(title= "Please select a file:")
#     return xl_file_path
# def get_folder():
#     root = tk.Tk()
#     root.withdraw()
#     folder_path = filedialog.askdirectory(title= "Please select a Directory:")
#     return folder_path   

# Replace the filename below.
if __name__ == "__main__":

    required_video_files = get_files()
    MAX_DURATION = 60*120-1
    # with open("times.txt") as f:
    #   times = f.readlines()

    # times = [x.strip() for x in times] 
    times = list()
    # path = Path(required_video_file)
    dir = get_folder()
    print("hey")
    for required_video_file in required_video_files:
        clip = VideoFileClip(required_video_file)
        print("hey")
        duration = clip.duration
        num_of_parts = ceil(duration/MAX_DURATION)
        for i in range(num_of_parts):
            times.append((i*MAX_DURATION,(i+1)*MAX_DURATION))
        drive = required_video_file.split(':')[0]
        #if it's not on the computer, then choose the folder
        audio_file=convert_video_to_audio_moviepy(required_video_file,output_folder=dir)
        # clip = AudioFileClip(audio_file)
        # path = Path(audio_file)
        # filename=path.name.split(".")[0]
        # dir = str(path.parent)+"\\"+filename
        # if not os.path.exists(dir):
        #     os.mkdir(dir)
        # for time in times:
        #   starttime = time[0]
        #   endtime = min((time[1],duration))
        #   output=dir+"\\"+str(times.index(time)+1)+str(path.suffix)
        #   print("hey")
        #   split_sound=clip.subclip(starttime, endtime)
        #   split_sound.write_audiofile(filename=output,bitrate='500k')
    os.startfile(dir)