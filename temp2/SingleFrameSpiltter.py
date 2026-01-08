import os
import cv2
from CheckCreate import check_create
import glob
import tqdm
from pathlib import Path
from SplitToFrames import *
import pandas as pd
from pickle import load,dump
SOURCE_DIR = 'G:/My Drive/ADOS/Videos/Unsorted/'
OUTPUT_DIR = 'c:/MainFrameDirSingle/'
check_create(OUTPUT_DIR)
FPS = 30
SECONDS_FROM_BEGINING = 75
subject_str = str(254)
subject_dir = SOURCE_DIR+subject_str+'/'
subject_vid_files = glob.glob(subject_dir+'*.mp4')
subject_output_dir = OUTPUT_DIR+subject_str+'/'
check_create(subject_output_dir)
vid_file = subject_dir + 'WIN_20230802_16_03_41_Pro.mp4'
vid_file_path = Path(vid_file)
vid_file_name = vid_file_path.name.split('.')[0]
# print(subject,vid_file_name)
vid_file_dir = subject_output_dir+vid_file_name+'/'
check_create(vid_file_dir)
print(robustSplitXFrame(vid_file,FPS*SECONDS_FROM_BEGINING,vid_file_dir))

