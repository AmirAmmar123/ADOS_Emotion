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
OUTPUT_DIR = 'c:/MainFrameDir2/'
check_create(OUTPUT_DIR)
FPS = 30
SECONDS_FROM_BEGINING = 120
FINISHED_DICT_PICK = 'finished_dict_frame_spilts2'
FINISHED_DICT_AVG_PICK = 'finished_dict_frame_spilts2_avg'
finished_dict = {}
avg_dict = {}
if os.path.exists(FINISHED_DICT_PICK):
    with open(FINISHED_DICT_PICK,'rb') as f:
        finished_dict = load(f)    
if os.path.exists(FINISHED_DICT_AVG_PICK):
    with open(FINISHED_DICT_AVG_PICK,'rb') as f:
        avg_dict = load(f)
    pd.DataFrame.from_dict(avg_dict,orient='index').to_csv('avgs.csv')
for subject in range(244,246):
    subject_str = str(subject)
    subject_dir = SOURCE_DIR+subject_str+'/'
    subject_vid_files = glob.glob(subject_dir+'*.mp4')
    subject_output_dir = OUTPUT_DIR+subject_str+'/'
    check_create(subject_output_dir)
    for vid_file in subject_vid_files:
        if vid_file in finished_dict:
            if finished_dict[vid_file]:
                continue
        vid_file_path = Path(vid_file)
        vid_file_name = vid_file_path.name.split('.')[0]
        print(subject,vid_file_name)
        vid_file_dir = subject_output_dir+vid_file_name+'/'
        check_create(vid_file_dir)
        avg_dict[vid_file_path]=robustSplitXFrame(vid_file,FPS*SECONDS_FROM_BEGINING,vid_file_dir)
        finished_dict[vid_file] = True
        with open(FINISHED_DICT_PICK,'wb') as f:
            dump(finished_dict,f)
        with open(FINISHED_DICT_AVG_PICK,'wb') as f:
            dump(avg_dict,f)
        pd.DataFrame.from_dict(avg_dict,orient='index').to_csv('avgs.csv')
