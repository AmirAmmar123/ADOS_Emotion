import glob
import os
import pandas as pd
import shutil
import pathlib
from CheckCreate import check_create

def get_video_catagory(video):
    if 'emo' in video.lower():
        return 'EMO'
    if 'story' in video.lower() or 'book' in video.lower():
        return 'STORY'
    if 'TB' in video or 'tooth' in video.lower():
        return 'TB'
    if 'pen1' in video.lower() or 'first' in video.lower():
        return 'PEN#1'
    if 'pen2' in video.lower() or 'second' in video.lower():
        return 'PEN#2'
    if 'pen3' in video.lower():
        return 'PEN#3'
    if 'calib' in video.lower() or 'camera' in video.lower():
        return 'CALIBRATION'
    return None
def get_camera(name):
    name = name.split('.mp4')[0]
    if 'both' in name.lower() or 'B' ==name.split('_')[-1]:
        return 'both'
    if 'subject' in  name.lower()  or 'S' == name.split('_')[-1]:
        return 'subject'
    if 'marissa' in name.lower() or 'I'==name.split('_')[-1]:
        return 'marissa'
    print(name)
    input()
    return None
SOUCE_DIR = 'G:/My Drive/ADOS/'
OUTDIR = 'c:/video files for sort/'

check_create(OUTDIR)
video_files = glob.glob(SOUCE_DIR+"/**/*.mp4",recursive=True)
asaf_files = [vid for vid in video_files if 'asaf' in vid.lower()]
faisal_files = [vid for vid in video_files if 'faisal' in vid.lower()]
fileNames = [] 
subject_dict = {}
subject_sorted_dict = {}
subjects_ns = set()
for file in asaf_files+faisal_files:
    filePath = pathlib.Path(file)
    fileName = filePath.name
    subject_n = int(fileName.split('_')[0])
    subjects_ns.add(subject_n)
for sn in subjects_ns:
    subject_dict[sn] = []
for file in asaf_files+faisal_files:
    filePath = pathlib.Path(file)
    fileName = filePath.name
    subject_n = int(fileName.split('_')[0])
    subject_dict[subject_n].append(str(filePath))
for sn in subjects_ns:
    subject_sorted_dict[sn]  = {}
    subject_sorted_dict[sn]['EMO'] = {}
    subject_sorted_dict[sn]['TB'] = {}
    subject_sorted_dict[sn]['STORY'] = {}
    subject_sorted_dict[sn]['CALIBRATION'] = {}
    subject_sorted_dict[sn]['PENS'] = {'PEN#1':{},'PEN#2':{},'PEN#3':{}}
for sn in subjects_ns:
    s_videos = subject_dict[sn]
    for video_p in s_videos:
        video  = pathlib.Path(video_p).name
        if "#" in video:
            continue
        camera_type  = get_camera(video)
        task = get_video_catagory(video)
        if 'PEN' not in task:
            if camera_type in subject_sorted_dict[sn][task].keys():
                subject_sorted_dict[sn][task][camera_type] = [subject_sorted_dict[sn][task][camera_type]]
                subject_sorted_dict[sn][task][camera_type].append(video_p)
            else:
                subject_sorted_dict[sn][task][camera_type] = video_p
        else:
            if camera_type in subject_sorted_dict[sn]['PENS'][task].keys():
                subject_sorted_dict[sn]['PENS'][task][camera_type] = [subject_sorted_dict[sn]['PENS'][task][camera_type]]
                subject_sorted_dict[sn]['PENS'][task][camera_type].append(video_p)
            else:
                subject_sorted_dict[sn]['PENS'][task][camera_type] = video_p
for sn in subjects_ns:
    print(subject_sorted_dict[sn])
from pickle import dump
with open('dict of sorted videos 24.5.23.pickle','wb') as f:
    dump(subject_sorted_dict,f)


    
