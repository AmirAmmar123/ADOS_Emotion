import pandas as pd
from CheckCreate import check_create
import shutil
import os
import cv2 as cv
from moviepy.editor import *
import numpy as np
import tqdm

TEMP_VID_DIR = 'c:/temp_vid_dir/'
SOURCE_VID_PATH = 'G:/My Drive/ADOS/Videos/Unsorted/'
OUTPUT_DIR = 'c:/newClips/'

check_create(TEMP_VID_DIR)
check_create(OUTPUT_DIR)

dfs = pd.read_excel('c:/times.xlsx',['sync','segmentTimes','filenames'],dtype={'subject_N':str})
sync_df = dfs['sync']
times_df = dfs['segmentTimes']
filenames_df = dfs['filenames']

created = set()

for df in [sync_df,times_df,filenames_df]:
    df.set_index(list(df.columns)[0],inplace=True)
    df.index = df.index.map(str)
    print(df)

sync_df /=30 #move to seconds format, 30 fps

for index,row in tqdm.tqdm(times_df.iterrows(),position=1):

    subject_n = str(index).split('_')[0]
    if(int(subject_n) in range(236,251)):
        print('skip',index)
        continue
    sub_cam_fn = filenames_df.at[index,'cam_subject']
    both_cam_fn = filenames_df.at[index,'cam_both']
    mar_cam_fn = filenames_df.at[index,'cam_marissa']

    index = str(index)
    subject_dir = SOURCE_VID_PATH+subject_n+'/'

    sub_cam_src_vid_path = subject_dir+sub_cam_fn+'.mp4'
    both_cam_src_vid_path = subject_dir+both_cam_fn+'.mp4'
    mar_cam_src_vid_path = subject_dir+mar_cam_fn+'.mp4'

    sub_cam_vid_path = TEMP_VID_DIR+ sub_cam_fn+'.mp4'
    if not os.path.exists(TEMP_VID_DIR+ sub_cam_fn+'.mp4'):
        sub_cam_vid_path = shutil.copy(sub_cam_src_vid_path,TEMP_VID_DIR)

    both_cam_vid_path = TEMP_VID_DIR+ both_cam_fn+'.mp4'
    if not os.path.exists(TEMP_VID_DIR+ both_cam_fn+'.mp4'):
        both_cam_vid_path = shutil.copy(both_cam_src_vid_path,TEMP_VID_DIR)

    mar_cam_vid_path = TEMP_VID_DIR+ mar_cam_fn+'.mp4'
    if not os.path.exists(TEMP_VID_DIR+ mar_cam_fn+'.mp4'):
        mar_cam_vid_path = shutil.copy(mar_cam_src_vid_path,TEMP_VID_DIR)


    
    sub_videoclip = VideoFileClip(sub_cam_vid_path)
    both_videoclip = VideoFileClip(both_cam_vid_path)
    mar_videoclip = VideoFileClip(mar_cam_vid_path)

    for task in tqdm.tqdm(['pen1','pen2','TB','Story','Emotions','calibration'],position=2):
        
        task_start_time_str = row[task+' start']
        task_end_time_str = row[task+' end']

        if pd.isna(task_end_time_str) or pd.isna(task_end_time_str) or not task_end_time_str or not task_end_time_str:
            continue

        task_start_time = int(task_start_time_str.split(':')[0])*60 + int(task_start_time_str.split(':')[1])
        task_end_time = int(task_end_time_str.split(':')[0])*60 + int(task_end_time_str.split(':')[1])

        task_str = task
        if subject_n+task in created:
            task_str+='2'
        if not os.path.exists(OUTPUT_DIR+subject_n+'_'+task_str+'_both.mp4'):
            task_both_subclip = both_videoclip.subclip(task_start_time,task_end_time)
            task_both_subclip.write_videofile(OUTPUT_DIR+subject_n+'_'+task_str+'_both.mp4')
            task_both_subclip.close()
        if not os.path.exists(OUTPUT_DIR+subject_n+'_'+task_str+'_subject.mp4'):
            both_sub_diff = float(sync_df.at[index,'Subject Dark'])-float(sync_df.at[index,'Both Dark'])
            task_sub_subclip = sub_videoclip.subclip(task_start_time+both_sub_diff,task_end_time+both_sub_diff)
            task_sub_subclip.write_videofile(OUTPUT_DIR+subject_n+'_'+task_str+'_subject.mp4')
            task_sub_subclip.close()
        if not os.path.exists(OUTPUT_DIR+subject_n+'_'+task_str+'_marissa.mp4'):
            both_mar_diff = float(sync_df.at[index,'Marissa Dark'])-float(sync_df.at[index,'Both Dark'])
            task_mar_subclip = mar_videoclip.subclip(task_start_time+both_mar_diff,task_end_time+both_mar_diff)
            task_mar_subclip.write_videofile(OUTPUT_DIR+subject_n+'_'+task_str+'_marissa.mp4')
            task_mar_subclip.close()

        created.add(subject_n+task)
    sub_videoclip.close()
    both_videoclip.close()
    mar_videoclip.close()
    # os.remove(sub_cam_vid_path)
    # os.remove(both_cam_vid_path)
    # os.remove(mar_cam_vid_path)



