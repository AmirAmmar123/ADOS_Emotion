import pandas as pd
from CheckCreate import check_create
import shutil
import os
import cv2 as cv

TEMP_VID_DIR = 'c:/temp_vid_dir/'
SOURCE_VID_PATH = 'G:/My Drive/ADOS/Videos/Unsorted/'
check_create(TEMP_VID_DIR)
dfs = pd.read_excel('c:/times.xlsx',['sync','segmentTimes','filenames'])
sync_df = dfs['sync']
times_df = dfs['segmentTimes']
filenames_df = dfs['filenames']
for df in [sync_df,times_df,filenames_df]:
    df.set_index(list(df.columns)[0],inplace=True)

sync_df /=30 #move to seconds format, 30 fps

for index,row in times_df.iterrows():
    sub_cam_fn = filenames_df[index]['cam_subject']
    both_cam_fn = filenames_df[index]['cam_both']
    mar_cam_fn = filenames_df[index]['cam_marissa']

    subject_n = index.split('_')[0]
    subject_dir = SOURCE_VID_PATH+subject_n+'/'

    sub_cam_src_vid_path = subject_dir+sub_cam_fn+'.mp4'
    both_cam_src_vid_path = subject_dir+both_cam_fn+'.mp4'
    mar_cam_src_vid_path = subject_dir+mar_cam_fn+'.mp4'

    sub_cam_vid_path = shutil.copy(sub_cam_src_vid_path,TEMP_VID_DIR)
    both_cam_vid_path = shutil.copy(both_cam_src_vid_path,TEMP_VID_DIR)
    mar_cam_vid_path = shutil.copy(mar_cam_src_vid_path,TEMP_VID_DIR)

    sub_cam_vid_cap = cv.VideoCapture(sub_cam_vid_path)
    both_cam_vid_cap = cv.VideoCapture(both_cam_vid_path)
    mar_cam_vid_cap = cv.VideoCapture(mar_cam_vid_path)


    os.remove(sub_cam_vid_path)
    os.remove(both_cam_vid_path)
    os.remove(mar_cam_vid_path)



