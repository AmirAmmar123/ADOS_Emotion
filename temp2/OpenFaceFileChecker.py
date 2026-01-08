import glob
import shutil
import pandas as pd
import numpy as np
from VideoDictLoader import VideosDictLoader
import cv2 as cv
from pathlib import Path
import os
def get_basic_file_path(videosDict,subject,task,camera):
    subject_dict = videosDict[int(subject)]
    if 'pen' in task.lower():
        subject_dict = subject_dict['PENS']
        task = task.replace('pen','PEN#')
    return Path(subject_dict[task][camera]).name.split('.')[0],subject_dict[task][camera]
    
videosDict = VideosDictLoader.loadVideosDict()
subject_list = VideosDictLoader.get_gaze_subject_list()
for subject in subject_list:
    if subject not in range(224,228) :
        continue
    subject = str(subject)
    print(subject)
    openface_source = 'c:/openface_dired/'+subject+'/'
    for task in ['TB','EMO','pen1','pen2']:
        s_c = get_basic_file_path(videosDict,subject,task,'subject')
        S_openFace_File_Cropped  = openface_source +s_c[0]+'_cropped.csv'
        if not os.path.exists(S_openFace_File_Cropped):
            S_openFace_File_Cropped = openface_source +subject+'_'+task+'_'+'subject_cropped.csv'
        s = get_basic_file_path(videosDict,subject,task,'subject')
        S_openFace_File  = openface_source + s[0]+'.csv'

        if not os.path.exists(S_openFace_File):
            S_openFace_File = openface_source +subject+'_'+task+'_'+'subject.csv'
        b_s = get_basic_file_path(videosDict,subject,task,'both')
        B_openFace_File_s  = openface_source +b_s[0]+ '_Subject.csv'
        if not os.path.exists(B_openFace_File_s):
            B_openFace_File_s = openface_source +subject+'_'+task+'_'+'both_Subject.csv'
        b_m = get_basic_file_path(videosDict,subject,task,'both')
        B_openFace_File_m  = openface_source +b_m[0]+ '_Marissa.csv'
        if not os.path.exists(B_openFace_File_m):
            B_openFace_File_m = openface_source +subject+'_'+task+'_'+'both_Marissa.csv'
        m = get_basic_file_path(videosDict,subject,task,'marissa')
        M_openFace_File = openface_source +m[0]+ '.csv'
        if not os.path.exists(M_openFace_File):
            M_openFace_File = openface_source +subject+'_'+task+'_'+'marissa.csv'
        openface_files = [S_openFace_File,S_openFace_File_Cropped,B_openFace_File_s,B_openFace_File_m,M_openFace_File]
        dfs = [pd.read_csv(file) for file in openface_files]
        print(task)
        print('s,s_c,b_s,b_m,m')
        max_frame = [(of_df['frame'].max()) for of_df in dfs]
        min_max_frame = min(max_frame)
        print(max_frame)
        print(min_max_frame)
        # for i,video_origin in enumerate([s,s_c,b_s,b_m,m]):
        #     video = video_origin[1]
        #     # print()
        #     cap = cv.VideoCapture(video)
        #     # count the number of frames
        #     fps = cap.get(cv.CAP_PROP_FPS)
        #     totalNoFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        #     durationInSeconds = totalNoFrames / fps
        #     of_df = pd.read_csv(openface_files[i])
        #     # print(durationInSeconds*30)
        #     # print(of_df['frame'].max())
        #     if(int(durationInSeconds*30)!=int(of_df['frame'].max())):
        #         print('error', video_origin,int(durationInSeconds*30),int(of_df['frame'].max()) )
        #     cap.release()

            
