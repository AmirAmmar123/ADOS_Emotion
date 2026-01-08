from VideoCropper import *
import pandas as pd
import numpy as np
from pickle import load
from pathlib import Path
import os
def facial_landmarks():
    N_FACIAL_LNDMKS = 68
    N_EYES_LNDMKS = 56
    L_PUPIL_1 = 20
    L_PUPIL_2 = 28
    R_PUPIL_1 = 48
    R_PUPIL_2 = 56
    face_lndmks_cols = []
    x_face_lndmks_cols =[]
    x_eye_lndmks_cols=[]
    y_face_lndmks_cols=[]
    y_eye_lndmks_cols=[]
    for i in range(N_FACIAL_LNDMKS):
        x_str = 'x_'+str(i)
        y_str = 'y_'+str(i)
        x_face_lndmks_cols+=[x_str]
        y_face_lndmks_cols+=[y_str]
    eye_lndmks_cols = []
    for i in range(N_EYES_LNDMKS):
        x_str = 'eye_lmk_x_'+str(i)
        y_str = 'eye_lmk_y_'+str(i)
        # x_eye_lndmks_cols+=[x_str]
        # y_eye_lndmks_cols+=[y_str]
    return x_face_lndmks_cols+x_eye_lndmks_cols,y_face_lndmks_cols+y_eye_lndmks_cols

def get_min_max__in_columns(df:pd.DataFrame):

    """return the minimun value found in the dataframe and the max value\n
    ret_val = min,max
      """
    min,max = df.to_numpy().min(),df.to_numpy().max()
    return min,max

def check_each_key(subject_dict):
    flag = True
    for task in ['EMO','TB','STORY']:
        if len(list(subject_dict[task].keys()))!=3:
            flag=False
    return flag
with open('dict of sorted videos 24.5.23.pickle','rb') as f:
    VIDEO_FILES = load(f)

to_deal_later = []
X,Y = facial_landmarks()
# VIDEO_FILES = [229]
for subject in VIDEO_FILES:
    if subject not in range(224,228):
        continue
    print(subject)
    # subject = 235
    #1. check if has both+subject+marissa (len(dict)==3)
    subject_dict = VIDEO_FILES[subject]
    if not check_each_key(subject_dict):
        continue
    for task in ['TB','EMO']:
        task_subject_video_file_str_path = subject_dict[task]['subject']
        if(type(task_subject_video_file_str_path)==list):
            to_deal_later.append(task_subject_video_file_str_path)
            continue

        video_file_path= Path(task_subject_video_file_str_path)
        video_file_name = video_file_path.name.split('.')[0]
        if(os.path.exists('c:/croppedVids2/'+video_file_name+'_cropped.mp4')):
            continue

        openface_source = 'c:/openface_dired/'+str(subject)+'/'
         #2. for each task - get subject openface file 
        S_openFace_File  = openface_source +video_file_name+'.csv'
        S_openface = pd.read_csv(S_openFace_File).set_index('frame')

        cols = S_openface.columns
        new_cols = [col.strip() for col in cols]
        S_openface.columns = new_cols
        #3. find min max
        minx,maxx = get_min_max__in_columns(S_openface[X])
        miny,maxy = get_min_max__in_columns(S_openface[Y])
        #4. crop and save
        print((minx-20,maxx+20,miny-20,maxy+20))
        cropVideo(task_subject_video_file_str_path,(int(minx-20),int(maxx+20),int(miny-40),int(maxy+40)),'c:/croppedVids2/'+video_file_name+'_cropped.mp4')
