from SplitToFrames import *
from pickle import load
import cv2 as cv
import pandas as pd
from VideoDictLoader import VideosDictLoader
from SplitToFrames import *
from CheckCreate import *
import glob
import pathlib
import tqdm
with open('dict of sorted videos 24.5.23.pickle','rb') as f:
    VIDEO_FILES = load(f)
VIDEO_FILES =VideosDictLoader.loadVideosDict()
from pickle import dump,load
PICK_FILE = 'to_fix_list'
to_fix_list = []
# if os.path.exists(PICK_FILE):
#     with open(PICK_FILE,'rb') as f:
#         to_fix_list = load(f)
if not True:
    print('Hey')
else:
    for subject in range(224,228):
    # print(VIDEO_FILES[subject])
        subject_dict = VIDEO_FILES[subject]
        for key in subject_dict:
            if 'pen' in key.lower() or 'story' in key.lower():
                continue
            print(subject_dict)
            for key2 in subject_dict[key]:
                vid_file = subject_dict[key][key2]
                print(vid_file)
                video = cv.VideoCapture(vid_file)
                fps = video.get(cv.CAP_PROP_FPS)
                if fps!=30:
                    to_fix_list.append(vid_file)
                video.release()
            subject_dict = VIDEO_FILES[subject]
    for subject in range(224,228):
    # print(VIDEO_FILES[subject])
        subject_dict = VIDEO_FILES[subject]['PENS']
        for key in subject_dict:
            for key2 in subject_dict[key]:
                vid_file = subject_dict[key][key2]
                print(vid_file)
                video = cv.VideoCapture(vid_file)
                fps = video.get(cv.CAP_PROP_FPS)
                if fps!=30:
                    to_fix_list.append(vid_file)
    with open(PICK_FILE,'wb') as f:
        print(to_fix_list)
        dump(to_fix_list,f)
VID_TEMP = 'c:/temp_vid_frames/'
OUT_DIR = 'c:/new_vids_fps2/'
check_create(OUT_DIR)
import os
to_fix_list2 = [fxf for fxf in to_fix_list if 'calib' not in fxf]
print(len(to_fix_list2))
for fxf in to_fix_list2:
    # if 'calib' in fxf:
    #     continue
    print(fxf)
for fix_video_file in tqdm.tqdm(to_fix_list,position=0, desc="vid_file", leave=False, colour='green', ncols=80):
    fvxp = pathlib.Path(fix_video_file)    
    if 'calib' in fvxp.name:
        continue
    if not os.path.exists( 'c:/vids_temp/'+fvxp.name):
       shutil.copy(fix_video_file,'c:/vids_temp/')
    vid_file_path = pathlib.Path('c:/vids_temp/'+fvxp.name)
    # vid_file_path = pathlib.Path(fix_video_file)
    # print(vid_file_path.name)
    output_video_name =OUT_DIR+vid_file_path.name
    # if '224' in  vid_file_path.name:
        # continue
    # check_create(VID_TEMP)
    # robustSplitToFrames(fix_video_file,VID_TEMP)
    if os.path.exists(output_video_name):
        continue
    fix_vid_cap = cv.VideoCapture(fix_video_file)
    width  = fix_vid_cap.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = fix_vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    dim = (int(width), int(height))
    out = cv2.VideoWriter(output_video_name,fourcc, 30,dim)
    # frame = glob.glob(VID_TEMP+'/*.png')
    sec = 0
    frameRate = 1/30 # Change this number to 1 for each 1 second
    
    fps = fix_vid_cap.get(cv2.CAP_PROP_FPS)
    if(fps==0):
        continue
    totalNoFrames = fix_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames / fps
    success = True
    with tqdm.tqdm(total=int(durationInSeconds*30)) as pbar:
        count=0
        while success:
            count = count + 1
            if(count%10==0):
                pbar.update(10)
            sec = sec + frameRate
            fix_vid_cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = fix_vid_cap.read()
            if hasFrames:
                out.write(image)
            success = hasFrames
                # cv2.imwrite(output_dir_path+str(count).zfill(len(str(max_frame)))+".png", image) # Save frame as JPG file
    # for image in tqdm.tqdm(frame,position=1, desc="frames", leave=False, colour='red', ncols=80):
        # print(image)
    cv2.destroyAllWindows()
    out.release()
    fix_vid_cap.release()
    # check_delete(VID_TEMP)
    # exit(1)
