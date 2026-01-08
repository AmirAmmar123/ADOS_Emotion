import os
import glob
from SplitToFrames import extract_image_one_fps
from CheckCreate import check_create
VIDEOS_SOURCE_DIR  = 'c:/new_videos/'
OUTPUT_FRAMES_ROOT = 'c:/new_frames/'
check_create(OUTPUT_FRAMES_ROOT)
for video_dir in os.listdir(VIDEOS_SOURCE_DIR):
    
    #### ONLY BECAUSE OF 233 STORY PROBLEM, Delete later
    if '233' not in video_dir:
        continue
    ###

    for video in glob.glob(VIDEOS_SOURCE_DIR+video_dir+'/*.mp4'):
        if 'both' not in video: 
            continue
        # print(video)
        if 'pen' in video:
            continue

        #### ONLY BECAUSE OF 233 STORY PROBLEM, Delete later
        if 'story' not in video:
            continue
        ###


        video_name = video.split('\\')[-1].split('.')[0]
        # print(video_name)
        if(int(video_name.split('_')[0])<228):
            continue
        cur_frames_dir = OUTPUT_FRAMES_ROOT+video_name+'/'
        if os.path.isdir(cur_frames_dir):
            continue
        print(cur_frames_dir)
        check_create(cur_frames_dir)
        extract_image_one_fps(video,cur_frames_dir)
