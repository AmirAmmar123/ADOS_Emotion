import glob
import os
import numpy as np
import cv2 as cv
import tqdm
from CheckCreate import check_create
from pathlib import Path
SOURCE_DIR = 'c:/new_vids_fps2/'
OUTPUT_DIR = 'c:/black_vids_new/'
check_create(OUTPUT_DIR)
vids = glob.glob(SOURCE_DIR+'*.mp4')
vids = [vid for vid in vids if 'both'  in vid.lower()]

for vid in vids:
    source_vid_path = Path(vid)
    if 'pen' in source_vid_path.name:
        continue
    source_vid_cap = cv.VideoCapture(vid)
    width  = source_vid_cap.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = source_vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    dim = (int(width), int(height))
    output_video_name_left = OUTPUT_DIR+source_vid_path.name.split('.')[0]+'_Subject.mp4'
    output_video_name_right = OUTPUT_DIR+source_vid_path.name.split('.')[0]+'_Marissa.mp4'

    out_vid_writer_left = cv.VideoWriter(output_video_name_left,fourcc, 30,dim)
    out_vid_writer_right = cv.VideoWriter(output_video_name_right,fourcc, 30,dim)

    # frame = glob.glob(VID_TEMP+'/*.png')
    sec = 0
    frameRate = 1/30 # Change this number to 1 for each 1 second
    
    fps = source_vid_cap.get(cv.CAP_PROP_FPS)
    if(fps==0):
        continue
    totalNoFrames = source_vid_cap.get(cv.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames / fps
    success = True
    color = (0,0,0)
    with tqdm.tqdm(total=int(durationInSeconds*30)) as pbar:
        count=0
        while success:
            count = count + 1
            if(count%10==0):
                pbar.update(10)
            sec = sec + frameRate
            success,image = source_vid_cap.read()
            if success:
                temp_image = image.copy
                temp_image_left  = cv.rectangle(image.copy(),(0,0),(int(image.shape[1]/2),int(image.shape[0])), color,-1)
                temp_image_right = cv.rectangle(image.copy(),(int(image.shape[1]/2),0),(int(image.shape[1]),int(image.shape[0])), color,-1)
                out_vid_writer_left.write(temp_image_left)
                out_vid_writer_right.write(temp_image_right)

    cv.destroyAllWindows()
    out_vid_writer_right.release()
    out_vid_writer_left.release()
    source_vid_cap.release()