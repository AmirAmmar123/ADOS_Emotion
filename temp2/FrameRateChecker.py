from pickle import load
import cv2 as cv
import glob
import os
import tqdm
from VideoDictLoader import VideosDictLoader
VIDEO_FILES = VideosDictLoader.loadVideosDict()
for subject in tqdm.tqdm(VIDEO_FILES.keys()):
    if int(subject)<236:
        continue
    for task in VIDEO_FILES[subject].keys():
        if 'pen' in task.lower():
            continue
        for camera in VIDEO_FILES[subject][task].keys():
            if type(VIDEO_FILES[subject][task][camera]) is not list:
                vid_file = VIDEO_FILES[subject][task][camera]
                print('checking ',vid_file)
                video = cv.VideoCapture(vid_file)
                fps = video.get(cv.CAP_PROP_FPS)
                if fps!=30:
                    print(fps,vid_file)


                continue
            for vid in VIDEO_FILES[subject][task][camera]:
                vid_file = vid
                video = cv.VideoCapture(vid_file)
                fps = video.get(cv.CAP_PROP_FPS)
                if fps!=30:
                    print(fps,vid_file)