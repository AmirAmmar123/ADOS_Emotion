import os
import Files
import cv2 

video_source_path = Files.get_file()
count = 0
vidcap = cv2.VideoCapture(video_source_path)
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

sec = 1366
frameRate = 1/30 # Change this number to 1 for each 1 second

vidcap.set(cv2.CAP_PROP_POS_MSEC,sec)
sucsses,frame = vidcap.read()
if sucsses:
    cv2.imshow('A',frame)
    cv2.waitKey(0)
while sucsses:
    sucsses,frame = vidcap.read()
    cv2.imshow('A',frame)
    cv2.waitKey(0)