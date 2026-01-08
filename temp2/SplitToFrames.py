import cv2
import numpy as np
def getFPS_and_FrameCount(vid_file):
    video = cv2.VideoCapture(vid_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    totalNoFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.release()
    return fps,totalNoFrames
def extract_image_one_fps(video_source_path, output_dir_path):

    vidcap = cv2.VideoCapture(video_source_path)
    count = 0
    success,image = vidcap.read()
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = str(length)
    # while success:
    #   vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000/30))      
    #   success,image = vidcap.read()

    #   ## Stop when last frame is identified
    #   image_last = cv2.imread(output_dir_path+str(count-1).zfill(len(max_frame))+'.png')
    #   if np.array_equal(image,image_last):
    #       break
    #   print(str(count).zfill(len(max_frame)))
    #   cv2.imwrite(output_dir_path+str(count).zfill(len(max_frame))+'.png', image)     # save frame as PNG file
    # #   print '{}.sec reading a new frame: {} '.format(count,success)
    #   count += 1
    frames_buffer_count = 100
    frames_buffer = []
    buffer_counter = 0
    with tqdm.tqdm(total = length) as pbar:
        while success:
            #add to buffer
            frames_buffer.append((output_dir_path+str(count).zfill(len(max_frame))+'.png',image))
            buffer_counter+=1
            # cv2.imwrite(output_dir_path+str(count).zfill(len(max_frame))+'.png', image)     # save frame as JPEG file      
            success,image = vidcap.read()
        # print('Read a new frame: ', success)
            count += 1
            if(count%10==0):
                pbar.update(10)
            if buffer_counter == frames_buffer_count:
                #empty buffer 
                for path,frame in frames_buffer:
                    cv2.imwrite(path,frame)
                buffer_counter=0
                frames_buffer = []
    if buffer_counter>0:
        for path,frame in frames_buffer:
            cv2.imwrite(path,frame)
            buffer_counter=0
            frames_buffer = []
        # if count == 87:
        #     break
import tqdm
def robustSplitToFrames(video_source_path, output_dir_path):
    count = 0
    vidcap = cv2.VideoCapture(video_source_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    def getFrame(sec,max_frame = None):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(output_dir_path+str(count).zfill(len(str(max_frame)))+".png", image) # Save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = 1/30 # Change this number to 1 for each 1 second
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames / fps
    success = getFrame(sec,max_frame=int(durationInSeconds*30))
    frames_buffer_count = 100
    frames_buffer = []
    buffer_counter = 0
    total_frames = int(durationInSeconds*30)
    total_frames_str = str(total_frames)
    total_frames_len = len(total_frames_str)
    with tqdm.tqdm(total=total_frames) as pbar:
        while success:
            count = count + 1
            if count%10==0:
                pbar.update(10)
            sec = sec + frameRate
            # success = getFrame(sec,max_frame=int(durationInSeconds*30))
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            success,image = vidcap.read()
            if success:
                frames_buffer.append((output_dir_path+str(count).zfill(total_frames_len)+".png",image))
                buffer_counter+=1
                # cv2.imwrite(output_dir_path+str(count).zfill(len(str(int(durationInSeconds*30))))+".png", image)
            if buffer_counter == frames_buffer_count:
            #empty buffer 
                for path,frame in frames_buffer:
                    cv2.imwrite(path,frame)
                buffer_counter=0
                frames_buffer = []
    if buffer_counter>0:
        for path,frame in frames_buffer:
            cv2.imwrite(path,frame)
            buffer_counter=0
            frames_buffer = []


import os
def splitXFrame(vid_file:str,x:int,output_dir:str,check_exists = False):
    count = 0
    vidcap = cv2.VideoCapture(vid_file)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    hasFrames = True
    with tqdm.tqdm(total = x) as pbar:
        while hasFrames and (count<x):
            hasFrames,image = vidcap.read()
            if hasFrames:
                # if not check_exists:
                cv2.imwrite(output_dir+str(count).zfill(len(str(length)))+".jpg", image) # Save frame as JPG file
                count+=1
                pbar.update(1)
                # if check_exists:
                #     if not os.path.exists(output_dir+str(count).zfill(len(str(length)))+".png"):
                #         cv2.imwrite(output_dir+str(count).zfill(len(str(length)))+".png", image) # Save frame as JPG file
                #     count+=1
                #     pbar.update(1)

def robustSplitXFrame(vid_file:str,x:int,output_dir:str,check_exists = False,start_from = 0):
    count = 0
    vidcap = cv2.VideoCapture(vid_file)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    hasFrames = True
    fps = 30
    min_average = np.infty
    min_avg_arg = -1
    with tqdm.tqdm(total = x) as pbar:
        while hasFrames and (count<x):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(start_from+(1/fps)*count)*1000)
            hasFrames,image = vidcap.read()

            if hasFrames:
                if (np.average(image)<min_average):
                    min_avg_arg = count
                    min_average = np.average(image)
                # if not check_exists:
                image_copy = cv2.resize(image, (0,0), fx=0.20, fy=0.20,interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(output_dir+str(count).zfill(len(str(length)))+".bmp",image_copy) # Save frame as JPG file
                count+=1
                if(count%10==0):
                    pbar.update(10)
                # if check_exists:
                #     if not os.path.exists(output_dir+str(count).zfill(len(str(length)))+".png"):
                #         cv2.imwrite(output_dir+str(count).zfill(len(str(length)))+".png", image) # Save frame as JPG file
                #     count+=1
                #     pbar.update(1)
    return min_avg_arg

