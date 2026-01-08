import numpy as np
import cv2

# Open the video
def cropVideo(source_video_path,crop_cor,output_video_path,fps = 30):
    """source_video_path - str path to video file\n
    crop_cor - tupple of 4 values, x,y,h,w  - x,y\n
    output_video_path - str path to output file\n
    fps - 30 by default
    """
    cap = cv2.VideoCapture(source_video_path)

    # Initialize frame counter
    cnt = 0

    # Some characteristics from the original video
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Here you can define your croping values
    minx,maxx,miny,maxy = crop_cor
    # if maxx > h_frame:
    #     maxx = h_frame
    # if maxy > w_frame:
    #     maxy = w_frame
    x = minx
    y = miny
    h = maxy-y
    w = maxx-x
    x = max(0,x)
    y = max(0,y)
    if h>w: 
        w = h
    else:
        h = w
    # output
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    scale_percent = w_frame/w# percent of original size
    width = int(w * scale_percent/2)
    height = int(h * scale_percent/2)
    dim = (width, height)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, dim)

    # Now we start
    while(cap.isOpened()):
        ret, frame = cap.read()
        # cv2.imshow('croped',frame)
        # cv2.waitKey()
        cnt += 1 # Counting frames

        # Avoid problems when video finish
        if ret==True:
            # Croping the frame
            crop_frame = frame[y:y+h,x:x+w]     
            # cv2.imshow('croped',crop_frame)
            # cv2.waitKey()
            # scale_percent = w_frame/crop_frame.shape[1] # percent of original size
            # width = int(crop_frame.shape[1] * scale_percent/2)
            # height = int(crop_frame.shape[0] * scale_percent/2)
            # dim = (width, height)
            crop_frame = cv2.resize(crop_frame,dim,interpolation=cv2.INTER_CUBIC)
            # Percentage
            xx = int(cnt *100/frames)
            if(xx%10==0):
                print(int(xx),'%')

            # Saving from the desired frames
            #if 15 <= cnt <= 90:
            #    out.write(crop_frame)

            # I see the answer now. Here you save all the video
            out.write(crop_frame)

            # Just to see the video in real time          
            # cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()