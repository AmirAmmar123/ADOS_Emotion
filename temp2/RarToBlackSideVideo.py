import patoolib
import os
import os.path
from os import mkdir
from os.path import exists
import glob
import cv2
import shutil
import tqdm
black_frames_dir = 'c:/black_frames/'
black_videos_dir = 'c:/black_videos/'
reg_frames_dir = 'c:/reg_frames/'
frame_ext = '.png' #will change to png?
if not os.path.exists(black_frames_dir):
    os.mkdir(black_frames_dir)
if not os.path.exists(reg_frames_dir):
    os.mkdir(reg_frames_dir)
if not exists(black_videos_dir):
    mkdir(black_videos_dir)

zips = ['G:/My Drive/Bayan/Frames_zip/206_First_Pen_B.zip']
zips = glob.glob('F:/new_frames_zipps/*.zip')
print(zips)
input()
for zip_name in os.listdir('F:/new_frames_zipps/'):
    if '224' not in zip_name or 'emotions' in zip_name:
        continue
    #extract
    file_name = zip_name.split('.')[0]
    # print(file_name)
    # continue
    if 'camera' in file_name.lower() or 'desktop' in file_name.lower() or 'calib' in file_name.lower():
        continue
    zip_path = 'F:/new_frames_zipps/'+zip_name
    zip_output_path = reg_frames_dir+file_name
    black_base = black_out_dir = black_frames_dir+file_name
    video_base = black_videos_dir+file_name
    if exists(video_base+'Right.mp4') and exists(video_base+'Left.mp4'):
        print('skip ',file_name)
        continue
    if not exists(zip_output_path):
       patoolib.extract_archive(zip_path, outdir=zip_output_path)
    for side in ['Right','Left']:
        black_out_dir = black_base+side+'/'
        if not exists(black_out_dir):
            mkdir(black_out_dir)

    frames = glob.glob(zip_output_path+'/*'+frame_ext)
    frame_shape = None
    flag = True
    for frame in tqdm.tqdm(frames):
        frame_n = frame.split('\\')[-1].split('.')[0]
        if exists(black_out_dir+frame_n+frame_ext):
            if flag:
                image = cv2.imread(frame)   
                frame_shape = image.shape
                flag = False
            continue
        image = cv2.imread(frame)   
        frame_shape = image.shape
        color = (0, 0, 0)
        for side in ['Left','Right']:
            black_out_dir = black_base+side+'/'
            if side == 'Left':
                temp_image  = cv2.rectangle(image.copy(),(0,0),(int(image.shape[1]/2),int(image.shape[0])), color,-1)
            else:
                temp_image = cv2.rectangle(image.copy(),(int(image.shape[1]/2),0),(int(image.shape[1]),int(image.shape[0])), color,-1)
            cv2.imwrite(black_out_dir+frame_n+frame_ext,temp_image)

    #to video
    for side in ['Left','Right']:
        black_out_dir = black_base+side+'/'
        height, width, layers = frame_shape
        video_name = video_base+side+'.mp4'
        video = cv2.VideoWriter(video_name,0,30,(width,height))
        b_frames = glob.glob(black_out_dir+'/*'+frame_ext)
        for image in b_frames:
            # print(image)
            video.write(cv2.imread(image))
        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(black_out_dir)
        shutil.rmtree(zip_output_path)
