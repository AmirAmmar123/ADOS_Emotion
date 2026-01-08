import os
import shutil
import glob
import subprocess
import time
from CheckCreate import check_create
from pickle import load
def get_camera_params(subject,camera):
    MetaData = 'c:/MetaData/'
    subject_meta_dir = MetaData+subject+'/openCV/'
    if not os.path.exists(subject_meta_dir):
        return ""
    with open(subject_meta_dir+'s_b_stereoParams','rb') as f:
        s_b_stereo_params =  load(f)
    iMatS,dc_S,iMatB,dc_B,R_B,T_B = s_b_stereo_params
    command = f'-fx {iMatB[0,0]} -fy {iMatB[1,1]} -cx {iMatB[0,2]} -cy {iMatB[1,2]}'
    print(command)
    exit(1)
    return command
VIDEOS_DIR = 'f:/black_videos/'
TEMP_DIR = 'c:/temp_vid_dir/'
check_create(TEMP_DIR)
OPENFACE_SOURCE = 'C:/Users/cs-lab/Desktop/OpenFace_2.2.0_win_x64/'
OPENFACE_EXE = 'FeatureExtraction.exe'
last = -1
videos = glob.glob(VIDEOS_DIR+'**/*.mp4',recursive=True)
# print(videos)
start = time.time()
for video in videos:
    # print(video.split('\\')[-1])
    # shutil.copy(video,TEMP_DIR)
    new_path = TEMP_DIR+video.split('\\')[-1]
    if 'calib' in video:
        continue
    # if 'TB' not in video:
    #     continue
    video_name = video.split('\\')[-1].split('.')[0].replace('Left',"_Subject").replace('Right','_Marissa')
    current = video_name.split('_')[0]
    if last!=-1:
        if last != current:
            end = time.time()
            # print('time passed for subject: ', end - start)
            start = end
    else:
        last = current 
    openface_command = OPENFACE_SOURCE+OPENFACE_EXE+' -f "'+new_path+'" -gaze -2Dfp -3Dfp -pose -tracked -of '+video_name+'.mp4 -multi-view 1 -wild '+get_camera_params(current,None)
    # if os.path.exists('processed/'+video_name+'.avi'):
    #     continue
    print(openface_command)
    subprocess.run(openface_command)
    os.remove(new_path)
    # break
