import os
import shutil
import glob
import subprocess
import time
from CheckCreate import check_create
from pickle import load
from pathlib import Path
def get_camera_params(subject,camera = 'subject'):
    MetaData = 'c:/MetaData/'
    subject_meta_dir = MetaData+subject+'/openCV/'
    if not os.path.exists(subject_meta_dir):
        return ""
    if camera == 'subject':
        if not os.path.exists(subject_meta_dir+'s_b_stereoParams'):
            return ""
        with open(subject_meta_dir+'s_b_stereoParams','rb') as f:
            s_b_stereo_params =  load(f)
        iMatS,dc_S,iMatB,dc_B,R_B,T_B = s_b_stereo_params
        command = f'-fx {iMatS[0,0]} -fy {iMatS[1,1]} -cx {iMatS[0,2]} -cy {iMatS[1,2]}'
        return command
    if camera == 'both':
        if not os.path.exists(subject_meta_dir+'s_b_stereoParams'):
            return ""
        with open(subject_meta_dir+'s_b_stereoParams','rb') as f:
            s_b_stereo_params =  load(f)
        iMatS,dc_S,iMatB,dc_B,R_B,T_B = s_b_stereo_params
        command = f'-fx {iMatB[0,0]} -fy {iMatB[1,1]} -cx {iMatB[0,2]} -cy {iMatB[1,2]}'
        return command
    if camera == 'marissa':
        if not os.path.exists(subject_meta_dir+'m_b_stereoParams'):
            return ""
        with open(subject_meta_dir+'m_b_stereoParams','rb') as f:
            m_b_stereo_params =  load(f)
        iMatM,dc_S,iMatB,dc_B,R_B,T_B = m_b_stereo_params
        command = f'-fx {iMatM[0,0]} -fy {iMatM[1,1]} -cx {iMatM[0,2]} -cy {iMatM[1,2]}'
        return command
    return ""

with open('dict of sorted videos 24.5.23.pickle','rb') as f:
    videos_dict = load(f)
# for subject_n in videos_dict.keys():
#     print(videos_dict[subject_n]['CALIBRATION'])
# exit(1)    
# for subject_n in videos_dict.keys():
#     for pen in videos_dict[subject_n]['PENS']:
#         if videos_dict[subject_n]['PENS'][pen]:
#             print(subject_n,pen,videos_dict[subject_n]['PENS'][pen])
# print(videos_dict)

OPENFACE_SOURCE = 'C:/Users/cs-lab/Desktop/OpenFace_2.2.0_win_x64/'
OPENFACE_EXE = 'FeatureExtraction.exe'
SOURCE_DIR = 'c:/croppedVids2/'
vids = glob.glob(SOURCE_DIR+'*.mp4')
vids.sort()
for vid in vids:

    vid_path = Path(vid)
    # if vid_path.name !='222_pen1_subject_cropped.mp4':
    #     continue
    openface_command = OPENFACE_SOURCE+OPENFACE_EXE+' -f "'+vid+'" -gaze -2Dfp -3Dfp -pose -tracked  -aus -of '+vid_path.name+' -multi-view 1'
    # if os.path.exists('processed/'+video_name+'.avi'):
    #     continue
    print(openface_command)
    subprocess.run(openface_command)

    # break
