import os
import shutil
import glob
import subprocess
import time
from CheckCreate import check_create
from pickle import load
from VideoDictLoader import VideosDictLoader
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

# with open('dict of sorted videos 24.5.23.pickle','rb') as f:
#     videos_dict = load(f)
videos_dict = VideosDictLoader.loadVideosDict()
# for subject_n in videos_dict.keys():
#     print(videos_dict[subject_n]['CALIBRATION'])
# exit(1)    
# for subject_n in videos_dict.keys():
#     for pen in videos_dict[subject_n]['PENS']:
#         if videos_dict[subject_n]['PENS'][pen]:
#             print(subject_n,pen,videos_dict[subject_n]['PENS'][pen])
# print(videos_dict)
TEMP_DIR = 'c:/TempClips/'
check_create(TEMP_DIR)
OPENFACE_SOURCE = 'C:/Users/cs-lab/Desktop/OpenFace_2.2.0_win_x64/'
OPENFACE_EXE = 'FeatureExtraction.exe'
last = -1
# print(videos) 
start = time.time()
for subject_n in videos_dict.keys():
    if(int(subject_n) <236):
        continue
    print(subject_n)
    # for pen in videos_dict[subject_n]['PENS']:
    for pen in ['PEN#1','PEN#2']:
        if videos_dict[subject_n]['PENS'][pen]:
            # for camera_type in videos_dict[subject_n]['PENS'][pen]:
            for camera_type in ['subject','marissa']:

                video = videos_dict[subject_n]['PENS'][pen][camera_type]
                print(video)
    # print(video.split('\\')[-1])
                
                new_path = shutil.copy(video,TEMP_DIR)
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
                if camera_type == 'both':
                    continue
                openface_command = OPENFACE_SOURCE+OPENFACE_EXE+' -f "'+new_path+'" -gaze -2Dfp -3Dfp -pose -tracked  -aus -of '+video_name+'.mp4 -multi-view 1 -wild '+get_camera_params(current,camera_type)
                # if os.path.exists('processed/'+video_name+'.avi'):
                #     continue
                print(video_name)
                subprocess.run(openface_command)
                os.remove(new_path)
    # for task in ['EMO','TB','STORY']:
    for task in ['TB','EMO']:
        # for camera_type in videos_dict[subject_n][task]:
        for camera_type in ['subject','marissa']:
            video = videos_dict[subject_n][task][camera_type]
# print(video.split('\\')[-1])
            
            new_path = shutil.copy(video,TEMP_DIR)
            if 'calib' in video:
                continue
            # if 'TB' not in video:
            #     continue
            # video_name = video.split('\\')[-1].split('.')[0].replace('Left',"_Subject").replace('Right','_Marissa')
            current = video_name.split('_')[0]
            if last!=-1:
                if last != current:
                    end = time.time()
                    # print('time passed for subject: ', end - start)
                    start = end
            else:
                last = current 
            if camera_type == 'both':
                continue
            openface_command = OPENFACE_SOURCE+OPENFACE_EXE+' -f "'+new_path+'" -gaze -2Dfp -3Dfp -pose -tracked  -aus -of '+video_name+'.mp4 -multi-view 1 -wild '+get_camera_params(current,camera_type)
            # if os.path.exists('processed/'+video_name+'.avi'):
            #     continue
            print(video_name)
            subprocess.run(openface_command)
            os.remove(new_path)

    # break
