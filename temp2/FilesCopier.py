import os
import glob
import shutil
def check_create(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
SOURCE_DIR = 'G:/My Drive/ADOS/Bayan/Faisal/'
OUTPUT_DIR = 'C:/new_videos/'
check_create(OUTPUT_DIR)
for dir in os.listdir(SOURCE_DIR):
    dir_path = SOURCE_DIR+dir
    if not os.path.isdir(dir_path):
        continue
    if 'record' not in dir:
        continue
    subject_n = int(dir.split('record')[-1])
    if subject_n>221:
        continue
    subject_dir_path = OUTPUT_DIR+str(subject_n)+'/'
    check_create(subject_dir_path)
    videos_of_subject = glob.glob(dir_path+'/*.mp4')
    for video in videos_of_subject:
        print(video)
        # shutil.copy(video,subject_dir_path)