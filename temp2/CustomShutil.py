import os
import shutil
import glob
import pathlib
SOURCE_VID_PATH = 'G:/My Drive/ADOS/Videos/Unsorted/'
TEMP_VID_DIR = 'c:/temp_vid_dir/'

for subject in range(252,253):
    subject_str = str(subject)
    subject_dir = SOURCE_VID_PATH+subject_str+'/'
    subject_vid_files = glob.glob(subject_dir+'*.mp4')
    for svf in subject_vid_files:
        svf_name = pathlib.Path(svf).name
        if os.path.exists(TEMP_VID_DIR+svf_name):
            continue
        shutil.copy(svf,TEMP_VID_DIR)