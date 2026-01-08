import os
import glob
import pandas as pd
import shutil
from CheckCreate import check_create
SOURCE = 'C:/Users/cs-lab/Desktop/תזה-20230206T093737Z-001/תזה/Code/Excel parsing/processed/'
SOURCE = 'G:/My Drive/ADOS/Videos/OpenFace_output/'
OPENFACE_DIR = 'c:/openface_all/'
OPENFACE_OT = 'c:/openface_dired/'
check_create(OPENFACE_DIR)
check_create(OPENFACE_OT)
csv_files = glob.glob(OPENFACE_DIR+'*.csv',recursive=True)
for file in os.listdir(OPENFACE_DIR):
    file_path = OPENFACE_DIR+file
    # print(file)
    subject = file.split('_')[0]
    # print(subject)
    check_create(OPENFACE_OT+subject)
    shutil.copy(file_path,OPENFACE_OT+subject)