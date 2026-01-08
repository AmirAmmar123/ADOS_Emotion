import shutil
import os
import glob
import datetime
import pathlib
SOURCE_DIR = 'C:/Users/cs-lab/Desktop/תזה-20230206T093737Z-001/תזה/Code/Excel parsing/processed/'
OUTPUT_DIR = 'c:/openface_dired/'

of_files = glob.glob(SOURCE_DIR+'*.csv')
print(of_files)

for of_file in of_files:
    if 'cropped' not in of_file:
        continue
    of_file_path = pathlib.Path(of_file)
    of_file_name = of_file_path.name
    subject = of_file_name.split('_')[0]
    subject_of_dir = OUTPUT_DIR+subject+'/'
    if not os.path.isdir(subject_of_dir):
        print('ERROR',subject_of_dir)
        exit(1)
    shutil.copy(of_file,subject_of_dir)
    # c_time = os.path.getmtime(of_file)
    # dt_c = datetime.datetime.fromtimestamp(c_time)
    # print(dt_c)