import os
import glob
import shutil

SOURCE = 'c:/MetaData/'
OUTPUT = 'G:/My Drive/ADOS/Videos/MetaData/'
print(os.listdir(SOURCE))

subjects_dirs = [dir for dir in os.listdir(SOURCE) if dir.isnumeric()]
subjects_dirs = [dir for dir in os.listdir(SOURCE) if os.path.exists(SOURCE+dir+'/openCV/') and dir.isnumeric()]
print(subjects_dirs)
for subject in subjects_dirs:
    new_subject_dir = OUTPUT+subject+'/'
    new_subject_dir_opencv = new_subject_dir+'openCV/'
    # os.mkdir(new_subject_dir)
    # os.mkdir(new_subject_dir_opencv)
    files_to_copy = glob.glob(SOURCE+subject+'/openCV/*')
    for file in files_to_copy:
        shutil.copy(file,new_subject_dir_opencv)

