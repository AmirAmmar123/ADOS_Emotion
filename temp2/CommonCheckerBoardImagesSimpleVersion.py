import glob
import os
import shutil
from Files import get_folder,get_file
from CheckCreate import check_create
import random
import cv2
from pathlib import Path,PurePath
from tqdm.notebook import trange, tqdm

folder1 = 'C:/235_calibration_subject_dark/'
folder2 = 'C:/235_calibration_both_dark/'
out_folder1 = folder1[:-1]+'_stereo/'
out_folder2 = folder2[:-1]+'_stereo/'
timeout = 5
check_create(out_folder1)
check_create(out_folder2)
timeout = 0
for img_name in os.listdir(folder1):
    if (img_name.split('.')[-1]!='png'):
        continue
    img1_path = folder1+img_name
    img2_path = folder2+img_name
    img1 = cv2.imread(img1_path,0)
    img2 = cv2.imread(img2_path,0)
    if cv2.checkChessboard(img1,(7,9)) and cv2.checkChessboard(img2,(7,9)):
        if timeout==0:
            shutil.copy(img1_path,out_folder1)
            shutil.copy(img2_path,out_folder2)
            timeout=5
        else:
            timeout-=1
    else:
        timeout=0
