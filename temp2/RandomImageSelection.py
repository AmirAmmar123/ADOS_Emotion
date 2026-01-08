import glob
import os
import shutil
from Files import get_folder
from CheckCreate import check_create
import random
import cv2
MODE = 1
if MODE:
    # folder1 = 'D:/new_frames_zipps/235_calibration_both'
    folder1 = 'C:/235_calibration_marissa'
    folder2 = 'C:/235_calibration_subject'

    TEMP1 = 'c:/temp1/'
    TEMP2 = 'c:/temp2/'
    check_create(TEMP1)
    check_create(TEMP2)

    images1 = os.listdir(folder1)
    images2 = os.listdir(folder2)
    color = (0, 0, 0)
    for i in range(0,20):
        r1 = random.randint(0,len(images1))
        i1 = images1[r1]
        i2 = images2[r1]
        if i < 20:
            shutil.copy(folder1+'/'+i1,TEMP1)
            shutil.copy(folder2+'/'+i2,TEMP2)
            continue
        image = cv2.imread(folder2+'/'+i2)
        cv2.imwrite(TEMP2+i2,cv2.rectangle(image,(250,500),(425,700), color,-1))    
        image = cv2.imread(folder1+'/'+i1)
        cv2.imwrite(TEMP1+i1,cv2.rectangle(image,(900,400),(1030,530), color,-1))
else:
    folder1 = 'c:/235_calibration_subject/'

    TEMP1 = 'c:/temp1/'
    check_create(TEMP1)

    images1 = os.listdir(folder1)
    color = (0, 0, 0)
    for i in range(0,100):
        r1 = random.randint(0,len(images1))
        i1 = images1[r1]
        if i < 20:
            shutil.copy(folder1+'/'+i1,TEMP1)
            continue
        image = cv2.imread(folder1+'/'+i1)
        cv2.imwrite(TEMP1+i1,cv2.rectangle(image,(250,500),(425,700), color,-1))    

