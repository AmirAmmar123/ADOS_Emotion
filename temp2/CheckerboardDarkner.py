import glob
import os
import shutil
from Files import get_folder,get_file
from CheckCreate import check_create
import random
import cv2
from pathlib import Path,PurePath
rs = []
input_folders = []
output_folders = []
for i in range(3):
    roi_img = get_file('select image for roi')
    input_folder = str(Path(roi_img).parent)+'/'
    input_folder = input_folder.replace('\\','/')
    input_folders.append(input_folder)
    # input_folder = 'c:/235_calibration_subject/'
    OUTPUT_FOLDER = 'c:/'+input_folder.split('/')[-2]+'_dark/'
    check_create(OUTPUT_FOLDER)
    output_folders.append(OUTPUT_FOLDER)
    img = cv2.imread(roi_img)
    resolution = (1920, 1080)

    # Create window
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    rs.append(cv2.selectROI("Output",img))
for i in range(3):
    for img_name in os.listdir(input_folders[i]):
        img_path = input_folders[i]+img_name
        image = cv2.imread(img_path)
        color = (0,0,0)
        cv2.imwrite(output_folders[i]+img_name,cv2.rectangle(image,rs[i], color,-1))    

print(OUTPUT_FOLDER)