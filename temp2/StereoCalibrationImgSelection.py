import glob
import os
import shutil
from Files import get_folder,get_file
from CheckCreate import check_create
import random
import cv2
from pathlib import Path,PurePath
from tqdm import tqdm
from pickle import dump,load
from sklearn.cluster import KMeans
def get_camera(name):
    if 'both' in name:
        return 'both'
    if 'subject' in  name:
        return 'subject'
    if 'marissa' in name:
        return 'marissa'
    return None

## setup
BOARD_SIZE = (6,8)
SUBJECT_N = 235
camera1_folder = 'C:/235_calibration_both_dark/'
camera2_folder = 'C:/235_calibration_subject_dark/'
camera_1_type  = get_camera(camera1_folder)
camera_2_type  = get_camera(camera2_folder)
if not camera_1_type:
    print('camera1 type is not clear')
    exit(1)
if not camera_2_type:
    print('camera2 type is not clear')
    exit(1)
board_imgs = []
timeout = 0
IMAGES_PICKLE_FILE = f'checked_images_list_{SUBJECT_N}'
CORNERS_PICKLE_FILE = f'checked_corners_list_{SUBJECT_N}'
checked_imgs = []
corners_list = []
if os.path.exists(IMAGES_PICKLE_FILE):
    with open(IMAGES_PICKLE_FILE,'rb') as img_pick:
        checked_imgs = load(img_pick)
    with open(CORNERS_PICKLE_FILE,'rb') as cor_pick:
        corners_list = load(cor_pick)
if len(checked_imgs)==0:
