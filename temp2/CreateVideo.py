import cv2
import numpy as np
import glob
from Files import get_files
frameSize = (640, 480)
import datetime

datetime_object = datetime.datetime.now()

out = cv2.VideoWriter(datetime_object.strftime("%m_%d_%Y, %H_%M_%S.avi"),cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)

for filename in get_files('select'):
    img = cv2.imread(filename)
    out.write(img)

out.release()