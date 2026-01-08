from Files import get_file
from VideoFrameSplit import extract_image_one_fps
from CheckCreate import check_create
import shutil
video_file = get_file('Select video file to split to images')
output_dir = 'c:/'+video_file.split('/')[-1].split('.')[0]+'/'
if video_file.split(':')[0]!='c':
    shutil.copy(video_file,'c:/')
    video_file = 'c:/'+video_file.split('/')[-1]
check_create(output_dir)
extract_image_one_fps(video_file,output_dir)
print(output_dir)
import os
os.remove(video_file)