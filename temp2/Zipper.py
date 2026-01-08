import glob
import os
import os.path
import shutil
path = 'C:/new_frames/'
zipped = 'F:/new_frames_zipps/'
if not os.path.exists(zipped):
    os.mkdir(zipped)
folders = os.listdir(path)
for folder in folders:
    folder_path = path+folder
    print(folder)
    if os.path.exists(zipped+folder+'.zip'):
        continue
    # print(zipped+folder+'.zip')
    # break
    shutil.make_archive(zipped+folder, 'zip', folder_path)
