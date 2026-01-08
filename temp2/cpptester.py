import os
from Files import get_folder
import shutil
import subprocess
import patoolib

import time
from pyunpack import Archive
import zipfile
dir = get_folder()
if not (os.path.exists(dir+"\\"+"rars")):
    os.mkdir(dir+"\\"+"rars")
rar_dir = dir+"\\"+"rars"
folders = (os.listdir(dir))
for folder in folders:
    if(os.path.isdir(dir+"\\"+folder)):
        files = os.listdir(dir+"\\"+folder)
        for file in files:
            rar_folder = file.split(".")[0]
            # if not os.path.exists(rar_dir+"\\"+rar_folder):
            #     os.mkdir(rar_dir+"/"+rar_folder)
            # shutil.copyfile(dir+"\\"+folder+"\\"+file, rar_dir+"\\"+rar_folder+"\\"+file)
            # time.sleep(1)
            # print("attempting unrar ",dir+"/"+folder+"/"+file)
            # with zipfile.ZipFile(dir+"/"+folder+"/"+file, 'r') as zip_ref:
            #     zip_ref.extractall(dir+"/"+folder+"/"+file+"/")
            patoolib.extract_archive(dir+"/"+folder+"/"+file, outdir=dir+"/"+folder+"/"+file+"/")

            # Archive(dir+"\\"+folder+"\\"+file).extractall(dir+"\\"+folder+"\\"+file+"\\")
    



