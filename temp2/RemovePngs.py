import os
import glob

SOURCE_DIR = 'c:/MainFrameDir2/'
pngs = glob.glob(SOURCE_DIR+'*/**/*.jpg')
print(pngs)
for png in pngs:
    os.remove(png)