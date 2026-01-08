import os
import glob
from pathlib import Path
dir = "C:/Users/cs-lab/Desktop/תזה-20230206T093737Z-001/תזה/Code/Excel parsing/tempFigs/"

pngs = glob.glob(dir+"*.png")
print(pngs)

for png in pngs:
    pngPath = Path(png)
    # print(pngPath.name)
    if(len(pngPath.name)!=7):
        continue
    print(pngPath.name)
    os.remove(png)