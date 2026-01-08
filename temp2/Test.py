from Files import *
from Excel import *
import os
import pandas as pd

fil_dict = dict()
max_videos = 0
for path, subdirs, files in os.walk(get_folder()):
    for name in files:
        ext = name.split(".")[-1]
        if ext=="mp4":
            record  = path.split("\\")[-1]
            if record not in list(fil_dict.keys()):
                fil_dict[record]=list()
            fil_dict[record].append(name)
for key in list(fil_dict.keys()):
    if(len(fil_dict[key])>max_videos):
        max_videos=len(fil_dict[key])
for key in list(fil_dict.keys()):
    while(len(fil_dict[key])<max_videos):
        fil_dict[key].append("")

df = pd.DataFrame.from_dict(fil_dict)
df.T.to_excel("subject videos.xlsx")
print(fil_dict)
