import os 
import shutil 
from Files import *
records = [26,30,32,33,36,38]
record = [32,36]
path = get_folder()
for item in os.listdir(path):
  if 'record' not in item:
    continue
  item_path= path+'/'+item
  if os.path.isdir(item_path):
    record_name = item
    record_num = record_name.split('record')[-1]
    if(int(record_num) not in records ):
        continue
    for sub_item in os.listdir(item_path):
      sub_item_path = item_path+"/"+sub_item
      if not os.path.isdir(sub_item_path):
        continue
      part = sub_item
      for video_file in os.listdir(sub_item_path):
        if video_file.split('.')[-1]!="mp4":
          continue
        video_file_path = sub_item_path+"/"+video_file
        new_video_file_name = part+"_"+video_file
        new_video_file_path =item_path+"/"+record_num+"_"+new_video_file_name
        #final test
        print("rename: ",video_file_path,new_video_file_path)
        os.rename(video_file_path,new_video_file_path)