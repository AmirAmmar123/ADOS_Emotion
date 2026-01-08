from pickle import dump,load
from unittest import result
import numpy as np
import pandas as pd 
from Files import *
from Excel import *
import matplotlib.pyplot as plt

# POSE  = [' pose_Tx',' pose_Ty',' pose_Tz']
# # openface_out = pd.read_csv(get_file())
# # print(openface_out)
# myopen = read_file_line_to_dict(get_file())
# # print(myopen)
# myopen = pd.DataFrame.from_dict(myopen['excel_dict'])
# # print(myopen)
# pose = myopen[POSE].astype(float)
# print(pose)
# print(pose.std( skipna = True).to_dict())
file_list = list()
file = "x"
while file:
    file = get_file("Please select your looking")
    if file:
        file_list.append(file)
print(file_list)