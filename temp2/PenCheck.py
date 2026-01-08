from pickle import load,dump
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sympy import Point3D,Plane,Line3D
import matplotlib
import tqdm
from RIGID3D import *
from MathCode import intersect_vector_plane,find_x_value_point
from StereoCalibrate import stereo_calibrate,calibrate_camera
from Files import *
import os
   
def get_frames_in_range_ms(start,end,df):
    start = start/1000
    end = end/1000
    min_set_flag = False
    min_f = float('inf')
    max_f = -1
    for index,row in df.iterrows():
        if float(row['timestamp']) > end:
            break
        if float(row['timestamp']) < start:
            continue
        if not min_set_flag:
            min_f = index
            min_set_flag = True
        max_f = index
    return (min_f,max_f)

def load_time_calibration_centers(csv_file,openface_df):
    df = pd.read_csv(csv_file)
    df['time'] = df['End'] - df['Start']
    df['time'] = df['time']*(df['Type']=='Center')
    centers = df.copy()
    centers = centers[centers['Type']=='Center']
    centerId = 1
    new_indicies = []
    for index,row in centers.iterrows():
        if index == centers['time'].idxmax():
            new_indicies.append('Center')
            continue
        new_indicies.append('Center'+str(centerId))
        centerId+=1
    centers.reset_index()
    centers['Type'] = new_indicies
    # print(centers)
    row = df.loc[df['time'].idxmax()]
    df = df[df.Type != 'Center']
    # df.loc[len(df.index)] = row
    df = pd.concat([df,centers],axis=0)
    df.drop(['time'],axis=1,inplace=True)
    df.sort_values('Start',inplace=True)
    print(df)
    calibration = []
    for index,row in df.iterrows():
        min_f,max_f = get_frames_in_range_ms(row['Start'],row['End'],openface_df)
        calibration.append((row['Type'],min_f,max_f))
    return calibration

fig = plt.figure()
# ax = Axes3D(fig)
# print(intersection_points)
subject = '234'
with open('3Dpickles/intr_'+subject+'_pen1','rb') as f:
    intersection_points = load(f)
calibration_file = 'G:/My Drive/ADOS/Videos/Faisal/record234/234_pen1_marissa 25-06-2023 00-13-44 Questions.csv'
task = 'pen1'
openface_source = 'c:/openface_dired/'+subject+'/'

S_openFace_File  = openface_source + subject+'_'+task+'_subject.csv'
S_openface = pd.read_csv(S_openFace_File).set_index('frame')
for df in [S_openface]:
    cols = df.columns
    new_cols = [col.strip() for col in cols]
    df.columns = new_cols
calibartion_frames =load_time_calibration_centers(calibration_file,S_openface)
print(calibartion_frames)
pen_phase = 0
colors_map = {'Up':'maroon','Down':'blue','Center':'purple','Left':'orange','Right':'olive'}
colors_map.update({'Center1':'pink','Center2':'fuchsia','Center3':'deeppink','Center4':'plum'})
phase_to_sum = {}
phase_to_counter = {}
for key in list(colors_map.keys()):
    phase_to_sum[key] = [0,0]
    phase_to_counter[key] = 0
for i,ip in enumerate(intersection_points):
#     print(ip)
    if(i < calibartion_frames[0][1]):
        continue
    
    if(i > calibartion_frames[pen_phase][2]):
        pen_phase+=1
        if(pen_phase>8):
            break
#         continue
    color = calibartion_frames[pen_phase][0]
#     print(i)
    phase_to_sum[color][0]+= ip[2]
    phase_to_sum[color][1]+= ip[1]
    phase_to_counter[color]+=1
#     pen_phase_to_name[pen_phase] = color
    plt.scatter(ip[2],-ip[1],color = colors_map[color],s=30)
for i in range(9):
    color = calibartion_frames[i][0]
#     print(color)
#     print(phase_to_sum[color][0],phase_to_counter[color])
#     print(phase_to_sum[color][0]/phase_to_counter[color])
    plt.scatter(phase_to_sum[color][0]/phase_to_counter[color],-phase_to_sum[color][1]/phase_to_counter[color],color = colors_map[color],s = 1000)
    anno = color[0]
    if "Center" in  color:
        anno = "C"
        anno += color.split("Center")[-1]
    if color == "Center":
        anno ="C0"
    plt.annotate(anno,(phase_to_sum[color][0]/phase_to_counter[color],-phase_to_sum[color][1]/phase_to_counter[color]))
#     plt.annotate()
plt.show()
# plt.savefig('test.png')
plt.clf()