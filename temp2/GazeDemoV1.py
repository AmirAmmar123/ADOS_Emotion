import cv2 as cv
import numpy as np 
import pandas as pd
from pickle import load
from MathCode import *
from RIGID3D import *

def get_currect_range_for_subject_given_start_and_end_of_pen_calib(subject,task,start,end):
    openface_source = 'c:/openface_dired/'+subject+'/'
    S_openFace_File  = openface_source + subject+'_'+task+'_subject.csv'
    S_openface = pd.read_csv(S_openFace_File).set_index('frame')
    cols = S_openface.columns
    new_cols = [col.strip() for col in cols]
    S_openface.columns = new_cols
    range = get_frames_in_range_ms(start,end,S_openface)

    return range
task = 'TB'
subject = '210'
with open('3Dpickles/intrDict_'+subject+'_pen1','rb') as f:
    intrPoints1Dict = load(f)
with open('3Dpickles/sFaceDict_'+subject+'_pen1','rb') as f:
    subject_faces1Dict = load(f)
with open('3Dpickles/sEyesDict_'+subject+'_pen1','rb') as f:
    subject_eyes1Dict = load(f)
with open('3Dpickles/mFace_'+subject+'_pen1','rb') as f:
    marissa_faces1 = load(f)
with open('3Dpickles/linesUsed_'+subject+'_'+'pen1','rb') as f:
    lines_used1 = load(f)
with open('3Dpickles/intrDict_'+subject+'_'+task,'rb') as f:
    intrPoints2Dict = load(f)
with open('3Dpickles/sFaceDict_'+subject+'_'+task,'rb') as f:
    subject_faces2Dict = load(f)
with open('3Dpickles/sEyesDict_'+subject+'_'+task,'rb') as f:
    subject_eyes2Dict = load(f)
with open('3Dpickles/mFace_'+subject+'_'+task,'rb') as f:
    marissa_faces2 = load(f)
with open('3Dpickles/linesUsed_'+subject+'_'+task,'rb') as f:
    lines_used2 = load(f)

video_file = 'c:/210_TB_subject_cropped.mp4'
key = (True,True,False,False)


intrPoints2 = intrPoints2Dict[key]
intrPoints1 = intrPoints1Dict[key]
intrPoints2 = intrPoints2Dict[key]
subject_range = None
marissa_faces_array1 = np.array(marissa_faces1)
if subject =='230':
    subject_range = get_currect_range_for_subject_given_start_and_end_of_pen_calib(subject,task,7650,29915)
    intrPoints1 = intrPoints1[subject_range[0]:subject_range[1]]
if subject == '210':
    subject_range = get_currect_range_for_subject_given_start_and_end_of_pen_calib(subject,task,2650,24401)
    intrPoints1 = intrPoints1[subject_range[0]:subject_range[1]]
hps_array1 = np.array(intrPoints1)
mean_marissa_face1 = marissa_faces_array1.mean(axis=0)
marissa_faces_array2 = np.array(marissa_faces2)
hps_array2 = np.array(intrPoints2)
print(marissa_faces_array2.shape)
print(len(lines_used2))
    # print(marissa_faces)


    # fig = plt.figure()
    # ax = Axes3D(fig)

lines_used_array = np.array(lines_used1[key])
if subject_range:
    lines_used_array = lines_used_array[subject_range[0]:subject_range[1]]
eyes_used = lines_used_array[:,0,:]
mean_eye = eyes_used.mean(axis=0)
    
mean_marissa_face2 = marissa_faces_array2.mean(axis=0)
transformed_hps = []
face_cloud_from_cropped,gaze_from_cropped,gaze_from_combined,intersecionPlane = key
square = 10
frames_draw = []
for i,mf2 in enumerate(marissa_faces2):
    hp = intrPoints2[i]
    d = mean_marissa_face1.mean(axis=0) - mf2.mean(axis=0)
    eye,vector = lines_used2[key][i]
    v1 = mean_marissa_face1.mean(axis=0) - mean_eye 
    # v1 = normalize(v1)
    v2 = hps_array1.mean(axis=0) - mean_eye
    # v2 = normalize(v2)
    v3 = hp - eye
    v3 = normalize(v3)
    vector = normalize(vector)
    if ((v3-vector)>0.000001).any():
        print(vector,v3)
        print(v3-vector)
        input()
    # v3 = v3 / np.linalg.norm(v3)
    v1 = normalize(v1)
    v2 = normalize(v2)
    v4 = find_vector_with_same_angle(v1,v2,v3)
    v4 = normalize(v4)
    # print(np.degrees(angle_between_vectors(v4,v3)),np.degrees(angle_between_vectors(v1,v2)))
    # print(np.dot(v4,v3),np.dot(v2,v1))
    # input()
    # v4 = v4 / np.linalg.norm(v4)
    eye_vect = mf2.mean(axis=0) - eye
    # print(eye_vect)
    # input()
    x_delta = eye_vect[0]
    # eye_vect = normalize(eye_vect)
    t = x_delta/v4[0]
    # transformed_hp = eye+t*v4
    transformed_hp = point_line_distance(mf2.mean(axis=0),(eye,v4))
    faceDefiningKeyPoints = [0,8,16]
    mf2PlanePoints = mf2[faceDefiningKeyPoints]
    # from sympy import Point3D
    # p1 = Point3D(mf2PlanePoints[0,0],mf2PlanePoints[0,1],mf2PlanePoints[0,2])
    # p2 = Point3D(mf2PlanePoints[1,0],mf2PlanePoints[1,1],mf2PlanePoints[1,2])
    # p3 = Point3D(mf2PlanePoints[2,0],mf2PlanePoints[2,1],mf2PlanePoints[2,2])
    # mPlane = sympy.Plane(p1,p2,p3)
    # thpPoint = Point3D(transformed_hp[0],transformed_hp[1],transformed_hp[2])
    # transformed_hp = mPlane.projection(thpPoint)
    # transformed_hp = np.array([transformed_hp[0],transformed_hp[1],transformed_hp[2]],dtype=np.float64)
    transformed_hp = project_point_onto_plane(transformed_hp,mf2PlanePoints.tolist())
    # print(transformed_hp)
    # input()
    if(np.isnan(transformed_hp).any()):
        continue
    transformed_hps.append(transformed_hp)
    
    mf2xMax = np.array([mf2.max(axis=0)[2],-mf2.min(axis=0)[1]])
    mf2xMin = np.array([mf2.min(axis=0)[2],-mf2.max(axis=0)[1]]) 
    tph = np.array([transformed_hp[2],-transformed_hp[1]])
    # print(tph,mf2xMax,mf2xMin)
    frames_draw.append((tph<mf2xMax).all() and (tph>mf2xMin).all())
    # if transformed_hp[2]
print('done')
cap = cv.VideoCapture(video_file)
index = 0
center_coordinates = (10, 10)
   
# Radius of circle
radius = 100
    
# Red color in BGR
color = (0, 255, 0)
    
# Line thickness of -1 px
thickness = -1
    
# Using cv2.circle() method
# Draw a circle of red color of thickness -1 px
format ='XVID'
# format = 'H264'
fourcc = cv.VideoWriter_fourcc(*format)
w_frame, h_frame = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
x, frames = cap.get(cv.CAP_PROP_FPS), cap.get(cv.CAP_PROP_FRAME_COUNT)
resize = 1
dim = (int(w_frame/resize), int(h_frame/resize))
out = cv.VideoWriter("c:/tb_gaze_demo210s.avi", fourcc, 30, dim)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        image = frame
        if frames_draw[index]:
            image = cv.circle(frame, center_coordinates, radius, color, thickness)
        image = cv.resize(image,dim)
        xx = int(index *100/frames)
        if(xx%10==0):
            print(int(xx),'%')
        out.write(image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        index+=1
    else:
        break
cap.release()
out.release()
cv.destroyAllWindows()
