subject_folder = 'c:/MetaData/16/openCV/'
openface_source = 'c:/openface_dired/16/'
from pickle import load
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
    return Vh[3,0:3]/Vh[3,3]

def triangulate(P1,P2,cloud1,cloud2):
    p3ds = []
    for uv1, uv2 in zip(cloud1, cloud2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
    return p3ds


face_lndmks_cols = []
for i in range(68):
    x_str = ' x_'+str(i)
    y_str = ' y_'+str(i)
    face_lndmks_cols+=[x_str,y_str]
face_lndmks_dict = {}
for i in range(68):
    x_str = ' x_'+str(i)
    y_str = ' y_'+str(i)
    face_lndmks_dict[i] = {'x': x_str, 'y':y_str}
eye_lndmks_cols = []
for i in range(56):
    x_str = ' eye_lmk_x_'+str(i)
    y_str = ' eye_lmk_y_'+str(i)
    eye_lndmks_cols+=[x_str,y_str]
eye_lndmks_dict = {}
for i in range(56):
    x_str = ' eye_lmk_x_'+str(i)
    y_str = ' eye_lmk_y_'+str(i)
    eye_lndmks_dict[i] = {'x': x_str, 'y':y_str}
    
with open(subject_folder+'s_b_stereoParams','rb') as f:
    s_b_stereo_params =  load(f)
K1,D1,K2,D2,R_bs,T_bs = s_b_stereo_params
T_bs = T_bs.T
# R_sb = R_bs.T
with open(subject_folder+'m_b_stereoParams','rb') as f:
    m_b_stereo_params =  load(f)
K3,D3,x,y,R_bm,T_bm =m_b_stereo_params
T_bm = T_bm.T
R_mb = R_bm.T
S_openFace_File  = openface_source + '16_pen1_Subject_1.csv'
B_openFace_File_s  = openface_source + '16_pen1_Both_1_Subject.csv'
B_openFace_File_m  = openface_source + '16_pen1_Both_1_Marissa.csv'
M_openFace_File = openface_source + '16_pen1_Marissa_1.csv'

S_openface = pd.read_csv(S_openFace_File).set_index('frame')
B_openface_s = pd.read_csv(B_openFace_File_s).set_index('frame')
B_openface_m = pd.read_csv(B_openFace_File_m).set_index('frame')
M_openface = pd.read_csv(M_openFace_File).set_index('frame')

#subject face triangulation

S_openface_face = S_openface[face_lndmks_cols]
B_openface_s_face  = B_openface_s[face_lndmks_cols]
B_openface_m_face = B_openface_m[face_lndmks_cols]
M_openface_face = M_openface[face_lndmks_cols]

S_openface_eye = S_openface[eye_lndmks_cols]
B_openface_s_eye  = B_openface_s[eye_lndmks_cols]
B_openface_m_eye  = B_openface_m[eye_lndmks_cols]
M_openface_eye = M_openface[eye_lndmks_cols]



face_array =  np.array(S_openface_face.loc[1])
reshaped = np.reshape(face_array,(68,2))
udp = cv.undistortPoints(reshaped,K1,D1,None,K1)
s_face_s = udp.squeeze()

face_array =  np.array(B_openface_s_face.loc[1])
reshaped = np.reshape(face_array,(68,2))
udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
s_face_b = udp.squeeze()

print(str(M_openface_face))
face_array =  np.array(M_openface_face.loc[1])
reshaped = np.reshape(face_array,(68,2))
udp = cv.undistortPoints(reshaped,K3,D3,None,K3)
m_face_m = udp.squeeze()

face_array =  np.array(B_openface_m_face.loc[1])
reshaped = np.reshape(face_array,(68,2))
udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
m_face_b = udp.squeeze()

eye_array =  np.array(S_openface_eye.loc[1])
reshaped = np.reshape(eye_array,(56,2))
udp = cv.undistortPoints(reshaped,K1,D1,None,K1)
s_eyes_s = udp.squeeze()

eye_array =  np.array(B_openface_s_eye.loc[1])
reshaped = np.reshape(eye_array,(56,2))
udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
s_eyes_b = udp.squeeze()

eye_array =  np.array(M_openface_eye.loc[1])
reshaped = np.reshape(eye_array,(56,2))
udp = cv.undistortPoints(reshaped,K3,D3,None,K3)
m_eyes_m = udp.squeeze()

eye_array =  np.array(B_openface_m_eye.loc[1])
reshaped = np.reshape(eye_array,(56,2))
udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
m_eyes_b = udp.squeeze()

RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = K2 @ RT1 #projection matrix for C1
 
#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R_bs, T_bs], axis = -1)
P2 = K1 @ RT2 #projection matrix for C2

P3= P1

RT3 = np.concatenate([R_bm, T_bm], axis = -1)
P4 = K3@RT3

face  = triangulate(P1,P2,s_face_b,s_face_s)
mface = triangulate(P3,P4,m_face_b,m_face_m)
# mface_in_both_world = R_bm@(mface+T_bm.T).T
# mface_in_subject_world = (R_bs.T)@(mface_in_both_world-T_bs)
# mface = mface_in_subject_world.T
# print(mface_in_both_world.shape)
fig = plt.figure()
ax = Axes3D(fig)

# creating the plot
plot_geeks = ax.scatter(face[:,0], -face[:,1], face[:,2], color='green')
plot_geeks = ax.scatter(mface[:,0], -mface[:,1], mface[:,2], color='blue')

plt.show()