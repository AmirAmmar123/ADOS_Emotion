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
# matplotlib.use("Agg")

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

def load_time_calibration(csv_file,openface_df):
    df = pd.read_csv(csv_file)
    df['time'] = df['End'] - df['Start']
    df['time'] = df['time']*(df['Type']=='Center')
    row = df.loc[df['time'].idxmax()]
    df = df[df.Type != 'Center']
    df.loc[len(df.index)] = row
    df.drop(['time'],axis=1,inplace=True)
    df.sort_values('Start',inplace=True)
    calibration = []
    for index,row in df.iterrows():
        min_f,max_f = get_frames_in_range_ms(row['Start'],row['End'],openface_df)
        calibration.append((row['Type'],min_f,max_f))
    return calibration
        


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

def move_view(event):
    ax.autoscale(enable=False, axis='both') 
    koef = 8
    zkoef = (ax.get_zbound()[0] - ax.get_zbound()[1]) / koef
    xkoef = (ax.get_xbound()[0] - ax.get_xbound()[1]) / koef
    ykoef = (ax.get_ybound()[0] - ax.get_ybound()[1]) / koef
    ## Map an motion to keyboard shortcuts
    if event.key == "ctrl+down":
        ax.set_ybound(ax.get_ybound()[0] + xkoef, ax.get_ybound()[1] + xkoef)
    if event.key == "ctrl+up":
        ax.set_ybound(ax.get_ybound()[0] - xkoef, ax.get_ybound()[1] - xkoef)
    if event.key == "ctrl+right":
        ax.set_xbound(ax.get_xbound()[0] + ykoef, ax.get_xbound()[1] + ykoef)
    if event.key == "ctrl+left":
        ax.set_xbound(ax.get_xbound()[0] - ykoef, ax.get_xbound()[1] - ykoef)
    if event.key == "down":
        ax.set_zbound(ax.get_zbound()[0] - zkoef, ax.get_zbound()[1] - zkoef)
    if event.key == "up":
        ax.set_zbound(ax.get_zbound()[0] + zkoef, ax.get_zbound()[1] + zkoef)
    # zoom option
    if event.key == "alt+up":
        ax.set_xbound(ax.get_xbound()[0]*0.90, ax.get_xbound()[1]*0.90)
        ax.set_ybound(ax.get_ybound()[0]*0.90, ax.get_ybound()[1]*0.90)
        ax.set_zbound(ax.get_zbound()[0]*0.90, ax.get_zbound()[1]*0.90)
    if event.key == "alt+down":
        ax.set_xbound(ax.get_xbound()[0]*1.10, ax.get_xbound()[1]*1.10)
        ax.set_ybound(ax.get_ybound()[0]*1.10, ax.get_ybound()[1]*1.10)
        ax.set_zbound(ax.get_zbound()[0]*1.10, ax.get_zbound()[1]*1.10)
    
    # Rotational movement
    elev=ax.elev
    azim=ax.azim
    if event.key == "shift+up":
        elev+=10
    if event.key == "shift+down":
        elev-=10
    if event.key == "shift+right":
        azim+=10
    if event.key == "shift+left":
        azim-=10

    ax.view_init(elev= elev, azim = azim)

    # print which ever variable you want 
    if event.key == "1":
        print(elev,azim,"x: ",ax.get_xbound()[0],ax.get_xbound()[1]," y: ",ax.get_ybound()[0],ax.get_ybound()[1]," z: ",ax.get_zbound()[0],ax.get_zbound()[1])

    ax.figure.canvas.draw()
        

def find_intersection_point(vector1_point, vector1_direction, vector2_point, vector2_direction):
    # Convert inputs to numpy arrays
    vector1_point = np.array(vector1_point)
    vector1_direction = np.array(vector1_direction)
    vector2_point = np.array(vector2_point)
    vector2_direction = np.array(vector2_direction)
    
    # Calculate the intersection point of the two vectors
    t = np.linalg.det([vector2_point - vector1_point, vector2_direction, np.cross(vector1_direction, vector2_direction)]) / np.linalg.det([vector1_direction, vector2_direction, np.cross(vector1_direction, vector2_direction)])
    intersection_point = vector1_point + t * vector1_direction
    
    return intersection_point.tolist()
def create_combined_vector(intersection_point, vector1_direction, vector2_direction):
    # Convert inputs to numpy arrays
    intersection_point = np.array(intersection_point)
    vector1_direction = np.array(vector1_direction)
    vector2_direction = np.array(vector2_direction)
    
    # Calculate the combined direction of the two vectors
    combined_direction = vector1_direction + vector2_direction
    
    # Create the new vector using the intersection point and combined direction
    new_vector = {'point': intersection_point.tolist(), 'direction': combined_direction.tolist()}
    
    return new_vector

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

subject_folder = 'c:/MetaData/234/openCV/'
openface_source = 'c:/openface_dired/234/'


N_FACIAL_LNDMKS = 68
N_EYES_LNDMKS = 56
L_PUPIL_1 = 20
L_PUPIL_2 = 28
R_PUPIL_1 = 48
R_PUPIL_2 = 56


face_lndmks_cols = []
for i in range(N_FACIAL_LNDMKS):
    x_str = 'x_'+str(i)
    y_str = 'y_'+str(i)
    face_lndmks_cols+=[x_str,y_str]
face_lndmks_dict = {}
for i in range(N_FACIAL_LNDMKS):
    x_str = 'x_'+str(i)
    y_str = 'y_'+str(i)
    face_lndmks_dict[i] = {'x': x_str, 'y':y_str}
eye_lndmks_cols = []
for i in range(N_EYES_LNDMKS):
    x_str = 'eye_lmk_x_'+str(i)
    y_str = 'eye_lmk_y_'+str(i)
    eye_lndmks_cols+=[x_str,y_str]
eye_lndmks_dict = {}
for i in range(N_EYES_LNDMKS):
    x_str = 'eye_lmk_x_'+str(i)
    y_str = 'eye_lmk_y_'+str(i)
    eye_lndmks_dict[i] = {'x': x_str, 'y':y_str}
face3D_lndmks_cols = []
for i in range(N_FACIAL_LNDMKS):
    x_str = 'X_'+str(i)
    y_str = 'Y_'+str(i)
    z_str = 'Z_'+str(i)
    face3D_lndmks_cols+=[x_str,y_str,z_str]
face3D_lndmks_dict = {}
for i in range(N_FACIAL_LNDMKS):
    x_str = 'X_'+str(i)
    y_str = 'Y_'+str(i)
    z_str = 'Z_'+str(i)
    face3D_lndmks_dict[i] = {'X': x_str, 'Y':y_str,'Z':z_str}
eye3D_lndmks_cols = []
for i in range(N_EYES_LNDMKS):
    x_str = 'eye_lmk_X_'+str(i)
    y_str = 'eye_lmk_Y_'+str(i)
    z_str = 'eye_lmk_Z_'+str(i)
    eye3D_lndmks_cols+=[x_str,y_str,z_str]
eye3D_lndmks_dict = {}
for i in range(N_EYES_LNDMKS):
    x_str = ' eye_lmk_X_'+str(i)
    y_str = ' eye_lmk_Y_'+str(i)
    z_str = ' eye_lmk_Z_'+str(i)
    eye3D_lndmks_dict[i] = {'X': x_str, 'Y':y_str,'Z': z_str}
    
    
    
    
with open(subject_folder+'s_b_stereoParams','rb') as f:
    s_b_stereo_params =  load(f)
K1,D1,K2,D2,R_bs,T_bs = s_b_stereo_params
T_bs = T_bs.T
# R_sb = R_bs.T
with open(subject_folder+'m_b_stereoParams','rb') as f:
    m_b_stereo_params =  load(f)
K3,D3,x,y,R_bm,T_bm =m_b_stereo_params
T_bm = T_bm.T
# R_bm = R_bm.T

with open(subject_folder+'s_m_stereoParams','rb') as f:
    s_m_stereo_params =  load(f)
w,z,x,y,R_ms,T_ms =s_m_stereo_params
T_ms = T_ms.T

import glob


camera1_frames = glob.glob("C:/MetaData/234/s_b_stereo_frames/*.png")
camera2_frames = glob.glob("C:/MetaData/234/b_s_stereo_frames/*.png")
# print(camera1_frames)
# print(camera2_frames)
# input()
# K1,D1 = calibrate_camera(camera1_frames)
# K2,D2 = calibrate_camera(camera2_frames)
# Rp, Tp = stereo_calibrate(K1,D1,K2,D2,camera1_frames,camera2_frames)
# print(Rp,R_bs)
# print(Tp,T_bs,T_bs/Tp)
# input()
# R_bs = Rp
# T_bs = Tp

# # input()
# Rp, Tp = stereo_calibrate(K3,D3,K2,D2,camera1_frames,camera2_frames)
# print(Rp,R_bs)
# print(Tp,T_bs)
# R_bm = Rp
# T_bm = Tp

RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = K1 @ RT1 #projection matrix for C1

#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R_bs, T_bs], axis = -1)
P2 = K2 @ RT2 #projection matrix for C2

P3= K3@RT1

RT3 = np.concatenate([R_bm, T_bm], axis = -1)
P4 = K2@RT3

#camera plot check


# S_openFace_File_Cropped  = openface_source + '234_pen1_subject_cropped.csv'
# S_openFace_File  = openface_source + '234_pen1_subject.csv'
# B_openFace_File_s  = openface_source + '234_pen1_both_Subject.csv'
# B_openFace_File_m  = openface_source + '234_pen1_both_Marissa.csv'
# M_openFace_File = openface_source + '234_pen1_marissa.csv'

S_openFace_File_Cropped  = openface_source + '234_TB_subject_cropped.csv'
S_openFace_File  = openface_source + '234_TB_subject.csv'
B_openFace_File_s  = openface_source + '234_TB_both_Subject.csv'
B_openFace_File_m  = openface_source + '234_TB_both_Marissa.csv'
M_openFace_File = openface_source + '234_TB_marissa.csv'


calibration_file = 'G:/My Drive/ADOS/Videos/Unsorted/207/Asaf/207_First_Pen1_I 30-04-2023 12-59-44 Questions.csv'

S_openface = pd.read_csv(S_openFace_File).set_index('frame')
S_openface_cropped = pd.read_csv(S_openFace_File_Cropped).set_index('frame')
B_openface_s = pd.read_csv(B_openFace_File_s).set_index('frame')
B_openface_m = pd.read_csv(B_openFace_File_m).set_index('frame')
M_openface = pd.read_csv(M_openFace_File).set_index('frame')

both = (942,1778)
subjec = (822,1212)
#subject face triangulation
for df in [S_openface,B_openface_s,B_openface_m,M_openface,S_openface_cropped]:
    cols = df.columns
    new_cols = [col.strip() for col in cols]
    df.columns = new_cols

S_openface_face = S_openface[face_lndmks_cols]
B_openface_s_face  = B_openface_s[face_lndmks_cols]
B_openface_m_face = B_openface_m[face_lndmks_cols]
M_openface_face = M_openface[face_lndmks_cols]

S_openface_eye = S_openface[eye_lndmks_cols]
B_openface_s_eye  = B_openface_s[eye_lndmks_cols]
B_openface_m_eye  = B_openface_m[eye_lndmks_cols]
# M_openface_eye = M_openface[eye_lndmks_cols]

selectedKeyPointsSubjectFace = [16,26,27,33,45,54]
faceDefiningKeyPoints = [0,8,16]
selectedKeyPointsMarissaFace = [0,17,27,33,36,48]

hps = []
hps2 = []
subject_faces = []
subject_eyes = [] 
marissa_faces = []

calibartion_frames =load_time_calibration(calibration_file,S_openface)
#example : [('Up', 209, 290), ('Center', 298, 380), ('Left', 396, 485), ('Right', 583, 673), ('Down', 763, 845)]

pen_phase = 0

intersection_points = []
for frame in tqdm.tqdm(S_openface.index.values.tolist()):
    # continue
    #check frame in part of face
    # if frame < calibartion_frames[pen_phase][1]:
    #     continue
    # if frame > calibartion_frames[pen_phase][2]:
    #     #frame out of range for sure
    #     pen_phase+=1
    #     if pen_phase == len(calibartion_frames):
    #         break
    #     continue
    
    face_array =  np.array(S_openface_face.loc[frame])
    reshaped = np.reshape(face_array,(N_FACIAL_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K1,D1,None,K1)
    s_face_s_2d = udp.squeeze()
    # s_face_s = reshaped

    face_array =  np.array(B_openface_s_face.loc[frame])
    reshaped = np.reshape(face_array,(N_FACIAL_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
    s_face_b_2d = udp.squeeze()
    # s_face_b = reshaped

    face_array =  np.array(M_openface_face.loc[frame])
    reshaped = np.reshape(face_array,(N_FACIAL_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K3,D3,None,K3)
    m_face_m_2d = udp.squeeze()
    # m_face_m = reshaped

    face_array =  np.array(B_openface_m_face.loc[frame])
    reshaped = np.reshape(face_array,(N_FACIAL_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
    m_face_b_2d = udp.squeeze()
    # m_face_b = reshaped


    eye_array =  np.array(S_openface_eye.loc[frame])
    reshaped = np.reshape(eye_array,(N_EYES_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K1,D1,None,K1)
    s_eyes_s_2d = udp.squeeze()

    eye_array =  np.array(B_openface_s_eye.loc[frame])
    reshaped = np.reshape(eye_array,(N_EYES_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
    s_eyes_b_2d = udp.squeeze()

    # eye_array =  np.array(M_openface_eye.loc[frame])
    # reshaped = np.reshape(eye_array,(N_EYES_LNDMKS,2))
    # udp = cv.undistortPoints(reshaped,K3,D3,None,K3)
    # m_eyes_m = udp.squeeze()

    eye_array =  np.array(B_openface_m_eye.loc[frame])
    reshaped = np.reshape(eye_array,(N_EYES_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
    m_eyes_b = udp.squeeze()


    selected_left_eye_kp_for_RT = np.array([8,11,14,17])
    selected_right_eye_kp_for_RT = selected_left_eye_kp_for_RT+(36-8)


# 



    s_face_traingluated = triangulate(P1,P2,s_face_s_2d,s_face_b_2d)
    s_eyes_traingluated = triangulate(P1,P2,s_eyes_s_2d,s_eyes_b_2d)
    mface3D_tri = triangulate(P3,P4,m_face_m_2d,m_face_b_2d)
    mface3D_tri = triangulate(P3,P4,m_face_m_2d,m_face_b_2d)


    
    s_OF_3D_FacePointCloud  = np.reshape(np.array(S_openface[face3D_lndmks_cols].loc[frame]),(N_FACIAL_LNDMKS,3))
    s_OF_3D_EyesPointCloud  = np.reshape(np.array(S_openface[eye3D_lndmks_cols].loc[frame]),(N_EYES_LNDMKS,3))





    s_OF_3D_FacePointCloud  = np.reshape(np.array(S_openface_cropped[face3D_lndmks_cols].loc[frame]),(N_FACIAL_LNDMKS,3))
    s_OF_3D_EyesPointCloud  = np.reshape(np.array(S_openface_cropped[eye3D_lndmks_cols].loc[frame]),(N_EYES_LNDMKS,3))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(s_OF_3D_FacePointCloud[:,0],s_OF_3D_FacePointCloud[:,1],s_OF_3D_FacePointCloud[:,2])
    # ax.scatter3D(s_OF_3D_EyesPointCloud[:,0],s_OF_3D_EyesPointCloud[:,1],s_OF_3D_EyesPointCloud[:,2])
    s_right_pupil3D = np.mean(s_OF_3D_EyesPointCloud[R_PUPIL_1:R_PUPIL_2,:],axis=0)
    ax.scatter3D(s_right_pupil3D[0],s_right_pupil3D[1],s_right_pupil3D[2])

    plt.show()
    # scale 3d point cloud to triangulated size
    # scale 3d point cloud to triangulated size
    dist1 = np.linalg.norm(s_face_traingluated[16]-s_face_traingluated[0])
    dist2 = np.linalg.norm(s_OF_3D_FacePointCloud[16]-s_OF_3D_FacePointCloud[0])

    #scale process - move cloud to origin (subtract the mean),scale, return to original position (add the mean)
    s_OF_3D_FacePointCloudMean = s_OF_3D_FacePointCloud.mean(axis=0)
    s_OF_3D_FacePointCloud -= s_OF_3D_FacePointCloudMean
    s_OF_3D_FacePointCloud/= (dist2/dist1)
    s_OF_3D_FacePointCloud+=s_OF_3D_FacePointCloudMean

    dist1 = np.linalg.norm(s_eyes_traingluated[42]-s_eyes_traingluated[36])
    dist2 = np.linalg.norm(s_OF_3D_EyesPointCloud[42]-s_OF_3D_EyesPointCloud[36])

    s_OF_3D_EyesPointCloudMean = s_OF_3D_EyesPointCloud.mean(axis=0)
    s_OF_3D_EyesPointCloud -= s_OF_3D_EyesPointCloudMean
    s_OF_3D_EyesPointCloud/= (dist2/dist1)
    s_OF_3D_EyesPointCloud+=s_OF_3D_EyesPointCloudMean


    #find transformation from that takes from "General" 3D world to triangulated "correct" world
    R,t = rigid_transform_3D(s_OF_3D_FacePointCloud.T,s_face_traingluated.T)
    s_OF_3D_FacePointCloud =  (R@s_OF_3D_FacePointCloud.T + t).T


    s_final_face_3D = s_OF_3D_FacePointCloud
    transformedEyes = (R@s_OF_3D_EyesPointCloud.T + t).T
    s_final_eyes_3D = transformedEyes






    s_face_traingluated = s_OF_3D_FacePointCloud
    s_eyes_traingluated = s_OF_3D_EyesPointCloud

    #bring marissa face to currect location
    s_final_face_3D =( R_bs@s_final_face_3D.T +T_bs).T
    transformedEyes = (R_bs@transformedEyes.T+T_bs).T
    m_final_3d_face = ( R_bm@mface3D_tri.T +T_bm).T


    subject_eyes_KP_8 = s_OF_3D_FacePointCloud[36+6]
    subject_eyes_KP_14 = s_OF_3D_FacePointCloud[39+6]
    subject_eyes_KP_11 = (s_OF_3D_FacePointCloud[37+6]+s_OF_3D_FacePointCloud[38+6])/2
    subject_eyes_KP_17 = (s_OF_3D_FacePointCloud[40+6]+s_OF_3D_FacePointCloud[41+6])/2
    eyes = [subject_eyes_KP_8,subject_eyes_KP_14,subject_eyes_KP_11,subject_eyes_KP_17]


    fig = plt.figure()
    ax = Axes3D(fig)


   #camera 
    # orientation1 = orientation2 * R
    # location1 = orientation2 * Translation + location2
    #location2 = -orientation2*Translation
    # location1 = 0,0,0
    # -> location2 =  -(orientation2*Translation)
    # ->location2  = -(orientation1*R^-1)


    #orientation3 = orientation2*R_bm
    #location3 = orientation2*T_bm + location2
    # = orientation2*T_bm
    #oriention
    #camera marissa - 000

    camera = np.array([0,0,0])
    ax.scatter3D(0,0,0,c = 'red')
    camera2 = T_bs # T - from camera 1 to camera 2 , but in camera2 world
    camera3inCamera2 = T_bm
    ax.scatter3D(camera2[0],camera2[1],camera2[2],c='blue')
    ax.scatter3D(camera3inCamera2[0],camera3inCamera2[1],camera3inCamera2[2],c = 'black')
    # plt.show()
    mface_in_both_world = R_bm@(mface3D_tri-T_bm.T).T


    ax.scatter3D(T_ms[0],T_ms[1],T_ms[2],c='pink')
    # ax.scatter3D(camera3b[0],camera3b[1],camera3b[2],c='orange')

    # print(sticker)
    # input()
    # ax.scatter3D(sticker[0],sticker[1],sticker[2],c='purple')
    #
    # c1 -> c2 | R@c1+t 
    # c1 : camera of marissa
    # c2 : camera of both 
    # 0,0,0 -> c2 -> c3 :  R_bm.T@((R_bs@O+T_bs)-T_bm)
    # mface_in_subject_world = (R_bs.T)@(camera2-T_bm)
    camera_direction = np.array([0,0,1])
    ax.quiver3D(0,0,0,camera_direction[0],camera_direction[1],camera_direction[2],length=100)
    camera_direction = R_bs@camera_direction
    ax.quiver3D(camera2[0],camera2[1],camera2[2],camera_direction[0],camera_direction[1],camera_direction[2],length=100)
    camera_direction = np.array([0,0,1])
    camera_direction = R_bm@camera_direction
    ax.quiver3D(camera3inCamera2[0],camera3inCamera2[1],camera3inCamera2[2],camera_direction[0],camera_direction[1],camera_direction[2],length=100)


    ax.scatter3D(s_final_face_3D[:,0],s_final_face_3D[:,1],s_final_face_3D[:,2],color = 'orange')
    ax.scatter3D(m_final_3d_face[:,0],m_final_3d_face[:,1],m_final_3d_face[:,2],color = 'black')
    ax.scatter3D(transformedEyes[:,0],transformedEyes[:,1],transformedEyes[:,2])

    left_gaze_vector = R_bs@S_openface_cropped[['gaze_0_x','gaze_0_y','gaze_0_z']].loc[frame] 
    right_gaze_vector = R_bs@S_openface_cropped[['gaze_1_x','gaze_1_y','gaze_1_z']].loc[frame]
    # s_left_old_pupil3D = np.mean(transformedEyes[L_PUPIL_1:L_PUPIL_2,:],axis=0)
    # print(sEyes3D_tri[L_PUPIL_1:L_PUPIL_2,:])
    # print(np.mean(sEyes3D_tri[L_PUPIL_1:L_PUPIL_2,:],axis=0))
    s_left_pupil3D = np.mean(transformedEyes[L_PUPIL_1:L_PUPIL_2,:],axis=0)
    s_right_pupil3D = np.mean(transformedEyes[R_PUPIL_1:R_PUPIL_2,:],axis=0)
    left_line_end = s_left_pupil3D + S_openface_cropped[['gaze_0_x','gaze_0_y','gaze_0_z']].loc[frame] *50
    line_mat = np.array([[s_left_pupil3D[0],left_line_end[0]],[s_left_pupil3D[1],left_line_end[1]],[s_left_pupil3D[2],left_line_end[2]]])
    # ax.plot3D([s_left_old_pupil3D[0],left_line_end[0]],[s_left_old_pupil3D[1],left_line_end[1]],[s_left_old_pupil3D[2],left_line_end[2]],color = 'black')
    line_mat = ( R_bs@line_mat +T_bs).T

    # ax.plot3D([line_mat[0][0],line_mat[1][0]],[line_mat[0][1],line_mat[1][1]],[line_mat[0][2],line_mat[1][2]],color = 'red',linewidth=10)
    ax.quiver3D(s_left_pupil3D[0],s_left_pupil3D[1],s_left_pupil3D[2],left_gaze_vector[0],left_gaze_vector[1],left_gaze_vector[2],length=100)
    ax.quiver3D(s_right_pupil3D[0],s_right_pupil3D[1],s_right_pupil3D[2],right_gaze_vector[0],right_gaze_vector[1],right_gaze_vector[2],length=100)
    # ax.scatter3D(sface3D_tri[36,0],sface3D_tri[36,1],sface3D_tri[36,2],color = 'blue')
    
    gaze_vector = Line3D([s_right_pupil3D[0], s_right_pupil3D[1], s_right_pupil3D[2]],direction_ratio=[right_gaze_vector[0].item(), right_gaze_vector[1].item(), right_gaze_vector[2].item()])
    # mFacePlane = Plane(mFacePlanePoints[0],mFacePlanePoints[1],mFacePlanePoints[2])
    
    

    middle_eye = (s_right_pupil3D+s_left_pupil3D)/2
    combined_vector = left_gaze_vector+right_gaze_vector
    ax.quiver3D(middle_eye[0],middle_eye[1],middle_eye[2],combined_vector[0],combined_vector[1],combined_vector[2],length=50,color='purple')

    
    mFacePlanePoints = [Point3D(p) for p in m_final_3d_face[faceDefiningKeyPoints]]
    mFacePlane = Plane(mFacePlanePoints[0],mFacePlanePoints[1],mFacePlanePoints[2])
    line = Line3D([middle_eye[0], middle_eye[1], middle_eye[2]],direction_ratio=[combined_vector[0], combined_vector[1], combined_vector[2]])
    line 
    intr = mFacePlane.intersection(line)[0].coordinates
    intersection_points.append(intr)



    ax.scatter3D(intr[0],intr[1],intr[2],s=10)




    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    ax.view_init(-77,-90) #elev,azim
    # plt.show()
    from CheckCreate import check_create
    check_create('tempFigs/')
    plt.savefig('tempFigs/'+str(frame).zfill(len(str(len(S_openface.index.values.tolist()))))+'.png')
    plt.close()
    continue

    # plt.show()




    # ax.scatter3D(subject_point_cloud[:,0],-subject_point_cloud[:,1],-subject_point_cloud[:,2])
    # ax.scatter3D(subject_eyes_point_cloud[:,0],-subject_eyes_point_cloud[:,1],-subject_eyes_point_cloud[:,2])
    eyes_kps =  [8,11,14,17]
    A = s_OF_3D_EyesPointCloud[selected_right_eye_kp_for_RT]
    B = np.array([subject_eyes_KP_8,subject_eyes_KP_11,subject_eyes_KP_14,subject_eyes_KP_17])
    # ax.scatter3D(A[:,0],-A[:,1],-A[:,2])
    # ax.scatter3D(B[:,0],-B[:,1],-B[:,2])
    A = np.array(A)
    R,t = rigid_transform_3D(A.T,B.T)
    NB = ((R@s_OF_3D_EyesPointCloud.T)+t).T
    # ax.scatter3D(NB[:,0],-NB[:,1],-NB[:,2])
    # ax.scatter3D(sface3D_tri[:,0],sface3D_tri[:,1],sface3D_tri[:,2])
    


    # ax.scatter3D(subject_eyes_KP_8[0],-subject_eyes_KP_8[1],-subject_eyes_KP_8[2])
    # ax.scatter3D(subject_eyes_KP_11[0],-subject_eyes_KP_11[1],-subject_eyes_KP_11[2])
    # ax.scatter3D(subject_eyes_KP_14[0],-subject_eyes_KP_14[1],-subject_eyes_KP_14[2])
    # ax.scatter3D(subject_eyes_KP_17[0],-subject_eyes_KP_17[1],-subject_eyes_KP_17[2])

    # subject_eyes_point_cloud = ((R@subject_eyes_point_cloud.T)+t).T
    # subject_eyes_point_cloud = subject_eyes_point_cloud[:int(subject_eyes_point_cloud.shape[0]/2)]
    
    # print(subject_eyes_point_cloud.shape[0])
    # for i in range(int(len(eye3D_lndmks_cols)/2)):
    #     ax.text(subject_eyes_point_cloud[i,0], -subject_eyes_point_cloud[i,1], -subject_eyes_point_cloud[i,2],str(i))
    # s_left_pupil3D = np.mean(subject_eyes_point_cloud[L_PUPIL_1:L_PUPIL_2,:],axis=0)
    left_gaze_vector = S_openface[['gaze_0_x','gaze_0_y','gaze_0_z']].loc[frame]

    # ax.quiver3D(s_left_pupil3D[0], -s_left_pupil3D[1], -s_left_pupil3D[2],left_gaze_vector[0],-left_gaze_vector[1],-left_gaze_vector[2],normalize=True,length=300)

    # plt.show()
    dist1 = np.linalg.norm(s_face_traingluated[selectedKeyPointsSubjectFace][0]-s_face_traingluated[selectedKeyPointsSubjectFace][1])
    dist2 = np.linalg.norm(s_OF_3D_FacePointCloud[selectedKeyPointsSubjectFace][0]-s_OF_3D_FacePointCloud[selectedKeyPointsSubjectFace][1])
    s_OF_3D_FacePointCloud/= (dist2/dist1)
    R,t = rigid_transform_3D(s_OF_3D_FacePointCloud[selectedKeyPointsSubjectFace].T,s_face_traingluated[selectedKeyPointsSubjectFace].T)
    sFace3D = ((R@s_OF_3D_FacePointCloud.T)+t).T

    sFace3D = ((R_bs@s_OF_3D_FacePointCloud.T) +T_bs).T
    ax.scatter3D(s_face_traingluated[:,0],s_face_traingluated[:,1],s_face_traingluated[:,2])
    ax.scatter3D(sFace3D[:,0],sFace3D[:,1],sFace3D[:,2])
    plt.show()
    # subject_eyes_point_cloud/= (dist2/dist1)
    sEyes3D = ((R@s_OF_3D_EyesPointCloud.T)+t).T
    sEyes3D = s_eyes_traingluated
    marissa_point_cloud  = np.reshape(np.array(M_openface[face3D_lndmks_cols].loc[frame]),(N_FACIAL_LNDMKS,3))
    dist1 = np.linalg.norm(mface3D_tri[selectedKeyPointsMarissaFace][0]-mface3D_tri[selectedKeyPointsMarissaFace][1])
    dist2 = np.linalg.norm(marissa_point_cloud[selectedKeyPointsMarissaFace][0]-marissa_point_cloud[selectedKeyPointsMarissaFace][1])
    marissa_point_cloud/= (dist2/dist1)
    R,t = rigid_transform_3D(marissa_point_cloud[selectedKeyPointsMarissaFace].T,mface3D_tri[selectedKeyPointsMarissaFace].T)
    mFace3D = (R@(marissa_point_cloud.T+t)).T
    mFace3D = R_bm@(mFace3D+T_bm.T).T
    mFace3D = (R_bs.T)@(mFace3D-T_bs)
    mFace3D = mFace3D.T

    mface_in_both_world = R_bm@(mface3D_tri+T_bm.T).T
    mface_in_subject_world = (R_bs.T)@(mface_in_both_world-T_bs)
    mface3D_tri = mface_in_subject_world.T

    # creating the plot
    # from sklearn.preprocessing import normalize

    # face = normalize(face,axis=0)
    # mface = normalize(mface,axis=0)
    # plot_geeks = ax.scatter3D(subject_point_cloud[:,0],subject_point_cloud[:,1],subject_point_cloud[:,2])
    # plot_geeks = ax.scatter3D(subject_eyes_point_cloud[:,0],subject_eyes_point_cloud[:,1],subject_eyes_point_cloud[:,2])

    # plt.show()
    subject_faces +=[sFace3D]
    subject_eyes +=[sEyes3D]
    marissa_faces+=[mFace3D]

    # for i in range(N_FACIAL_LNDMKS):
    #     ax.text(sface3D[i,0], -sface3D[i,1], -sface3D[i,2],str(i))
    s_left_pupil3D = np.mean(sEyes3D[L_PUPIL_1:L_PUPIL_2,:],axis=0)
    s_right_pupil3D = np.mean(sEyes3D[R_PUPIL_1:R_PUPIL_2,:],axis=0)
    left_gaze_vector = S_openface[['gaze_0_x','gaze_0_y','gaze_0_z']].loc[frame]
    right_gaze_vector = S_openface[['gaze_1_x','gaze_1_y','gaze_1_z']].loc[frame]

    # mFacePlanePoints = [Point3D(p) for p in mFace3D[faceDefiningKeyPoints]]
    # mFacePlane = Plane(mFacePlanePoints[0],mFacePlanePoints[1],mFacePlanePoints[2])
    # line = Line3D([s_right_pupil3D[0], s_right_pupil3D[1], s_right_pupil3D[2]],direction_ratio=[right_gaze_vector[0].item(), right_gaze_vector[1].item(), right_gaze_vector[2].item()])
    # intr = mFacePlane.intersection(line)[0].coordinates
    # hps += [(intr,calibartion_frames[pen_phase][0],frame)]
    vec_rep = ((s_right_pupil3D[0], s_right_pupil3D[1], s_right_pupil3D[2]),(right_gaze_vector[0].item(), right_gaze_vector[1].item(), right_gaze_vector[2].item()))
    plane_rep = [p.tolist() for p in mFace3D[faceDefiningKeyPoints]]
    intr = intersect_vector_plane(vec_rep,plane_rep[0],plane_rep[1],plane_rep[2])
    hps += [(intr,calibartion_frames[pen_phase][0],frame)]
    
    fig = plt.figure()
    ax = Axes3D(fig)
    plot_geeks = ax.scatter3D(s_face_traingluated[:,0], -s_face_traingluated[:,1], -s_face_traingluated[:,2],  color=mcolors.CSS4_COLORS['lightgreen'],alpha = 0.4)
    plot_geeks = ax.scatter3D(mface3D_tri[:,0], -mface3D_tri[:,1], -mface3D_tri[:,2],  color=mcolors.CSS4_COLORS['coral'],alpha=0.4)
    plot_geeks = ax.scatter3D(sFace3D[:,0], -sFace3D[:,1], -sFace3D[:,2], color=mcolors.CSS4_COLORS['darkgreen'])
    plot_geeks = ax.scatter3D(mFace3D[:,0], -mFace3D[:,1], -mFace3D[:,2], color= mcolors.CSS4_COLORS['crimson'])
    plot_geeks = ax.scatter3D(sEyes3D[:,0], -sEyes3D[:,1], -sEyes3D[:,2], color='red')
    plot_geeks = ax.scatter3D(s_left_pupil3D[0], -s_left_pupil3D[1], -s_left_pupil3D[2], color='purple')
    plot_geeks = ax.scatter3D(s_right_pupil3D[0], -s_right_pupil3D[1], -s_right_pupil3D[2], color='purple')
    ax.quiver3D(s_left_pupil3D[0], -s_left_pupil3D[1], -s_left_pupil3D[2],left_gaze_vector[0],-left_gaze_vector[1],-left_gaze_vector[2],normalize=True,length=300)
    ax.quiver3D(s_right_pupil3D[0], -s_right_pupil3D[1], -s_right_pupil3D[2],right_gaze_vector[0],-right_gaze_vector[1],-right_gaze_vector[2],normalize=True,length=300)
    ax.scatter3D(intr[0],-intr[1],-intr[2],s=100)

    ax.set_xlim(-130,20)
    ax.set_ylim(-100,3  )
    ax.set_zlim(-200,-100)
    ax.view_init(800,90) #elev,azim

    #camera level meshgrid plot
    # x_ = np.linspace(-200, 100, 5)
    # y_ = np.zeros_like(x_)
    # z_ = np.linspace(-200, 100, 5)
    # x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    # ax.scatter3D(x, y, z, alpha=0.01,color='black')
    fig.canvas.mpl_connect("key_press_event", move_view)
    plt.show()
    del fig
    del ax
    plt.savefig('3d2/'+str(frame)+'.png')
    plt.cla()
    plt.clf()
    plt.close()
import os
fig = plt.figure()
for ip in intersection_points:
    fig.scatter
plt.show()
plt.savefig('TB.png')
plt.clf()
images = [img for img in os.listdir('tempFigs/') if img.endswith(".png")]
frame = cv.imread(os.path.join('tempFigs/', images[0]))
height, width, layers = frame.shape
fourcc = cv.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
video = cv.VideoWriter('TB_234.mp4', fourcc, 20, (width,height))

for image in tqdm.tqdm(images):
    video.write(cv.imread(os.path.join('tempFigs/', image)))

cv.destroyAllWindows()
video.release()
# with open('3Dpickles/hps_2','wb') as f:
#     dump(hps,f)
# with open('3Dpickles/sface_2','wb') as f:
#     dump(subject_faces,f)
# with open('3Dpickles/seye_2','wb') as f:
#     dump(subject_eyes,f)
# with open('3Dpickles/mface_2','wb') as f:
#     dump(marissa_faces,f)
