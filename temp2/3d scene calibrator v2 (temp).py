subject_folder = 'c:/MetaData/207/openCV/'
openface_source = 'c:/openface_dired/207/'
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
from MathCode import intersect_vector_plane

matplotlib.use("Agg")

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

RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = K1 @ RT1 #projection matrix for C1

#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R_bs, T_bs], axis = -1)
P2 = K2 @ RT2 #projection matrix for C2

P3= K3@RT1

RT3 = np.concatenate([R_bm, T_bm], axis = -1)
P4 = K2@RT3

S_openFace_File  = openface_source + '207_Second_Pen_S.csv'
B_openFace_File_s  = openface_source + '207_Second_Pen_B_Subject.csv'
B_openFace_File_m  = openface_source + '207_Second_Pen_B_Marissa.csv'
M_openFace_File = openface_source + '207_Second_Pen_I.csv'

calibration_file = 'G:/My Drive/ADOS/Videos/Unsorted/207/Asaf/207_Second_Pen_I 30-04-2023 16-46-10 Questions.csv'

S_openface = pd.read_csv(S_openFace_File).set_index('frame')
B_openface_s = pd.read_csv(B_openFace_File_s).set_index('frame')
B_openface_m = pd.read_csv(B_openFace_File_m).set_index('frame')
M_openface = pd.read_csv(M_openFace_File).set_index('frame')

#subject face triangulation
for df in [S_openface,B_openface_s,B_openface_m,M_openface]:
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
M_openface_eye = M_openface[eye_lndmks_cols]

selectedKeyPointsSubjectFace = [16666666666666666666,26,27,33,45,54]
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


for frame in tqdm.tqdm(S_openface.index.values.tolist()):

    #check frame in part of face
    if frame < calibartion_frames[pen_phase][1]:
        continue
    if frame > calibartion_frames[pen_phase][2]:
        #frame out of range for sure
        pen_phase+=1
        if pen_phase == len(calibartion_frames):
            break
        continue
    
    face_array =  np.array(S_openface_face.loc[frame])
    reshaped = np.reshape(face_array,(N_FACIAL_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K1,D1,None,K1)
    s_face_s = udp.squeeze()
    # s_face_s = reshaped

    face_array =  np.array(B_openface_s_face.loc[frame])
    reshaped = np.reshape(face_array,(N_FACIAL_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
    s_face_b = udp.squeeze()
    # s_face_b = reshaped

    face_array =  np.array(M_openface_face.loc[frame])
    reshaped = np.reshape(face_array,(N_FACIAL_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K3,D3,None,K3)
    m_face_m = udp.squeeze()
    # m_face_m = reshaped

    face_array =  np.array(B_openface_m_face.loc[frame])
    reshaped = np.reshape(face_array,(N_FACIAL_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
    m_face_b = udp.squeeze()
    # m_face_b = reshaped


    eye_array =  np.array(S_openface_eye.loc[frame])
    reshaped = np.reshape(eye_array,(N_EYES_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K1,D1,None,K1)
    s_eyes_s = udp.squeeze()

    eye_array =  np.array(B_openface_s_eye.loc[frame])
    reshaped = np.reshape(eye_array,(N_EYES_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
    s_eyes_b = udp.squeeze()

    eye_array =  np.array(M_openface_eye.loc[frame])
    reshaped = np.reshape(eye_array,(N_EYES_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K3,D3,None,K3)
    m_eyes_m = udp.squeeze()

    eye_array =  np.array(B_openface_m_eye.loc[frame])
    reshaped = np.reshape(eye_array,(N_EYES_LNDMKS,2))
    udp = cv.undistortPoints(reshaped,K2,D2,None,K2)
    m_eyes_b = udp.squeeze()



    #2d camera 1 camera 3 -> 3d camera 1
    #3d camera 1 from openface -  precise 
    sface3D_tri = triangulate(P1,P2,s_face_s,s_face_b)
    sEyes3D_tri = triangulate(P1,P2,s_eyes_s,s_eyes_b)
    mface3D_tri = triangulate(P3,P4,m_face_m,m_face_b)
    mface3D_tri = triangulate(P3,P4,m_face_m,m_face_b)
    subject_point_cloud  = np.reshape(np.array(S_openface[face3D_lndmks_cols].loc[frame]),(N_FACIAL_LNDMKS,3))
    subject_eyes_point_cloud  = np.reshape(np.array(S_openface[eye3D_lndmks_cols].loc[frame]),(N_EYES_LNDMKS,3))

    dist1 = np.linalg.norm(sface3D_tri[selectedKeyPointsSubjectFace][0]-sface3D_tri[selectedKeyPointsSubjectFace][1])
    dist2 = np.linalg.norm(subject_point_cloud[selectedKeyPointsSubjectFace][0]-subject_point_cloud[selectedKeyPointsSubjectFace][1])
    subject_point_cloud/= (dist2/dist1)
    R,t = rigid_transform_3D(subject_point_cloud[selectedKeyPointsSubjectFace].T,sface3D_tri[selectedKeyPointsSubjectFace].T)
    sFace3D = ((R@subject_point_cloud.T)+t).T

    # subject_eyes_point_cloud/= (dist2/dist1)
    sEyes3D = ((R@subject_eyes_point_cloud.T)+t).T
    sEyes3D = sEyes3D_tri
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
    plot_geeks = ax.scatter3D(sface3D_tri[:,0], -sface3D_tri[:,1], -sface3D_tri[:,2],  color=mcolors.CSS4_COLORS['lightgreen'],alpha = 0.4)
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

    del fig
    del ax
    plt.savefig('3d2/'+str(frame)+'.png')
    plt.cla()
    plt.clf()
    plt.close()
with open('3Dpickles/hps - second pen','wb') as f:
    dump(hps,f)
with open('3Dpickles/sface - second pen','wb') as f:
    dump(subject_faces,f)
with open('3Dpickles/seye - second pen','wb') as f:
    dump(subject_eyes,f)
with open('3Dpickles/mface - second pen','wb') as f:
    dump(marissa_faces,f)
