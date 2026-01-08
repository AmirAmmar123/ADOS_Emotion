from pickle import load,dump
import pandas as pd
import cv2 as cv
import numpy as np
from sympy import Point3D,Plane,Line3D
import matplotlib
import tqdm
from RIGID3D import *
from MathCode import intersect_vector_plane,find_x_value_point,point_line_distance
from StereoCalibrate import stereo_calibrate,calibrate_camera
from Files import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
from itertools import product
subjects_list = ['209','232','235','234','230','212','210']
subjects_list = [subjects_list[-1]]
for subject,task in set(product(set(subjects_list),set(('pen1','pen2','TB')))):
    print(subject,task)
    ######################################################################################
    # subject = '210'
    # task = 'pen1'
    ######################################################################################

    subject_folder = 'c:/MetaData/'+subject+'/openCV/'
    openface_source = 'c:/openface_dired/'+subject+'/'


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

    # with open(subject_folder+'s_m_stereoParams','rb') as f:
    #     s_m_stereo_params =  load(f)
    # w,z,x,y,R_ms,T_ms =s_m_stereo_params
    # T_ms = T_ms.T

    import glob


    camera1_frames = glob.glob('C:/MetaData/'+subject+'/s_b_stereo_frames/*.png')
    camera2_frames = glob.glob('C:/MetaData/'+subject+'/b_s_stereo_frames/*.png')
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


    S_openFace_File_Cropped  = openface_source + subject+'_'+task+'_subject_cropped.csv'
    S_openFace_File  = openface_source + subject+'_'+task+'_subject.csv'
    B_openFace_File_s  = openface_source + subject+'_'+task+'_both_Subject.csv'
    B_openFace_File_m  = openface_source + subject+'_'+task+'_both_Marissa.csv'
    M_openFace_File = openface_source + subject+'_'+task+'_marissa.csv'

    # S_openFace_File_Cropped  = openface_source + '234_pen1_subject_cropped.csv'
    # S_openFace_File  = openface_source + '234_pen1_subject.csv'
    # B_openFace_File_s  = openface_source + '234_pen1_both_Subject.csv'
    # B_openFace_File_m  = openface_source + '234_pen1_both_Marissa.csv'
    # M_openFace_File = openface_source + '234_pen1_marissa.csv'

    # S_openFace_File_Cropped  = openface_source + '234_TB_subject_cropped.csv'
    # S_openFace_File  = openface_source + '234_TB_subject.csv'
    # B_openFace_File_s  = openface_source + '234_TB_both_Subject.csv'
    # B_openFace_File_m  = openface_source + '234_TB_both_Marissa.csv'
    # M_openFace_File = openface_source + '234_TB_marissa.csv'

    # S_openFace_File_Cropped  = openface_source + subject+'_TB_subject_cropped.csv'
    # S_openFace_File  = openface_source + subject+'_TB_subject.csv'
    # B_openFace_File_s  = openface_source + subject+'_TB_both_Subject.csv'
    # B_openFace_File_m  = openface_source + subject+'_TB_both_Marissa.csv'
    # M_openFace_File = openface_source + subject+'_TB_marissa.csv'

    calibration_file = 'G:/My Drive/ADOS/Videos/Unsorted/207/Asaf/207_First_Pen1_I 30-04-2023 12-59-44 Questions.csv'

    S_openface = pd.read_csv(S_openFace_File).set_index('frame')
    S_openface_cropped = pd.read_csv(S_openFace_File_Cropped).set_index('frame')
    B_openface_s = pd.read_csv(B_openFace_File_s).set_index('frame')
    B_openface_m = pd.read_csv(B_openFace_File_m).set_index('frame')
    M_openface = pd.read_csv(M_openFace_File).set_index('frame')

    both = (942,1778)
    subjec = (822,1212)
    # subject face triangulation
    for df in [S_openface,B_openface_s,B_openface_m,M_openface,S_openface_cropped]:
        cols = df.columns
        new_cols = [col.strip() for col in cols]
        df.columns = new_cols
    # print(list(S_openface.columns))
    # exit(1)

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

    selected_left_eye_kp_for_RT = np.array([8,11,14,17])
    selected_right_eye_kp_for_RT = selected_left_eye_kp_for_RT+(36-8)
    intersection_points = []

    intersection_points_dict =create_binary_dict(size = 4)
    face_points_dict = {True:[],False:[]}
    eyes_points_dict = {True:[],False:[]}
    lines_used = create_binary_dict(size=4)
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




    # 



        s_face_traingluated = triangulate(P1,P2,s_face_s_2d,s_face_b_2d)
        s_eyes_traingluated = triangulate(P1,P2,s_eyes_s_2d,s_eyes_b_2d)
        mface3D_tri = triangulate(P3,P4,m_face_m_2d,m_face_b_2d)
        mface3D_tri = triangulate(P3,P4,m_face_m_2d,m_face_b_2d)

        m_final_3d_face = ( R_bm@mface3D_tri.T +T_bm).T

        #this is where to change the parameters of running
        for face_cloud_from_cropped in [True,False]:


            if not face_cloud_from_cropped:
                s_OF_3D_FacePointCloud  = np.reshape(np.array(S_openface[face3D_lndmks_cols].loc[frame]),(N_FACIAL_LNDMKS,3))
                s_OF_3D_EyesPointCloud  = np.reshape(np.array(S_openface[eye3D_lndmks_cols].loc[frame]),(N_EYES_LNDMKS,3))

            else:
                s_OF_3D_FacePointCloud  = np.reshape(np.array(S_openface_cropped[face3D_lndmks_cols].loc[frame]),(N_FACIAL_LNDMKS,3))
                s_OF_3D_EyesPointCloud  = np.reshape(np.array(S_openface_cropped[eye3D_lndmks_cols].loc[frame]),(N_EYES_LNDMKS,3))



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
            s_OF_3D_FacePointCloud =  ((R@s_OF_3D_FacePointCloud.T) +t).T


            s_final_face_3D = s_OF_3D_FacePointCloud

            R,t = rigid_transform_3D(s_OF_3D_EyesPointCloud.T,s_eyes_traingluated.T)
            transformedEyes = ((R@s_OF_3D_EyesPointCloud.T)+t).T
            s_final_eyes_3D = transformedEyes



            s_face_traingluated = s_OF_3D_FacePointCloud
            s_eyes_traingluated = s_OF_3D_EyesPointCloud

            #bring marissa face to currect location
            s_final_face_3D =( R_bs@s_final_face_3D.T +T_bs).T
            transformedEyes = (R_bs@transformedEyes.T+T_bs).T
            for gaze_from_cropped in [True,False]:


                if gaze_from_cropped:
                    left_gaze_vector = R_bs@S_openface_cropped[['gaze_0_x','gaze_0_y','gaze_0_z']].loc[frame] 
                    right_gaze_vector = R_bs@S_openface_cropped[['gaze_1_x','gaze_1_y','gaze_1_z']].loc[frame]

                else:
                    left_gaze_vector = R_bs@S_openface[['gaze_0_x','gaze_0_y','gaze_0_z']].loc[frame] 
                    right_gaze_vector = R_bs@S_openface[['gaze_1_x','gaze_1_y','gaze_1_z']].loc[frame]
                for gaze_from_combined in [True,False]:
                    
                    

                    s_left_pupil3D = np.mean(transformedEyes[L_PUPIL_1:L_PUPIL_2,:],axis=0)
                    s_right_pupil3D = np.mean(transformedEyes[R_PUPIL_1:R_PUPIL_2,:],axis=0)
                    middle_eye = (s_right_pupil3D+s_left_pupil3D)/2
                    combined_vector = left_gaze_vector+right_gaze_vector
                    combined_vector = combined_vector/np.linalg.norm(combined_vector)
                    fig = plt.figure()
                    ax = Axes3D(fig)
                    ax.scatter3D(s_left_pupil3D[0],s_left_pupil3D[1],s_left_pupil3D[2],color='red')
                    ax.scatter3D(s_right_pupil3D[0],s_right_pupil3D[1],s_right_pupil3D[2],color='blue')
                    ax.quiver3D(middle_eye[0],middle_eye[1],middle_eye[2],combined_vector[0],combined_vector[1],combined_vector[2],length=50,color='purple')
                    ax.quiver3D(s_left_pupil3D[0],s_left_pupil3D[1],s_left_pupil3D[2],left_gaze_vector[0],left_gaze_vector[1],left_gaze_vector[2],length=50,color='red')
                    ax.quiver3D(s_right_pupil3D[0],s_right_pupil3D[1],s_right_pupil3D[2],right_gaze_vector[0],right_gaze_vector[1],right_gaze_vector[2],length=50,color='blue')
                    ax.scatter3D(m_final_3d_face[:,0],m_final_3d_face[:,1],m_final_3d_face[:,2],color='green')
                    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
                    ax.view_init(-77,-90) #elev,azim

                    
                    mFacePlanePoints = m_final_3d_face[faceDefiningKeyPoints]
                    marissa_face_mean = m_final_3d_face.mean(axis=0)
                    for intersect_vector in [True,False]:
                        if gaze_from_combined:
                            if intersect_vector:
                                intr = intersect_vector_plane((middle_eye,combined_vector),mFacePlanePoints)
                            else:
                                # ax.scatter3D(marissa_face_mean[0],marissa_face_mean[1],marissa_face_mean[2])
                                # ax.scatter3D(m_final_3d_face[:,0],m_final_3d_face[:,1],m_final_3d_face[:,2])

                                intr = point_line_distance(marissa_face_mean,(middle_eye,combined_vector))
                                lines_used[(face_cloud_from_cropped,gaze_from_cropped,gaze_from_combined,intersect_vector)].append((middle_eye,combined_vector))
                                ax.scatter3D(intr[0],intr[1],intr[2])
                                print('combined', intr[1])
                                # plt.show()
                        else:
                            if intersect_vector:
                                intr = intersect_vector_plane((s_right_pupil3D,right_gaze_vector),mFacePlanePoints)
                            else:
                                intr = point_line_distance(marissa_face_mean,(s_right_pupil3D,right_gaze_vector))
                                lines_used[(face_cloud_from_cropped,gaze_from_cropped,gaze_from_combined,intersect_vector)].append((s_right_pupil3D,right_gaze_vector))
                                ax.scatter3D(*intr)
                                print('right', intr[1])
                                plt.show()
                        intersection_points_dict[(face_cloud_from_cropped,gaze_from_cropped,gaze_from_combined,intersect_vector)].append(intr)
            eyes_points_dict[face_cloud_from_cropped].append((s_left_pupil3D,s_right_pupil3D))
            face_points_dict[face_cloud_from_cropped].append(s_final_face_3D)
        marissa_faces.append(m_final_3d_face)

    # with open('3Dpickles/intr_'+subject+'_'+task+'','wb') as f:
    #     dump(intersection_points,f)
    # with open('3Dpickles/sFace_'+subject+'_'+task+'','wb') as f:
    #     dump(subject_faces,f)
    # with open('3Dpickles/sEyes_'+subject+'_'+task+'','wb') as f:
    #     dump(subject_eyes,f)
    from CheckCreate import check_create
    check_create('3Dpickles')
    with open('3Dpickles/intrDict_'+subject+'_'+task+'','wb') as f:
        dump(intersection_points_dict,f)
    with open('3Dpickles/sFaceDict_'+subject+'_'+task+'','wb') as f:
        dump(face_points_dict,f)
    with open('3Dpickles/sEyesDict_'+subject+'_'+task+'','wb') as f:
        dump(eyes_points_dict,f)
    with open('3Dpickles/mFace_'+subject+'_'+task+'','wb') as f:
        dump(marissa_faces,f)
    with open('3Dpickles/linesUsed_'+subject+'_'+task+'','wb') as f:
        dump(lines_used,f)
    # with open('3Dpickles/intr_234_TB','wb') as f:
    #     dump(intersection_points,f)
    # with open('3Dpickles/sFace_234_TB','wb') as f:
    #     dump(subject_faces,f)
    # with open('3Dpickles/sEyes_234_TB','wb') as f:
    #     dump(subject_eyes,f)
    # with open('3Dpickles/mFace_234_TB','wb') as f:
    #     dump(marissa_faces,f)

    # with open('3Dpickles/intr_234_pen1','wb') as f:
    #     dump(intersection_points,f)
    # with open('3Dpickles/sFace_234_pen1','wb') as f:
    #     dump(subject_faces,f)
    # with open('3Dpickles/sEyes_234_pen1','wb') as f:
    #     dump(subject_eyes,f)
    # with open('3Dpickles/mFace_234_pen1','wb') as f:
    #     dump(marissa_faces,f)


    # with open('3Dpickles/intr_'+subject+'_TB','wb') as f:
    #     dump(intersection_points,f)
    # with open('3Dpickles/sFace_'+subject+'_TB','wb') as f:
    #     dump(subject_faces,f)
    # with open('3Dpickles/sEyes_'+subject+'_TB','wb') as f:
    #     dump(subject_eyes,f)
    # with open('3Dpickles/mFace_'+subject+'_TB','wb') as f:
    #     dump(marissa_faces,f)