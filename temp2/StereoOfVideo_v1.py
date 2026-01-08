from turtle import shape
import cv2
import numpy as np
from Files import get_folder,get_file,get_files,load_pickle,save_pickle_file
import scipy.io as sio
from scipy import ndimage
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from RIGID3D import *
from os.path import isfile
from sympy import Point3D,Plane,Line3D

# from NewCalibration import calc_frame
# from vpython import *
FACE_LANDMARKS = [8,12,54,33,35,45,16]
FACE_LANDMARKS.sort() # or centroid
def calc_frame(sp,mp,gazeVecL,gazeVecR,by='dist'):
    if by == 'plane':
        return calc_frame_byPlane(sp,mp,gazeVecL,gazeVecR)
    else:
        return calc_frame_byDist(sp,mp,gazeVecL,gazeVecR)
def calc_frame_byDist(sp,mp,gazeVecL,gazeVecR):
    xs,ys,zs = sp
    xm,ym,zm = mp
    xsArray = np.array(xs)
    ysArray = np.array(ys)
    zsArray = np.array(zs)
    xmArray = np.array(xm)
    ymArray = np.array(ym)
    zmArray = np.array(zm)
    xDist = np.max(xsArray-xmArray)
    yDist = np.max(ysArray - ymArray)
    zDist = np.max(zsArray- zmArray)
    zDistI = np.where((zsArray-zmArray)==zDist)

    t = (zmArray[zDistI[0]] - zsArray[zDistI[0]])/gazeVecR[2]
    # X = X0 + t · Dx
    # Y = Y0 + t · Dy
    # Z = Z0 + t · Dz
    # use zDist 
    hitPoints = list()
    rightPupil = ((xsArray[49]+xsArray[53]+xsArray[51]+xsArray[55])/4,(ysArray[49]+ysArray[53]+ysArray[51]+ysArray[55])/4
    ,(zsArray[49]+zsArray[53]+zsArray[51]+zsArray[55])/4)
    pTx = rightPupil[0]+t*gazeVecR[0]
    pTy = rightPupil[1]+t*gazeVecR[1]
    pTz = rightPupil[2]+t*gazeVecR[2]
    hitPoints.append((pTx,pTy,pTz))
    return hitPoints
def calc_frame_byPlane(sp,mp,gazeVecL,gazeVecR):
    xs,ys,zs = sp
    xm,ym,zm = mp
    # xsArray = np.array(xs)
    # ysArray = np.array(ys)
    # zsArray = np.array(zs)
    # xmArray = np.array(xm)
    # ymArray = np.array(ym)
    # zmArray = np.array(zm)
    # xDist = np.max(xsArray-xmArray)
    # yDist = np.max(ysArray - ymArray)
    # zDist = np.max(zsArray- zmArray)
    # zDistI = np.where((zsArray-zmArray)==zDist)

    # t = (zmArray[zDistI[0]] - zsArray[zDistI[0]])/gazeVecR[2]
    # # X = X0 + t · Dx
    # # Y = Y0 + t · Dy
    # # Z = Z0 + t · Dz
    # # use zDist 


    facePlanePointsX =[xm[56+0].item(),xm[56+8].item(),xm[56+16].item()]
    facePlanePointsY =[ym[56+0].item(),ym[56+8].item(),ym[56+16].item()]
    facePlanePointsZ =[zm[56+0].item(),zm[56+8].item(),zm[56+16].item()]
    faceVerts =[list(zip(facePlanePointsX, facePlanePointsY, facePlanePointsZ))]


    points = [[facePlanePointsX[0],facePlanePointsY[0], facePlanePointsZ[0]],
           [facePlanePointsX[1], facePlanePointsY[1], facePlanePointsZ[1]],
           [facePlanePointsX[2], facePlanePointsY[2], facePlanePointsZ[2]]]
    p0, p1, p2 = points
    hitPoints = list()
    rightPupil = ((xs[49]+xs[53]+xs[51]+xs[55])/4,(ys[49]+ys[53]+ys[51]+ys[55])/4
    ,(zs[49]+zs[53]+zs[51]+zs[55])/4)
    plane = Plane(p0,p1,p2)
    line = Line3D([rightPupil[0], rightPupil[1], rightPupil[2]],direction_ratio=[gazeVecR[0].item(), gazeVecR[1].item(), gazeVecR[2].item()])
    intr = plane.intersection(line)
    hp = np.array(intr[0],dtype=float)
    hitPoints.append((hp[0],hp[1],hp[2]))
    return hitPoints
def loadCameraParam(folder=None):
    MATLAB_NAMES  = ["intrinsicMatrix1","distortionCoefficients1","intrinsicMatrix2","distortionCoefficients2","rotationOfCamera2","translationOfCamera2"]
    PARAMS_NAMES = ['CameraParameters1','CameraParameters2','RotationOfCamera2','TranslationOfCamera2']
    if not folder:
        matlabVarsFolder = get_folder("Select matlab folder")
    else:
        matlabVarsFolder = folder
    K1 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[0]+".mat")[MATLAB_NAMES[0]]
    D1= sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[1]+".mat")[MATLAB_NAMES[1]]
    K2 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[2]+".mat")[MATLAB_NAMES[2]]
    D2 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[3]+".mat")[MATLAB_NAMES[3]]
    R = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[4]+".mat")[MATLAB_NAMES[4]]
    T = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[5]+".mat")[MATLAB_NAMES[5]]
    F = sio.loadmat(matlabVarsFolder+"/"+"F"+".mat")['fund']
    E = sio.loadmat(matlabVarsFolder+"/"+"E"+".mat")['essen']
    # openface = pd.read_csv(matlabVarsFolder+"/"+"faceOF.csv")
    params = dict()
    params['K1'] = K1
    params['D1'] = D1
    params['K2'] = K2
    params['D2'] = D2
    params['R'] = R
    T = np.rot90(T,3)
    params['T'] = T
    params['F'] = F
    params['E'] = E
    return params
def getPoints(params_dict = None,bothOF = None,marrisaOF = None,row_i = 0,face = 0):
    # MATLAB_NAMES  = ["intrinsicMatrix1","distortionCoefficients1","intrinsicMatrix2","distortionCoefficients2","rotationOfCamera2","translationOfCamera2"]
    # PARAMS_NAMES = ['CameraParameters1','CameraParameters2','RotationOfCamera2','TranslationOfCamera2']
    # matlabVarsFolder = get_folder("Select matlab folder")
    # K1 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[0]+".mat")[MATLAB_NAMES[0]]
    # D1= sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[1]+".mat")[MATLAB_NAMES[1]]
    # K2 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[2]+".mat")[MATLAB_NAMES[2]]
    # D2 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[3]+".mat")[MATLAB_NAMES[3]]
    # R = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[4]+".mat")[MATLAB_NAMES[4]]
    # T = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[5]+".mat")[MATLAB_NAMES[5]]
    # # T = np.transpose(T)
    # T = np.rot90(T,3)
    # # R = np.transpose(R)
    # F = sio.loadmat(matlabVarsFolder+"/"+"F"+".mat")['fund']
    # E = sio.loadmat(matlabVarsFolder+"/"+"E"+".mat")['essen']
    if not params_dict:
        params_dict = loadCameraParam()
    K1 = params_dict['K1']
    D1 = params_dict['D1']
    K2 = params_dict['K2']
    D2 = params_dict['D2']
    R = params_dict['R']
    T = params_dict['T']
    # T = np.transpose(T)

    # R = np.transpose(R)
    F = params_dict['F']
    E = params_dict['E']
    xs = list()
    ys = list()
    zs = list()
    xs2 = list()
    ys2= list()
    zs2 = list()
    other_OF = marrisaOF
    main_OF = bothOF
    gazeVec_l = np.expand_dims( np.array([other_OF.at[row_i,' gaze_0_x'],other_OF.at[row_i, ' gaze_0_y'],other_OF.at[row_i, ' gaze_0_z']]),1) #left
    gazeVec_r = np.expand_dims( np.array([other_OF.at[row_i,' gaze_1_x'],other_OF.at[row_i, ' gaze_1_y'],other_OF.at[row_i, ' gaze_1_z']]),1) #left

    for i in range(56):
        x_str = ' eye_lmk_X_'+str(i)
        y_str = ' eye_lmk_Y_'+str(i)
        z_str = ' eye_lmk_Z_'+str(i)
        xs.append(other_OF.at[row_i,x_str])
        ys.append(other_OF.at[row_i,y_str])
        zs.append(other_OF.at[row_i,z_str])
    for i in range(68):
        x_str = ' X_'+str(i)
        y_str = ' Y_'+str(i)
        z_str = ' Z_'+str(i)
        xs.append(other_OF.at[row_i,x_str])
        ys.append(other_OF.at[row_i,y_str])
        zs.append(other_OF.at[row_i,z_str])
    for i in range(56):
        x_str = ' eye_lmk_X_'+str(i)
        y_str = ' eye_lmk_Y_'+str(i)
        z_str = ' eye_lmk_Z_'+str(i)
        xs2.append(main_OF.at[face,x_str])
        ys2.append(main_OF.at[face,y_str])
        zs2.append(main_OF.at[face,z_str])
    for i in range(68):
        x_str = ' X_'+str(i)
        y_str = ' Y_'+str(i)
        z_str = ' Z_'+str(i)
        xs2.append(main_OF.at[face,x_str])
        ys2.append(main_OF.at[face,y_str])
        zs2.append(main_OF.at[face,z_str])
    p3ds = []
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = K1 @ RT1 #projection matrix for C1
    
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = K2 @ RT2 #projection matrix for C2
    xs_a = list()
    ys_a= list()
    zs_a =list()
    for i in range(len(xs)):
        uv1 = (xs[i],ys[i],zs[i])
        uv2 = (xs2[i],ys2[i],zs2[i])
        uv1 = Project(uv1,K2)
        uv2 = Project(uv2,K1)
        # print(uv1,uv2)
        _p3d = DLT(P1, P2, uv2,uv1)
        # uv1  = np.ndarray([int(uv1[0]),int(uv1[1])])
        # uv2  = np.ndarray([int(uv2[0]),int(uv2[1])])
        # p1 = np.ndarray(shape=(2,1))
        # p2 = np.ndarray(shape=(2,1))
        # p1[0]  = int(uv1[0])
        # p1[1] = int(uv1[1])
        # p2[0]  = int(uv1[0])
        # p2[1] = int(uv1[1])
        # # uv2  = np.ndarray([int(uv2[0]),int(uv2[1])])
        # _p3d = cv2.triangulatePoints(P1,P2,p1,p2)
        # print("p3ds",i)
        xs_a.append(_p3d[0].item())
        ys_a.append(_p3d[1].item())
        zs_a.append(_p3d[2].item())
        p3ds.append(_p3d)
    return xs_a,ys_a,zs_a,(R@gazeVec_l+T),(R@gazeVec_r+T)
def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]
def Project_1(point,fx,fy,cx,cy):
    projected = np.zeros((3,1))
    X = point[0]
    Y = point[1]
    Z = point[2]
    x=0
    y=0
    if not Z==0:
        x = ((X * fx / Z) + cx);
        y = ((Y * fy / Z) + cy);
    else:
        x=X
        y=Y
    return (x,y,0)
def Project(point,K):
    projected = np.zeros((3,1))
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    X = point[0]
    Y = point[1]
    Z = point[2]
    x=0
    y=0
    if not Z==0:
        x = ((X * fx / Z) + cx);
        y = ((Y * fy / Z) + cy);
    else:
        x=X
        y=Y
    return (x,y,0)
CONVERTED = True


# R = np.transpose(R)
# pixleVector = np.expand_dims(np.array([806,388,1]),1)
def get3DPointsFromVideo(save_video = True,reload =False, drawHitPoints=False):
    if drawHitPoints:
        hitPoints,(meanx,meany,meanz) = load_pickle()
    state = None
    if reload and isfile("get3DPointsFromVideo_state.pickle"):
        state = load_pickle("get3DPointsFromVideo_state.pickle")
    # state ={'c1ParamsF': 'C:/Users/user/Desktop/calib subject both', 'c2ParamsF': 'C:/Users/user/Desktop/calib Both Marisa', 'subjectOF_file': 'C:/Users/user/Desktop/OpenFace_2.2.0_win_x64/OpenFace_2.2.0_win_x64/processed/pen subject.csv', 'marrisaOF_file': 'C:/Users/user/Desktop/OpenFace_2.2.0_win_x64/OpenFace_2.2.0_win_x64/processed/pen marrisa.csv', 'bothCSVFolder': 'F:/ALL VIDEOS/record16/sync/pen frames both openface output'}
    PointOfFrames = list()
    if state:
        c1ParamsF = state['c1ParamsF']
        c2ParamsF = state['c2ParamsF']
        c1Params = loadCameraParam(c1ParamsF)
        c2Params = loadCameraParam(c2ParamsF)
        subjectOF_file = state['subjectOF_file']
        marrisaOF_file = state['marrisaOF_file']
        subjectOF = pd.read_csv(subjectOF_file)
        marrisaOF = pd.read_csv(marrisaOF_file)
        bothCSVFolder = state['bothCSVFolder']
    else:
        c1ParamsF = get_folder("Select camera 1 camera 3 folder ")
        c2ParamsF = get_folder("Select camera 3 camera 2 folder ")
        c1Params = loadCameraParam(c1ParamsF)
        c2Params = loadCameraParam(c2ParamsF)
        subjectOF_file = get_file("Please select subject openface file of pen")
        marrisaOF_file = get_file("Please select marissa openface file of pen")
        subjectOF = pd.read_csv(subjectOF_file)
        marrisaOF = pd.read_csv(marrisaOF_file)
        bothCSVFolder = get_folder("Please select camera3 CSVs folder")
        state ={'c1ParamsF': c1ParamsF , 'c2ParamsF': c2ParamsF, 'subjectOF_file': subjectOF_file, 'marrisaOF_file': marrisaOF_file, 'bothCSVFolder': bothCSVFolder}
        save_pickle_file(state)
    

    print(subjectOF.at[0,'frame'])
    openface2 = pd.read_csv(bothCSVFolder+"/"+"frame"+str(0)+".csv")
    r1_nose_x0 = openface2.at[0," eye_lmk_X_0"]
    r2_nose_x0 = openface2.at[1," eye_lmk_X_0"]
    if(r1_nose_x0 > r2_nose_x0):
        face0=0
        subject_nose_x0 = r1_nose_x0
        marissa_nose_x0 = r2_nose_x0
    else:
        face0=1
        subject_nose_x0 = r2_nose_x0
        marissa_nose_x0 = r1_nose_x0

    counter = 0
    filenames = list()
    for index, row in subjectOF.iterrows():
        counter +=1
        gazeVec_l = np.expand_dims( np.array([row[' gaze_0_x'],row[ ' gaze_0_y'],row[ ' gaze_0_z']]),1) #left
        gazeVec_r = np.expand_dims( np.array([row[' gaze_1_x'],row[ ' gaze_1_y'],row[ ' gaze_1_z']]),1) #left
        p2Vector  =np.expand_dims( np.array([1430,533,1]),1)

        # Rinv = np.linalg.inv(R)
        # gazeVec_r_RT = np.matmul(Rinv,gazeVec_r)-T
        num_of_f = 55
        xs = list()
        ys = list()
        zs = list()
        xs2 = list()
        ys2= list()
        zs2 = list()
        openface2 = pd.read_csv(bothCSVFolder+"/"+"frame"+str(counter-1)+".csv")
        r1_nose_x0 = openface2.at[0," eye_lmk_X_0"]
        r2_nose_x0 = openface2.at[1," eye_lmk_X_0"]
        if(r1_nose_x0 > r2_nose_x0):
            face=0
        else:
            face=1
        for i in range(56):
            x_str = ' eye_lmk_X_'+str(i)
            y_str = ' eye_lmk_Y_'+str(i)
            z_str = ' eye_lmk_Z_'+str(i)
            xs.append(row[x_str])
            ys.append(row[y_str])
            zs.append(row[z_str])
        for i in range(68):
            x_str = ' X_'+str(i)
            y_str = ' Y_'+str(i)
            z_str = ' Z_'+str(i)
            xs.append(row[x_str])
            ys.append(row[y_str])
            zs.append(row[z_str])
        for i in range(56):
            x_str = ' eye_lmk_X_'+str(i)
            y_str = ' eye_lmk_Y_'+str(i)
            z_str = ' eye_lmk_Z_'+str(i)
            xs2.append(openface2.at[face,x_str])
            ys2.append(openface2.at[face,y_str])
            zs2.append(openface2.at[face,z_str])
        for i in range(68):
            x_str = ' X_'+str(i)
            y_str = ' Y_'+str(i)
            z_str = ' Z_'+str(i)
            xs2.append(openface2.at[face,x_str])
            ys2.append(openface2.at[face,y_str])
            zs2.append(openface2.at[face,z_str])

        
        p3ds = []
        #RT matrix for C1 is identity.
        RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        P1 = c1Params['K1'] @ RT1 #projection matrix for C1
        
        #RT matrix for C2 is the R and T obtained from stereo calibration.
        RT2 = np.concatenate([c1Params['R'], c1Params['T']], axis = -1)
        P2 = c1Params['K2'] @ RT2 #projection matrix for C2
        # table_edge1 = (168,941)

        # table_edge2 = (1153,785)
        xs_a = list()
        ys_a= list()
        zs_a =list()
        for i in range(len(xs)):
            uv1 = (xs[i],ys[i],zs[i])
            uv2 = (xs2[i],ys2[i],zs2[i])
            uv1 = Project(uv1,c1Params['K1'])
            uv2 = Project(uv2,c1Params['K2'])
            # print(uv1,uv2)
            # print(uv1,uv2)
            _p3d = DLT(P1, P2, uv1, uv2)
            # p1 = np.ndarray(shape=(2,1))
            # p2 = np.ndarray(shape=(2,1))
            # p1[0]  = int(uv1[0])
            # p1[1] = int(uv1[1])
            # p2[0]  = int(uv1[0])
            # p2[1] = int(uv1[1])
            # # uv2  = np.ndarray([int(uv2[0]),int(uv2[1])])
            # _p3d = cv2.triangulatePoints(P1,P2,p1,p2)
            # print("p3ds",i)
            xs_a.append(_p3d[0].item())
            ys_a.append(_p3d[1].item())
            zs_a.append(_p3d[2].item())
            p3ds.append(_p3d)
        # _p3d = DLT(P1, P2, table_edge1, table_edge2)
        print("p3ds",i)
        xs_a.append(_p3d[0])
        ys_a.append(_p3d[1])
        zs_a.append(_p3d[2])
        # p3ds.append(_p3d)
        p3ds = np.array(p3ds)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.set_xlim3d(-15, 5)
        # ax.set_ylim3d(-10, 10)
        # ax.set_zlim3d(10, 30)
        
        # connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
        # for _c in connections:
            # print(p3ds[_c[0]])
            # print(p3ds[_c[1]])
            # ax.scatter(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
        
        # plt.show()
        # xs_a = np.array(xs)
        # # xs_a =xs_a-xs_a.min()
        # # xs_a = xs_a/xs_a.sum()
        # ys_a = np.array(ys)
        # # ys_a =ys_a-ys_a.min()
        # # ys_a = ys_a/ys_a.sum()
        # zs_a = np.array(zs)
        # # zs_a =zs_a-zs_a.min()
        # # zs_a = zs_a/zs_a.sum()
        # # print(xs)
        # print(ys)
        # print(zs)
        # print(xs,ys,zs)
        # xs1,ys1,zs1,gvl1,gvr1 = getPoints()
        # Rinv = np.linalg.inv(c1Params['R'])
        # for i in range(len(xs1)):
        #     v = np.expand_dims(np.array([xs1[i],ys1[i],zs1[i]]),1)
        #     v = Rinv@v -c1Params['T']
        #     xs1[i] = v[0]
        #     ys1[i] = v[1]
        #     zs1[i] = v[2]
        xs1,ys1,zs1,gvl1,gvr1 = getPoints(c2Params,openface2,marrisaOF,counter-1,1-face)
        Rinv = np.linalg.inv(c1Params['R'])
        for i in range(len(xs1)):
            v = np.expand_dims(np.array([xs1[i],ys1[i],zs1[i]]),1)
            v = Rinv@(v -c1Params['T'])
            # v = c2Params['R']@(v +c2Params['T'])
            xs1[i] = v[0]
            ys1[i] = v[1]
            zs1[i] = v[2]

        gvr1 = Rinv@(gvr1 -c1Params['T'])
        gvl1 = Rinv@(gvl1 -c1Params['T'])

        
        # Create Data

        # Voxels is used to customizations of the
        # sizes, positions and colors.
        PointOfFrames.append(((xs_a,ys_a,zs_a),(xs1,ys1,zs1),gazeVec_l,gazeVec_r,row[' timestamp']))
        if drawHitPoints:
            # hitPoints,(meanx,meany,meanz) = load_pickle()
            facePoints_0 = np.ndarray(shape=(3,len(FACE_LANDMARKS)))
            facePoints_1 = np.ndarray(shape=(3,len(FACE_LANDMARKS)))
            for i,lnd in enumerate(FACE_LANDMARKS):
                facePoints_0[0][i] = xs1[lnd]
                facePoints_0[1][i] = ys1[lnd]
                facePoints_0[2][i] = zs1[lnd]
                facePoints_1[0][i] = meanx[lnd]
                facePoints_1[1][i] = meany[lnd]
                facePoints_1[2][i] = meanz[lnd]
                # print(s_x[lnd],s_x1[lnd])
                # facePoints_0[X][i] = s_x[lnd]
                # facePoints_0[Y][i] = s_y[lnd]
                # facePoints_0[Z][i] = s_z[lnd]
                # facePoints_1[X][i] = s_x1[lnd]
                # facePoints_1[Y][i] = s_y1[lnd]
                # facePoints_1[Z][i] = s_z1[lnd]

            fR,fT = rigid_transform_3D(facePoints_0,facePoints_1)
        COLOR = {'Center':'r','Up' : 'g','Down':'b','Right':'y','Left' : 'm'}
        print("save video",save_video)
        if save_video:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(xs_a, ys_a,zs_a,color = 'g')
            ax.scatter3D(xs1[:20], ys1[:20],zs1[:20],color ='b')
            ax.scatter3D(xs1[21], ys1[21],zs1[21],color ='r')
            ax.scatter3D(xs1[22:], ys1[22:],zs1[22:],color ='b')
            if drawHitPoints:
                ax.scatter3D(meanx,meany,meanz,color='c')
                for key in hitPoints.keys():
                    for point in hitPoints[key]:
                        hx, hy,hz = zip(*point)
                        #why s hx all the same?
                        face_matrix = np.ndarray(shape = (3,len(hx)))
                        for l in range(len(hx)):
                            face_matrix[0][l]=hx[l].item()
                            face_matrix[1][l]=hy[l].item()
                            face_matrix[2][l]=hz[l].item()
                            # p0 = np.transpose(p0)
                        p0 = fR@face_matrix +fT
                        ax.scatter(p0[0,:],p0[1,:],p0[2,:],color=COLOR[key])
            hx, hy,hz = zip(*calc_frame((xs_a,ys_a,zs_a),(xs1,ys1,zs1),gazeVec_l,gazeVec_r))
            ax.scatter(hx,hy,hz,color='y')
            # ax.scatter3D(xs1, ys1,zs1,color ='b')
            axes = [5, 5, 5]
            data = np.ones(axes, dtype=np.bool)
            
            # Control Transparency
            alpha = 0.9
            
            # Control colour
            colors = np.empty(axes + [4], dtype=np.float32)
            
            colors[:] = [1, 0, 0, alpha]  # red
            ax.voxels(data, facecolors=colors)
            ax.quiver(xs_a[49], ys_a[49], zs_a[49], gazeVec_r[0], gazeVec_r[1], gazeVec_r[2], length=500 , normalize=True,color='g')
            ax.quiver(xs_a[21], ys_a[21], zs_a[21], gazeVec_l[0], gazeVec_l[1], gazeVec_l[2], length=500 , normalize=True,color='r')
            # ax.plot([xs_a[21], xs_a[21]+100*gazeVec_l[0]], [ys_a[21], ys_a[21]+100*gazeVec_l[1]],zs=[zs_a[21], zs_a[21]+100*gazeVec_l[2]])
            elev = 0
            azim = 0
            ax.view_init(-90,-80) #elev,azim
            ax.set_xlim3d(-75,0)
            ax.set_ylim3d(-40, 80)
            ax.set_zlim3d(-70, 220)

            # plt.axis('square')
            filename = f'{counter}.png'
            filenames.append(filename)
            
            # save frame
            plt.savefig('NewgifOutput3/'+filename)
            if counter == 1:
                plt.show()  
                break
            else:
                plt.close()
            plt.close()
        # plt.close()
    print(len(PointOfFrames))
    return PointOfFrames

        # for i in range(len(xs_a)):
        #     points(pos=vector(xs_a[i],ys_a[i],zs_a[i]), radius=10, color=color.yellow)
        # for i in range(len(xs1)):
        #     points(pos=vector(xs1[i],ys1[i],zs1[i]), radius=10, color=color.blue)   
        # d =20 
        # points(pos=vector(0,0,0), radius=10, color=color.red)   
        # curve(vector(xs_a[49], ys_a[49], zs_a[49]),vector(xs_a[49]+d*gazeVec_r[0], ys_a[49]+d*gazeVec_r[1], zs_a[49]+d*gazeVec_r[2]))
        # curve(vector(xs_a[21], ys_a[21], zs_a[21]),vector(xs_a[21]+d*gazeVec_l[0], ys_a[21]+d*gazeVec_l[1], zs_a[21]+d*gazeVec_l[2]))
        # # curve(vector(xs1[49], -ys1[49], zs1[49]),vector(xs1[49]-d*gvr1[0], -ys1[49]-d*gvr1[1], zs1[49]-d*gvr1[2]))
        # # curve(vector(xs1[21], -ys1[21], zs1[21]),vector(xs1[21]-d*gvl1[0], -ys1[21]-d*gvl1[1], zs1[21]-d*gvl1[2]))
        # scene.camera.pos = vector(0,0,0)
        # break

