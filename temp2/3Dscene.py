import cv2
import numpy as np
from Files import get_folder
import scipy.io as sio
from scipy import ndimage
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
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
def getPoints():
    MATLAB_NAMES  = ["intrinsicMatrix1","distortionCoefficients1","intrinsicMatrix2","distortionCoefficients2","rotationOfCamera2","translationOfCamera2"]
    PARAMS_NAMES = ['CameraParameters1','CameraParameters2','RotationOfCamera2','TranslationOfCamera2']
    matlabVarsFolder = get_folder("Select matlab folder")
    K1 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[0]+".mat")[MATLAB_NAMES[0]]
    D1= sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[1]+".mat")[MATLAB_NAMES[1]]
    K2 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[2]+".mat")[MATLAB_NAMES[2]]
    D2 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[3]+".mat")[MATLAB_NAMES[3]]
    R = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[4]+".mat")[MATLAB_NAMES[4]]
    T = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[5]+".mat")[MATLAB_NAMES[5]]
    T = np.transpose(T)

    F = sio.loadmat(matlabVarsFolder+"/"+"F"+".mat")['fund']
    E = sio.loadmat(matlabVarsFolder+"/"+"E"+".mat")['essen']

    xs = list()
    ys = list()
    zs = list()
    xs2 = list()
    ys2= list()
    zs2 = list()
    other_OF = pd.read_csv(matlabVarsFolder+"/"+"faceOF.csv")
    main_OF = pd.read_csv(matlabVarsFolder+"/"+"frame2.csv")
    gazeVec_l = np.expand_dims( np.array([other_OF.at[0,' gaze_0_x'],other_OF.at[0, ' gaze_0_y'],other_OF.at[0, ' gaze_0_z']]),1) #left
    gazeVec_r = np.expand_dims( np.array([other_OF.at[0,' gaze_1_x'],other_OF.at[0, ' gaze_1_y'],other_OF.at[0, ' gaze_1_z']]),1) #left
    face = 1
    for i in range(56):
        x_str = ' eye_lmk_X_'+str(i)
        y_str = ' eye_lmk_Y_'+str(i)
        z_str = ' eye_lmk_Z_'+str(i)
        xs.append(other_OF.at[0,x_str])
        ys.append(other_OF.at[0,y_str])
        zs.append(other_OF.at[0,z_str])
    for i in range(68):
        x_str = ' X_'+str(i)
        y_str = ' Y_'+str(i)
        z_str = ' Z_'+str(i)
        xs.append(other_OF.at[0,x_str])
        ys.append(other_OF.at[0,y_str])
        zs.append(other_OF.at[0,z_str])
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
        # print("p3ds",i)
        xs_a.append(_p3d[0])
        ys_a.append(_p3d[1])
        zs_a.append(_p3d[2])
        p3ds.append(_p3d)
    return xs_a,ys_a,zs_a,(R@gazeVec_l+T),(R@gazeVec_r+T)
# # _p3d = DLT(P1, P2, table_edge1, table_edge2)
# print("p3ds",i)
# xs_a.append(_p3d[0])
# ys_a.append(_p3d[1])
# zs_a.append(_p3d[2])
# p3ds.append(_p3d)

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
fig = plt.figure()
# print(xs)
# print(ys)
# print(zs)
xs1,ys1,zs1,gvl1,gvr1 = getPoints()
xs2,ys2,zs2,gvl2,gvr2 = getPoints()
print(gvl1,gvr1)
ax = plt.axes(projection='3d')
ax.scatter3D(xs1, ys1,zs1,color='g')
ax.quiver(xs1[49], ys1[49], zs1[49], -gvr1[0], -gvr1[1], -gvr1[2], length=100 , normalize=True,color='g')
ax.quiver(xs1[21], ys1[21], zs1[21], -gvl1[0], -gvl1[1], -gvl1[2], length=100 , normalize=True,color='g')
ax.scatter3D(xs2, ys2,zs2,color='b')
ax.quiver(xs2[49], ys2[49], zs2[49], -gvr2[0], -gvr2[1], -gvr2[2], length=100 , normalize=True,color='r')
ax.quiver(xs2[21], ys2[21], zs2[21], -gvl2[0], -gvl2[1], -gvl2[2], length=100 , normalize=True,color='r')
plt.axis('square')

plt.show()