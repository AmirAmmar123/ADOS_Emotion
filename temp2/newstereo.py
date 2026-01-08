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
from StereoOfVideo_v1 import calc_frame
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sympy import Point3D,Plane,Line3D
FACE_LANDMARKS = [8,12,54,33,35,45,16].sort()
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
    # T = np.transpose(T)
    T = np.rot90(T,3)
    # R = np.transpose(R)
    F = sio.loadmat(matlabVarsFolder+"/"+"F"+".mat")['fund']
    E = sio.loadmat(matlabVarsFolder+"/"+"E"+".mat")['essen']

    xs = list()
    ys = list()
    zs = list()
    xs2 = list()
    ys2= list()
    zs2 = list()
    xs2d = list()
    ys2d = list()
    xs22d = list()
    ys22d = list()
    other_OF = pd.read_csv(matlabVarsFolder+"/"+"faceOF.csv")
    main_OF = pd.read_csv(matlabVarsFolder+"/"+"frame2.csv")
    gazeVec_l = np.expand_dims( np.array([other_OF.at[0,' gaze_0_x'],other_OF.at[0, ' gaze_0_y'],other_OF.at[0, ' gaze_0_z']]),1) #left
    gazeVec_r = np.expand_dims( np.array([other_OF.at[0,' gaze_1_x'],other_OF.at[0, ' gaze_1_y'],other_OF.at[0, ' gaze_1_z']]),1) #left
    # openface2 = pd.read_csv(bothCSVFolder+"/"+"frame"+str(counter-1)+".csv")
    r1_nose_x0 = main_OF.at[0," eye_lmk_X_0"]
    r2_nose_x0 = main_OF.at[1," eye_lmk_X_0"]
    if(r1_nose_x0 < r2_nose_x0):
        face=0
    else:
        face=1
    for i in range(56):
        x_str = ' eye_lmk_X_'+str(i)
        y_str = ' eye_lmk_Y_'+str(i)
        z_str = ' eye_lmk_Z_'+str(i)
        x2d_str = ' eye_lmk_x_'+str(i)
        y2d_str = ' eye_lmk_y_'+str(i)
        xs.append(other_OF.at[0,x_str])
        ys.append(other_OF.at[0,y_str])
        zs.append(other_OF.at[0,z_str])
        xs2d.append(other_OF.at[0,x2d_str])
        ys2d.append(other_OF.at[0,y2d_str])
    for i in range(68):
        x_str = ' X_'+str(i)
        y_str = ' Y_'+str(i)
        z_str = ' Z_'+str(i)
        x2d_str = ' x_'+str(i)
        y2d_str = ' y_'+str(i)
        xs.append(other_OF.at[0,x_str])
        ys.append(other_OF.at[0,y_str])
        zs.append(other_OF.at[0,z_str])
        xs2d.append(other_OF.at[0,x2d_str])
        ys2d.append(other_OF.at[0,y2d_str])
    for i in range(56):
        x_str = ' eye_lmk_X_'+str(i)
        y_str = ' eye_lmk_Y_'+str(i)
        z_str = ' eye_lmk_Z_'+str(i)
        x2d_str = ' eye_lmk_x_'+str(i)
        y2d_str = ' eye_lmk_y_'+str(i)
        xs2.append(main_OF.at[face,x_str])
        ys2.append(main_OF.at[face,y_str])
        zs2.append(main_OF.at[face,z_str])
        xs22d.append(main_OF.at[face,x2d_str])
        ys22d.append(main_OF.at[face,y2d_str])
    for i in range(68):
        x_str = ' X_'+str(i)
        y_str = ' Y_'+str(i)
        z_str = ' Z_'+str(i)
        x2d_str = ' x_'+str(i)
        y2d_str = ' y_'+str(i)
        xs2.append(main_OF.at[face,x_str])
        ys2.append(main_OF.at[face,y_str])
        zs2.append(main_OF.at[face,z_str])
        xs22d.append(main_OF.at[face,x2d_str])
        ys22d.append(main_OF.at[face,y2d_str])
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
        # uv1 = (xs[i],ys[i],zs[i])
        # uv2 = (xs2[i],ys2[i],zs2[i])
        # uv1 = Project(uv1,K2)
        # uv2 = Project(uv2,K1)
        uv1 = (xs2d[i],ys2d[i])
        uv2 = (xs22d[i],ys22d[i])
        # print(uv1,uv2)
        _p3d = DLT(P1, P2, uv2,uv1)
        # print("p3ds",i)
        xs_a.append(_p3d[0])
        ys_a.append(_p3d[1])
        zs_a.append(_p3d[2])
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
MATLAB_NAMES  = ["intrinsicMatrix1","distortionCoefficients1","intrinsicMatrix2","distortionCoefficients2","rotationOfCamera2","translationOfCamera2"]
PARAMS_NAMES = ['CameraParameters1','CameraParameters2','RotationOfCamera2','TranslationOfCamera2']
matlabVarsFolder = get_folder("Select matlab folder")
if CONVERTED:
    K1 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[0]+".mat")[MATLAB_NAMES[0]]
    D1= sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[1]+".mat")[MATLAB_NAMES[1]]
    K2 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[2]+".mat")[MATLAB_NAMES[2]]
    D2 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[3]+".mat")[MATLAB_NAMES[3]]
    R = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[4]+".mat")[MATLAB_NAMES[4]]
    T = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[5]+".mat")[MATLAB_NAMES[5]]
    F = sio.loadmat(matlabVarsFolder+"/"+"F"+".mat")['fund']
    E = sio.loadmat(matlabVarsFolder+"/"+"E"+".mat")['essen']
    openface = pd.read_csv(matlabVarsFolder+"/"+"faceOF.csv")
    # IMG_POINTS = sio.loadmat(matlabVarsFolder+"/"+"imagePoints"+".mat")['imagePoints']
    # print(IMG_POINTS.shape)

else:
    print(sio.loadmat(matlabVarsFolder+"/"+"stereo.mat").keys())
    print(sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[0]+".mat").keys())
    K1 = sio.loadmat(matlabVarsFolder+"/"+"all.mat")['stereoParams'][PARAMS_NAMES[0]]['IntrinsicMatrix']
    D1= sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[1]+".mat")[MATLAB_NAMES[1]]
    K2 = sio.loadmat(matlabVarsFolder+"/"+"all.mat")['stereoParams'][PARAMS_NAMES[1]]['IntrinsicMatrix']
    D2 = sio.loadmat(matlabVarsFolder+"/"+MATLAB_NAMES[3]+".mat")[MATLAB_NAMES[3]]
    R = sio.loadmat(matlabVarsFolder+"/"+"all.mat")['stereoParams'][PARAMS_NAMES[2]]
    T = sio.loadmat(matlabVarsFolder+"/"+"all.mat")['stereoParams'][PARAMS_NAMES[3]]
# T = np.transpose(T)
T = np.rot90(T,3)

# R = np.transpose(R)
pixleVector = np.expand_dims(np.array([806,388,1]),1)
# gazeVec_l = np.expand_dims( np.array([-0.27616,0.176646,-0.944739]),1) #left
# gazeVec_r = np.expand_dims( np.array([-0.44487,0.158656,-0.88143]),1) #right
gazeVec_l = np.expand_dims( np.array([openface.at[0,' gaze_0_x'],openface.at[0, ' gaze_0_y'],openface.at[0, ' gaze_0_z']]),1) #left
gazeVec_r = np.expand_dims( np.array([openface.at[0,' gaze_1_x'],openface.at[0, ' gaze_1_y'],openface.at[0, ' gaze_1_z']]),1) #left
p2Vector  =np.expand_dims( np.array([1430,533,1]),1)
# pixleVector = np.rot90(pixleVector)
# p2Vector = np.rot90(p2Vector)
# R = np.transpose(R)
print(p2Vector.shape,T.shape)
print(np.matmul(R,np.subtract(p2Vector,T))  )
subjectImage = cv2.imread(matlabVarsFolder+"/lframe2.jpg", cv2.IMREAD_UNCHANGED)
bothImage = cv2.imread(matlabVarsFolder+"/frame2.jpg", cv2.IMREAD_UNCHANGED)
# print(F)
# E = np.rot90(E)
epiLine = np.matmul(F,pixleVector)
vecLine0 = np.matmul(R,np.subtract(gazeVec_l,T))
vecLine1 = np.matmul(R,np.subtract(gazeVec_r,T))
# print( epiLine, F.shape,pixleVector.shape,pixleVector)
# p1 = cv2.Point
# p2 = cv2.Point(bothImage.cols,-(epiLine[2]+epiLine[0]*bothImage.cols)/epiLine[1])
p2 = ()
# cv2.line(bothImage, (0,int(-epiLine[2]/epiLine[1])),(bothImage.shape[1],int(-(epiLine[2]+epiLine[0]*bothImage.shape[1])/epiLine[1])), (1024,0,0), 2)
epiLine = gazeVec_r
# eye1=(1468,508)
# eye2=(1444,502)
# iris_l = (openface.at[0,' eye_lmk_X_21'],openface.at[0,' eye_lmk_X_21'],openface.at[0,' eye_lmk_X_21'])

# exit(1)
 #plot faces
x =openface.at[0,' eye_lmk_X_21']+ openface.at[0, ' eye_lmk_X_25']
y =openface.at[0,' eye_lmk_X_21'] + openface.at[0, ' eye_lmk_Y_25']
z =openface.at[0,' eye_lmk_X_21']+ openface.at[0, ' eye_lmk_Z_25']
x /= 2
y /= 2
z /= 2
iris_l = (int(x),int(y),int(z))
x =openface.at[0,' eye_lmk_X_49']+ openface.at[0, ' eye_lmk_X_53']
y =openface.at[0,' eye_lmk_Y_49'] + openface.at[0, ' eye_lmk_Y_53']
z =openface.at[0,' eye_lmk_Z_49']+ openface.at[0, ' eye_lmk_Z_53']
x /= 2
y /= 2
z /= 2
iris_r = (int(x),int(y),int(z))
projected_iris = Project_1(iris_r,1448.50869806072,1447.12208343430,939.845887677898,541.375717465825)
projected_iris = Project(iris_r,K1)
print(projected_iris)
edge_r = (iris_r[0]+50*gazeVec_r[0],iris_r[1]+50*gazeVec_r[1],iris_r[2]+50*gazeVec_r[2])
projected_edge = Project_1(edge_r,1448.50869806072,1447.12208343430,939.845887677898,541.375717465825)
projected_edge = Project(edge_r,K1)

projected_iris_vect  = np.expand_dims(np.array([projected_iris[0],projected_iris[1],1]),1)
projected_edge_vect  = np.expand_dims(np.array([projected_edge[0],projected_edge[1],1]),1)
epiline1 = np.matmul(F,projected_iris_vect)
epiline2 = np.matmul(F,projected_edge_vect)
print(iris_r)
iris_2_vect = np.expand_dims(np.array([iris_r[0]/iris_r[2],iris_r[1]/iris_r[2],1]),1)

test1 = np.matmul(E,iris_2_vect)
# print(Project_1(test1,1446.08691386883,1444.43423207416,972.279407095570,525.100703666741))
cv2.line(bothImage, (0,int(-epiline1[2]/epiline1[1])),(bothImage.shape[1],int(-(epiline1[2]+epiline1[0]*bothImage.shape[1])/epiline1[1])), (1024,0,0), 2)
cv2.line(bothImage, (0,int(-epiline2[2]/epiline2[1])),(bothImage.shape[1],int(-(epiline2[2]+epiline2[0]*bothImage.shape[1])/epiline2[1])), (255, 250, 255), 2)
# imS = cv2.resize(bothImage, (960, 540))  
# cv2.imshow("Image", imS)
# cv2.imwrite("frame%d.jpg" % 1000, bothImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Rinv = np.linalg.inv(R)
# gazeVec_r_RT = np.matmul(Rinv,gazeVec_r)-T
num_of_f = 55
xs = list()
ys = list()
zs = list()
xs2 = list()
ys2= list()
zs2 = list()
openface2 = pd.read_csv(matlabVarsFolder+"/"+"frame2.csv")
# openface2 = pd.read_csv(bothCSVFolder+"/"+"frame"+str(counter-1)+".csv")
r1_nose_x0 = openface2.at[0," eye_lmk_X_0"]
r2_nose_x0 = openface2.at[1," eye_lmk_X_0"]
if(r1_nose_x0 > r2_nose_x0):
    face=0
else:
    face=1
xs2d = list()
ys2d = list()
zs2d = list()
xs22d = list()
ys22d = list()
zs22d = list()
for i in range(56):
    x_str = ' eye_lmk_X_'+str(i)
    y_str = ' eye_lmk_Y_'+str(i)
    z_str = ' eye_lmk_Z_'+str(i)
    x2d_str = ' eye_lmk_x_'+str(i)
    y2d_str = ' eye_lmk_y_'+str(i)
    xs.append(openface.at[0,x_str])
    ys.append(openface.at[0,y_str])
    zs.append(openface.at[0,z_str])
    xs2d.append(openface.at[0,x2d_str])
    ys2d.append(openface.at[0,y2d_str])
for i in range(68):
    x_str = ' X_'+str(i)
    y_str = ' Y_'+str(i)
    z_str = ' Z_'+str(i)
    x2d_str = ' x_'+str(i)
    y2d_str = ' y_'+str(i)
    xs.append(openface.at[0,x_str])
    ys.append(openface.at[0,y_str])
    zs.append(openface.at[0,z_str])
    xs2d.append(openface.at[0,x2d_str])
    ys2d.append(openface.at[0,y2d_str])
for i in range(56):
    x_str = ' eye_lmk_X_'+str(i)
    y_str = ' eye_lmk_Y_'+str(i)
    z_str = ' eye_lmk_Z_'+str(i)
    x2d_str = ' eye_lmk_x_'+str(i)
    y2d_str = ' eye_lmk_y_'+str(i)
    xs2.append(openface2.at[face,x_str])
    ys2.append(openface2.at[face,y_str])
    zs2.append(openface2.at[face,z_str])
    xs22d.append(openface2.at[face,x2d_str])
    ys22d.append(openface2.at[face,y2d_str])
for i in range(68):
    x_str = ' X_'+str(i)
    y_str = ' Y_'+str(i)
    z_str = ' Z_'+str(i)
    x2d_str = ' x_'+str(i)
    y2d_str = ' y_'+str(i)
    xs2.append(openface2.at[face,x_str])
    ys2.append(openface2.at[face,y_str])
    zs2.append(openface2.at[face,z_str])
    xs22d.append(openface2.at[face,x2d_str])
    ys22d.append(openface2.at[face,y2d_str])
p3ds = []
#RT matrix for C1 is identity.
RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = K1 @ RT1 #projection matrix for C1
 
#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R, T], axis = -1)
P2 = K2 @ RT2 #projection matrix for C2
table_edge1 = (168,941)

table_edge2 = (1153,785)
xs_a = list()
ys_a= list()
zs_a =list()
for i in range(len(xs)):
    # uv1 = (xs[i],ys[i],zs[i])
    # uv2 = (xs2[i],ys2[i],zs2[i])
    # uv1 = Project(uv1,K1)
    # uv2 = Project(uv2,K2)
    uv1 = (xs2d[i],ys2d[i])
    uv2 = (xs22d[i],ys22d[i])
    # print(uv1,uv2)
    _p3d = DLT(P1, P2, uv1, uv2)
    # print("p3ds",i)
    xs_a.append(_p3d[0])
    ys_a.append(_p3d[1])
    zs_a.append(_p3d[2])
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
fig = plt.figure()
# # print(xs)
# print(ys)
# print(zs)
print(xs,ys,zs)
ax = plt.axes(projection='3d')
xs1,ys1,zs1,gvl1,gvr1 = getPoints()
Rinv = np.linalg.inv(R)
for i in range(len(xs1)):
    v = np.expand_dims(np.array([xs1[i],ys1[i],zs1[i]]),1)
    v = Rinv@(v -T)
    xs1[i] = v[0]
    ys1[i] = v[1]
    zs1[i] = v[2]
ax.scatter3D(xs_a, ys_a,zs_a,color = 'g')
ax.scatter3D(xs1, ys1,zs1,color ='b')
hx, hy,hz = zip(*calc_frame((xs_a,ys_a,zs_a),(xs1,ys1,zs1),gazeVec_l,gazeVec_r,by='plane'))
ax.scatter(hx,hy,hz,color='y')
axes = [5, 5, 5]
 
# Create Data
data = np.ones(axes, dtype=np.bool)
 
# Control Transparency
alpha = 0.9
 
# Control colour
colors = np.empty(axes + [4], dtype=np.float32)
 
colors[:] = [1, 0, 0, alpha]  # red


facePlanePoints = np.array([(xs1[0].item(),ys1[0].item()),(xs1[8].item(),ys1[8].item()),(xs1[33].item(),ys1[33].item())])
facePlanePointsX =[xs1[56+0].item(),xs1[56+8].item(),xs1[56+33].item()]
facePlanePointsY =[ys1[56+0].item(),ys1[56+8].item(),ys1[56+33].item()]
facePlanePointsZ =[zs1[56+0].item(),zs1[56+8].item(),zs1[56+33].item()]
faceVerts =[list(zip(facePlanePointsX, facePlanePointsY, facePlanePointsZ))]


points = [[facePlanePointsX[0],facePlanePointsY[0], facePlanePointsZ[0]],
           [facePlanePointsX[1], facePlanePointsY[1], facePlanePointsZ[1]],
           [facePlanePointsX[2], facePlanePointsY[2], facePlanePointsZ[2]]]

p0, p1, p2 = points
x0, y0, z0 = p0
x1, y1, z1 = p1
x2, y2, z2 = p2

ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

point  = np.array(p0)
normal = np.array(u_cross_v)

d = -point.dot(normal)

xx, yy = np.meshgrid(range(int(facePlanePointsX[2]) - 50 ,int(facePlanePointsX[2]) +50), range(int(facePlanePointsY[2]) - 40 ,int(facePlanePointsY[2])+40))

z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

print('grid',xx,yy,z)

ax.plot_surface(xx, yy, z)



#line Points
# p0 = [0,0,0] #point in line
# v0 = [1,1,1] #line direction as vector

#create plane and line
plane = Plane(p0,p1,p2)
rightPupil = ((xs_a[49]+xs_a[53]+xs_a[51]+xs_a[55])/4,(ys_a[49]+ys_a[53]+ys_a[51]+ys_a[55])/4
,(zs_a[49]+zs_a[53]+zs_a[51]+zs_a[55])/4)
line = Line3D([rightPupil[0], rightPupil[1], rightPupil[2]],direction_ratio=[gazeVec_r[0].item(), gazeVec_r[1].item(), gazeVec_r[2].item()])

print('point v',[xs_a[49].item(), ys_a[49].item(), zs_a[49].item()])
print('point r',[gazeVec_r[0].item(), gazeVec_r[1].item(), gazeVec_r[2].item()])
print(f"plane equation: {plane.equation()}")
print(f"line equation: {line.equation()}")

#find intersection:

intr = plane.intersection(line)
hp = np.array(intr[0],dtype=float)
print(f"intersection: {intr}")
print(hp)
# ax.scatter(hp[0],hp[1],hp[2],color='m')


# Voxels is used to customizations of the
# sizes, positions and colors.
ax.voxels(data, facecolors=colors)
print([(xs1[0].item(),ys1[0].item()),(xs1[8].item(),ys1[8].item()),(xs1[33].item(),ys1[33].item())])
# facePlane = Polygon(facePlanePoints,facecolor= 'k')
# ax.add_patch(facePlane)
ax.add_collection3d(Poly3DCollection(faceVerts))
ax.quiver(xs_a[49], ys_a[49], zs_a[49], gazeVec_r[0], gazeVec_r[1], gazeVec_r[2], length=50 , normalize=True,color='g')
ax.quiver(xs_a[21], ys_a[21], zs_a[21], gazeVec_l[0], gazeVec_l[1], gazeVec_l[2], length=50 , normalize=True,color='r')
# ax.plot([xs_a[21], xs_a[21]+100*gazeVec_l[0]], [ys_a[21], ys_a[21]+100*gazeVec_l[1]],zs=[zs_a[21], zs_a[21]+100*gazeVec_l[2]])
elev = 0
azim = 0
ax.view_init(elev, azim)
ax.set_box_aspect((np.ptp(xs1), np.ptp(ys1), np.ptp(zs1))) 
plt.show()

# print(iris_vec)
# print(Project(iris_r_2,1446.08691386883,1444.43423207416,972.279407095570,525.100703666741))
exit(1)
print(gazeVec_r[0]+gazeVec_r[1]+gazeVec_r[2])
dist = sqrt(pow(gazeVec_r[0],2)+pow(gazeVec_r[1],2)+pow(gazeVec_r[2],2))
# print(dist)
Dx = gazeVec_r[0]
Dy = gazeVec_r[1]
t  = 50 
X = iris_r[0]+t*Dx 
Y = iris_r[1] + t*Dy
bothImage = subjectImage
print(vecLine0)
cv2.line(bothImage, (int(X),int(Y)),iris_r, (1024,0,0), 2)
mtx = [[1448.50869806072,0,939.845887677898],[0,1447.12208343430,541.375717465825],[0,0,1]]
val = Project(gazeVec_r,1448.50869806072,1447.12208343430,939.845887677898,541.375717465825)
print(val)
mtx = np.array(mtx)
# val = cv2.projectPoints(edge_r,(0,0,0),(0,0,0),mtx,0)
print(val)
print(int(iris_r[1]+t*gazeVec_r[1]))
epiLine = gazeVec_l
cv2.line(bothImage, iris_l,(iris_l[0]-24,int(iris_l[1]-24*gazeVec_r[1])), (1024,0,0), 2)

cv2.line(bothImage, (0,0), (bothImage.shape[1], bothImage.shape[0]), (0,0,1024), 12)
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)  
imS = cv2.resize(bothImage, (960, 540))  
cv2.imshow("Image", imS)
cv2.imwrite("frame%d.jpg" % 1000, bothImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(1)
# K1 = np.transpose(K1)
# K2 = np.transpose(K2)
# R = np.transpose(R)
# T = np.transpose(T)
D1 = np.transpose(D1)
D2 = np.transpose(D2)
points1 =cv2.findChessboardCorners(subjectImage,(8,6))
points2 =cv2.findChessboardCorners(subjectImage,(8,6))
# print(val)
# exit(1)
params = [K1,K2,D1,D2,R,T]
for param in params:
    print(param.shape)
print(params)
image_size = subjectImage.shape[::-1][1:3]
print(image_size)
# flags = cv2.CALIB_ZERO_DISPARITY
FUND = True
if FUND :
    rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(points1, points2, F, image_size)
    rec_im1 = cv2.warpPerspective(subjectImage, rectmat1, (len(subjectImage), len(subjectImage[0])))
    rec_im2 = cv2.warpPerspective(bothImage, rectmat1, (len(subjectImage), len(subjectImage[0])))
    cv2.imshow('im1', rec_im1)
    cv2.waitKey(0)
    cv2.imshow('im2', rec_im2)
    cv2.waitKey(0)
else :    
    R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(K1,D1,K2,D2,image_size,R,T)
    leftmapX, leftmapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    rightmapX, rightmapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    left_remap = cv2.remap(subjectImage, leftmapX, leftmapY, cv2.INTER_LANCZOS4)
    right_remap = cv2.remap(bothImage, leftmapX, rightmapY, cv2.INTER_LANCZOS4)


    # For some reason, the images get rotated upside down after remapping, and I have to invert them back
    left_remap = ndimage.rotate(left_remap,0)
    right_remap = ndimage.rotate(right_remap,0)

    for line in range(0, int(right_remap.shape[0] / 20)):
        left_remap[line * 20, :] = 0
        right_remap[line * 20, :] = 0

    cv2.namedWindow('output images', cv2.WINDOW_NORMAL)
    cv2.imshow('output images', np.hstack([left_remap, right_remap]))
    cv2.waitKey(0) 