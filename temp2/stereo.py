from lib2to3.pytree import convert
import cv2
import numpy as np
from Files import get_folder
import scipy.io as sio
from scipy import ndimage
from math import sqrt
import pandas as pd
def Project(point,fx,fy,cx,cy):
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
    openface = pd.read_csv(matlabVarsFolder+"/"+"lframe2OF.csv")
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

T = np.rot90(T)
pixleVector = np.expand_dims(np.array([806,388,1]),1)
# gazeVec_l = np.expand_dims( np.array([-0.27616,0.176646,-0.944739]),1) #left
# gazeVec_r = np.expand_dims( np.array([-0.44487,0.158656,-0.88143]),1) #right
gazeVec_l = np.expand_dims( np.array([-0.299976,0.170632,-0.938562]),1) #left
gazeVec_r = np.expand_dims( np.array([-0.466253,0.151944,-0.871505]),1) #right
p2Vector  =np.expand_dims( np.array([1430,533,1]),1)
# pixleVector = np.rot90(pixleVector)
# p2Vector = np.rot90(p2Vector)
# R = np.transpose(R)
print(p2Vector.shape,T.shape)
print(np.matmul(R,np.subtract(p2Vector,T)))
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
# iris_r = (852,400)
print(openface)
exit(1)
# iris_l = (795,409)
edge_r = (818,404)
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