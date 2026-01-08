import os
import glob
from pickle import dump,load
from CheckCreate import check_create,check_delete
from SplitToFrames import extract_image_one_fps
import shutil
from SelectImgsForCalibration import chess_board_image_selection,kmeans_imgs_selection 
import cv2
from pathlib import Path,PurePath
from tqdm import tqdm
from pickle import dump,load
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
import matlab.engine
import tqdm
def kmeans_imgs_selection_Matlab(corners_list,k = 40,errors = None,count = 3,error_threshold = 1.5,recursive_error_fix =False):
    images_first_corners = [corner[0][:] for corner in corners_list]
#     corners_first_cord = [cl[0] for cl in corners_list ]
#     corners_first_cord = [cfc[0] for cfc in corners_first_cord]
    corners_first_cord = images_first_corners
    corners_first_cord = np.array(corners_first_cord,dtype=np.double)
    corners_first_cord=np.nan_to_num(corners_first_cord)
    h = 0.5 # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = corners_first_cord[:, 0].min() - 1, corners_first_cord[:, 0].max() + 1
    y_min, y_max = corners_first_cord[:, 1].min() - 1, corners_first_cord[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_s = [cfc[0] for cfc in corners_first_cord]
    y_s = [cfc[1] for cfc in corners_first_cord]
    kmeans = KMeans(n_clusters=k,n_init='auto')
    train_labels = kmeans.fit_predict(corners_first_cord)
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.figure(1)
    plt.clf()
    Z = np.expand_dims(Z, axis=0)
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(corners_first_cord[:, 0], corners_first_cord[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the digits dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    closest_data = []
    m_clusters = kmeans.labels_.tolist()
    all_data = [ corners_first_cord.shape[0]]

    centers = np.array(kmeans.cluster_centers_)
    if errors:
        closest = []
        labels =train_labels
        array = np.full((k,len(corners_list)),fill_value = np.inf)
        for i,label in enumerate(labels):
            array[label][i] = errors[i]
        argsorted = np.argsort(array)
        for i in range(k):
            for j in range(count):
                if array[i][argsorted[i][j]] < error_threshold:
                    closest.append(argsorted[i][j])
        for clo in closest:
            plt.scatter(
                corners_first_cord[clo][0],
                corners_first_cord[clo][1],
                marker="x",
                s=50,
                linewidths=3,
                color="black",
                zorder=10,
            )
    if not errors:
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, corners_first_cord)
        for clo in closest:
            plt.scatter(
                corners_first_cord[clo][0],
                corners_first_cord[clo][1],
                marker="x",
                s=50,
                linewidths=3,
                color="black",
                zorder=10,
            )

    plt.savefig('--------temp.png')
    plot_img = cv2.imread('--------temp.png',-1)
    os.remove('--------temp.png')
    if len(closest)==0:
        if recursive_error_fix:
            return kmeans_imgs_selection_Matlab(corners_list,k,errors,count,error_threshold+1,True)
    return closest,plot_img
OLD_SQUARE_SIZE = 4
DAN_LAB_SQAURE_SIZE = 5
NEW_SQAURE_SIZE = 3
def calibrate_from_img_list(eng,imgs,square_size='auto',subject_n = 0):
    if square_size == 'auto':
        if subject_n  < 221:
            square_size = OLD_SQUARE_SIZE
        elif subject_n <228:
            square_size = DAN_LAB_SQAURE_SIZE
        elif subject_n <255:
            square_size = NEW_SQAURE_SIZE
    f_detector = eng.vision.calibration.monocular.CheckerboardDetector()
    ret_val=eng.detectPatternPoints(f_detector,imgs,nargout=2)
    chess_points = ret_val[0]
    chess_images_used = ret_val[1]
    originalImage = eng.imread(imgs[0],nargout=1)
    (mrows,ncols,o)= eng.size(originalImage,nargout=3)
    squareSize = square_size;  # in units of 'centimeters'
    worldPoints = eng.generateWorldPoints(f_detector, 'SquareSize', squareSize,nargout=1)
    (cameraParams3coeff, imagesUsed, estimationErrors) = eng.estimateCameraParameters(chess_points, worldPoints, 
        'EstimateSkew', False, 'EstimateTangentialDistortion', False, 
        'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'centimeters', 
        'InitialIntrinsicMatrix', matlab.double(vector=[]  ), 'InitialRadialDistortion', matlab.double(vector=[]  ),
        'ImageSize', matlab.double(vector=[mrows,ncols]  ),nargout = 3)
    return cameraParams3coeff,imagesUsed,estimationErrors,chess_images_used,chess_points,worldPoints,mrows,ncols
def get_estimations_errors(eng,cameraParams):
    eng1 = matlab.engine.start_matlab()
    eng1.workspace['CameraParams'] = cameraParams
    repErros = eng1.eval('CameraParams.ReprojectionErrors')
    imgs = permute_matlab_point_cloud(repErros)
    errors = []
    for img in imgs:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        errors.append(errors_sum/len(img))
    eng1.quit()
    return errors
def permute_matlab_point_cloud(points):
    """gets points shape [x,y,z] return [z,x,y]
    example [48,2,107] -> [107,48,2]"""
    points_rot = []
    for o in range(points.size[2]):
        img = []
        for i in range(points.size[0]):
            point = []
            for j in range(points.size[1]):
                point.append(points[i][j][o])
            img.append(point)
        points_rot.append(img)
    return points_rot
def chess_board_check(img_list,BOARD_SIZE = (6,8)):
    results = []
    final_list = [] 
    indicies = []
    reverse = {}
    import random
    for i,img_path in enumerate(tqdm.tqdm(img_list)):
        reverse[img_path] = i
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(img, (600,400), interpolation= cv2.INTER_LINEAR)
        if cv2.checkChessboard(img,BOARD_SIZE):
            results.append(True)
            indicies.append(i)
            final_list.append(img_path)
        else:   
            results.append(False)
    new_final_list = []
    new_results = []
    if len(final_list)>1000:
        if len(final_list)<2000:
            ratio = int(len(final_list)/(len(final_list)-1000))            
            for i in range(len(final_list)):
                if (i%ratio)>0:
                    new_final_list.append(final_list[i])
                else:
                    results[reverse[final_list[i]]] = False 
        else:
            results_true_indicies = [i for i in range(0,len(results)) if results[i]]
            selected = random.sample(results_true_indicies,k=len(results_true_indicies)-1000)
            for s in selected:
                results[s] = False
            new_final_list = [img_list[i] for i in range(len(results)) if results[i]]
    else:
        new_final_list = final_list
            
    return new_final_list,results
def calibrate_from_points(eng,imagePoints,worldPoints,mrows,ncols):
        (cameraParams3coeff, imagesUsed, estimationErrors) = eng.estimateCameraParameters(imagePoints, worldPoints, 
        'EstimateSkew', False, 'EstimateTangentialDistortion', False, 
        'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'centimeters', 
        'InitialIntrinsicMatrix', matlab.double(vector=[]  ), 'InitialRadialDistortion', matlab.double(vector=[]  ),
        'ImageSize', matlab.double(vector=[mrows,ncols]  ),nargout = 3)
        return (cameraParams3coeff, imagesUsed, estimationErrors)

def get_stereo_errors_estimation(eng,stereo_params):
    old_ws_var = None
    old_ws_flag = True
    eng1 =matlab.engine.start_matlab()
    try:
        old_ws_var=eng.workspace['stereoParams']
    except:
        old_ws_flag = False
    eng1.workspace['stereoParams'] = stereo_params
    repErros_camera1 = eng1.eval('stereoParams.CameraParameters1.ReprojectionErrors')
    repErros_camera2 = eng1.eval('stereoParams.CameraParameters2.ReprojectionErrors')
    imgs1 = permute_matlab_point_cloud(repErros_camera1)
    imgs2 = permute_matlab_point_cloud(repErros_camera2)

    errors1 = []
    errors2 = []
    for img in imgs1:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        errors1.append(errors_sum/len(img))
    for img in imgs2:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        errors2.append(errors_sum/len(img))
    eng1.quit()

    return errors1,errors2