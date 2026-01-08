import glob
import os
import shutil
from Files import get_folder,get_file
from CheckCreate import check_create
import random
import cv2
from pathlib import Path,PurePath
from tqdm import tqdm
from pickle import dump,load
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
OLD_SQUARE_SIZE = 0.4
DAN_LAB_SQAURE_SIZE = 0.5
NEW_SQAURE_SIZE = 0.3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def get_camera(name):
    if 'both' in name:
        return 'both'
    if 'subject' in  name:
        return 'subject'
    if 'marissa' in name:
        return 'marissa'
    return None
def chess_board_image_selection(folder = None,subject_n = 'auto',camera_type = 'auto',BOARD_SIZE = (6,8),pickle_files_dir = '',n_wanted_image = 30,square_size = 'auto',chess_board_imgs_folder = None):
    #arguemnts check&fix
    if not folder:
        folder = get_folder()+'/'
    if subject_n == 'auto':
        folder_name = os.path.dirname(folder)
        for split in folder_name.split('_'):
            if split.isdigit():
                subject_n = int(split)
                break
        print('could not resolve subject number in: '+folder)
        exit(1)
    if camera_type == 'auto':
        folder_name = os.path.dirname(folder)
        camera_type = get_camera(folder_name)
        if not camera_type:
            print('could not resolve camera type in: '+folder)
            exit(1)
    if square_size == 'auto':
        if subject_n  < 221:
            square_size = OLD_SQUARE_SIZE
        elif subject_n <228:
            square_size = DAN_LAB_SQAURE_SIZE
        elif subject_n <240:
            square_size = NEW_SQAURE_SIZE

    check_create(pickle_files_dir)
    #*********************************************************

    # code from https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
    #find chessboard images
    rows = BOARD_SIZE[0] #number of checkerboard rows.
    columns = BOARD_SIZE[1] #number of checkerboard columns.
    world_scaling = square_size*10 #cm  #change this to the real world square size. Or not.
    
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    board_imgs = []
    timeout = 0
    IMAGES_PICKLE_FILE = pickle_files_dir+f'{subject_n}_checked_images_list_'+camera_type
    CORNERS_PICKLE_FILE =pickle_files_dir+f'{subject_n}checked_corners_list_'+camera_type
    imgpts_PICKLE_FILE = pickle_files_dir+f'{subject_n}_imgspts_list_'+camera_type
    objectpts_PICKLE_FILE =pickle_files_dir+f'{subject_n}objtpts_list_'+camera_type
    checked_imgs = []
    corners_list = []
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
    
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    if os.path.exists(IMAGES_PICKLE_FILE):
        with open(IMAGES_PICKLE_FILE,'rb') as img_pick:
            checked_imgs = load(img_pick)
        with open(CORNERS_PICKLE_FILE,'rb') as cor_pick:
            corners_list = load(cor_pick)
        with open(imgpts_PICKLE_FILE,'rb') as img_pick:
            imgpoints = load(img_pick)
        with open(objectpts_PICKLE_FILE,'rb') as cor_pick:
            objpoints = load(cor_pick)
    i=0
    if len(checked_imgs)==0:
        for img_name in tqdm(os.listdir(folder)):
            i+=1
            if (img_name.split('.')[-1]!='png'):
                continue
            img_path = folder+img_name
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            if cv2.checkChessboard(img,BOARD_SIZE):
                ret, corners = cv2.findChessboardCorners(img, BOARD_SIZE)
                if ret:
                    left = corners[0][0][0]
                    right = corners[0][0][0]
                    bottom = corners[0][0][1]
                    top = corners[0][0][1]
                    for corner in corners:
                        c_x = corner[0][0]
                        c_y = corner[0][1]
                        if c_x < left: 
                            left = c_x
                        if c_x > right:
                            right = c_x
                        if c_y <top:
                            top = c_y
                        if c_y > bottom:
                            bottom = c_y
                    h = int(bottom-top)+1
                    w = int(right-left)+1
                    left = int(left)
                    top = int(top)
                    cropped_chessboard = img[top:top+h,left:left+w].copy()
                    fm = cv2.Laplacian(cropped_chessboard, cv2.CV_64F).var()
                    if fm <150:
                        continue
                    checked_imgs.append((img_path,i))
                    corners_list.append(corners)
                    conv_size = (11, 11)

                    #opencv can attempt to improve the checkerboard coordinates
                    corners = cv2.cornerSubPix(img, corners, conv_size, (-1, -1), criteria)
                    if chess_board_imgs_folder:
                        img_to_draw_on = img.copy()
                        cv2.drawChessboardCorners(img_to_draw_on, (rows,columns), corners, ret)
                        drawn_img_filename = chess_board_imgs_folder + img_name
                        drawn_chess_filename = chess_board_imgs_folder + 'c_'+img_name
                        cv2.putText(cropped_chessboard, "{}: {:.2f}".format("Blur: ", fm), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        cv2.imwrite(drawn_img_filename,img_to_draw_on)
                        cv2.imwrite(drawn_chess_filename,cropped_chessboard)


                    objpoints.append(objp)
                    imgpoints.append(corners)

                board_imgs.append(img_path)
                # if len(board_imgs)==30:
                #     break
        with open(IMAGES_PICKLE_FILE,'wb') as img_pick:
            dump(checked_imgs,img_pick)
        with open(CORNERS_PICKLE_FILE,'wb') as cor_pick:
            dump(corners_list,cor_pick)
        with open(imgpts_PICKLE_FILE,'wb') as img_pick:
            dump(imgpoints,img_pick)
        with open(objectpts_PICKLE_FILE,'wb') as cor_pick:
            dump(objpoints,cor_pick)
#********************************************************88


    corners_first_cord = [cl[0] for cl in corners_list ]
    corners_first_cord = [cfc[0] for cfc in corners_first_cord]
    corners_first_cord = np.array(corners_first_cord,dtype=np.double)
    h = 0.5 # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = corners_first_cord[:, 0].min() - 1, corners_first_cord[:, 0].max() + 1
    y_min, y_max = corners_first_cord[:, 1].min() - 1, corners_first_cord[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_s = [cfc[0] for cfc in corners_first_cord]
    y_s = [cfc[1] for cfc in corners_first_cord]
    kmeans = KMeans(n_clusters=n_wanted_image,n_init='auto')
    kmeans.fit(corners_first_cord)
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
    selected_images = []
    for clo in closest:
        selected_images.append((checked_imgs[clo],clo))

    # seleted_images  - list of paths to the images selected by the chessboard image selection algorithm and it's index in closest
    # closest - list of indicies relative to the checked_imgs list, of the selected images , i.e - selected_images[0] == checked_imgs[closest[0]]
    # kmeans - kmeans sklearn object used to select the chessboard images
    # plot_img - visual plot of the entire images real-world "speard" and the marked selected chessboard images
    # checked_imgs - list of paths to the images in which cv2 detected there is a chessboard frame
    # corners_list - list of detected chessboard corners, len(corners_list) == len(checked_imgs)
    # objpoints - same as in "Temuge Batpurev" format  
    # imgpoints - same as in "Temuge Batpurev" format  
    return selected_images,closest,kmeans,plot_img,checked_imgs,corners_list,objpoints,imgpoints

def kmeans_imgs_selection(corners_list,k = 30):
    corners_first_cord = [cl[0] for cl in corners_list ]
    corners_first_cord = [cfc[0] for cfc in corners_first_cord]
    corners_first_cord = np.array(corners_first_cord,dtype=np.double)
    h = 0.5 # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = corners_first_cord[:, 0].min() - 1, corners_first_cord[:, 0].max() + 1
    y_min, y_max = corners_first_cord[:, 1].min() - 1, corners_first_cord[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_s = [cfc[0] for cfc in corners_first_cord]
    y_s = [cfc[1] for cfc in corners_first_cord]
    kmeans = KMeans(n_clusters=k,n_init='auto')
    kmeans.fit(corners_first_cord)
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
    return closest,plot_img
