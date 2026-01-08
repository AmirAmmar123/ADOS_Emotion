

    #imports
import glob
from pickle import dump,load
from CheckCreate import check_create,check_delete
import shutil
import matlab.engine
import numpy as np
import numpy as np
from MatlabCalibration import *
import cv2
import pathlib
import os
from SplitToFrames import *
from VideoDictLoader import VideosDictLoader
import time
#constants and preperations
def check_fps_multiple_vids(vids):
    fpss =set()
    framess = set()
    for vid in vids:
        fps,frame = getFPS_and_FrameCount(vid)
        fpss.add(fps)
        framess.add(frame)
    if len(fpss)!=1 or len(framess)!=1:
        return False
    return True
    
def process_subjects(VIDEO_FILES):
    subjects_ns = list(VIDEO_FILES.keys())
    counter = 0
    has_subject = subjects_ns.copy()
    has_marissa = subjects_ns.copy()
    has_both = subjects_ns.copy()
    for subject_n in subjects_ns:
        try:
            calib_subject = VIDEO_FILES[subject_n]['CALIBRATION']['subject']
        except:
            has_subject.remove(subject_n)
        try:
            calib_marissa = VIDEO_FILES[subject_n]['CALIBRATION']['marissa']
        except:
            has_marissa.remove(subject_n)
        try:
            calib_both = VIDEO_FILES[subject_n]['CALIBRATION']['both']
        except:
            has_both.remove(subject_n)
        counter+=1
    possible_subjects = set(has_subject).intersection(has_both)
    full_method_subjects = set(has_subject).intersection(has_both).intersection(has_marissa)
    semi_method_subjects = possible_subjects.difference(full_method_subjects)
    return full_method_subjects,semi_method_subjects


F_HEIGHT  =1080
F_WIDTH = 1920
VIDEO_FILES = None
S_TEMP = 'c:/$subject_temp_frames$/'
M_TEMP = 'c:/$marissa_temp_frames$/'
B_TEMP = 'c:/$both_temp_frames$/'
SVID_TEMP = 'c:/$temp_subject_vid$.mp4'
MVID_TEMP = 'c:/$temp_marissa_vid$.mp4'
BVID_TEMP = 'c:/$temp_both_vid$.mp4'
BLACKEN = range(229,235+1)
PICKLES_DIR = 'c:/pickles/'
META_DATA = 'c:/MetaData2/'
check_create(META_DATA)
check_create(PICKLES_DIR)

with open('dict of sorted videos 4.4.23.pickle','rb') as f:
    VIDEO_FILES = load(f)
VIDEO_FILES = VideosDictLoader.loadVideosDict()
for subject_n in VIDEO_FILES.keys():
    if subject_n<222 or subject_n>227:
        continue
    
    for task in VIDEO_FILES[subject_n].keys():
        for camera in VIDEO_FILES[subject_n][task].keys():
            entry = VIDEO_FILES[subject_n][task][camera]
            if type(entry) is str:
                continue
            if type(entry) is list:
                entry_set = set(entry)
                entry_set_list = list(entry_set)
                print(entry)
                VIDEO_FILES[subject_n][task][camera]= entry_set_list[0]


if not VIDEO_FILES:
    exit(0)
fm_sbjs, sm_sbjs = process_subjects(VIDEO_FILES)
fm_sbjs = list(fm_sbjs)
fm_sbjs = list(range(249,250))
splitFirstSubjectInList = False #if False, skips frame spliting 1 time
for fm_sbj in fm_sbjs:
    print(fm_sbj)
    shutil.copy(VIDEO_FILES[fm_sbj]['CALIBRATION']['subject'],SVID_TEMP)
    shutil.copy(VIDEO_FILES[fm_sbj]['CALIBRATION']['marissa'],MVID_TEMP)
    shutil.copy(VIDEO_FILES[fm_sbj]['CALIBRATION']['both'],BVID_TEMP)
    check_create(S_TEMP)
    check_create(M_TEMP)
    check_create(B_TEMP)

    if splitFirstSubjectInList:
        if check_fps_multiple_vids([SVID_TEMP,BVID_TEMP,MVID_TEMP]):
            extract_image_one_fps(SVID_TEMP,S_TEMP)
            extract_image_one_fps(BVID_TEMP,B_TEMP)
            extract_image_one_fps(MVID_TEMP,M_TEMP)
        else:
            print('videos fps does not match, robust split needed...')
            robustSplitToFrames(SVID_TEMP,S_TEMP)
            robustSplitToFrames(BVID_TEMP,B_TEMP)
            robustSplitToFrames(MVID_TEMP,M_TEMP)
    if fm_sbj in BLACKEN:
        print("DARKEN!!")
        input()
    splitFirstSubjectInList = True
    if len(os.listdir(S_TEMP))!=len(os.listdir(B_TEMP)) or len(os.listdir(S_TEMP))!=len(os.listdir(B_TEMP)):
            check_delete(S_TEMP)
            check_delete(M_TEMP)
            check_delete(B_TEMP)
            check_delete(SVID_TEMP)
            check_delete(MVID_TEMP)
            check_delete(BVID_TEMP)
            check_create(META_DATA+'FRAME COUNT ISSUE WITH '+str(fm_sbj))
            continue
    eng =  matlab.engine.start_matlab()
    s_initial_imgs = glob.glob(S_TEMP+"*.png")
    subject_meta_dir = META_DATA+str(fm_sbj)+'/'
    check_create(subject_meta_dir)
    meta_data = dict()

    #calculations
    s_chess_board_imgs,s_chess_results = chess_board_check(s_initial_imgs)   
    print('A',end=',')  
    s_calibration = calibrate_from_img_list(eng,s_chess_board_imgs,'auto',int(fm_sbj))
    print('B',end=',')
    #x_calibration = cameraParams3coeff,imagesUsed,estimationErrors,chess_images_used,chess_points,worldPoints,mrows,ncols
    s_images_points = permute_matlab_point_cloud(s_calibration[4])
    eng.workspace['CameraParams']=s_calibration[0]
    #change to inplace error:
    eng.workspace['CameraParams'] = s_calibration[0]
    repErros = eng.eval('CameraParams.ReprojectionErrors')
    imgs = permute_matlab_point_cloud(repErros)
    s_initial_error_list = []
    for img in imgs:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        s_initial_error_list.append(errors_sum/len(img))
    s_selected,s_plot_img = kmeans_imgs_selection_Matlab(s_images_points,k=12,errors=s_initial_error_list)
    print('C',end=',')
    s_final_chessPoints = [s_images_points[i] for i  in s_selected]
    s_final_chessPoints = np.array(s_final_chessPoints)
    s_final_chessPoints = np.moveaxis(s_final_chessPoints,0,2)
    s_final_chessPoints = matlab.double([item for item in s_final_chessPoints])
    s_final_calibration = calibrate_from_points(eng,s_final_chessPoints,s_calibration[5],s_calibration[6],s_calibration[7])
    print('D',end=',')
    #meta data
    meta_data['s_chess_board_images'] = s_chess_board_imgs
    meta_data['s_chess_results'] = s_chess_results
    eng.workspace['s_calibration_initial'] = s_calibration
    eng.save(subject_meta_dir+'s_calibration_initial.mat', 's_calibration_initial',nargout = 0)
    meta_data['s_initial_error_list'] = s_initial_error_list
    meta_data['s_selected'] = s_selected
    eng.workspace['s_final_calibration'] = s_final_calibration
    eng.save(subject_meta_dir+'s_final_calibration.mat', 's_final_calibration',nargout = 0)
    cv2.imwrite(subject_meta_dir+'s_camera_kmeans_selection.png',s_plot_img)
    check_create(subject_meta_dir+'s_camera_frames/')
    s_used_imgs = [s_chess_board_imgs[i] for i in range(len(s_chess_board_imgs)) if s_calibration[3][i][0]]
    s_used_imgs = [s_used_imgs[i] for i in range(len(s_used_imgs)) if s_calibration[1][i][0]]
    s_used_imgs = [s_used_imgs[i] for i in s_selected]    
    for imgTC in s_used_imgs:
        shutil.copy(imgTC,subject_meta_dir+'s_camera_frames/')
    meta_data['s_final_used_imgs'] = s_used_imgs
    eng.workspace['CameraParams'] = s_final_calibration[0]
    repErros = eng.eval('CameraParams.ReprojectionErrors')
    imgs = permute_matlab_point_cloud(repErros)
    errors = []
    for img in imgs:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        errors.append(errors_sum/len(img))
    meta_data['s_final_error_mean'] = np.array(errors).mean()
    with open(subject_meta_dir+'meta_data_pickle','wb') as f:
        dump(meta_data,f)
    print('BB',end=',')
    b_initial_imgs = glob.glob(B_TEMP+"*.png")
    #calculations
    b_chess_board_imgs,b_chess_results = chess_board_check(b_initial_imgs)     
    print('BA',end=',')
    b_calibration = calibrate_from_img_list(eng,b_chess_board_imgs,'auto',int(fm_sbj))
    print('BB',end=',')
    #x_calibration = cameraParams3coeff,imagesUsed,estimationErrors,chess_images_used,chess_points,worldPoints,mrows,ncols
    b_images_points = permute_matlab_point_cloud(b_calibration[4])
    eng.workspace['CameraParams']=b_calibration[0]
    #change to inplace error:
    eng.workspace['CameraParams'] = b_calibration[0]
    repErros = eng.eval('CameraParams.ReprojectionErrors')
    imgs = permute_matlab_point_cloud(repErros)
    b_initial_error_list = []
    for img in imgs:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        b_initial_error_list.append(errors_sum/len(img))    
    b_selected,b_plot_img = kmeans_imgs_selection_Matlab(b_images_points,k=12,errors=b_initial_error_list)
    print('BC',end=',')
    b_final_chessPoints = [b_images_points[i] for i  in b_selected]
    b_final_chessPoints = np.array(b_final_chessPoints)
    b_final_chessPoints = np.moveaxis(b_final_chessPoints,0,2)
    b_final_chessPoints = matlab.double([item for item in b_final_chessPoints])
    b_final_calibration = calibrate_from_points(eng,b_final_chessPoints,b_calibration[5],b_calibration[6],b_calibration[7])
    print('BD',end=',')
    #meta data
    meta_data['b_chess_board_images'] = b_chess_board_imgs
    meta_data['b_chess_results'] = b_chess_results
    eng.workspace['b_calibration_initial'] = b_calibration
    eng.save(subject_meta_dir+'b_calibration_initial.mat', 'b_calibration_initial',nargout = 0)
    meta_data['b_initial_error_list'] = b_initial_error_list
    meta_data['b_selected'] = b_selected
    eng.workspace['b_final_calibration'] = b_final_calibration
    eng.save(subject_meta_dir+'b_final_calibration.mat', 'b_final_calibration',nargout = 0)
    cv2.imwrite(subject_meta_dir+'b_camera_kmeans_selection.png',b_plot_img)
    check_create(subject_meta_dir+'b_camera_frames/')
    b_used_imgs = [b_chess_board_imgs[i] for i in range(len(b_chess_board_imgs)) if b_calibration[3][i][0]]
    b_used_imgs = [b_used_imgs[i] for i in range(len(b_used_imgs)) if b_calibration[1][i][0]]
    b_used_imgs = [b_used_imgs[i] for i in b_selected]    
    for imgTC in b_used_imgs:
        shutil.copy(imgTC,subject_meta_dir+'b_camera_frames/')
    meta_data['b_final_used_imgs'] = b_used_imgs
    eng.workspace['CameraParams'] = b_final_calibration[0]
    repErros = eng.eval('CameraParams.ReprojectionErrors')
    imgs = permute_matlab_point_cloud(repErros)
    errors = []
    for img in imgs:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        errors.append(errors_sum/len(img))
    meta_data['b_final_error_mean'] = np.array(errors).mean()
    with open(subject_meta_dir+'meta_data_pickle','wb') as f:
        dump(meta_data,f)

    print('CC',end=',')
    m_initial_imgs = glob.glob(M_TEMP+"*.png")
    #calculations
    m_chess_board_imgs,m_chess_results = chess_board_check(m_initial_imgs)     
    print('CA',end=',')
    m_calibration = calibrate_from_img_list(eng,m_chess_board_imgs,'auto',int(fm_sbj))
    print('CB',end=',')
    #x_calibration = cameraParams3coeff,imagesUsed,estimationErrors,chess_images_used,chess_points,worldPoints,mrows,ncols
    m_images_points = permute_matlab_point_cloud(m_calibration[4])
    eng.workspace['CameraParams']=m_calibration[0]
    #change to inplace error:
    eng.workspace['CameraParams'] = m_calibration[0]
    repErros = eng.eval('CameraParams.ReprojectionErrors')
    imgs = permute_matlab_point_cloud(repErros)
    m_initial_error_list = []
    for img in imgs:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        m_initial_error_list.append(errors_sum/len(img))    
    m_selected,m_plot_img = kmeans_imgs_selection_Matlab(m_images_points,k=12,errors=m_initial_error_list)
    print('CC',end=',')
    m_final_chessPoints = [m_images_points[i] for i  in m_selected]
    m_final_chessPoints = np.array(m_final_chessPoints)
    m_final_chessPoints = np.moveaxis(m_final_chessPoints,0,2)
    m_final_chessPoints = matlab.double([item for item in m_final_chessPoints])
    m_final_calibration = calibrate_from_points(eng,m_final_chessPoints,m_calibration[5],m_calibration[6],m_calibration[7])
    print('CD',end=',')
    #meta data
    meta_data['m_chess_board_images'] = m_chess_board_imgs
    meta_data['m_chess_results'] = m_chess_results
    eng.workspace['m_calibration_initial'] = m_calibration
    eng.save(subject_meta_dir+'m_calibration_initial.mat', 'm_calibration_initial',nargout = 0)
    meta_data['m_initial_error_list'] = m_initial_error_list
    meta_data['m_selected'] = m_selected
    eng.workspace['m_final_calibration'] = m_final_calibration
    eng.save(subject_meta_dir+'m_final_calibration.mat', 'm_final_calibration',nargout = 0)
    cv2.imwrite(subject_meta_dir+'m_camera_kmeans_selection.png',m_plot_img)
    check_create(subject_meta_dir+'m_camera_frames/')
    m_used_imgs = [m_chess_board_imgs[i] for i in range(len(m_chess_board_imgs)) if m_calibration[3][i][0]]
    m_used_imgs = [m_used_imgs[i] for i in range(len(m_used_imgs)) if m_calibration[1][i][0]]
    m_used_imgs = [m_used_imgs[i] for i in m_selected]    
    for imgTC in m_used_imgs:
        shutil.copy(imgTC,subject_meta_dir+'m_camera_frames/')
    meta_data['m_final_used_imgs'] = m_used_imgs
    eng.workspace['CameraParams'] = m_final_calibration[0]
    repErros = eng.eval('CameraParams.ReprojectionErrors')
    imgs = permute_matlab_point_cloud(repErros)
    errors = []
    for img in imgs:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        errors.append(errors_sum/len(img))
    meta_data['m_final_error_mean'] = np.array(errors).mean()
    with open(subject_meta_dir+'meta_data_pickle','wb') as f:
        dump(meta_data,f)
    #calculations
    print('DD',end=',')
    b_chess_name_only = set()
    s_chess_name_only = set()
    m_chess_name_only = set()
    b_chess_name_to_img_index = dict()
    s_chess_name_to_img_index = dict()
    m_chess_name_to_img_index = dict()

    b_chess_board_imgs_used = [b_chess_board_imgs[i] for i in range(len(b_chess_board_imgs)) if b_calibration[3][i][0]]
    s_chess_board_imgs_used = [s_chess_board_imgs[i] for i in range(len(s_chess_board_imgs)) if s_calibration[3][i][0]]
    m_chess_board_imgs_used = [m_chess_board_imgs[i] for i in range(len(m_chess_board_imgs)) if m_calibration[3][i][0]]

    for i,img_p in enumerate(b_chess_board_imgs_used):
        img_path = pathlib.Path(img_p)
        img_name = img_path.name
        b_chess_name_to_img_index[img_name] = i
        b_chess_name_only.add(img_name)
    for i,img_p in enumerate(s_chess_board_imgs_used):
        img_path = pathlib.Path(img_p)
        img_name = img_path.name
        s_chess_name_to_img_index[img_name] = i
        s_chess_name_only.add(img_name)
    for i,img_p in enumerate(m_chess_board_imgs_used):
        img_path = pathlib.Path(img_p)
        img_name = img_path.name
        m_chess_name_to_img_index[img_name] = i
        m_chess_name_only.add(img_name)
    # print(b_chess_name_only,s_chess_name_only)
    # b_chess_name_only=list(b_chess_name_only)
    # s_chess_name_only=list(s_chess_name_only)
    # b_chess_name_only.sort()
    # s_chess_name_only.sort()
    # for i in range(len(b_chess_name_only)):
    #     if i>len(s_chess_name_only):
    #         continue
    #     print(b_chess_name_only[i],s_chess_name_only[i])
    s_b_common = b_chess_name_only.intersection(s_chess_name_only)
    m_b_common = b_chess_name_only.intersection(m_chess_name_only)
    s_m_common = m_chess_name_only.intersection(s_chess_name_only)
    # print(m_b_common)
    s_b_initial_stereo_points = []
    b_s_initial_stereo_points = []
    #find image points of common
    s_b_after_clean_common = []
    check_create(subject_meta_dir+'s_b_initial_stereo_frames/')
    check_create(subject_meta_dir+'b_s_initial_stereo_frames/')
    for common_img in s_b_common:
        s_b_to_add = s_images_points[s_chess_name_to_img_index[common_img]]
        b_s_to_add = b_images_points[b_chess_name_to_img_index[common_img]]
        #before combining - clean!
        if  not np.any(np.isnan(s_b_to_add)) and not np.any(np.isnan(b_s_to_add)):
            s_b_initial_stereo_points.append(s_b_to_add)
            b_s_initial_stereo_points.append(b_s_to_add)
            s_b_after_clean_common.append(common_img)
            s_img = S_TEMP+common_img
            b_img = B_TEMP+common_img
            shutil.copy(s_img,subject_meta_dir+'s_b_initial_stereo_frames/')
            shutil.copy(b_img,subject_meta_dir+'b_s_initial_stereo_frames/')
    s_b_initial_stereo_points = np.array(s_b_initial_stereo_points)
    b_s_initial_stereo_points = np.array(b_s_initial_stereo_points)
    m_b_initial_stereo_points = []
    b_m_initial_stereo_points = []
    #find image points of common
    m_b_after_clean_common = []
    check_create(subject_meta_dir+'m_b_initial_stereo_frames/')
    check_create(subject_meta_dir+'b_m_initial_stereo_frames/')
    for common_img in m_b_common:
        m_b_to_add = m_images_points[m_chess_name_to_img_index[common_img]]
        b_m_to_add = b_images_points[b_chess_name_to_img_index[common_img]]
        #before combining - clean!
        if  not np.any(np.isnan(m_b_to_add)) and not np.any(np.isnan(b_m_to_add)):
            m_b_initial_stereo_points.append(m_b_to_add)
            b_m_initial_stereo_points.append(b_m_to_add)
            m_b_after_clean_common.append(common_img)
            m_img = M_TEMP+common_img
            b_img = B_TEMP+common_img
            shutil.copy(m_img,subject_meta_dir+'m_b_initial_stereo_frames/')
            shutil.copy(b_img,subject_meta_dir+'b_m_initial_stereo_frames/')
    m_b_initial_stereo_points = np.array(m_b_initial_stereo_points)
    b_m_initial_stereo_points = np.array(b_m_initial_stereo_points)
    print(s_b_initial_stereo_points.shape)
    print(b_s_initial_stereo_points.shape)
    #initial stereo calibration
    print('DDD',end=',')
    s_b_combined = np.moveaxis(np.moveaxis(np.array([s_b_initial_stereo_points,b_s_initial_stereo_points]),0,3),0,2)
    s_b_combined = matlab.double([item for item in s_b_combined])
    s_b_stereo = eng.estimateStereoBaseline(s_b_combined,s_calibration[5],s_final_calibration[0],b_final_calibration[0],'WorldUnits', 'centimeters',nargout=3)
    eng.workspace['stereoParams'] = s_b_stereo[0]
    repErros_camera1 = eng.eval('stereoParams.CameraParameters1.ReprojectionErrors')
    repErros_camera2 = eng.eval('stereoParams.CameraParameters2.ReprojectionErrors')
    imgs1 = permute_matlab_point_cloud(repErros_camera1)
    imgs2 = permute_matlab_point_cloud(repErros_camera2)
    initial_errors_s_b = []
    initial_errors_b_s = []
    for img in imgs1:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        initial_errors_s_b.append(errors_sum/len(img))
    for img in imgs2:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        initial_errors_b_s.append(errors_sum/len(img))
    #kmeans+error minimizing image reselection (2X10 from each camera)
    s_b_stereo_selected,s_b_stereo_plot = kmeans_imgs_selection_Matlab(s_b_initial_stereo_points,k = 10,errors = initial_errors_s_b,count = 2,error_threshold = 3)
    b_s_stereo_selected,b_s_stereo_plot = kmeans_imgs_selection_Matlab(b_s_initial_stereo_points,k = 10,errors = initial_errors_b_s,count = 2,error_threshold = 3)
    print('DDA',end=',')
    s_b_final_images = set(s_b_stereo_selected).union(b_s_stereo_selected)
    s_b_final_points = [] 
    b_s_final_points = []
    for img in s_b_final_images:
        s_b_final_points.append(s_b_initial_stereo_points[img])
        b_s_final_points.append(b_s_initial_stereo_points[img])
    s_b_stereo_combined_final_points = np.moveaxis(np.moveaxis( np.array([s_b_final_points,b_s_final_points]),0,3),0,2)
    s_b_stereo_combined_final_points = matlab.double([item for item in s_b_stereo_combined_final_points])
    #final stereo calibration
    s_b_final_stereo = eng.estimateStereoBaseline(s_b_stereo_combined_final_points,s_calibration[5],s_final_calibration[0],b_final_calibration[0],'WorldUnits', 'centimeters',nargout=3)
    print('DDB',end=',')
    #meta data
    meta_data['s_b_common'] = s_b_common
    eng.workspace['s_b_intial_stereo'] = s_b_stereo
    eng.save(subject_meta_dir+'s_b_intial_stereo.mat', 's_b_intial_stereo',nargout = 0)
    meta_data['initial_errors_s_b'] = (initial_errors_s_b,initial_errors_b_s)
    meta_data['s_b_stereo_selected'] = s_b_stereo_selected
    meta_data['b_s_stereo_selected'] = b_s_stereo_selected
    eng.workspace['s_b_final_stereo'] = s_b_final_stereo
    eng.save(subject_meta_dir+'s_b_final_stereo.mat', 's_b_final_stereo',nargout = 0)
    
    eng.workspace['stereoParams'] = s_b_final_stereo[0]
    repErros_camera1 = eng.eval('stereoParams.CameraParameters1.ReprojectionErrors')
    repErros_camera2 = eng.eval('stereoParams.CameraParameters2.ReprojectionErrors')
    imgs1 = permute_matlab_point_cloud(repErros_camera1)
    imgs2 = permute_matlab_point_cloud(repErros_camera2)
    final_errors_s_b = []
    final_errors_b_s = []
    for img in imgs1:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        final_errors_s_b.append(errors_sum/len(img))
    for img in imgs2:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        final_errors_b_s.append(errors_sum/len(img))
    meta_data['s_b_final_error_mean'] = np.array([final_errors_s_b,final_errors_b_s]).mean()

    cv2.imwrite(subject_meta_dir+'s_b_stereo_kmeans_selection.png',s_b_stereo_plot)
    cv2.imwrite(subject_meta_dir+'b_s_stereo_kmeans_selection.png',b_s_stereo_plot)
    s_b_used_final_images = [s_b_after_clean_common[i] for i in list(s_b_final_images)]
    check_create(subject_meta_dir+'s_b_stereo_frames/')
    check_create(subject_meta_dir+'b_s_stereo_frames/')
    for sbufi in s_b_used_final_images:
        shutil.copy(s_chess_board_imgs_used[s_chess_name_to_img_index[sbufi]],subject_meta_dir+'s_b_stereo_frames/')
        shutil.copy(b_chess_board_imgs_used[b_chess_name_to_img_index[sbufi]],subject_meta_dir+'b_s_stereo_frames/')
    with open(subject_meta_dir+'meta_data_pickle','wb') as f:
        dump(meta_data,f)

    #initial stereo calibration
    print('EEE',end=',')
    m_b_combined = np.moveaxis(np.moveaxis(np.array([m_b_initial_stereo_points,b_m_initial_stereo_points]),0,3),0,2)
    m_b_combined = matlab.double([item for item in m_b_combined])
    m_b_stereo = eng.estimateStereoBaseline(m_b_combined,m_calibration[5],m_final_calibration[0],b_final_calibration[0],'WorldUnits', 'centimeters',nargout=3)
    eng.workspace['stereoParams'] = m_b_stereo[0]
    repErros_camera1 = eng.eval('stereoParams.CameraParameters1.ReprojectionErrors')
    repErros_camera2 = eng.eval('stereoParams.CameraParameters2.ReprojectionErrors')
    imgs1 = permute_matlab_point_cloud(repErros_camera1)
    imgs2 = permute_matlab_point_cloud(repErros_camera2)

    initial_errors_m_b = []
    initial_errors_b_m = []
    for img in imgs1:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        initial_errors_m_b.append(errors_sum/len(img))
    for img in imgs2:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        initial_errors_b_m.append(errors_sum/len(img))
    #kmeans+error minimizing image reselection (2X10 from each camera)
    m_b_stereo_selected,m_b_stereo_plot = kmeans_imgs_selection_Matlab(m_b_initial_stereo_points,k = 10,errors = initial_errors_m_b,count = 2,error_threshold = 3)
    b_m_stereo_selected,b_m_stereo_plot = kmeans_imgs_selection_Matlab(b_m_initial_stereo_points,k = 10,errors = initial_errors_b_m,count = 2,error_threshold = 3)
    print('EEA',end=',')
    m_b_final_images = set(m_b_stereo_selected).union(b_m_stereo_selected)
    m_b_final_points = [] 
    b_m_final_points = []
    for img in m_b_final_images:
        m_b_final_points.append(m_b_initial_stereo_points[img])
        b_m_final_points.append(b_m_initial_stereo_points[img])
    m_b_stereo_combined_final_points = np.moveaxis(np.moveaxis( np.array([m_b_final_points,b_m_final_points]),0,3),0,2)
    m_b_stereo_combined_final_points = matlab.double([item for item in m_b_stereo_combined_final_points])
    #final stereo calibration
    m_b_final_stereo = eng.estimateStereoBaseline(m_b_stereo_combined_final_points,m_calibration[5],m_final_calibration[0],b_final_calibration[0],'WorldUnits', 'centimeters',nargout=3)
    print('EEB',end=',')
    #meta data
    meta_data['m_b_common'] = m_b_common
    eng.workspace['m_b_intial_stereo'] = m_b_stereo
    eng.save(subject_meta_dir+'m_b_intial_stereo.mat', 'm_b_intial_stereo',nargout = 0)
    meta_data['initial_errors_m_b'] = (initial_errors_m_b,initial_errors_b_m)
    meta_data['m_b_stereo_selected'] = m_b_stereo_selected
    meta_data['b_m_stereo_selected'] = b_m_stereo_selected
    eng.workspace['m_b_final_stereo'] = m_b_final_stereo
    eng.save(subject_meta_dir+'m_b_final_stereo.mat', 'm_b_final_stereo',nargout = 0)
    
    eng.workspace['stereoParams'] = m_b_final_stereo[0]
    repErros_camera1 = eng.eval('stereoParams.CameraParameters1.ReprojectionErrors')
    repErros_camera2 = eng.eval('stereoParams.CameraParameters2.ReprojectionErrors')
    imgs1 = permute_matlab_point_cloud(repErros_camera1)
    imgs2 = permute_matlab_point_cloud(repErros_camera2)
    final_errors_m_b = []
    final_errors_b_m = []
    for img in imgs1:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        final_errors_m_b.append(errors_sum/len(img))
    for img in imgs2:
        errors_sum = 0 
        for point in img:
            point_error = (point[0]**2+point[1]**2)**0.5
            errors_sum+=point_error
        final_errors_b_m.append(errors_sum/len(img))
    meta_data['m_b_final_error_mean'] = np.array([final_errors_m_b,final_errors_b_m]).mean()

    cv2.imwrite(subject_meta_dir+'m_b_stereo_kmeans_selection.png',m_b_stereo_plot)
    cv2.imwrite(subject_meta_dir+'b_m_stereo_kmeans_selection.png',b_m_stereo_plot)
    m_b_used_final_images = [m_b_after_clean_common[i] for i in list(m_b_final_images)]
    check_create(subject_meta_dir+'m_b_stereo_frames/')
    check_create(subject_meta_dir+'b_m_stereo_frames/')
    for sbufi in m_b_used_final_images:
        shutil.copy(m_chess_board_imgs_used[m_chess_name_to_img_index[sbufi]],subject_meta_dir+'m_b_stereo_frames/')
        shutil.copy(b_chess_board_imgs_used[b_chess_name_to_img_index[sbufi]],subject_meta_dir+'b_m_stereo_frames/')

    with open(subject_meta_dir+'meta_data_pickle','wb') as f:
        dump(meta_data,f)
    try:
        print('O',end=',')
        if len(s_m_common) > 0:
            s_m_initial_stereo_points = []
            m_s_initial_stereo_points = []
            #find image points of common
            s_m_after_clean_common = []
            check_create(subject_meta_dir+'s_m_initial_stereo_frames/')
            check_create(subject_meta_dir+'m_s_initial_stereo_frames/')
            for common_img in s_m_common:
                s_m_to_add = s_images_points[s_chess_name_to_img_index[common_img]]
                m_s_to_add = m_images_points[m_chess_name_to_img_index[common_img]]
                #before combining - clean!
                if  not np.any(np.isnan(s_m_to_add)) and not np.any(np.isnan(m_s_to_add)):
                    s_m_initial_stereo_points.append(s_m_to_add)
                    m_s_initial_stereo_points.append(m_s_to_add)
                    s_m_after_clean_common.append(common_img)
                    s_img = S_TEMP+common_img
                    m_img = M_TEMP+common_img
                    shutil.copy(s_img,subject_meta_dir+'s_m_initial_stereo_frames/')
                    shutil.copy(m_img,subject_meta_dir+'m_s_initial_stereo_frames/')
            s_m_initial_stereo_points = np.array(s_m_initial_stereo_points)
            m_s_initial_stereo_points = np.array(m_s_initial_stereo_points)
            #initial stereo calibration
            s_m_combined = None
            s_m_combined = np.moveaxis(np.moveaxis(np.array([s_m_initial_stereo_points,m_s_initial_stereo_points]),0,3),0,2)
            s_m_combined = matlab.double([item for item in s_m_combined])
            s_m_stereo = eng.estimateStereoBaseline(s_m_combined,s_calibration[5],s_final_calibration[0],m_final_calibration[0],'WorldUnits', 'centimeters',nargout=3)
            eng.workspace['stereoParams'] = s_m_stereo[0]
            repErros_camera1 = eng.eval('stereoParams.CameraParameters1.ReprojectionErrors')
            repErros_camera2 = eng.eval('stereoParams.CameraParameters2.ReprojectionErrors')
            imgs1 = permute_matlab_point_cloud(repErros_camera1)
            imgs2 = permute_matlab_point_cloud(repErros_camera2)

            initial_errors_s_m = []
            initial_errors_m_s = []
            for img in imgs1:
                errors_sum = 0 
                for point in img:
                    point_error = (point[0]**2+point[1]**2)**0.5
                    errors_sum+=point_error
                initial_errors_s_m.append(errors_sum/len(img))
            for img in imgs2:
                errors_sum = 0 
                for point in img:
                    point_error = (point[0]**2+point[1]**2)**0.5
                    errors_sum+=point_error
                initial_errors_m_s.append(errors_sum/len(img))
            #kmeans+error minimizing image reselection (2X10 from each camera)
            s_m_stereo_selected,s_m_stereo_plot = kmeans_imgs_selection_Matlab(s_m_initial_stereo_points,k = min(10,len(s_m_common)),errors = initial_errors_s_m,count = 2,error_threshold = 5)
            m_s_stereo_selected,m_s_stereo_plot = kmeans_imgs_selection_Matlab(m_s_initial_stereo_points,k = min(10,len(s_m_common)),errors = initial_errors_m_s,count = 2,error_threshold = 5)

            s_m_final_images = set(s_m_stereo_selected).union(m_s_stereo_selected)
            s_m_final_points = [] 
            m_s_final_points = []
            for img in s_m_final_images:
                s_m_final_points.append(s_m_initial_stereo_points[img])
                m_s_final_points.append(m_s_initial_stereo_points[img])
            s_m_stereo_combined_final_points = np.moveaxis(np.moveaxis( np.array([s_m_final_points,m_s_final_points]),0,3),0,2)
            s_m_stereo_combined_final_points = matlab.double([item for item in s_m_stereo_combined_final_points])
            #final stereo calibration
            s_m_final_stereo = eng.estimateStereoBaseline(s_m_stereo_combined_final_points,s_calibration[5],s_final_calibration[0],m_final_calibration[0],'WorldUnits', 'centimeters',nargout=3)

            #meta data
            meta_data['s_m_common'] = s_m_common
            eng.workspace['s_m_intial_stereo'] = s_m_stereo
            eng.save(subject_meta_dir+'s_m_intial_stereo.mat', 's_m_intial_stereo',nargout = 0)
            meta_data['initial_errors_s_m'] = (initial_errors_s_m,initial_errors_m_s)
            meta_data['s_m_stereo_selected'] = s_m_stereo_selected
            meta_data['m_s_stereo_selected'] = m_s_stereo_selected
            eng.workspace['s_m_final_stereo'] = s_m_final_stereo
            eng.save(subject_meta_dir+'s_m_final_stereo.mat', 's_m_final_stereo',nargout = 0)
            eng.workspace['stereoParams'] = s_m_final_stereo[0]
            repErros_camera1 = eng.eval('stereoParams.CameraParameters1.ReprojectionErrors')
            repErros_camera2 = eng.eval('stereoParams.CameraParameters2.ReprojectionErrors')
            imgs1 = permute_matlab_point_cloud(repErros_camera1)
            imgs2 = permute_matlab_point_cloud(repErros_camera2)
            final_errors_m_s = []
            final_errors_s_m = []
            for img in imgs1:
                errors_sum = 0 
                for point in img:
                    point_error = (point[0]**2+point[1]**2)**0.5
                    errors_sum+=point_error
                final_errors_m_s.append(errors_sum/len(img))
            for img in imgs2:
                errors_sum = 0 
                for point in img:
                    point_error = (point[0]**2+point[1]**2)**0.5
                    errors_sum+=point_error
                final_errors_s_m.append(errors_sum/len(img))
            meta_data['s_m_final_error_mean'] = np.array([final_errors_m_s,final_errors_s_m]).mean()

            cv2.imwrite(subject_meta_dir+'s_m_stereo_kmeans_selection.png',s_m_stereo_plot)
            cv2.imwrite(subject_meta_dir+'m_s_stereo_kmeans_selection.png',m_s_stereo_plot)
            s_m_used_final_images = [s_m_after_clean_common[i] for i in list(s_m_final_images)]
            check_create(subject_meta_dir+'s_m_stereo_frames/')
            check_create(subject_meta_dir+'m_s_stereo_frames/')
            for sbufi in s_m_used_final_images:
                shutil.copy(s_chess_board_imgs_used[s_chess_name_to_img_index[sbufi]],subject_meta_dir+'s_m_stereo_frames/')
                shutil.copy(m_chess_board_imgs_used[m_chess_name_to_img_index[sbufi]],subject_meta_dir+'m_s_stereo_frames/')
    except:
        print("S_M failed")
    #end of subject iteration
    with open(subject_meta_dir+'meta_data_pickle','wb') as f:
        dump(meta_data,f)
    eng.quit()
    check_delete(S_TEMP)
    check_delete(M_TEMP)
    check_delete(B_TEMP)
    check_delete(SVID_TEMP)
    check_delete(MVID_TEMP)
    check_delete(BVID_TEMP)
    print('Finished, Wating 5 seconds...')
    time.sleep(5)

