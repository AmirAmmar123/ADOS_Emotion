import os
import glob
from pickle import dump,load
from CheckCreate import check_create,check_delete
from SplitToFrames import extract_image_one_fps
import shutil
from SelectImgsForCalibration import chess_board_image_selection,kmeans_imgs_selection 
from VideoDictLoader import VideosDictLoader
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
import cv2
import pathlib

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
check_create(PICKLES_DIR)
VIDEO_FILES = VideosDictLoader.loadVideosDict()
if not VIDEO_FILES:
    exit(0)
fm_sbjs, sm_sbjs = process_subjects(VIDEO_FILES)
for sbjs in fm_sbjs:
    print(sbjs)
    #split frame for each video
    shutil.copy(VIDEO_FILES[sbjs]['CALIBRATION']['subject'],SVID_TEMP)
    shutil.copy(VIDEO_FILES[sbjs]['CALIBRATION']['marissa'],MVID_TEMP)
    shutil.copy(VIDEO_FILES[sbjs]['CALIBRATION']['both'],BVID_TEMP)
    check_create(S_TEMP)
    check_create(M_TEMP)
    check_create(B_TEMP)
    # extract_image_one_fps(SVID_TEMP,S_TEMP)
    # extract_image_one_fps(BVID_TEMP,B_TEMP)
    # extract_image_one_fps(MVID_TEMP,M_TEMP)
    check_create('c:/chess_images/')
    check_create('c:/chess_images/'+str(sbjs)+'/')
    check_create('c:/chess_images/'+str(sbjs)+'/subject/')
    check_create('c:/chess_images/'+str(sbjs)+'/marissa/')
    check_create('c:/chess_images/'+str(sbjs)+'/both/')


    #detect images for each folder
    #  x_ret = selected_images 0, closest 1, kmeans 2, plot_img 3, checked_imgs4 
    # , corners_list 5, objpoints 6, imgpoints 7
    # full explanation in end of function
    s_ret = chess_board_image_selection(S_TEMP,sbjs,'subject',(6,8),PICKLES_DIR,30,'auto','c:/chess_images/'+str(sbjs)+'/subject/')
    cv2.imwrite('c:/chess_images/'+str(sbjs)+'/s_selected_plot.png',s_ret[3])
    print(len(s_ret[1]),(len(s_ret[6])))
    s_selected_objpoints = [s_ret[6][i] for i in s_ret[1]]
    s_selected_imgpoints = [s_ret[7][i] for i in s_ret[1]]
    #calibrate camera
    s_rmse, s_mtx, s_dist, s_rvecs, s_tvecs = cv2.calibrateCamera(s_selected_objpoints, s_selected_imgpoints, (F_WIDTH, F_HEIGHT), None, None)
    print(s_rmse)

    m_ret = chess_board_image_selection(M_TEMP,sbjs,'marissa',(6,8),PICKLES_DIR,30,'auto','c:/chess_images/'+str(sbjs)+'/marissa/')
    cv2.imwrite('c:/chess_images/'+str(sbjs)+'/m_selected_plot.png',m_ret[3])
    m_selected_objpoints = [m_ret[6][i] for i in m_ret[1]]
    m_selected_imgpoints = [m_ret[7][i] for i in m_ret[1]]
    #calibrate camera
    m_rmse, m_mtx, m_dist, m_rvecs, m_tvecs = cv2.calibrateCamera(m_selected_objpoints, m_selected_imgpoints, (F_WIDTH, F_HEIGHT), None, None)

    b_ret = chess_board_image_selection(B_TEMP,sbjs,'both',(6,8),PICKLES_DIR,30,'auto','c:/chess_images/'+str(sbjs)+'/both/')
    cv2.imwrite('c:/chess_images/'+str(sbjs)+'/b_selected_plot.png',b_ret[3])
    b_selected_objpoints = [b_ret[6][i] for i in b_ret[1]]
    b_selected_imgpoints = [b_ret[7][i] for i in b_ret[1]]
    #calibrate camera
    b_rmse, b_mtx, b_dist, b_rvecs, b_tvecs = cv2.calibrateCamera(b_selected_objpoints, b_selected_imgpoints, (F_WIDTH, F_HEIGHT), None, None)

    #stereo calibration 
    s_set = set()
    m_set = set()
    b_set = set()
    s_set_reverse = dict()
    m_set_reverse = dict()
    b_set_reverse = dict()
    s_img_to_imgpts = {}
    b_img_to_imgpts = {}
    m_img_to_imgpts = {}
    for i,img_path in enumerate(s_ret[4]):
        img_p = pathlib.Path(img_path[0])
        img_name = img_p.name
        s_set.add(img_name)
        s_set_reverse[img_name] = (img_path,i)
        s_img_to_imgpts[img_name] = s_ret[7][i]
    for i,img_path in enumerate(b_ret[4]):
        img_p = pathlib.Path(img_path[0])
        img_name = img_p.name
        b_set.add(img_name)
        b_set_reverse[img_name] = (img_path,i)
        b_img_to_imgpts[img_name] = b_ret[7][i]
    for i,img_path in enumerate(m_ret[4]):
        img_p = pathlib.Path(img_path[0])
        img_name = img_p.name
        m_set.add(img_name)
        m_set_reverse[img_name] = (img_path,i)
        m_img_to_imgpts[img_name] = m_ret[7][i]

    b_and_s_imgs = b_set.intersection(s_set)
    m_and_s_imgs = m_set.intersection(s_set)
    common_mb_flag = len(m_and_s_imgs)>0
    m_and_b_imgs = m_set.intersection(b_set)

    bs_inter_indx = [b_set_reverse[img][1] for img in b_and_s_imgs]
    bs_inter_corners_list = [b_ret[7][i] for i in bs_inter_indx]
    bs_stereo_imgs,img = kmeans_imgs_selection(bs_inter_corners_list)
    cv2.imwrite('c:/chess_images/'+str(sbjs)+'/b selected from b&s plot.png',img)
    sb_inter_indx = [s_set_reverse[img][1] for img in b_and_s_imgs]
    sb_inter_corners_list = [s_ret[7][i] for i in sb_inter_indx]
    sb_stereo_imgs,img = kmeans_imgs_selection(sb_inter_corners_list)
    cv2.imwrite('c:/chess_images/'+str(sbjs)+'/s selected from b&s plot.png',img)
    b_s_selected_for_stereo = set(bs_stereo_imgs).union(sb_stereo_imgs)
    if(common_mb_flag):
        ms_inter_indx = [m_set_reverse[img][1] for img in m_and_s_imgs]
        ms_inter_corners_list = [m_ret[7][i] for i in ms_inter_indx]
        ms_stereo_imgs,img = kmeans_imgs_selection(ms_inter_corners_list)
        cv2.imwrite('c:/chess_images/'+str(sbjs)+'/m selected from m&s plot.png',img)
        sm_inter_indx = [s_set_reverse[img][1] for img in m_and_s_imgs]
        sm_inter_corners_list = [s_ret[7][i] for i in sm_inter_indx]
        sm_stereo_imgs,img = kmeans_imgs_selection(sm_inter_corners_list)
        cv2.imwrite('c:/chess_images/'+str(sbjs)+'/s selected from m&s plot.png',img)
        m_s_selected_for_stereo = set(ms_stereo_imgs).union(sm_stereo_imgs)

    m_and_b_imgs = list(m_and_b_imgs)
    m_and_b_imgs.sort()
    mb_inter_indx_imgs = {}
    mb_inter_indx = []
    for img in m_and_b_imgs:
        mb_inter_indx.append(m_set_reverse[img][1])
        mb_inter_indx_imgs[mb_inter_indx[-1]]=img
    mb_inter_indx.sort()
    mb_inter_corners_rev = {}
    mb_inter_corners_list = []
    for j,i in enumerate(mb_inter_indx):
        mb_inter_corners_list.append(m_ret[7][i])
        mb_inter_corners_rev[j]=i
    mb_stereo_imgs,img = kmeans_imgs_selection(mb_inter_corners_list)
    cv2.imwrite('c:/chess_images/'+str(sbjs)+'/m selected from m&b plot.png',img)
    bm_inter_indx = []
    bm_inter_indx_imgs= {}
    for img in m_and_b_imgs:
        bm_inter_indx.append(b_set_reverse[img][1])
        bm_inter_indx_imgs[bm_inter_indx[-1]]=img
    bm_inter_indx.sort()
    bm_inter_corners_list = []
    bm_inter_corners_rev ={}
    for j,i in enumerate(bm_inter_indx):
        bm_inter_corners_list.append(b_ret[7][i])
        bm_inter_corners_rev[j]=i
    bm_stereo_imgs,img = kmeans_imgs_selection(bm_inter_corners_list)
    cv2.imwrite('c:/chess_images/'+str(sbjs)+'/b selected from m&b plot.png',img)
    bm_stereo_real_inx = [bm_inter_indx_imgs[bm_inter_corners_rev[i]] for i in bm_stereo_imgs]
    mb_stereo_real_inx = [mb_inter_indx_imgs[mb_inter_corners_rev[i]]for i in mb_stereo_imgs]
    bm_stereo_real_inx.sort()
    mb_stereo_real_inx.sort()

    m_b_selected_for_stereo = set(bm_stereo_real_inx).union(mb_stereo_real_inx)
    m_b_selected_for_stereo = list(m_b_selected_for_stereo)
    m_b_selected_for_stereo.sort()
    stereo_objpts = []
    for i in m_b_selected_for_stereo:
        stereo_objpts.append(m_ret[6][m_set_reverse[i][1]])
    # stereo_objpts = [stereo_objpts[i] for i in m_b_selected_for_stereo]
    mb_stereo_imgpts = []
    bm_stereo_imgpts = []
    for i in m_b_selected_for_stereo:
        mb_stereo_imgpts.append(m_img_to_imgpts[i])
        bm_stereo_imgpts.append(b_img_to_imgpts[i])

    for img in m_b_selected_for_stereo:
        
        shutil.copy(m_set_reverse[img][0][0] ,'c:/temp1/')
        shutil.copy(b_set_reverse[img][0][0] ,'c:/temp2/')

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC

    # ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
    # #                                                                  mtx2, dist2, (F_WIDTH, F_HEIGHT), criteria = criteria, flags = stereocalibration_flags)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
    stereo_params_s_m = cv2.stereoCalibrate(stereo_objpts, mb_stereo_imgpts, bm_stereo_imgpts, m_mtx, m_dist,
                                            b_mtx, b_dist, (F_WIDTH, F_HEIGHT), criteria = criteria, flags = stereocalibration_flags)
    print(b_mtx)
    print(stereo_params_s_m[0])
    # stereo_params = {}
    #delete folders
    # check_delete(S_TEMP)
    # check_delete(M_TEMP)
    # check_delete(B_TEMP)
    # check_delete(SVID_TEMP)
    # check_delete(MVID_TEMP)
    # check_delete(BVID_TEMP)
    break
