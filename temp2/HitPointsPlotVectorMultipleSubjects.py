from pickle import load
import matplotlib.pyplot as plt
import numpy as np
from RIGID3D import *
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
import matplotlib
# matplotlib.use("Ag0")
import pandas as pd
from MathCode import *
import sympy
from CheckCreate import check_create
from VideoDictLoader import VideosDictLoader
import os
def get_currect_range_for_subject_given_start_and_end_of_pen_calib(subject,task,start,end):
    openface_source = 'c:/openface_dired/'+subject+'/'
    S_openFace_File  = openface_source + subject+'_'+task+'_subject.csv'
    S_openface = pd.read_csv(S_openFace_File).set_index('frame')
    cols = S_openface.columns
    new_cols = [col.strip() for col in cols]
    S_openface.columns = new_cols
    range = get_frames_in_range_ms(start,end,S_openface)

    return range

# for subject in ['209','232','235','234','230','212','210']:
subjects = VideosDictLoader.get_gaze_subject_list()
# subjects = [224,225,226,227]
for subject in subjects:
    try:
            
        subject = str(subject)
        task = 'pen2'
        if not os.path.exists('3Dpickles/intrDict_'+subject+'_pen1') or not os.path.exists('3Dpickles/intrDict_'+subject+'_'+task):
            continue
        try: 
            with open('3Dpickles/intrDict_'+subject+'_pen1','rb') as f:
                intrPoints1Dict = load(f)
            with open('3Dpickles/sFaceDict_'+subject+'_pen1','rb') as f:
                subject_faces1Dict = load(f)
            with open('3Dpickles/sEyesDict_'+subject+'_pen1','rb') as f:
                subject_eyes1Dict = load(f)
            with open('3Dpickles/mFace_'+subject+'_pen1','rb') as f:
                marissa_faces1 = load(f)
            with open('3Dpickles/linesUsed_'+subject+'_'+'pen1','rb') as f:
                lines_used1 = load(f)
            if subject == '227':
                subject_range = get_currect_range_for_subject_given_start_and_end_of_pen_calib(subject,task,18000,25000)
                marissa_faces1 = marissa_faces1[subject_range[0]:subject_range[1]]   
            with open('3Dpickles/intrDict_'+subject+'_'+task,'rb') as f:
                intrPoints2Dict = load(f)
            with open('3Dpickles/sFaceDict_'+subject+'_'+task,'rb') as f:
                subject_faces2Dict = load(f)
            with open('3Dpickles/sEyesDict_'+subject+'_'+task,'rb') as f:
                subject_eyes2Dict = load(f)
            with open('3Dpickles/mFace_'+subject+'_'+task,'rb') as f:
                marissa_faces2 = load(f)
            with open('3Dpickles/linesUsed_'+subject+'_'+task,'rb') as f:
                lines_used2 = load(f)
        except:
            print('exception ',subject)
            continue
        #intrdict = (face_cloud_from_cropped,gaze_from_cropped,gaze_from_combined)

        for key in intrPoints1Dict:
            print(key)
            intrPoints1 = intrPoints1Dict[key]
            intrPoints2 = intrPoints2Dict[key]
            subject_range = None
            if subject =='230':
                subject_range = get_currect_range_for_subject_given_start_and_end_of_pen_calib(subject,task,7650,29915)
                intrPoints1 = intrPoints1[subject_range[0]:subject_range[1]]
            if subject == '234':
                subject_range = get_currect_range_for_subject_given_start_and_end_of_pen_calib(subject,task,3656,28753)
                intrPoints1 = intrPoints1[subject_range[0]:subject_range[1]]
            if subject == '232':
                subject_range = get_currect_range_for_subject_given_start_and_end_of_pen_calib(subject,task,4151,26000)
                intrPoints1 = intrPoints1[subject_range[0]:subject_range[1]]   
            if subject == '235':
                subject_range = get_currect_range_for_subject_given_start_and_end_of_pen_calib(subject,task,12000,27000)
                intrPoints1 = intrPoints1[subject_range[0]:subject_range[1]]   
            if subject == '227':
                subject_range = get_currect_range_for_subject_given_start_and_end_of_pen_calib(subject,task,18000,25000)
                intrPoints1 = intrPoints1[subject_range[0]:subject_range[1]]   
                # marissa_faces1 = marissa_faces1[subject_range[0]:subject_range[1]]   
                # lines_used1 = lines_used1[key]
            marissa_faces_array1 = np.array(marissa_faces1)
            hps_array1 = np.array(intrPoints1)
            mean_marissa_face1 = marissa_faces_array1.mean(axis=0)
            marissa_faces_array2 = np.array(marissa_faces2)
            hps_array2 = np.array(intrPoints2)
            print(marissa_faces_array2.shape)
            print(len(lines_used2))
            # print(marissa_faces)


            # fig = plt.figure()
            # ax = Axes3D(fig)

            lines_used_array = np.array(lines_used1[key])
            if subject_range:
                lines_used_array = lines_used_array[subject_range[0]:subject_range[1]]
            if key[-1]:
                continue
            eyes_used = lines_used_array[:,0,:]
            mean_eye = eyes_used.mean(axis=0)
            
            mean_marissa_face2 = marissa_faces_array2.mean(axis=0)
            transformed_hps = []
            face_cloud_from_cropped,gaze_from_cropped,gaze_from_combined,intersecionPlane = key
            if not face_cloud_from_cropped or not gaze_from_cropped:
                continue
            for i,mf2 in enumerate(marissa_faces2):
                hp = intrPoints2[i]
                d = mean_marissa_face1.mean(axis=0) - mf2.mean(axis=0)
                eye,vector = lines_used2[key][i]
                v1 = mean_marissa_face1.mean(axis=0) - mean_eye 
                # v1 = normalize(v1)
                v2 = hps_array1.mean(axis=0) - mean_eye
                # v2 = normalize(v2)
                v3 = hp - eye
                v3 = normalize(v3)

                vector = normalize(vector)
                if ((v3-vector)>0.000001).any():
                    print(vector,v3)
                    print(v3-vector)
                    input()
                # v3 = v3 / np.linalg.norm(v3)
                v1 = normalize(v1)
                v2 = normalize(v2)
                v4 = find_vector_with_same_angle(v1,v2,v3)
                v4 = normalize(v4)
                # print(np.degrees(angle_between_vectors(v4,v3)),np.degrees(angle_between_vectors(v1,v2)))
                # print(np.dot(v4,v3),np.dot(v2,v1))
                # input()
                # v4 = v4 / np.linalg.norm(v4)
                eye_vect = mf2.mean(axis=0) - eye
                # print(eye_vect)
                # input()
                x_delta = eye_vect[0]
                # eye_vect = normalize(eye_vect)
                t = x_delta/v4[0]
                # transformed_hp = eye+t*v4
                transformed_hp = point_line_distance(mf2.mean(axis=0),(eye,v4))
                faceDefiningKeyPoints = [36,45,8]
                mf2PlanePoints = mf2[faceDefiningKeyPoints]
                # from sympy import Point3D
                # p1 = Point3D(mf2PlanePoints[0,0],mf2PlanePoints[0,1],mf2PlanePoints[0,2])
                # p2 = Point3D(mf2PlanePoints[1,0],mf2PlanePoints[1,1],mf2PlanePoints[1,2])
                # p3 = Point3D(mf2PlanePoints[2,0],mf2PlanePoints[2,1],mf2PlanePoints[2,2])
                # mPlane = sympy.Plane(p1,p2,p3)
                # thpPoint = Point3D(transformed_hp[0],transformed_hp[1],transformed_hp[2])
                # transformed_hp = mPlane.projection(thpPoint)
                # transformed_hp = np.array([transformed_hp[0],transformed_hp[1],transformed_hp[2]],dtype=np.float64)
                # transformed_hp = project_point_onto_plane(transformed_hp,mf2PlanePoints.tolist())
                # print(transformed_hp)
                # input()
                if(np.isnan(transformed_hp).any()):
                    continue
                transformed_hps.append(transformed_hp)
        
                # transformed_hp = hp.T
                # plt.scatter(transformed_hp[2],-transformed_hp[1],color = 'blue')
                # ax.scatter3D(transformed_hp[0],-transformed_hp[1],transformed_hp[2],color = 'blue')
                # plt.show()
            mean_marissa_face2_plane = mean_marissa_face2[faceDefiningKeyPoints]
            rot_mat = get_plane_rot_matrix(mean_marissa_face2[0],mean_marissa_face2_plane.tolist())

            transformed_hps_list = transformed_hps
            # ax = sns.heatmap(uniform_data, linewidth=0.5)
            transformed_hps = np.array(transformed_hps)
            transformed_hps = transformed_hps.squeeze()
            transformed_hps = (rot_mat@transformed_hps.T).T
            print(transformed_hps.shape)
            x = transformed_hps[:,0]
            y = -transformed_hps[:,1]
            plt.scatter(x,y)
            # print(x)  
            # print(y)
            # print(list(y))
            heatmap, xedges, yedges = np.histogram2d(x,y)
            plt.pcolormesh(xedges, yedges, heatmap.T)

            # from matplotlib import cm
            # f = plt.figure()
            # ax = f.add_subplot(111)
            # ax.pcolormesh(x,y,f(x,y), cmap = cm.Blues)
            # plt.imshow(heatmap)


            # for hp in intrPoints1:
            #     plt.scatter(hp[2],-hp[1],color = 'red',alpha=0.1)
                # ax.scatter3D(hp[0],-hp[1],hp[2],color = 'red',alpha = 0.1)
            # plt.scatter(x,y)
            # intrPoints2 = np.array(intrPoints2)
            # plt.scatter(intrPoints2[:,2],-intrPoints2[:,1],color = 'magenta',alpha=1)
            # for hp in intrPoints2:
            #     plt.scatter(hp[2],-hp[1],color = 'magenta',alpha=1)
            # print(hps_array2.mean(axis=0).shape)
            # plt.scatter(hps_array2.mean(axis=0)[2],-hps_array2.mean(axis=0)[1],color = 'blue',edgecolors='green',s=100)
            # print(mean_marissa_face2.shape)
            # plt.scatter(mean_marissa_face2[:,2],-mean_marissa_face2[:,1],color = 'black')
            # plt.scatter(mean_marissa_face2.mean(axis=0)[2],-mean_marissa_face2.mean(axis=0)[1],color = 'black',edgecolors='green',s=100)

            # plt.scatter(mean_marissa_face1[:,2],-mean_marissa_face1[:,1],color = 'black')
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.scatter3D(mean_marissa_face2[:,2],-mean_marissa_face2[:,1])
            # plt.show()
            # input()
            num = mean_marissa_face2.shape[0]
            mean_marissa_face2 = (rot_mat@mean_marissa_face2.T).T
            plt.scatter(mean_marissa_face2[:,0],-mean_marissa_face2[:,1],color = 'gray')
            # exit(1)
            # print(range(num))
            # for i in range(num):
                # mf2Planed = project_point_onto_plane(mean_marissa_face2[i],mean_marissa_face2_plane.tolist())
                # ax.scatter3D(mean_marissa_face2[:,0],mean_marissa_face2[:,1],mean_marissa_face2[:,2])
                # print('c')
            # plt.show()
            str1 = 'FaceCloudFromCropped-'+str(face_cloud_from_cropped)+'\n'
            str1 += 'GazeFromCropped-'+str(gaze_from_cropped)+'\n'
            str1+=  'GazeFromCombined-'+str(gaze_from_combined)
            str1+= 'IntersetionPlane- '+str(intersecionPlane)
            plt.title(str1)
            base_dir = 'figs5pen/'
            subject_dir = base_dir+str(subject)+'/'
            check_create(subject_dir)
            plt.savefig(subject_dir+'pens_'+subject+str1.replace('\n','_')+'.png')
            plt.clf()
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        continue
# exit(1)
import tqdm
frame = 0
exit(1)
for transformed_hp in tqdm.tqdm(transformed_hps_list):
    plt.ylim((0,100))
    plt.xlim((125,300))
    if frame > 0:
        # plt.scatter(mean_marissa_face1[:,2],-mean_marissa_face1[:,1],color = 'black')
        plt.scatter(marissa_faces2[frame][:,2],-marissa_faces2[frame][:,1],color = 'gray')
        # for hp in intrPoints1:
        #     plt.scatter(hp[2],-hp[1],color = 'red',alpha=0.1)
    plt.scatter(transformed_hp[2],-transformed_hp[1],color = 'blue')
    from CheckCreate import check_create
    check_create('tempFigsForTB/')
    plt.savefig('tempFigsForTB/'+str(frame).zfill(len(str(len(transformed_hps_list))))+'.png')
    plt.close()
    frame+=1
    continue
# ax.scatter3D(mean_marissa_face1[:,0],-mean_marissa_face1[:,1],mean_marissa_face1[:,2],color = 'black')
# ax.scatter3D(mean_marissa_face2[:,0],-mean_marissa_face2[:,1],mean_marissa_face2[:,2],color = 'gray')
# ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
# ax.view_init(-77,-90) #elev,azim

# plt.show()
import os
import cv2 as cv
images = [img for img in os.listdir('tempFigsForTB/') if img.endswith(".png")]
frame = cv.imread(os.path.join('tempFigsForTB/', images[0]))
height, width, layers = frame.shape
fourcc = cv.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
video = cv.VideoWriter(subject+'_TBGaze.mp4', fourcc, 30, (width,height))
import tqdm
for image in tqdm.tqdm(images):
    video.write(cv.imread(os.path.join('tempFigsForTB/', image)))