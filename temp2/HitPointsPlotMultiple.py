from pickle import load
import matplotlib.pyplot as plt
import numpy as np
from RIGID3D import *
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
import matplotlib
# matplotlib.use("Ag0")

for subject in ['234','230','212','210']:

    with open('3Dpickles/intrDict_'+subject+'_pen1','rb') as f:
        intrPoints1Dict = load(f)
    with open('3Dpickles/sFaceDict_'+subject+'_pen1','rb') as f:
        subject_faces1Dict = load(f)
    with open('3Dpickles/sEyesDict_'+subject+'_pen1','rb') as f:
        subject_eyes1Dict = load(f)
    with open('3Dpickles/mFace_'+subject+'_pen1','rb') as f:
        marissa_faces1 = load(f)

    with open('3Dpickles/intrDict_'+subject+'_pen2','rb') as f:
        intrPoints2Dict = load(f)
    with open('3Dpickles/sFaceDict_'+subject+'_pen2','rb') as f:
        subject_faces2Dict = load(f)
    with open('3Dpickles/sEyesDict_'+subject+'_pen2','rb') as f:
        subject_eyes2Dict = load(f)
    with open('3Dpickles/mFace_'+subject+'_pen2','rb') as f:
        marissa_faces2 = load(f)

    #intrdict = (face_cloud_from_cropped,gaze_from_cropped,gaze_from_combined)
    for key in intrPoints1Dict:
        print(key)
        intrPoints1 = intrPoints1Dict[key]
        intrPoints2 = intrPoints2Dict[key]

        marissa_faces_array1 = np.array(marissa_faces1)
        hps_array1 = np.array(intrPoints1)
        mean_marissa_face1 = marissa_faces_array1.mean(axis=0)
        marissa_faces_array2 = np.array(marissa_faces2)
        hps_array2 = np.array(intrPoints2)
        print(marissa_faces_array2.shape)
        # print(marissa_faces)


        # fig = plt.figure()
        # ax = Axes3D(fig)

        mean_marissa_face2 = marissa_faces_array2.mean(axis=0)
        transformed_hps = []
        face_cloud_from_cropped,gaze_from_cropped,gaze_from_combined,intersecionPlane = key
        if not face_cloud_from_cropped or not gaze_from_cropped:
            continue
        for i,mf2 in enumerate(marissa_faces2):
            # break
            dist1 = np.linalg.norm(mean_marissa_face1[16]-mean_marissa_face1[0])
            dist2 = np.linalg.norm(mf2[16]-mf2[0])
            old = mf2.mean(axis=0)
            mf2 = mf2- old
            mf2/= (dist2/dist1)
            mf2+=old

            # R,t = rigid_transform_3D(mf2.T,mean_marissa_face1.T)
            # transformed_mf2 = ((R@mf2.T)+t).T
            # plt.scatter(mean_marissa_face1[:,2],-mean_marissa_face1[:,1],color = 'black',s=100)
            # plt.scatter(mean_marissa_face1.mean(axis=0)[2],-mean_marissa_face1.mean(axis=0)[1],color = 'black',edgecolors='green',s=300)
            # plt.scatter(transformed_mf2.mean(axis=0)[2],-transformed_mf2.mean(axis=0)[1],color = 'black',edgecolors='green',s=100)

            # plt.scatter(transformed_mf2[:,2],-transformed_mf2[:,1],color = 'gray')
            hp = intrPoints2[i]
            # old = mf2.mean(axis=0)
            # hp -= old
            # hp /=(dist2/dist1)
            # hp+=old
            # hp=np.expand_dims(hp,1)
            R = np.eye(3,3)
            t = mean_marissa_face1.mean(axis=0) - mf2.mean(axis=0)

            transformed_hp = hp-t.T
            if(np.isnan(transformed_hp).any()):
                continue
            transformed_hps.append(transformed_hp)
    
            # transformed_hp = hp.T
            # plt.scatter(transformed_hp[2],-transformed_hp[1],color = 'blue')
            # ax.scatter3D(transformed_hp[0],-transformed_hp[1],transformed_hp[2],color = 'blue')
            # plt.show()
        transformed_hps_list = transformed_hps
        # ax = sns.heatmap(uniform_data, linewidth=0.5)
        transformed_hps = np.array(transformed_hps)
        transformed_hps = transformed_hps.squeeze()
        print(transformed_hps.shape)
        x = transformed_hps[:,2]
        y = -transformed_hps[:,1]
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


        for hp in intrPoints1:
            plt.scatter(hp[2],-hp[1],color = 'red',alpha=0.1)
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

        plt.scatter(mean_marissa_face1[:,2],-mean_marissa_face1[:,1],color = 'black')
        plt.scatter(mean_marissa_face2[:,2],-mean_marissa_face2[:,1],color = 'gray')
        str1 = 'FaceCloudFromCropped-'+str(face_cloud_from_cropped)+'\n'
        str1 += 'GazeFromCropped-'+str(gaze_from_cropped)+'\n'
        str1+=  'GazeFromCombined-'+str(gaze_from_combined)
        str1+= 'IntersetionPlane- '+str(intersecionPlane)
        plt.title(str1)
        plt.savefig('figs2/pens_'+subject+str1.replace('\n','_')+'.png')
        plt.clf()
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