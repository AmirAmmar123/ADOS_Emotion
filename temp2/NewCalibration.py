from sqlite3 import Timestamp
from Files import *
import numpy as np
import pandas as pd
from StereoOfVideo_v1 import get3DPointsFromVideo,calc_frame
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from tqdm.notebook import trange, tqdm
FACE = {"Center" : (11900,13901),"Up":(8401,11401),"Down" : (14652,17152),"Right":(25150,27401),"Left" :(20400,22901)}
# FACE = {"Center" : (8400,9653),"Up":(5153,5901),"Down" : (7150,8150),"Right":(9900,10653),"Left" :(12152,12901)}
INV_FACE = {v: k for k, v in FACE.items()}
COLOR = {'Center':'r','Up' : 'g','Down':'b','Right':'y','Left' : 'm'}
# pointsFromFrames =  get3DPointsFromVideo(save_video=False,reload = False,drawHitPoints=False)
# save_pickle_file(pointsFromFrames,"Save pickle of list")

def process(plot = False):
    pointsFromFrames = load_pickle()
    points2Part = dict()
    for key in FACE.keys():
        points2Part[key] = list()
    mean_xm = np.zeros((len(pointsFromFrames[0][0][0])))
    mean_ym = np.zeros((len(pointsFromFrames[0][0][0])))
    mean_zm = np.zeros((len(pointsFromFrames[0][0][0])))
    counter =0 
    for i,frame in tqdm(enumerate(pointsFromFrames),desc = 'face'):
        print(i)
        (xs,ys,zs),(xm,ym,zm),gazeVecL,gazeVecR,timestamp = frame
        for key in FACE:
            start = FACE[key][0]
            end  = FACE[key][1]
            if start<=timestamp*1000<=end:
                # sp = 
                points2Part[key].append(((xs,ys,zs),(xm,ym,zm),gazeVecL,gazeVecR))
            for j in range(len(xm)):
                mean_xm[j]+=xm[j]
            for j in range(len(ym)):
                mean_ym[j]+=ym[j]
            for j in range(len(zm)):
                mean_zm[j]+=zm[j]
            counter+=1
    mean_xm/=counter
    mean_ym/=counter
    mean_zm/=counter
                
    hitPointsByPart = dict()
    for key in FACE.keys():
        hitPointsByPart[key]=list()
    for key in points2Part.keys():
        for  a,arguments in enumerate(points2Part[key]):
            print(a)
            hitPointsByPart[key].append(calc_frame(*arguments,by='plane'))
    if plot:
        fig = plt.figure()
        ax = plt.axes()
        for key in tqdm(hitPointsByPart.keys(),desc = "loop"):
            for point in hitPointsByPart[key]:
                hx, hy,hz = zip(*point)
                ax.scatter(hx,hy,color=COLOR[key])
        plt.show()
    save_pickle_file((hitPointsByPart,(mean_xm,mean_ym,mean_zm)))
    return hitPointsByPart,(mean_xm,mean_ym,mean_zm)

#return a list with [i] : ((xs : list,ys,zs),(xm,ym,zm),gazevecl :ndarray ,gazevecr) 
# openFacePen = pd.read_csv(get_file("Select openface of pen"))
# print(pointsFromFrames)
process(True)