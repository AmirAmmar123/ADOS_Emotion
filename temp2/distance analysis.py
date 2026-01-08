from operator import ge, indexOf
import statistics
from turtle import left
from pickle import dump,load
from types import TracebackType
import numpy as np
import pandas as pd 
import scipy as stats
from scipy.stats import chi2
from sqlalchemy import false, true
from Files import *
from Excel import read_file_line_to_dict
from enum import Enum
from GazeV1 import getGazexyAtTimes
import moviepy.editor as mp
from os import getenv
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_resize
class Face:
    center = 2
    up = 0
    down = 1
    right = 3
    left = 4
def render_text_on_image_2(smooth,timestamps,conf = None,extras = None):
    # loading video dsa gfg intro video 
    IMAGEMAGICK_BINARY = os.getenv('IMAGEMAGICK_BINARY', 'C:\Program Files\ImageMagick-7.0.11-Q16-HDRI\magick.exe')    
    clip = mp.VideoFileClip(get_file("select video file")) 
    txt_clip_lists = list()
    i=0
    for i in range(len(timestamps)-1):
        if i%100==0:
            print(i,"/",len(timestamps))
        start = timestamps[i]
        end = timestamps[i+1]
        time = (start,end)
        # print(i,start,end)
        if(smooth[i]): text ="up "
        else:
            if(not smooth[i]): text = "down "
            # else:
            #     text = "unclear"
        if conf:
            text+=str("%.2f" % conf[i])
        if extras:
            text+=" "+extras[i]
        # text = "up: "+str(upm[i])+" down: "+str(dom[i])
        txt_clip = mp.TextClip(text, fontsize = 75, color = 'Red') 
        txt_clip = txt_clip.set_position((10,10)).set_start(float(time[0])).set_duration(float(time[1])-float(time[0]))
        txt_clip_lists.append(txt_clip)
    text_clips = CompositeVideoClip(txt_clip_lists)
    video = mp.CompositeVideoClip([clip, text_clips]) 
    video = video.resize(height=360)
    video.write_videofile(save_get_file("save your video file",ext='mp4')) 

def render_text_on_image_3(text_list,timestamps=None,ignore=None):
    # loading video dsa gfg intro video 
    IMAGEMAGICK_BINARY = os.getenv('IMAGEMAGICK_BINARY', 'C:\Program Files\ImageMagick-7.0.11-Q16-HDRI\magick.exe')    
    clip = mp.VideoFileClip(get_file("select video file")) 
    txt_clip_lists = list()
    i=0
    if not timestamps:
        timestamps = list()
        for i  in range(len(text_list)):
            timestamps.append((i*1/30,(i+1)*1/30))
    for i in range(len(timestamps)-1):
        if i%100==0:
            print(i,"/",len(timestamps))
        start = timestamps[i]
        end = timestamps[i+1]
        time = (start,end)
        # print(i,start,end)
        flag = True
        if(ignore):
            if  text_list[i]==ignore:
                flag = False
        if flag:
            text = text_list[i]
            # text = "up: "+str(upm[i])+" down: "+str(dom[i])
            txt_clip = mp.TextClip(text, fontsize = 75, color = 'Red') 
            txt_clip = txt_clip.set_position((10,10)).set_start(float(time[0])).set_duration(float(time[1])-float(time[0]))
            txt_clip_lists.append(txt_clip)
    text_clips = CompositeVideoClip(txt_clip_lists)
    video = mp.CompositeVideoClip([clip, text_clips]) 
    video = video.resize(height=360)
    video.write_videofile(save_get_file("save your video file",ext='mp4')) 

def mahalanobis(x=None, data=None, cov=None):

    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()
# example  :
data = {'score': [91, 93, 72, 87, 86, 73, 68, 87, 78, 99, 95, 76, 84, 96, 76, 80, 83, 84, 73, 74],
        'hours': [16, 6, 3, 1, 2, 3, 2, 5, 2, 5, 2, 3, 4, 3, 3, 3, 4, 3, 4, 4],
        'prep': [3, 4, 0, 3, 4, 0, 1, 2, 1, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2],
        'grade': [70, 88, 80, 83, 88, 84, 78, 94, 90, 93, 89, 82, 95, 94, 81, 93, 93, 90, 89, 89]
        }

# df = pd.DataFrame(data,columns=['score', 'hours', 'prep','grade'])
# # print(df.head())
# df['mahalanobis'] = mahalanobis(x=df, data=df[['score', 'hours', 'prep', 'grade']])
# # print(df.head())
# df['p'] = 1 - chi2.cdf(df['mahalanobis'], 3)
def preprocess():
    time = list()
    th=1000
    time.insert(Face.center,(3900/th,4650/th))
    time.insert(Face.up,(900/th,1900/th)) #think about replacing up and center
    time.insert(Face.down,(2900/th,3650/th))
    time.insert(Face.right,(4900/th,5650/th))
    time.insert(Face.left,(6650/th, 7461/th))
    time_dict = dict()
    time_dict["Center"] = (Face.center,time[Face.center])
    time_dict["Up"] = (Face.up,time[Face.up])
    time_dict["Down"] = (Face.down,time[Face.down])
    time_dict["Right"] = (Face.right,time[Face.right])
    time_dict["Left"] = (Face.left,time[Face.left])
    metadf = pd.DataFrame()
    metadf = pd.DataFrame.from_dict([time_dict])
    print(time)
    gazex,gazey,timestamps = getGazexyAtTimes(time)
    print(gazex)
    data_up = {'x' : gazex[Face.up],'y':gazey[Face.up]}
    data_down = {'x' : gazex[Face.down],'y':gazey[Face.down]}
    data_left = {'x':gazex[Face.left],'y' :gazey[Face.left]}
    data_right = {'x':gazex[Face.right],'y' :gazey[Face.right]}
    data_center = {'x':gazex[Face.center],'y' :gazey[Face.center]}
    dfu = pd.DataFrame(data_up,columns=['x','y'])
    dfd = pd.DataFrame(data_down,columns=['x','y'])
    dfl = pd.DataFrame(data_left,columns=['x','y'])
    dfr = pd.DataFrame(data_right,columns=['x','y'])
    dfc = pd.DataFrame(data_center,columns=['x','y'])
    dfu['mahalanobis'] = mahalanobis(x=dfu, data=dfu[['x','y']])
    dfd['mahalanobis'] = mahalanobis(x=dfd, data=dfd[['x','y']])
    dfl['mahalanobis'] = mahalanobis(x=dfl, data=dfl[['x','y']])
    dfr['mahalanobis'] = mahalanobis(x=dfr, data=dfr[['x','y']])
    dfc['mahalanobis'] = mahalanobis(x=dfc, data=dfc[['x','y']])
    dfu['p'] = 1 - chi2.cdf(dfu['mahalanobis'], 1)
    dfd['p'] = 1 - chi2.cdf(dfd['mahalanobis'], 1)
    dfl['p'] = 1 - chi2.cdf(dfl['mahalanobis'], 1)
    dfr['p'] = 1 - chi2.cdf(dfr['mahalanobis'], 1)
    dfc['p'] = 1 - chi2.cdf(dfc['mahalanobis'], 1)
    folder = get_folder()
    print(folder)
    f = open(folder+"//meta.pickle", 'wb')
    dump(time_dict, f)
    f.close()
    metadf.to_excel(folder+"//meta.xlsx")
    dfu.to_excel(folder+"//up cloud.xlsx")
    dfd.to_excel(folder+"//down cloud.xlsx")
    dfc.to_excel(folder+"//center cloud.xlsx")
    dfr.to_excel(folder+"//right cloud.xlsx")
    dfl.to_excel(folder+"//left cloud.xlsx")
    dfu.to_pickle(folder+"//upCloud")
    dfd.to_pickle(folder+"//downCloud")
    dfr.to_pickle(folder+"//rightCloud")
    dfl.to_pickle(folder+"//leftCloud")
    dfc.to_pickle(folder+"//centerCloud")
# preprocess()
# exit(1)
def test_preprocess():
    folder = get_folder("select folder of preprocess pickels")
    dfu=pd.read_pickle(folder+"//upCloud")
    dfd=pd.read_pickle(folder+"//downCloud")
    dfr=pd.read_pickle(folder+"//rightCloud")
    dfl=pd.read_pickle(folder+"//leftCloud")
    dfc=pd.read_pickle(folder+"//centerCloud")
    allx,ally,timestamps = getGazexyAtTimes([(0,12546)])
    points = {'x':allx[0],'y':ally[0]}
    pointsdf = pd.DataFrame(points,columns=['x','y'])
    upm =  mahalanobis(x=pointsdf, data=dfu[['x','y']])
    dom = mahalanobis(x=pointsdf, data=dfd[['x','y']])
    lem = mahalanobis(x=pointsdf, data=dfl[['x','y']])
    rim = mahalanobis(x=pointsdf, data=dfr[['x','y']])
    cem = mahalanobis(x=pointsdf, data=dfc[['x','y']]) 

    text_list = list()
    for i in range(len(cem)):
        str = "center: "+ "{:.2f}".format(cem[i])+"\n"
        str += "up: "+ "{:.2f}".format(upm[i])+"\n"
        str += "down: "+ "{:.2f}".format(dom[i])+"\n"
        str += "right: "+ "{:.2f}".format(rim[i])+"\n"
        str += "left: "+ "{:.2f}".format(lem[i])+"\n"    
        text_list.append(str)

    render_text_on_image_3(text_list,timestamps)
    exit()

    #####
    # create vote system 
    window = 2
    prediction = list()
    extras=list()
    for i in range(len(upm)):
        if(upm[i]<3): text =1
        else:
            if(dom[i]<3): text = -1
            else:
                text = 0
        prediction.append(text)
        extras.append("up: "+str("%.2f" % upm[i])+"\n down: "+str("%.2f" % dom[i])+"\n left: "+str("%.2f" % lem[i])+"\n right: "+str("%.2f" % rim[i])+"\n center: "+str("%.2f" % cem[i]))
    render_text_on_image_3(extras,timestamps)
    return
    # smoth prediction
    smooth =list()
    smooth.append(0)
    smooth.append(0)
    smooth_left =list()
    smooth_left.append(lem[0]/5)
    smooth_left.append(lem[0]/5)
    smooth_right =list()
    smooth_right.append(rim[0]/5)
    smooth_right.append(rim[0]/5)
    smooth_center =list()
    smooth_center.append(cem[0]/5)
    smooth_center.append(cem[0]/5)
    for i in range(2,len(prediction)-2):
        sum = 0 
        sum+=prediction[i-2]
        sum+=prediction[i-1]
        sum+=prediction[i]
        sum+=prediction[i+1]
        sum+=prediction[i+2]
        sum/=5
        if sum>=0.5:
            smooth.append(1)
        else:
            if sum<=-0.5:
                smooth.append(-1)
            else:
                smooth.append(0)
        suml = (lem[i-2]+lem[i-1]+lem[i]+lem[i+1]+lem[i+2])/5
        sumr = (rim[i-2]+rim[i-1]+rim[i]+rim[i+1]+rim[i+2])/5
        sumc = (cem[i-2]+cem[i-1]+cem[i]+cem[i+1]+cem[i+2])/5
        smooth_left.append(suml)
        smooth_right.append(sumr)
        smooth_center.append(sumc)
    smooth_left.append(lem[-1]/5)
    smooth_left.append(lem[-1]/5)
    smooth_right.append(rim[-1]/5)
    smooth_right.append(rim[-1]/5)
    smooth_center.append(cem[-1]/5)
    smooth_center.append(cem[-1]/5)      
    extras = list()
    for i in range(len(smooth_center)):
        extras.append("left: "+str("%.2f" % smooth_left[i])+"right: "+str("%.2f" % smooth_right[i])+"center: "+str("%.2f" % smooth_center[i]))
    smooth.append(0)
    smooth.append(0)
    prediction = list()
    conf = list()
    extras = list()
    other_flag = list()
    for i in range(len(upm)):
        upmi =upm[i]
        domi = dom[i]
        prediction.append(1) if (upmi<domi)  else prediction.append(0)
        conf.append(dom[i]-upm[i])  if (upmi<domi) else conf.append(upm[i]-dom[i])
        if smooth_left[i]<2:
            extras.append("left")
            other_flag.append(True)
        elif smooth_right[i]<2:
            extras.append("right")
            other_flag.append(True)
        else:
            extras.append("")
            other_flag.append(False)
    final = list()
    for i in range(len(prediction)):
        if prediction[i]==1 and not other_flag[i]:
            final.append(True)

        else:
            final.append(False)
    sym = list( )
    sym.append(0)
    sym.append(0)
    confa = list()
    confa.append(0)
    confa.append(0)
    for i in range(2,len(prediction)-2):
        sum = 0 
        csum = 0
        sum+=prediction[i-2]
        sum+=prediction[i-1]
        sum+=prediction[i]
        sum+=prediction[i+1]
        sum+=prediction[i+2]
        sum/=5
        csum+=conf[i-2]
        csum+=conf[i-1]
        csum+=conf[i]
        csum+=conf[i+1]
        csum+=conf[i+2]
        csum/=5
        confa.append(csum)
        if sum>=0.5:
            sym.append(1)
        else:
            if sum<0.5:
                sym.append(-1)

    sym.append(0)
    sym.append(0)
    confa.append(0)
    confa.append(0)
    pointsdf['mahalanobis'] = mahalanobis(x=pointsdf, data=dfu[['x','y']])
    render_text_on_image_2(final,timestamps,conf,extras)
    # render_text_on_image_2(final,timestamps) 
# preprocess()
# test_preprocess()
# exit(1)
def preprocess_emo():
    folder = get_folder("select folder of preprocess pickels")
    folder2 = get_folder("select folder of emo pickels")
    dfu=pd.read_pickle(folder+"//upCloud")
    dfd=pd.read_pickle(folder+"//downCloud")
    dfr=pd.read_pickle(folder+"//rightCloud")
    dfl=pd.read_pickle(folder+"//leftCloud")
    dfc=pd.read_pickle(folder+"//centerCloud")
    offset = 0
    end = 60*1000*9.5
    allx,ally,timestamps = getGazexyAtTimes([(0,end-offset)])
    points = {'x':allx[0],'y':ally[0]}
    pointsdf = pd.DataFrame(points,columns=['x','y'])
    upm =  mahalanobis(x=pointsdf, data=dfu[['x','y']])
    dom = mahalanobis(x=pointsdf, data=dfd[['x','y']])
    lem = mahalanobis(x=pointsdf, data=dfl[['x','y']])
    rim = mahalanobis(x=pointsdf, data=dfr[['x','y']])
    cem = mahalanobis(x=pointsdf, data=dfc[['x','y']])
    np.save(folder2+"//EupCloud", upm)
    np.save(folder2+"//EdownCloud", dom)
    np.save(folder2+"//ErightCloud", rim)
    np.save(folder2+"//EleftCloud", lem)
    np.save(folder2+"//EcenterCloud", cem)
    f = open(str(folder2+"//timestamps"), 'wb')
    dump(timestamps, f)
    f.close()

def process_clip():
    folder = get_folder("select folder of emo pickels")
    upm=np.load(folder+"//EupCloud.npy")
    dom=np.load(folder+"//EdownCloud.npy")
    rim=np.load(folder+"//ErightCloud.npy")
    lem=np.load(folder+"//EleftCloud.npy")
    cem=np.load(folder+"//EcenterCloud.npy")
    with open(folder+"//timestamps", "rb") as input_file:
        timestamps = load(input_file)
    input_file =  open(get_file("select emo speak time)"), "rb") 
    speak_time = load(input_file)

    m_input_file = None
    m_file = get_file("select marrisa speak time") # story - click cancel
    if m_file:
        m_input_file = open(m_file, "rb")
    if m_input_file:
        marrisa_speak_time = load(m_input_file)
        marrisa_speak_time_d = marrisa_speak_time['excel_dict']
        print(marrisa_speak_time['keys'])
        # print(marrisa_speak_time_d)
        # exit(1)
    speak_dict = speak_time['excel_dict']
    times_list = list()
    cut_time = 2130908
    mar_cut = 2130908
    meta = dict()
    meta["s_speak"] = input_file.name
    if not m_file:
        m_file="X"
    meta["m_speak"] = m_file
    meta["folder"] = folder
    meta['cut_time'] = str(cut_time)
    meta['mar_cut'] = str(mar_cut)
    for row in range(int(speak_time['first_row']),int(speak_time['last_row'])):
        times_list.append((int(speak_dict['start'][row])-cut_time,int(speak_dict['end'][row])-cut_time))
    mar_times_list = list()
    if m_input_file:
        for row in range(int(marrisa_speak_time['first_row']),int(marrisa_speak_time['last_row'])):
            mar_times_list.append((int(marrisa_speak_time_d['start'][row])-mar_cut,int(marrisa_speak_time_d['end'][row])-mar_cut))       
    index = int(0)
    speaking_list = list()
    for time in timestamps:
        if(index>= len(times_list)):
            break
        if(time<times_list[index][0]/1000):
            speaking_list.append(0)
            continue
        if(time>times_list[index][1]/1000):
            index+=1
            continue
        speaking_list.append(1)
    index = int(0)
    mar_speaking_list = list()
    for time in timestamps:
        if(index>= len(mar_times_list)):
            break
        if(time<mar_times_list[index][0]/1000):
            mar_speaking_list.append(0)
            continue
        if(time>mar_times_list[index][1]/1000):
            index+=1
            continue
        mar_speaking_list.append(1)


    threshold = 200
    text_list = list()
    looking_list = list()
    for i in range(len(cem)):
        text="-"
        looking = 0
        meta["threshold"] = str(threshold)
        if(cem[i]<200): #
            # if(upm[i]<200):
                text = "eyes"
                looking = 1
        text_list.append(text)
        looking_list.append(looking)
    looking_Array = np.array(looking_list)
    speaking_Array = np.array(speaking_list)
    mar_speak_Array = np.array(mar_speaking_list)
    print(looking_Array)
    # np.rot90(looking_Array)
    df = pd.DataFrame(looking_Array)
    df = pd.concat([df,pd.DataFrame(speaking_Array)],axis=1,ignore_index=True)
    df = pd.concat([df,pd.DataFrame(mar_speak_Array)],axis=1,ignore_index=True)
    df.columns=['subject looking','subject speaking','marrisa speaking']
    df.to_excel(folder+"//looking.xlsx")
    df.to_pickle(folder+"//lookingpick")
    df = pd.DataFrame(cem)
    df.to_excel(folder+"//cem.xlsx")
    print(meta)
    metadf = pd.DataFrame()
    metadf = pd.DataFrame.from_dict([meta])
    metadf.to_excel(folder+"//fin_meta.xlsx")
    metadf.to_pickle(folder+"//fin_meta")
    # of = pd.read_csv(get_file('select of csv'))
    # df = pd.concat([df,of],axis=1,ignore_index=True)
    # df.to_excel(folder+"//all.xlsx")
    # df.to_pickle(folder+"//allpick")
    # df.to_pickle(folder+"//lookingpick")
    # render_text_on_image_3(text_list,timestamps,ignore="-")
# preprocess() 
# test_preprocess()
# exit(1)
# preprocess_emo()
# df = pd.DataFrame()
# d1= read_file_line_to_dict(get_file(("select openface output excel file")))
# d2 = read_file_line_to_dict(get_file(("select openface output excel file")))
# print(d1)
# exit(1)
# test_preprocess()
# process_clip()
#display p-values for first five rows in dataframe
# # print(df.head())
# time = list()
# th=1000
# time.insert(Face.center,(1150/th,2401/th))
# time.insert(Face.left,(2651/th,4401/th)) #think about replacing up and center
# time.insert(Face.down,(10650/th,12546/th))
# time.insert(Face.right,(6900/th,8150/th))
# time.insert(Face.left,(8560/th,10151/th))
# print(time)
# gazex,gazey,timestamps = getGazexyAtTimes(time)
# data_up = {'x' : gazex[Face.up],'y':gazey[Face.up]}
# data_down = {'x' : gazex[Face.down],'y':gazey[Face.down]}
# data_left = {'x':gazex[Face.left],'y' :gazey[Face.left]}
# data_right = {'x':gazex[Face.right],'y' :gazey[Face.right]}
# data_center = {'x':gazex[Face.center],'y' :gazey[Face.center]}
# dfu = pd.DataFrame(data_up,columns=['x','y'])
# dfd = pd.DataFrame(data_down,columns=['x','y'])
# dfl = pd.DataFrame(data_left,columns=['x','y'])
# dfr = pd.DataFrame(data_right,columns=['x','y'])
# dfc = pd.DataFrame(data_center,columns=['x','y'])
# dfu['mahalanobis'] = mahalanobis(x=dfu, data=dfu[['x','y']])
# dfd['mahalanobis'] = mahalanobis(x=dfd, data=dfd[['x','y']])
# dfl['mahalanobis'] = mahalanobis(x=dfl, data=dfl[['x','y']])
# dfr['mahalanobis'] = mahalanobis(x=dfr, data=dfr[['x','y']])
# dfc['mahalanobis'] = mahalanobis(x=dfc, data=dfc[['x','y']])
# dfu['p'] = 1 - chi2.cdf(dfu['mahalanobis'], 1)
# dfd['p'] = 1 - chi2.cdf(dfd['mahalanobis'], 1)
# dfl['p'] = 1 - chi2.cdf(dfl['mahalanobis'], 1)
# dfr['p'] = 1 - chi2.cdf(dfr['mahalanobis'], 1)
# dfc['p'] = 1 - chi2.cdf(dfc['mahalanobis'], 1)
# dfu.to_excel("up cloud.xlsx")
# dfd.to_excel("down cloud.xlsx")
# # dfu analysis
# maxm = max(list(dfu['mahalanobis']))
# minm = min(list(dfu['mahalanobis']))
# imax = list(dfu['mahalanobis']).index(maxm)
# imin = list(dfu['mahalanobis']).index(minm)
# pmax = dfu['p'][imax]
# pmin = dfu['p'][imin]
# print("up : ",maxm,pmax,minm,pmin)
# maxm = max(list(dfd['mahalanobis']))
# minm = min(list(dfd['mahalanobis']))
# imax = list(dfd['mahalanobis']).index(maxm)
# imin = list(dfd['mahalanobis']).index(minm)
# pmax = dfd['p'][imax]
# pmin = dfd['p'][imin]
# print("down : ",maxm,pmax,minm,pmin)
# maxm = max(list(dfl['mahalanobis']))
# minm = min(list(dfl['mahalanobis']))
# imax = list(dfl['mahalanobis']).index(maxm)
# imin = list(dfl['mahalanobis']).index(minm)
# pmax = dfl['p'][imax]
# pmin = dfl['p'][imin]
# print("left : ",maxm,pmax,minm,pmin)
# maxm = max(list(dfr['mahalanobis']))
# minm = min(list(dfr['mahalanobis']))
# imax = list(dfr['mahalanobis']).index(maxm)
# imin = list(dfr['mahalanobis']).index(minm)
# pmax = dfr['p'][imax]
# pmin = dfr['p'][imin]
# print("right : ",maxm,pmax,minm,pmin)
# maxm = max(list(dfc['mahalanobis']))
# minm = min(list(dfc['mahalanobis']))
# imax = list(dfc['mahalanobis']).index(maxm)
# imin = list(dfc['mahalanobis']).index(minm)
# pmax = dfc['p'][imax]
# pmin = dfc['p'][imin]
# print("center : ",maxm,pmax,minm,pmin)
# print("up : ",max(dfd['mahalanobis']),min(dfd['mahalanobis']))
# point = {'x': [-0.4915],'y':[0.3757]}
# pointdf = pd.DataFrame(point,columns=['x','y'])
# upm =  mahalanobis(x=pointdf, data=dfu[['x','y']])
# dom = mahalanobis(x=pointdf, data=dfd[['x','y']])

# allx,ally,timestamps = getGazexyAtTimes([(0,11500)])
# points = {'x':allx[0],'y':ally[0]}
# pointsdf = pd.DataFrame(points,columns=['x','y'])
# upm =  mahalanobis(x=pointsdf, data=dfu[['x','y']])
# dom = mahalanobis(x=pointsdf, data=dfd[['x','y']])
# lem = mahalanobis(x=pointsdf, data=dfl[['x','y']])
# rim = mahalanobis(x=pointsdf, data=dfr[['x','y']])
# cem = mahalanobis(x=pointsdf, data=dfc[['x','y']])
# # create vote system 
# window = 2
# prediction = list()
# extras=list()
# for i in range(len(upm)):
#     if(upm[i]<6): text =1
#     else:
#         if(dom[i]<4): text = -1
#         else:
#             text = 0
#     prediction.append(text)
#     extras.append("left: "+str("%.2f" % lem[i])+"right: "+str("%.2f" % rim[i])+"center: "+str("%.2f" % cem[i]))
# # smoth prediction
# smooth =list()
# smooth.append(0)
# smooth.append(0)
# smooth_left =list()
# smooth_left.append(lem[0]/5)
# smooth_left.append(lem[0]/5)
# smooth_right =list()
# smooth_right.append(rim[0]/5)
# smooth_right.append(rim[0]/5)
# smooth_center =list()
# smooth_center.append(cem[0]/5)
# smooth_center.append(cem[0]/5)
# for i in range(2,len(prediction)-2):
#     sum = 0 
#     sum+=prediction[i-2]
#     sum+=prediction[i-1]
#     sum+=prediction[i]
#     sum+=prediction[i+1]
#     sum+=prediction[i+2]
#     sum/=5
#     if sum>=0.5:
#         smooth.append(1)
#     else:
#         if sum<=-0.5:
#             smooth.append(-1)
#         else:
#             smooth.append(0)
#     suml = (lem[i-2]+lem[i-1]+lem[i]+lem[i+1]+lem[i+2])/5
#     sumr = (rim[i-2]+rim[i-1]+rim[i]+rim[i+1]+rim[i+2])/5
#     sumc = (cem[i-2]+cem[i-1]+cem[i]+cem[i+1]+cem[i+2])/5
#     smooth_left.append(suml)
#     smooth_right.append(sumr)
#     smooth_center.append(sumc)
# smooth_left.append(lem[-1]/5)
# smooth_left.append(lem[-1]/5)
# smooth_right.append(rim[-1]/5)
# smooth_right.append(rim[-1]/5)
# smooth_center.append(cem[-1]/5)
# smooth_center.append(cem[-1]/5)      
# extras = list()
# for i in range(len(smooth_center)):
#     extras.append("left: "+str("%.2f" % smooth_left[i])+"right: "+str("%.2f" % smooth_right[i])+"center: "+str("%.2f" % smooth_center[i]))
# smooth.append(0)
# smooth.append(0)
# prediction = list()
# conf = list()
# extras = list()
# other_flag = list()
# for i in range(len(upm)):
#     upmi =upm[i]
#     domi = dom[i]
#     prediction.append(1) if (upmi<domi)  else prediction.append(0)
#     conf.append(dom[i]-upm[i])  if (upmi<domi) else conf.append(upm[i]-dom[i])
#     if smooth_left[i]<2:
#         extras.append("left")
#         other_flag.append(True)
#     elif smooth_right[i]<2:
#         extras.append("right")
#         other_flag.append(True)
#     else:
#         extras.append("")
#         other_flag.append(False)
# final = list()
# for i in range(len(prediction)):
#     if prediction[i]==1 and not other_flag[i]:
#         final.append(True)

#     else:
#         final.append(False)
# sym = list( )
# sym.append(0)
# sym.append(0)
# confa = list()
# confa.append(0)
# confa.append(0)
# for i in range(2,len(prediction)-2):
#     sum = 0 
#     csum = 0
#     sum+=prediction[i-2]
#     sum+=prediction[i-1]
#     sum+=prediction[i]
#     sum+=prediction[i+1]
#     sum+=prediction[i+2]
#     sum/=5
#     csum+=conf[i-2]
#     csum+=conf[i-1]
#     csum+=conf[i]
#     csum+=conf[i+1]
#     csum+=conf[i+2]
#     csum/=5
#     confa.append(csum)
#     if sum>=0.5:
#         sym.append(1)
#     else:
#         if sum<0.5:
#             sym.append(-1)

# sym.append(0)
# sym.append(0)
# confa.append(0)
# confa.append(0)
# pointsdf['mahalanobis'] = mahalanobis(x=pointsdf, data=dfu[['x','y']])
# # render_text_on_image_2(smooth,timestamps,conf,extras)
# render_text_on_image_2(final,timestamps) 
# until here it's for preprocess 
# offset = 0
# end = 247000
# allx,ally,timestamps = getGazexyAtTimes([(0,end-offset)])
# # points = {'x':allx[0],'y':ally[0]}
# # pointsdf = pd.DataFrame(points,columns=['x','y'])
# # upm =  mahalanobis(x=pointsdf, data=dfu[['x','y']])
# # dom = mahalanobis(x=pointsdf, data=dfd[['x','y']])
# # lem = mahalanobis(x=pointsdf, data=dfl[['x','y']])
# # rim = mahalanobis(x=pointsdf, data=dfr[['x','y']])
# # cem = mahalanobis(x=pointsdf, data=dfc[['x','y']])
# # np.save(save_get_file(title="save pickle up,"), upm)
# # np.save(save_get_file(title="save pickle d,"), dom)
# # np.save(save_get_file(title="save pickle r,"), rim)
# # np.save(save_get_file(title="save pickle l,"), lem)
# # np.save(save_get_file(title="save pickle c,"), cem)
# # exit(1)
# upm =  np.load(get_file("select pickle file up"))
# dom = np.load(get_file("select pickle file down"))
# lem = np.load(get_file("select pickle file left"))
# rim = np.load(get_file("select pickle file right"))
# cem = np.load(get_file("select pickle file center"))
# print("end malanobis")
# # for t in gazex:
# #     print(len(t))
# window = 2
# prediction = list()
# extras=list()
# for i in range(len(upm)):
#     if(upm[i]<6): text =1
#     else:
#         if(dom[i]<4): text = -1
#         else:
#             text = 0
#     prediction.append(text)
#     extras.append("left: "+str("%.2f" % lem[i])+"right: "+str("%.2f" % rim[i])+"center: "+str("%.2f" % cem[i]))
# # smoth prediction
# smooth =list()
# smooth.append(0)
# smooth.append(0)
# smooth_left =list()
# smooth_left.append(lem[0]/5)
# smooth_left.append(lem[0]/5)
# smooth_right =list()
# smooth_right.append(rim[0]/5)
# smooth_right.append(rim[0]/5)
# smooth_center =list()
# smooth_center.append(cem[0]/5)
# smooth_center.append(cem[0]/5)
# for i in range(2,len(prediction)-2):
#     sum = 0 
#     sum+=prediction[i-2]
#     sum+=prediction[i-1]
#     sum+=prediction[i]
#     sum+=prediction[i+1]
#     sum+=prediction[i+2]
#     sum/=5
#     if sum>=0.5:
#         smooth.append(1)
#     else:
#         if sum<=-0.5:
#             smooth.append(-1)
#         else:
#             smooth.append(0)
#     suml = (lem[i-2]+lem[i-1]+lem[i]+lem[i+1]+lem[i+2])/5
#     sumr = (rim[i-2]+rim[i-1]+rim[i]+rim[i+1]+rim[i+2])/5
#     sumc = (cem[i-2]+cem[i-1]+cem[i]+cem[i+1]+cem[i+2])/5
#     smooth_left.append(suml)
#     smooth_right.append(sumr)
#     smooth_center.append(sumc)
# smooth_left.append(lem[-1]/5)
# smooth_left.append(lem[-1]/5)
# smooth_right.append(rim[-1]/5)
# smooth_right.append(rim[-1]/5)
# smooth_center.append(cem[-1]/5)
# smooth_center.append(cem[-1]/5)      
# extras = list()
# for i in range(len(smooth_center)):
#     extras.append("left: "+str("%.2f" % smooth_left[i])+"right: "+str("%.2f" % smooth_right[i])+"center: "+str("%.2f" % smooth_center[i]))
# smooth.append(0)
# smooth.append(0)
# prediction = list()
# conf = list()
# extras = list()
# other_flag = list()
# for i in range(len(upm)):
#     upmi =upm[i]
#     domi = dom[i]
#     prediction.append(1) if (upmi<domi)  else prediction.append(0)
#     conf.append(dom[i]-upm[i])  if (upmi<domi) else conf.append(upm[i]-dom[i])
#     if smooth_left[i]<2:
#         extras.append("left")
#         other_flag.append(True)
#     elif smooth_right[i]<2:
#         extras.append("right")
#         other_flag.append(True)
#     else:
#         extras.append("")
#         other_flag.append(False)
# final = list()
# final_with_time = list()
# for i in range(len(prediction)):
#     if prediction[i]==1 and not other_flag[i]:
#         final.append(True)
#     else:
#         final.append(False)
# sym = list( )
# sym.append(0)
# sym.append(0)
# confa = list()
# confa.append(0)
# confa.append(0)
# for i in range(2,len(prediction)-2):
#     sum = 0 
#     csum = 0
#     sum+=prediction[i-2]
#     sum+=prediction[i-1]
#     sum+=prediction[i]
#     sum+=prediction[i+1]
#     sum+=prediction[i+2]
#     sum/=5
#     csum+=conf[i-2]
#     csum+=conf[i-1]
#     csum+=conf[i]
#     csum+=conf[i+1]
#     csum+=conf[i+2]
#     csum/=5
#     confa.append(csum)
#     if sum>=0.5:
#         sym.append(1)
#     else:
#         if sum<0.5:
#             sym.append(-1)

# sym.append(0)
# sym.append(0)
# confa.append(0)
# confa.append(0)
# # pointsdf['mahalanobis'] = mahalanobis(x=pointsdf, data=dfu[['x','y']])
# # render_text_on_image_2(smooth,timestamps,conf,extras)
# # render_text_on_image_2(final,timestamps) #last one
# #statistics 
# #load speaking pickle
# # print(timestamps) 
# render_text_on_image_2(final,timestamps) 
# talking = list()
# speaking_times = read_file_line_to_dict(get_file("select speaking time pickle/excel"))
# speaking_times_d = speaking_times['excel_dict']
# row = int(speaking_times['first_row'])
# last_row = int(speaking_times['last_row'])
# for j in range(len(timestamps)-1):

#     start = (int(speaking_times_d['start'][row])-offset)/1000
#     end = (int(speaking_times_d['end'][row])-offset)/1000
#     if  start>= timestamps[j] and  row <= last_row:
#         talking.append(True)
#         if end<=timestamps[j+1]:
#             row+=1
#     else:
#         talking.append(False)
# talking.append(false)
# print(talking)
# # for i in range(timestamps):
# #     time
# print(len(timestamps),len(talking),len(final))
# data = {'Time':timestamps,'Talking':talking,'Eyes':final}
# table = pd.DataFrame(data)
# table.to_excel(save_get_file("save data frame","xlsx"))


