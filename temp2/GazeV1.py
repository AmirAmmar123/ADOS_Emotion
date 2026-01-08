from re import T
from turtle import color
from Files import * 
from Excel import read_file_line_to_dict
from Excel_To_Srt import millis_to_time_string
import matplotlib.pyplot as plt
import moviepy.editor as mp
from os import getenv
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_resize
from sklearn.cluster import KMeans
import numpy as np
# import cv2
L_B = -0.36
ML_B = -0.37
MR_B = -0.39
R_B = 0.41
U_B = 0.26
MU_B = 0.279
MD_B = 0.299
D_B = 0.33 
def plot_pen():
    TIMES = ['CENTER','TOP','DOWN','RIGHT','LEFT'] # subject's right&left 
    colors=['blue','green','red','yellow','black','cyan','magenta']
    gaze_xl_file = get_file("select openface output file")
    times_xl_file  = get_file("select gaze times file")
    print(gaze_xl_file)
    gaze_dict  = read_file_line_to_dict(gaze_xl_file)['excel_dict']
    # save_pickle_file(gaze_dict)
    # file = get_file()
    times_dict  = read_file_line_to_dict(times_xl_file)['excel_dict']
    # print(times_dict)
    times_list = list()
    for i in range(2,len(times_dict['start'])+2):
        times_list.append((times_dict['start'][i],times_dict['end'][i]))
    # print(times_list)
    rows_of_gaze = list(gaze_dict[' timestamp'].keys())
    gazes_list = list()
    print(times_list)
    for i in range(len(times_list)):
        gazes_list.append(list())
    print(len(gazes_list))
    i=0
    for row in rows_of_gaze:
        # print(len(times_list),i)
        cur_start = times_list[i][0]/1000
        cur_end  = times_list[i][1]/1000
        cur_time = float(gaze_dict[' timestamp'][row])
        if cur_time >=cur_start:
            if cur_time <=cur_end:
                gazes_list[i].append((float(gaze_dict[' gaze_angle_x'][row]),float(gaze_dict[' gaze_angle_y'][row])))
        if cur_time > cur_end:
            i+=1
            if i>=len(times_list):
                break
    print(len(gazes_list))
    avrages = list()
    for list in gazes_list:
        sumx = 0 
        sumy = 0
        for gaze in list : 
            sumx+=gaze[0]
            sumy+=gaze[1]
        avrages.append((sumx/len(list),sumy/len(list)))
    for c,list in enumerate(gazes_list):
        plt.scatter(*zip(*list),color=colors[c])
    for c,avrage in enumerate(avrages):
        plt.scatter(avrage[0],avrage[1],color=colors[c],s=512)
    plt.show()
def get_parsed_gaze():
    gaze_xl_file = get_file("select openface output file")
    gaze_dict  = read_file_line_to_dict(gaze_xl_file)['excel_dict']
    gazex_key =' gaze_angle_x'
    gazey_key =' gaze_angle_y'
    time_key =' timestamp'
    rows_of_gaze = list(gaze_dict[' timestamp'].keys())
    before_strings = list()
    for i,row in enumerate(rows_of_gaze):
        start = gaze_dict[time_key][row]
        end = gaze_dict[time_key][row]
        if row+1 in rows_of_gaze:
            end = gaze_dict[time_key][row+1]
        gazex= float(gaze_dict[gazex_key][row])
        gazey = float(gaze_dict[gazey_key][row])
        if gazex>=(-0.416) and gazex<=(-0.36) and gazey>=0.25 and gazey<=0.33:
            # strings.append(str(counter))
            # counter+=1
            # print(int(float(start)*1000),int(float(end)*1000),(millis_to_time_string(int(float(end)*1000))))
            # strings.append(millis_to_time_string(int(float(start)*1000))+' --> '+millis_to_time_string(int(float(end)*1000)))
            # strings.append("Looking at face")
            before_strings.append((start,end))
    parsed_before = list()
    start_b = before_strings[0][0]
    i=1
    while  i<len(before_strings):
        while before_strings[i][0]==before_strings[i-1][1]:
            i+=1
        parsed_before.append((start_b,before_strings[i-1][1]))
        start_b = before_strings[i][0]
        i+=1
    return parsed_before
def get_means(k=5):
    gaze_xl_file = get_file("select openface output file")
    gaze_dict  = read_file_line_to_dict(gaze_xl_file)['excel_dict']
    gazex_key =' gaze_angle_x'
    gazey_key =' gaze_angle_y'
    time_key =' timestamp'
    rows_of_gaze = list(gaze_dict[' timestamp'].keys())
    data_points = list()
    for i,row in enumerate(rows_of_gaze):
        start = gaze_dict[time_key][row]
        end = gaze_dict[time_key][row]
        if row+1 in rows_of_gaze:
            end = gaze_dict[time_key][row+1]
        gazex= float(gaze_dict[gazex_key][row])
        gazey = float(gaze_dict[gazey_key][row])
        data_points.append((gazex,gazey))
    data_arr  = np.array(data_points)
    k_means = KMeans(n_clusters=k,random_state=0).fit(data_arr)
    colors=['blue','green','red','yellow','black','cyan','magenta']
    for i in range(len(data_arr)):
        pred=k_means.labels_[i]
        plt.scatter(data_arr[i][0],data_arr[i][1],color=colors[pred])
    plt.show()



def get_parsed_blink():
    blink_xl_file = get_file("select openface output file")
    blink_dict  = read_file_line_to_dict(blink_xl_file)['excel_dict']
    blinkx_key =' AU45_r'
    time_key =' timestamp'
    rows_of_blink = list(blink_dict[' timestamp'].keys())
    before_strings = list()
    for i,row in enumerate(rows_of_blink):
        start = blink_dict[time_key][row]
        end = blink_dict[time_key][row]
        if row+1 in rows_of_blink:
            end = blink_dict[time_key][row+1]
        blinkx= float(blink_dict[blinkx_key][row])
        if blinkx >= 1:
            before_strings.append((start,end))
    parsed_before = list()
    start_b = before_strings[0][0]
    i=1
    while  i<len(before_strings):
        while before_strings[i][0]==before_strings[i-1][1]:
            i+=1
            if i>=len(before_strings):
                break
        parsed_before.append((start_b,before_strings[i-1][1]))
        if i<len(before_strings):
            start_b = before_strings[i][0]
        i+=1
    return parsed_before
def get_parsed_boxes():
    cam_xl_file = get_file("select openface output file")
    cam_dict  = read_file_line_to_dict(cam_xl_file)['excel_dict']
    gazex_key =' gaze_angle_x'
    gazey_key =' gaze_angle_y'
    time_key =' timestamp'
    rows_of_cam = list(cam_dict[' timestamp'].keys())
    data = list()
    eps = 10 # percent

    for i,row in enumerate(rows_of_cam):
        start = cam_dict[time_key][row]
        end = cam_dict[time_key][row]
        if row+1 in rows_of_cam:
            end = cam_dict[time_key][row+1]
        camx= float(cam_dict[gazex_key][row])
        camy= float(cam_dict[gazey_key][row])
        faceBoxY = None
        faceBoxX = None
        if camx <= (L_B+(L_B-R_B)/eps) and camx >= ML_B : #L
            faceBoxX = 'L'
        if camx < ML_B and camx >= (MR_B): #M
            faceBoxX = 'M'
        if camx <MR_B and camx >= (R_B+(R_B-L_B)/eps) : #R
            faceBoxX = 'R'
        if camy >= (U_B+(U_B-D_B)/eps) and camy <=MU_B:
            faceBoxY = 'U'
        if camy > MU_B and camy <= MD_B :
            faceBoxY= 'M'
        if camy > MD_B and camy<=(D_B+(D_B-U_B)/eps):
            faceBoxY = 'D'
        time = (float(start),float(end))
        box = (faceBoxX,faceBoxY)
        data.append((time,box))
    return data
def get_parsed_cam():
    cam_xl_file = get_file("select openface output file")
    cam_dict  = read_file_line_to_dict(cam_xl_file)['excel_dict']
    gazex_key =' gaze_angle_x'
    gazey_key =' gaze_angle_y'
    time_key =' timestamp'
    rows_of_cam = list(cam_dict[' timestamp'].keys())
    before_strings = list()
    for i,row in enumerate(rows_of_cam):
        start = cam_dict[time_key][row]
        end = cam_dict[time_key][row]
        if row+1 in rows_of_cam:
            end = cam_dict[time_key][row+1]
        camx= float(cam_dict[gazex_key][row])
        camy= float(cam_dict[gazey_key][row])
        if camx>=0.03 and camx<=0.15 and camy>=0.1 and camy<=0.2:
            before_strings.append((start,end))
    parsed_before = list()
    start_b = before_strings[0][0]
    i=1
    while  i<len(before_strings):
        while before_strings[i][0]==before_strings[i-1][1]:
            i+=1
            if i>=len(before_strings):
                break
        parsed_before.append((start_b,before_strings[i-1][1]))
        if i<len(before_strings):
            start_b = before_strings[i][0]
        i+=1
    return parsed_before


def looking_time_srt():
    gaze_xl_file = get_file("select openface output file")
    gaze_dict  = read_file_line_to_dict(gaze_xl_file)['excel_dict']
    counter =1
    gazex_key =' gaze_angle_x'
    gazey_key =' gaze_angle_y'
    time_key =' timestamp'
    rows_of_gaze = list(gaze_dict[' timestamp'].keys())
    f=save_get_file(title ="Please select where to save the srt file",ext='srt')
    strings = list()
    before_strings = list()
    row_to_get = len(rows_of_gaze)
    for i,row in enumerate(rows_of_gaze):
        start = gaze_dict[time_key][row]
        end = gaze_dict[time_key][row]
        if row+1 in rows_of_gaze:
            end = gaze_dict[time_key][row+1]
        gazex= float(gaze_dict[gazex_key][row])
        gazey = float(gaze_dict[gazey_key][row])
        if gazex>=(-0.416) and gazex<=(-0.36) and gazey>=0.25 and gazey<=0.33:
            # strings.append(str(counter))
            # counter+=1
            # print(int(float(start)*1000),int(float(end)*1000),(millis_to_time_string(int(float(end)*1000))))
            # strings.append(millis_to_time_string(int(float(start)*1000))+' --> '+millis_to_time_string(int(float(end)*1000)))
            # strings.append("Looking at face")
            before_strings.append((start,end))
    parsed_before = list()
    start_b = before_strings[0][0]
    i=1
    while  i<len(before_strings):
        while before_strings[i][0]==before_strings[i-1][1]:
            i+=1
        parsed_before.append((start_b,before_strings[i-1][1]))
        start_b = before_strings[i][0]
        i+=1
    for parsed in parsed_before:
        strings.append(str(counter))
        counter+=1
        # print(int(float(parsed[0])*1000),int(float(end)*1000),(millis_to_time_string(int(float(end)*1000))))
        strings.append(millis_to_time_string(int(float(parsed[0])*1000))+" Looking at face "+str(counter-1)+' ––> '+millis_to_time_string(int(float(parsed[1])*1000)))
        # strings.append() 
        strings.append(" ")

    write_list_to_file(strings,f)
        
def unify_parsed(parsed_before,factor=0.3):
    new_parsed = list()
    last_parsed=0
    next_parsed=0
    i = 0
    while i<len(parsed_before)-1:

        first_parse = float(parsed_before[i][0])
        if i<(len(parsed_before)-1):
            while float(parsed_before[i+1][0])-float(parsed_before[i][1])<factor and i+1<len(parsed_before): #close so unify
                i+=1
                if i>=len(parsed_before)-1:
                    break
        if i<len(parsed_before):
            new_parsed.append((first_parse,float(parsed_before[i][1])))
        i+=1
    return new_parsed

def render_text_on_image():
    # loading video dsa gfg intro video 
    IMAGEMAGICK_BINARY = getenv ('IMAGEMAGICK_BINARY', 'C:\Program Files\ImageMagick-7.0.8-Q16\convert.exe')
    clip = mp.VideoFileClip(get_file("select video file")) 
    parsed_before = get_parsed_gaze()
    print(parsed_before)
    parsed_before = unify_parsed(parsed_before,factor=0)
    # text  = "Face"
    text = "cam"
    txt_clip_lists = list()
    i=0
    for parsed in parsed_before:
        txt_clip = mp.TextClip(text, fontsize = 75, color = 'Red') 
        txt_clip = txt_clip.set_position((10,10)).set_start(float(parsed[0])).set_duration(float(parsed[1])-float(parsed[0]))
        txt_clip_lists.append(txt_clip)

    # parsed_before = get_parsed_gaze()
    # parsed_before = unify_parsed(parsed_before,factor=0.3)
    # text  = "Face"
    # # text = "Blink"
    # txt_clip_lists = list()
    # i=0
    # for parsed in parsed_before:
    #     txt_clip = mp.TextClip(text, fontsize = 75, color = 'Green',size=(200,None),method='caption',align='center') 
    #     txt_clip = txt_clip.set_position((250,250)).set_start(float(parsed[0])).set_duration(float(parsed[1])-float(parsed[0])+0.1)
    #     txt_clip_lists.append(txt_clip)
    text_clips = CompositeVideoClip(txt_clip_lists)
    video = mp.CompositeVideoClip([clip, text_clips]) 
    video = video.resize(height=360)
    video = video.subclip(0,25)    
# showing video 
    video.write_videofile(save_get_file("save your video file",ext='mp4')) 
def render_text_on_image_2():
    # loading video dsa gfg intro video 
    IMAGEMAGICK_BINARY = os.getenv('IMAGEMAGICK_BINARY', 'C:\Program Files\ImageMagick-7.0.11-Q16-HDRI\magick.exe')    
    clip = mp.VideoFileClip(get_file("select video file")) 
    parsed_before = get_parsed_boxes()
    txt_clip_lists = list()
    i=0
    for parsed in parsed_before:
        time,box = parsed
        b0 = box[0]
        b1 = box[1]
        if not box[0]:
            b0 = ""
        if not box[1]:
            b1 = ""
        text = b0+b1
        txt_clip = mp.TextClip(text, fontsize = 75, color = 'Red') 
        txt_clip = txt_clip.set_position((10,10)).set_start(float(time[0])).set_duration(float(time[1])-float(time[0]))
        txt_clip_lists.append(txt_clip)

    # parsed_before = get_parsed_gaze()
    # parsed_before = unify_parsed(parsed_before,factor=0.3)
    # text  = "Face"
    # # text = "Blink"
    # txt_clip_lists = list()
    # i=0
    # for parsed in parsed_before:
    #     txt_clip = mp.TextClip(text, fontsize = 75, color = 'Green',size=(200,None),method='caption',align='center') 
    #     txt_clip = txt_clip.set_position((250,250)).set_start(float(parsed[0])).set_duration(float(parsed[1])-float(parsed[0])+0.1)
    #     txt_clip_lists.append(txt_clip)
    text_clips = CompositeVideoClip(txt_clip_lists)
    video = mp.CompositeVideoClip([clip, text_clips]) 
    video = video.resize(height=360)
# showing video 
    video.write_videofile(save_get_file("save your video file",ext='mp4')) 
# looking_time_srt()
# render_text_on_image()
# get_means(2)
# print(get_parsed())
# print(get_parsed_boxes())
# render_text_on_image_2()
def getGazexyAtTimes(times):
    time_x_list = list()
    time_y_list = list()
    for i in range(len(times)):
        time_x_list.append(list())
        time_y_list.append(list())
    gaze_dict  =  read_file_line_to_dict(get_file(("select openface output excel file")))
    gaze_dict = gaze_dict['excel_dict']
    # print(gazeDict['excel_dict'])
    gazex_key =' gaze_angle_x'
    gazey_key =' gaze_angle_y'
    time_key =' timestamp'
    rows_of_gaze = list(gaze_dict[time_key].keys())
    # print(rows_of_gaze)
    cur_time = 0
    change_flag = False
    timestamps = list()
    for i,row in enumerate(rows_of_gaze):
        start = float(gaze_dict[time_key][row])
        timestamps.append(start)
        # print(start,cur_time)
        if start>=times[cur_time][0] :
            if start<=times[cur_time][1]:
                # in time range
                change_flag = True
                gazex = float(gaze_dict[gazex_key][row])
                gazey = float(gaze_dict[gazey_key][row])
                time_x_list[cur_time].append(gazex)
                time_y_list[cur_time].append(gazey)
            else:
                # out of time range
                if change_flag:
                    change_flag = False
                    cur_time += 1
                    cur_time%= len(times)

    return time_x_list,time_y_list,timestamps
    

            


