from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_resize
from Files import get_file,save_get_file
import moviepy.editor as mp
import os
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

