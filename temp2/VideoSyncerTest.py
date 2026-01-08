from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
from Files import *
def synchronize_videos(video_path1, video_path2, output_path, start_time=0, duration=None):
    video1 = VideoFileClip(video_path1).subclip(start_time, duration)
    video2 = VideoFileClip(video_path2).subclip(start_time, duration)
    
    audio1 = video1.audio
    audio2 = video2.audio
    
    audio1_array = np.array(audio1.to_soundarray())
    audio2_array = np.array(audio2.to_soundarray())
    
    correlation = np.correlate(audio1_array[:, 0], audio2_array[:, 0], mode='full')
    delay = len(audio1_array) - np.argmax(correlation)
    
    # if delay > 0:
    #     video2 = video2.subclip(0, video2.duration - delay / audio2.fps)
    # else:
    #     video2 = video2.subclip(-delay / audio2.fps, video2.duration)
    
    # synchronized_video = video1.set_audio(video2.audio)
    # synchronized_video.write_videofile(output_path)
    print(delay,delay/audio2.fps)
#checked 230 both and marissa
# -1177115
# Usage
import shutil
video_path1 =get_file()
video_path2 = get_file()
vid1 = shutil.copy(video_path1,'c:/')
# vid2 = shutil.copy(video_path2,'c:/')
vid2 = video_path2
output_path = 'output_synced_video.mp4'
start_time = 5  # Start synchronization from 5 seconds into the video
duration = 60   # Synchronize for a duration of 60 seconds

synchronize_videos(vid1, vid2, output_path, start_time, duration)