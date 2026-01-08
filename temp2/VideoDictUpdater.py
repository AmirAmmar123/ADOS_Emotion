from VideoDictLoader import VideosDictLoader
import pandas as pd
translate_task = {'TB':'TB','EMO':'emotions','STORY':'Story','CALIBRATION':'calibration','pen1':'PEN#1','pen2':'PEN#2'}
OUTPUT_DIR = 'c:/newClips/'
videosDict = VideosDictLoader.loadVideosDict()
print(videosDict)
for subject in range(236,255):
    videosDict[subject] = {}
    subject_str = str(subject)
    for task in ['TB','EMO','STORY','CALIBRATION']:
        subject_vid = OUTPUT_DIR+subject_str+'_'+translate_task[task]+'_'+'subject.mp4'
        both_vid = OUTPUT_DIR+subject_str+'_'+translate_task[task]+'_'+'both.mp4'
        marissa_vid = OUTPUT_DIR+subject_str+'_'+translate_task[task]+'_'+'marissa.mp4'
        videosDict[subject][task]={'subject':subject_vid,'both':both_vid,'marissa':marissa_vid}
    videosDict[subject]['PENS']={}
    for pen in ['pen1','pen2']:
        subject_vid = OUTPUT_DIR+subject_str+'_'+pen+'_'+'subject.mp4'
        both_vid = OUTPUT_DIR+subject_str+'_'+pen+'_'+'both.mp4'
        marissa_vid = OUTPUT_DIR+subject_str+'_'+pen+'_'+'marissa.mp4'
        videosDict[subject]['PENS'][translate_task[pen]]={'subject':subject_vid,'both':both_vid,'marissa':marissa_vid}
print(videosDict)
VideosDictLoader.saveVideoDict(videosDict)
# pd.DataFrame(videosDict).to_csv('c:/test.csv')