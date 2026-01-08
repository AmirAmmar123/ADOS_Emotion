from pickle import load,dump
import pandas as pd
class VideosDictLoader:
    def loadVideosDict():    
        with open('dict of sorted videos 24.5.23.pickle','rb') as f:
            videos_dict = load(f)
            return videos_dict
    def saveVideoDict(newVidDict):
        with open('dict of sorted videos 24.5.23.pickle','wb') as f:
            dump(newVidDict,f)
    def check_each_key(subject_dict):
        flag = True
        for task in ['EMO','TB','STORY']:
            if len(list(subject_dict[task].keys()))!=3:
                flag=False
        return flag
    def get_gaze_subject_list():
        videosDict = VideosDictLoader.loadVideosDict()
        subject_list = []
        for subject in videosDict:
            subject_dict = videosDict[subject]
            if not VideosDictLoader.check_each_key(subject_dict) or not subject_dict['PENS']['PEN#1'] :
                continue
            subject_list+=[subject]
        return subject_list
    def videoDictEdit(keys:list,value):
        videosDict = VideosDictLoader.loadVideosDict()
        tempDict = [videosDict.copy()]
        for i,key in enumerate(keys[:-1]):
            tempDict.append(tempDict[i][key])
        tempDict[-1][keys[-1]] = value
        # print(tempDict[0][222]['PENS'])
        VideosDictLoader.saveVideoDict(tempDict[0])


VideosDictLoader.loadVideosDict = staticmethod(VideosDictLoader.loadVideosDict)
VideosDictLoader.get_gaze_subject_list = staticmethod(VideosDictLoader.get_gaze_subject_list)
VideosDictLoader.check_each_key = staticmethod(VideosDictLoader.check_each_key)
VideosDictLoader.videoDictEdit = staticmethod(VideosDictLoader.videoDictEdit)


# VideosDictLoader.videoDictEdit([222,'PENS','PEN#1','marissa'],'G:\\My Drive\\ADOS\\Videos\\Faisal\\record222\\222_pen1_marissa.mp4')
# print(VideosDictLoader.get_gaze_subject_list())