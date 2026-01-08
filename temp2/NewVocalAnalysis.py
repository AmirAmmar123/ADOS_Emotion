from itertools import count
from pickle import NONE
from VocalFeatures import *
from Files import * 
import numpy as np
import pandas as pd
from scipy.io.wavfile import read,write
from LittleGui import *
from os import mkdir 
from os.path import exists
import moviepy.editor as mp
from os import getenv
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_resize
import matplotlib.patches as mpatches
from video2audioSplit import convert_video_to_audio_moviepy
taskNAME = None
filesNAME = None
FILES = {}
MARISSA_FILES = {}
col_names =['record','task','wav_file','speak_times']
ALL_FILES = pd.DataFrame(columns=col_names, dtype=object)
ALL_FILES.set_index(['record','task'],inplace=True)
MARISSA_ALL_FILES = pd.DataFrame(columns=col_names, dtype=object)
MARISSA_ALL_FILES.set_index(['record','task'],inplace=True)
labels = {'F0':'(Hz)', 'Intensity':'(dB)','HNR':'(dB)'}
# ASD = [4,8,12,13,20,144,201,202,210,211,212,36,216,26]
# NT = [9,10,11,16,17,21,22,29,31,18,19,205,206,207,208,217,219,221,30,145]
# ASD_PO = [1,2,3,5,6,27,148,14]
# NT_PO = [28,145,147]
# MALE = [1,3,4,5,12,13,14,16,17,19,20,21,22,23,24,25,27,151,152,201,202,206,207,210,211,212,208,217,219,221,36,26,216,30,145]
# FEMALE = [2,7,8,9,10,11,18,28,29,31,144,205]
# NOT_HEB = [219,221,210]
IGNORE = []
# IGNORE = [36,216]
TO_DROP= ['voiceID','record']
TO_DROP_2 = ['TB_voiceID','Emo_voiceID','Story_voiceID','TB_task','Emo_task','Story_task','Story_record','TB_record','Emo_record']
TASKS = ['TB','Story','Emo']
TASK_TO_NUM = {'TB':1,'Story':2,'Emo':3}
AS_TYPE = {'record':'int32','Type':'string','Gender':'string'}
RECORD2TYPE = pd.read_excel('record2type.xlsx',engine='openpyxl').astype(AS_TYPE).set_index('record')

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

def get_times_list(excel_file:str):
    excel_df = pd.read_excel(excel_file,engine='openpyxl')
    print(excel_df)
    times = []
    if 'start'in excel_df.columns:
        for index,row in excel_df.iterrows():
            times.append((row['start'],row['end']))
    if 'Start'in excel_df.columns:
        for index,row in excel_df.iterrows():
            times.append((row['Start'],row['End']))
    print(times)
    return times
def create_pickels_from_dataframe(df):
    ndf = df.reset_index()
    for subject in list(ndf['record']):
        if os.path.exists('VocalOutput2/'+str(subject)):
            print('no',subject)
            continue
        okay_flag = True
        for t in ['TB','Story','Emo']:
            if ((subject),t) not in df.index:
                okay_flag = False
        if not okay_flag:
            continue
        tb_df = subject_task_pitch_analysis("TB",save_file=True,audio_f= df.at[((subject),'TB'),'wav_file'],speak_f = df.at[((subject),'TB'),'speak_times'],wav_file_save=str(subject)+'TB.wav')
        story_df = subject_task_pitch_analysis("Story",save_file=True,audio_f= df.at[((subject),'Story'),'wav_file'],speak_f = df.at[((subject),'Story'),'speak_times'],wav_file_save=str(subject)+'Story.wav')
        emo_df = subject_task_pitch_analysis("Emo",save_file=True,repeat=0,audio_f= df.at[((subject),'Emo'),'wav_file'],speak_f = df.at[((subject),'Emo'),'speak_times'],wav_file_save=str(subject)+'Emo.wav')
        subject = str(subject)
        tb_df['record'] = int(subject)
        story_df['record'] = int(subject)
        emo_df['record'] = int(subject)
        all_df = pd.concat([tb_df,story_df,emo_df])
        cols = list(all_df)
        irr = {'task':'string','record':'int32','voiceID':'string'}
        floats = [item for item in cols if item not in list(irr.keys()) ]
        for floa in floats:
            irr[floa] = 'float64'
        all_df = all_df.astype(irr)
        if not os.path.exists('VocalOutput2'):
            os.mkdir('VocalOutput2')
        all_df.to_pickle('VocalOutput2/'+subject)
def create_pickls_from_dataframe_mar(df):
    print(str(df.index))
    for index,row in df.iterrows():
        subject_n = index[0]
        task = index[1]
        wav_file_path = row['wav_file']
        speak_time_path = row['speak_times']
        audio_file = wav_file_path
        if audio_file.split('.')[-1]=='mp4':
            audio_file = convert_video_to_audio_moviepy(audio_file,output_folder=None)
        print("selected "+audio_file)
        speaking_times = speak_time_path
        print("selected "+speaking_times)
        # input()
        times_df = pd.read_csv(speaking_times)
        times = []
        # print(times_df)
        for index,row in times_df.iterrows():
            if(-row['Start']+row['End'] < 500):
                continue
            times.append((row['Start'],row['End']))
            break
        # times = get_times_list(speaking_times)
        print(times)
        # input()
        # times = [(500,1000),(1000,1500),(1500,2000)]
        rate,data= read(audio_file)
        data = np.rot90(data)
        t = times[0]
        start = int((t[0]/1000)*rate)
        end = int(rate*(t[1]/1000))
        new_data = data[0,start:end]
        for t in times[1:]:
            if((t[1]-t[0]) < 40 ):
                continue
            start = int((t[0]/1000)*rate)
            end = int(rate*(t[1]/1000))
            new_data=np.concatenate([new_data,data[0,start:end]],axis=0)
            # print(analyze_dataFrame(data[0,start:end],rate))
        result_df = analyze_dataFrame(new_data,rate)
        result_df['task']=task
        tb_df = result_df
        tb_df['record'] = int(subject_n)
        all_df = pd.concat([tb_df])
        cols = list(all_df)
        irr = {'task':'string','record':'int32','voiceID':'string'}
        floats = [item for item in cols if item not in list(irr.keys()) ]
        for floa in floats:
            irr[floa] = 'float64'
        all_df = all_df.astype(irr)
        if not os.path.exists('VocalOutputMarissa2nd'):
            os.mkdir('VocalOutputMarissa2nd')
        all_df.to_pickle('VocalOutputMarissa2nd/'+str(subject_n))
        if not os.path.exists('VocalOutputMarissa2ndWAV'):
            os.mkdir('VocalOutputMarissa2ndWAV')
        write('VocalOutputMarissa2ndWAV/'+str(subject_n)+".wav",rate,new_data)



def render_pitch_on_video():
    audio_file = get_file("Select wav file ")
    print("selected "+audio_file)
    rate,data= read(audio_file)
    data = np.rot90(data)
    pitch_array = get_pitch(data,rate)
    df = analyze_dataFrame(data,rate)
    print(df['meanF0Hz'])
    print(np.mean(pitch_array))
    print(np.mean(pitch_array[pitch_array!=0]))
    seconds = data.shape[1] / rate
    print(pitch_array.shape,data.shape[1],data.shape[1]/pitch_array.shape[0],rate,seconds,seconds/pitch_array.shape[0])
    time = 0 
    pitch_txt = []
    timestamps = [0]
    for i in range(pitch_array.shape[0]):
        pitch_txt.append(str(pitch_array[i]))
        end = time + seconds/pitch_array.shape[0]
        timestamps.append(end)
        time = end
    print(pitch_txt)
    print(timestamps[-1])
    render_text_on_image_3(pitch_txt,timestamps)
def marissa_task_pitch_analysis(task:str,save_file = True,repeat = 0):
    audio_file = get_file("Select wav file "+task)
    if audio_file.split('.')[-1]=='mp4':
        audio_file = convert_video_to_audio_moviepy(audio_file,output_folder=None)
    print("selected "+audio_file)
    speaking_times = get_file("Select speak time (csv) "+task)
    print("selected "+speaking_times)
    MARISSA_FILES[task] = [audio_file,speaking_times]
    times_df = pd.read_csv(speaking_times)
    times = []
    for index,row in times_df.iterrows():
        times.append((row['Start'],row['End']))
    # times = get_times_list(speaking_times)
    print(times)
    # input()
    # times = [(500,1000),(1000,1500),(1500,2000)]
    rate,data= read(audio_file)
    data = np.rot90(data)
    t = times[0]
    start = int((t[0]/1000)*rate)
    end = int(rate*(t[1]/1000))
    new_data = data[0,start:end]
    for t in times[1:]:
        if((t[1]-t[0]) < 40 ):
            continue
        start = int((t[0]/1000)*rate)
        end = int(rate*(t[1]/1000))
        new_data=np.concatenate([new_data,data[0,start:end]],axis=0)
        # print(analyze_dataFrame(data[0,start:end],rate))
    result_df = analyze_dataFrame(new_data,rate)
    result_df['task']=task
    if(save_file):
        write("test"+task+".wav",rate,new_data)
    return result_df

def subject_task_pitch_analysis(task:str,save_file = False,repeat = 0,audio_f = None,speak_f = None,wav_file_save = None):
    if not audio_f:
        audio_file = get_file("Select wav file "+task)
    else:
        audio_file = audio_f
    print("selected "+audio_file)
    if not speak_f:
        speaking_times = get_file("Select speak time (xlsx) "+task)
    else:
        speaking_times = speak_f
    print("selected "+speaking_times)
    FILES[task] = [audio_file,speaking_times]
    times = get_times_list(speaking_times)
    # times = [(500,1000),(1000,1500),(1500,2000)]
    rate,data= read(audio_file)
    data = np.rot90(data)
    t = times[0]
    start = int((t[0]/1000)*rate)
    end = int(rate*(t[1]/1000))
    new_data = data[0,start:end]
    for t in times[1:]:
        if((t[1]-t[0]) < 40 ):
            continue
        start = int((t[0]/1000)*rate)
        end = int(rate*(t[1]/1000))
        new_data=np.concatenate([new_data,data[0,start:end]],axis=0)
        # print(analyze_dataFrame(data[0,start:end],rate))
    for i in range(repeat):
        audio_file = get_file("Select wav file "+task)
        print("selected "+audio_file)
        speaking_times = get_file("Select speak time (xlsx) "+task)
        print("selected "+speaking_times)
        times = get_times_list(speaking_times)
        # times = [(500,1000),(1000,1500),(1500,2000)]
        rate,data= read(audio_file)
        data = np.rot90(data)
        for t in times:
            if((t[1]-t[0]) < 40 ):
                continue
            start = int((t[0]/1000)*rate)
            end = int(rate*(t[1]/1000))
            new_data=np.concatenate([new_data,data[0,start:end]],axis=0)
    output_f = 'wav_folder/'
    if not exists(output_f):
        os.mkdir(output_f)
    if(save_file):
        if wav_file_save:
            write(output_f+wav_file_save,rate,new_data)
        else:
            write(output_f+"test"+task+".wav",rate,new_data)

    result_df = analyze_dataFrame(new_data,rate)
    result_df['task']=task
    return result_df
def marissa_pitch_analysis():
    tb_df = marissa_task_pitch_analysis("TB")
    subject = get_string()
    for task in list(MARISSA_FILES.keys()):
        MARISSA_ALL_FILES.at[(subject,task),'wav_file'] = MARISSA_FILES[task][0]
        MARISSA_ALL_FILES.at[(subject,task),'speak_times'] = MARISSA_FILES[task][1]
    # MARISSA_ALL_FILES.to_pickle('marissa_files.pickle')
    # MARISSA_ALL_FILES.to_csv('marrissa_files.csv')

    print(MARISSA_ALL_FILES)
    tb_df['record'] = int(subject)
    all_df = pd.concat([tb_df])
    cols = list(all_df)
    irr = {'task':'string','record':'int32','voiceID':'string'}
    floats = [item for item in cols if item not in list(irr.keys()) ]
    for floa in floats:
        irr[floa] = 'float64'
    all_df = all_df.astype(irr)
    if not os.path.exists('VocalOutputMarissa'):
        os.mkdir('VocalOutputMarissa')
    print(all_df)
    all_df.to_pickle('VocalOutputMarissa/'+subject)
def subject_pitch_analysis():

    tb_df = subject_task_pitch_analysis("TB",save_file=False)
    story_df = subject_task_pitch_analysis("Story",save_file=False)
    emo_df = subject_task_pitch_analysis("Emo",save_file=False,repeat=0)
    print(FILES)
    subject = get_string()
    for task in list(FILES.keys()):
        ALL_FILES.at[(subject,task),'wav_file'] = FILES[task][0]
        ALL_FILES.at[(subject,task),'speak_times'] = FILES[task][1]
    ALL_FILES.to_pickle('files.pickle')
    ALL_FILES.to_csv('files.csv')

    print(ALL_FILES)
    tb_df['record'] = int(subject)
    story_df['record'] = int(subject)
    emo_df['record'] = int(subject)
    all_df = pd.concat([tb_df,story_df,emo_df])
    cols = list(all_df)
    irr = {'task':'string','record':'int32','voiceID':'string'}
    floats = [item for item in cols if item not in list(irr.keys()) ]
    for floa in floats:
        irr[floa] = 'float64'
    all_df = all_df.astype(irr)
    if not os.path.exists('VocalOutput'):
        os.mkdir('VocalOutput')
    all_df.to_pickle('VocalOutput/'+subject)
    # all_df.reindex(['TB','Story','Emo'])
    # print(all_df)
    # return all_df
def analyze_marissa_files():
    files = get_files("select df pickles")
    all = pd.DataFrame()
    all.index.name = 'record'
    for file in files:
        subject_n = int(file.split('/')[-1].split('.')[0])
        if(subject_n in IGNORE):
            continue
        subject_df = pd.read_pickle(file)
        subject_df.drop(TO_DROP,axis=1,inplace=True)
        for index,row in subject_df.iterrows():
            for col in subject_df.columns:
                if(col != 'task'):
                    all.at[subject_n,row['task']+"_"+col] = float(row[col])
        all.at[subject_n,'Type'] = RECORD2TYPE.at[subject_n,'Type']
        all.at[subject_n,'Gender'] = RECORD2TYPE.at[subject_n,'Gender']
        all.at[subject_n,'Location'] = RECORD2TYPE.at[subject_n,'Location']
    all.to_csv('marissa_analyzed2nd.csv')
def analyze_files():
    files = get_files("select df pickles")
    all = pd.DataFrame()
    diffs = pd.DataFrame()
    all.index.name = 'record'
    diffs.index.name = 'record_task1_task2'
    diffs_cols_orig = []
    for file in files:
        subject_n = int(file.split('/')[-1].split('.')[0])
        if(subject_n in IGNORE):
            continue
        subject_df = pd.read_pickle(file)
        subject_df.drop(TO_DROP,axis=1,inplace=True)
        for index,row in subject_df.iterrows():
            for col in subject_df.columns:
                if(col != 'task'):
                    all.at[subject_n,row['task']+"_"+col] = float(row[col])
        all.at[subject_n,'Type'] = RECORD2TYPE.at[subject_n,'Type']
        all.at[subject_n,'Gender'] = RECORD2TYPE.at[subject_n,'Gender']
        all2 = all
        # without_task  =subject_df.drop(['task'],axis=1)
        n_subject_df =  subject_df.set_index('task')
        # print(n_subject_df)
        for col in n_subject_df.columns:
            if n_subject_df.dtypes[col]!='float64':
                continue
            for i,task_i in enumerate(TASKS):
                for j,task_j in enumerate(TASKS):
                    if task_i == task_j:
                        continue
                    diffs.at[str(subject_n)+"_"+task_i+"_"+task_j,col] = n_subject_df.at[task_i,col]-n_subject_df.at[task_j,col]
                    diffs.at[str(subject_n)+"_"+task_i+"_"+task_j,'record'] = subject_n
                    if i<=j:
                        all2.at[subject_n,task_i+"-"+task_j+"_"+col] = n_subject_df.at[task_i,col]-n_subject_df.at[task_j,col]
                        diffs_cols_orig.append(task_i+"-"+task_j+"_"+col)
                    diffs.at[str(subject_n)+"_"+task_i+"_"+task_j,'Type'] = RECORD2TYPE.at[subject_n,'Type']
                    diffs.at[str(subject_n)+"_"+task_i+"_"+task_j,'Gender'] = RECORD2TYPE.at[subject_n,'Gender']
                    diffs.at[str(subject_n)+"_"+task_i+"_"+task_j,'task1'] = task_i
                    diffs.at[str(subject_n)+"_"+task_i+"_"+task_j,'task2'] = task_j
        
                    # diffs = diffs.append()
    diffs.set_index(['record','task1','task2'],inplace=True)
    diffs.drop(diffs.columns[0],axis=1,inplace=True)

        # print(subject_n,type(subject_n))
    print(all)
    # all.drop(diffs_cols_orig,axis=1).to_excel("New vocal.xlsx")
    # all.drop(diffs_cols_orig,axis=1).to_pickle("NewVocal")
    # diffs.to_excel('New vocal diffs.xlsx')
    # diffs.to_pickle('newVocalDiffs')
    # all2.to_excel("NewVocalWithDiff.xlsx")
    all.drop(diffs_cols_orig,axis=1).to_excel("New vocal2.xlsx")
    all.drop(diffs_cols_orig,axis=1).to_pickle("NewVocal2")
    diffs.to_excel('New vocal diffs2.xlsx')
    diffs.to_pickle('newVocalDiffs2')
    all2.to_excel("NewVocalWithDiff2.xlsx")
def hist_pre():
    files = get_files("select df pickles")
    all = pd.DataFrame()
    all.index.name = 'record'
    counter = 0 
    for file in files:
        subject_n = int(file.split('/')[-1].split('.')[0])
        subject_df = pd.read_pickle(file)
        subject_df.drop(TO_DROP,axis=1,inplace=True)
        for index,row in subject_df.iterrows():
            for col in subject_df.columns:
                if(col != 'task'):
                    all.at[counter,col] = float(row[col])
                    all.at[counter,'Type'] = RECORD2TYPE.at[subject_n,'Type']
                    all.at[counter,'Gender'] = RECORD2TYPE.at[subject_n,'Gender']
                # all.at[counter,'Source'] = str(subject_n)+row['task']
            counter +=1
        print(subject_n,type(subject_n))
    print(all)
    all.to_excel("New hist.xlsx")

def histogram():
    all = pd.read_excel("New hist.xlsx",engine='openpyxl')
    all.drop('record',axis=1,inplace=True)
    MALE_MASK = all['Gender'] == 'M'
    FEMALE_MASK = all['Gender'] == 'F'
    ASD_mask = all['Type'] == 'ASD'
    NT_mask = all['Type'] == 'NT'
    male = all[MALE_MASK]
    female = all[FEMALE_MASK]
    print(male)
    print(female)
    M_ASD_mask = male['Type'] == 'ASD'
    M_NT_mask = male['Type'] == 'NT'
    F_ASD_mask = female['Type'] == 'ASD'
    F_NT_mask = female['Type'] == 'NT'
    for col in all.columns:
        if col == 'Type' or col =='Gender':
            continue
        all[NT_mask][col].plot.hist(bins=20,alpha=0.5,title='NT',legend= True)
        all[ASD_mask][col].plot.hist(bins=20,alpha=0.5,title='ASD',legend= True)
        mean_nt = all[NT_mask][col].mean()
        mean_asd = all[ASD_mask][col].mean()
        plt.text(mean_nt,plt.ylim()[1]/2,'mean NT: '+"{:.3f}".format(mean_nt),rotation=90)
        plt.text(mean_asd,plt.ylim()[1]/2,'mean ASD: '+"{:.3f}".format(mean_asd),rotation=90)
        plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
        ax = plt.gca()
        for key in labels.keys():
            if key in col:
                ax.set_xlabel(labels[key])
        counts = all["Type"].value_counts()
        plt.title(col + "(ASD: "+str(counts['ASD'])+", NT: "+str(counts['NT'])+")")
        if not exists('histograms/type'):
            mkdir('histograms/type')
        plt.savefig('histograms/type/'+col)
        plt.close()
    for col in all.columns:
        if col == 'Type' or col =='Gender':
            continue
        male[col].plot.hist(bins=10,alpha=0.5,title='M')
        female[col].plot.hist(bins=10,alpha=0.5,title='F')
        mean_nt =  male[col].mean()
        mean_asd = female[col].mean()
        plt.text(mean_nt,plt.ylim()[1]/2,'mean M: '+"{:.3f}".format(mean_nt),rotation=90)
        plt.text(mean_asd,plt.ylim()[1]/2,'mean F: '+"{:.3f}".format(mean_asd),rotation=90)
        plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
        plt.legend(['M','F'])
        counts = all["Gender"].value_counts()
        # plt.title(col + "(M: "+str(counts['M'])+", F: "+str(counts['F'])+")")
        plt.title(col)
        ax = plt.gca()
        for key in labels.keys():
            if key in col:
                ax.set_xlabel(labels[key])
        if not exists('histograms/gender'):
            mkdir('histograms/gender')
        plt.savefig('histograms/gender/'+col)
        plt.close()
    for col in male.columns:
        if col == 'Type' or col =='Gender':
            continue
        hnt = male[M_NT_mask][col].plot.hist(bins=10,alpha=0.5,title='NT',color = 'b')
        hasd = male[M_ASD_mask][col].plot.hist(bins=10,alpha=0.5,title='ASD',color ='orange')
        mean_nt =  male[M_NT_mask][col].mean()
        mean_asd = male[M_ASD_mask][col].mean()
        plt.text(mean_nt,plt.ylim()[1]/2,'mean NT: '+"{:.3f}".format(mean_nt),rotation=90)
        plt.text(mean_asd,plt.ylim()[1]/2,'mean ASD: '+"{:.3f}".format(mean_asd),rotation=90)
        plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
        ax = plt.gca()
        for key in labels.keys():
            if key in col:
                ax.set_xlabel(labels[key])
        
        asdp = mpatches.Patch(color='orange', label='ASD')
        ntp = mpatches.Patch(color='blue', label='NT')
        plt.legend(handles = [asdp,ntp])
        counts = male["Type"].value_counts()
        plt.title(col +" Male "+"(ASD: "+str(int(counts['ASD']/3))+", NT: "+str(int(counts['NT']/3))+")")
        if not exists('histograms/male'):
            mkdir('histograms/male')
        plt.savefig('histograms/male/male_'+col)
        plt.close()
    # for col in female.columns:
    #     if col == 'Type' or col =='Gender':
    #         continue
    #     female[F_NT_mask][col].plot.hist(bins=10,alpha=0.5,title='NT')
    #     female[F_ASD_mask][col].plot.hist(bins=10,alpha=0.5,title='ASD')
    #     mean_nt =  female[F_NT_mask][col].mean()
    #     mean_asd = female[F_ASD_mask][col].mean()
    #     plt.text(mean_nt,plt.ylim()[1]/2,'mean NT: '+"{:.3f}".format(mean_nt),rotation=90)
    #     plt.text(mean_asd,plt.ylim()[1]/2,'mean ASD: '+"{:.3f}".format(mean_asd),rotation=90)
    #     plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
    #     plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
    #     ax = plt.gca()
    #     for key in labels.keys():
    #         if key in col:
    #             ax.set_xlabel(labels[key])
    #     plt.legend(['NT','ASD'])
    #     counts = female["Type"].value_counts()
    #     plt.title(col +" Female "+"(ASD: "+str(counts['ASD'])+", NT: "+str(counts['NT'])+")")
    #     # plt.title(col + " Female")
    #     if not exists('histograms/female'):
    #         mkdir('histograms/female')
    #     plt.savefig('histograms/female/female_'+col)
    #     plt.close()
       
def cross_subject():
    diffs = pd.read_pickle('newVocalDiffs')
    diffs=diffs.reset_index(drop=False)
    # diffs.drop('record',axis=1,inplace=True)
    if not exists('histograms/diffs'):
        mkdir('histograms/diffs')
    for task_i in TASKS:
        for task_j in TASKS:
            if task_i==task_j:
                continue
            current_dir = 'histograms/diffs/'+task_i+"_"+task_j
            if not exists(current_dir):
                mkdir(current_dir)
            print(diffs)
            current_mask = (diffs['task1']==task_i) & (diffs['task2']==task_j)
            all = diffs[current_mask]
            print(all)
            MALE_MASK = all['Gender'] == 'M'
            FEMALE_MASK = all['Gender'] == 'F'
            ASD_mask = all['Type'] == 'ASD'
            NT_mask = all['Type'] == 'NT'
            male = all[MALE_MASK]
            female = all[FEMALE_MASK]
            M_ASD_mask = male['Type'] == 'ASD'
            M_NT_mask = male['Type'] == 'NT'
            F_ASD_mask = female['Type'] == 'ASD'
            F_NT_mask = female['Type'] == 'NT'
            for col in all.columns:
                if col == 'Type' or col =='Gender' or col=='task1' or col=='task2':
                    continue
                all[NT_mask][col].plot.hist(bins=10,alpha=0.5,title='NT')
                all[ASD_mask][col].plot.hist(bins=10,alpha=0.5,title='ASD')
                mean_nt = all[NT_mask][col].mean()
                mean_asd = all[ASD_mask][col].mean()
                plt.text(mean_nt,plt.ylim()[1]/2,'mean NT: '+"{:.3f}".format(mean_nt),rotation=90)
                plt.text(mean_asd,plt.ylim()[1]/2,'mean ASD: '+"{:.3f}".format(mean_asd),rotation=90)
                plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
                plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
                plt.legend(['NT','ASD'])
                counts = all["Type"].value_counts()
                plt.title(col + "(ASD: "+str(counts['ASD'])+", NT: "+str(counts['NT'])+")")
                if not exists(current_dir+'/type'):
                    mkdir(current_dir+'/type')
                plt.savefig(current_dir+'/type/'+col)
                plt.close()
            for col in all.columns:
                if col == 'Type' or col =='Gender' or col=='task1' or col=='task2':
                    continue
                male[col].plot.hist(bins=10,alpha=0.5,title='M')
                female[col].plot.hist(bins=10,alpha=0.5,title='F')
                mean_nt =  male[col].mean()
                mean_asd = female[col].mean()
                plt.text(mean_nt,plt.ylim()[1]/2,'mean M: '+"{:.3f}".format(mean_nt),rotation=90)
                plt.text(mean_asd,plt.ylim()[1]/2,'mean F: '+"{:.3f}".format(mean_asd),rotation=90)
                plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
                plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
                plt.legend(['M','F'])
                counts = all["Gender"].value_counts()
                # No F right now
                # plt.title(col + "(M: "+str(counts['M'])+", F: "+str(counts['F'])+")")
                plt.title(col)
                if not exists(current_dir+'/gender'):
                    mkdir(current_dir+'/gender')
                plt.savefig(current_dir+'/gender/'+col)
                plt.close()
            for col in male.columns:
                if col == 'Type' or col =='Gender' or col=='task1' or col=='task2':
                    continue
                male[M_NT_mask][col].plot.hist(bins=10,alpha=0.5,title='NT')
                male[M_ASD_mask][col].plot.hist(bins=10,alpha=0.5,title='ASD')
                mean_nt =  male[M_NT_mask][col].mean()
                mean_asd = male[M_ASD_mask][col].mean()
                plt.text(mean_nt,plt.ylim()[1]/2,'mean NT: '+"{:.3f}".format(mean_nt),rotation=90)
                plt.text(mean_asd,plt.ylim()[1]/2,'mean ASD: '+"{:.3f}".format(mean_asd),rotation=90)
                plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
                plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
                plt.legend(['NT','ASD'])
                counts = male["Type"].value_counts()
                plt.title(col +" Male "+"(ASD: "+str(counts['ASD'])+", NT: "+str(counts['NT'])+")")
                if not exists(current_dir+'/male'):
                    mkdir(current_dir+'/male')
                plt.savefig(current_dir+'/male/male_'+col)
                plt.close()
            # for col in female.columns:
            #     if col == 'Type' or col =='Gender' or col=='task1' or col=='task2':
            #         continue
            #     female[F_NT_mask][col].plot.hist(bins=10,alpha=0.5,title='NT')
            #     female[F_ASD_mask][col].plot.hist(bins=10,alpha=0.5,title='ASD')
            #     mean_nt =  female[F_NT_mask][col].mean()
            #     mean_asd = female[F_ASD_mask][col].mean()
            #     plt.text(mean_nt,plt.ylim()[1]/2,'mean NT: '+"{:.3f}".format(mean_nt),rotation=90)
            #     plt.text(mean_asd,plt.ylim()[1]/2,'mean ASD: '+"{:.3f}".format(mean_asd),rotation=90)
            #     plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
            #     plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
            #     plt.legend(['NT','ASD'])
            #     counts = female["Type"].value_counts()
            #     plt.title(col +" Female "+"(ASD: "+str(counts['ASD'])+", NT: "+str(counts['NT'])+")")
            #     # plt.title(col + " Female")
            #     if not exists(current_dir+'/female'):
            #         mkdir(current_dir+'/female')
            #     plt.savefig(current_dir+'/female/female_'+col)
            #     plt.close()
            
#change task string to number
def convert_dfs(what ='TASK_TO_NUMBER'):

    if what ==  'TASK_TO_NUMBER':
        files = get_files("select df pickles")
        for file in files:
            subject_n = int(file.split('/')[-1].split('.')[0])
            subject_df = pd.read_pickle(file)
            print(subject_df.to_string())
            cols = list(subject_df)
            irr = {'task':'string','record':'int32','voiceID':'string'}
            floats = [item for item in cols if item not in list(irr.keys()) ]
            for floa in floats:
                irr[floa] = 'float64'
            subject_df = subject_df.astype(irr)

            subject_df.to_pickle('VocalOutput/'+str(subject_n))

def marissa_hist():
    dir = 'marissa_hist2nd/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    all = pd.read_csv('marissa_analyzed2nd.csv')
    # all['Is_Old_Lab'] = all['Location'] == 'Old_Lab'
    for col in list(all.columns):
        if 'mean' not in col:
            continue
        if col.replace('mean','stdev') in list(all.columns):
            all[col.replace('mean','cv')] = all[col.replace('mean','stdev')]/all[col]
        if col.replace('mean','SD') in list(all.columns):
            all[col.replace('mean','cv')] = all[col.replace('mean','SD') ]/all[col]
    for index,row in all.iterrows():
        if row['Location'] !='Old_Lab':
            all.at[index,'Location'] = 'Other'
    for col in list(all.columns):
        if col == 'Type' or col=='Gender' or col=='Location':
            continue
        old_mask = all['Location'] == 'Old_Lab'
        other_mask = all['Location'] == 'Other'
        # all[NT_mask][col].plot.hist(bins=10,alpha=0.5,label='NT',legend= True)
        # all[ASD_mask][col].plot.hist(bins=10,alpha=0.5,label='ASD',legend= True)
        all.pivot(columns='Location', values=col).plot.hist(bins=10,alpha=0.5)
        # all.pivot(columns='Location', values=col).plot.kde()
        # counts = all["Location"].value_counts()
        # plt.title(col + "(Old: "+str(counts['ASD'])+", NT: "+str(counts['NT'])+")")
        # print(all[old_mask][col])
        old_vals = pd.Series.sort_values(all[old_mask][col]).iloc[0:-3]
        # print(old_vals)
        # input()
        other_vals = pd.Series.sort_values(all[other_mask][col]).iloc[0:-3]
        mean_old = all[old_mask][col].mean()
        mean_other = all[other_mask][col].mean()
        plt.text(mean_old,plt.ylim()[1]/2,'mean Old: '+"{:.3f}".format(mean_old),rotation=90)
        plt.text(mean_other,plt.ylim()[1]/2,'mean Other: '+"{:.3f}".format(mean_other),rotation=90)
        plt.text(other_vals.mean(),plt.ylim()[1]/4,'90% mean Other: '+"{:.3f}".format(other_vals.mean()),rotation=90)
        plt.text(old_vals.mean(),plt.ylim()[1]/4,'90% mean Old: '+"{:.3f}".format(old_vals.mean(),),rotation=90)
        plt.axvline(mean_old, color='b', linestyle='solid', linewidth=1)
        plt.axvline(mean_other, color='r', linestyle='solid', linewidth=1)
        plt.axvline(old_vals.mean(), color='b', linestyle='dashed', linewidth=1)
        plt.axvline(other_vals.mean(), color='r', linestyle='dashed', linewidth=1)
        ax = plt.gca()
        for key in labels.keys():
            if key in col:
                ax.set_xlabel(labels[key])
        from os.path import exists
        from os import mkdir
        plt.savefig(dir+col)
        plt.close()
from scipy.stats import ttest_ind
def ttest_marissa():
    for i,file in enumerate(['marissa_analyzed.csv','marissa_analyzed2nd.csv']):
        all = pd.read_csv(file)
        results = pd.DataFrame()
        for index,row in all.iterrows():
            if row['Location'] !='Old_Lab':
                all.at[index,'Location'] = 'Other'
        for col in list(all.columns):
            print(col)
            if col == 'Type' or col=='Gender' or col=='Location':
                continue
            old_mask = all['Location'] == 'Old_Lab'
            other_mask = all['Location'] == 'Other'
            t,p = ttest_ind(all[old_mask][col],all[other_mask][col])
            results.at[col,'t-statistic'] = t
            results.at[col,'p'] = p
        print(str(results))
        results.to_csv('t-test on Marrisa-'+str(i)+'.csv')
import scipy.stats
def ttest_fm():
    f1 = pd.read_excel('New vocal.xlsx',engine = 'openpyxl')
    f2 = pd.read_excel('New vocal2.xlsx',engine = 'openpyxl')
    ntm = f1[f1['Type']=='NT']
    ntf = f2[f2['Type']=='NT']
    print(scipy.stats.shapiro(ntm['TB_stdevF0Hz']/ntm['TB_meanF0Hz']))
    print(scipy.stats.shapiro(ntf['TB_stdevF0Hz']/ntf['TB_meanF0Hz']))
    print(scipy.stats.mannwhitneyu(ntm['TB_stdevF0Hz']/ntm['TB_meanF0Hz'],ntf['TB_stdevF0Hz']/ntf['TB_meanF0Hz']))

    print(len(ntm))
    print(len(ntf))

import os.path
if os.path.exists('files.pickle'):
    ALL_FILES = pd.read_pickle('files.pickle')
if os.path.exists('marissa_files.pickle'):
    MARISSA_ALL_FILES = pd.read_pickle('marissa_files.pickle')
MARISSA_ALL_FILES = pd.read_csv('marrissa_files.csv').set_index(['record','task'])
NEW_SUBJECT = pd.read_csv('subject_files2.csv').set_index(['record','task'])
# ttest_fm()
create_pickels_from_dataframe(NEW_SUBJECT)
# marissa_pitch_analysis()
# analyze_marissa_files()
# marissa_hist()
# ttest_marissa()
# subject_pitch_analysis()
# # df = pd.read_pickle("VocalOutput/1")
# # print(df)
# # result_df = subject_task_pitch_analysis()
# # result_df.to_pickle(save_get_file("Save pickle"))``
# # print(RECORD2TYPE)
# convert_dfs()
analyze_files()
# # hist_pre()
# histogram()
# cross_subject()
# render_pitch_on_video()