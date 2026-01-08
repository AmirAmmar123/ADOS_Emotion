from operator import contains
from pickle import dump,load
from unittest import result
import numpy as np
import pandas as pd 
from Files import *
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile
from Excel import read_file_line_to_dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
ASD = [4,8,12,13,20,144]
NT = [9,10,11,16,17,21,22,29,31]
ASD_PO = [1,2,3,5,6,27,148]
NT_PO = [28,145,147]
POSE  = [' pose_Tx',' pose_Ty',' pose_Tz']

# file_list = list()
# file = "x"
# while file:
#     file = get_file("Please select your looking")
#     if file:
#         file_list.append(file)
#         print(file)
# exit(1)
def create_frame(save = True, plot = True):
    file_list = ['G:/האחסון שלי/shared with saleh/gaze pickles/record 4/emo/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 4/story/pickles/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 8/emo pre/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 8/story/pickles/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 9/emo/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 9/story/pickles/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 12/emo/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 12/story/pickles/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 16/emo/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 16/story/pickles/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 17/emo/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 17/story/pickles/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 20/emo/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 20/story/pickles/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 21/emo/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 21/story/pickles/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 22/emo/lookingpick', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 22/story/pickles/lookingpick']
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 10/emo/lookingpick')
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 10/story/lookingpick')
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 11/emo/lookingpick')
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 11/story/lookingpick')
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 13/emo/lookingpick')
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 13/story/lookingpick')
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 31/emo/lookingpick')
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 31/story/lookingpick')
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 29/emo/lookingpick')
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 29/story/lookingpick')
    file_list.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 144/emo/lookingpick')
    file = "x"
    while file:
        file = get_file("Please select your looking- concat ")
        if file:
            file_list.append(file)
            print(file)
    dict_file_to_number = dict()
    dict_file_to_task = dict()
    numbers_list = list()
    file_list_emo = ['G:/האחסון שלי/shared with saleh/gaze pickles/record 4/emo_subject_openface.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 8/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 9/emo subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 10/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 11/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 12/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 16/emo subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 17/emo_subjec.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 20/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 21/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 22/emo_subject.pickle']
    file_list_emo.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 13/emo subject.pickle')
    file_list_emo.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 11/emo_subject.pickle')
    file_list_emo.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 10/emo_subject.pickle')
    file = "x"
    while file:
        file = get_file("Please select your emo pickle - concat ")
        if file:
            file_list_emo.append(file)
            print(file)
    emo_pickls_folder = 'x'
    emo_pickls_folder = get_folder("Select emo pickle folder")
    if emo_pickls_folder:
        file_list_emo  = listdir(emo_pickls_folder)
    for file in file_list:
        number = file.split("record")[-1]
        number = number.split("/")[0]
        if "emo" in file:
            task = "emo"
        if "story" in file:
            task = "story"
        number=number.strip()
        if not number in numbers_list:
            numbers_list.append(number)
        dict_file_to_number[file]=number
        dict_file_to_task[file]=task
    for file in file_list_emo:
        number = file.split("record")[-1]
        number = number.split("/")[0]
        number=number.strip()
        dict_file_to_number[file]=number 
    results_dict = dict()
    counter = 0 
    print(numbers_list)
    # columns = ['emo Look&Speak ratio','emo Look&Listen ratio','emo Look&Silence ratio','story Look&Speak ratio','story Look&Silence ratio','emo total speak','emo total listen','emo total silence','story total speak','story total silence']
    results_frame = pd.DataFrame(index = numbers_list,dtype=int)
    # results_frame.columns = columns
    # results_frame.index = numbers_list
    for number in numbers_list:
        if int(number) in ASD:
            results_frame.at[number,"Type"]= "ASD"
        if int(number) in NT:
            results_frame.at[number,"Type"] = "NT"

    print(results_frame)
    for file in file_list:
        
        looking = pd.read_pickle(file)
        if dict_file_to_task[file] == "emo":
            looking = looking.dropna()
            looking_c = looking.iloc[:,0].astype('bool')
            s_speak_c = looking.iloc[:,1].astype('bool')
            m_speak_c = looking.iloc[:,2].astype('bool')
            subject_look_speak = looking_c*s_speak_c
            subject_look_listen = looking_c*m_speak_c
            subject_look_silence = looking_c*(~(s_speak_c+m_speak_c))
            print(len(subject_look_silence) - subject_look_silence.sum())
            result_dict = dict()
            name_task =dict_file_to_task[file]
            # results_dict[name_task+" Look&Speak_ratio"] = (subject_look_speak.sum()/len(subject_look_speak))
            # results_dict[name_task+" Look&Listen_ratio"] = (subject_look_listen.sum()/len(subject_look_listen))
            # results_dict[name_task+" Look&Silence_ratio"] = (subject_look_silence.sum()/len(subject_look_silence))
            results_frame.at[dict_file_to_number[file],name_task+" Look&Speak ratio"] = (subject_look_speak.sum()/len(subject_look_speak))
            results_frame.at[dict_file_to_number[file],name_task+" Look&Listen ratio"] = (subject_look_listen.sum()/len(subject_look_listen))
            results_frame.at[dict_file_to_number[file],name_task+" Look&Silence ratio"] = (subject_look_silence.sum()/len(subject_look_silence))
            results_frame.at[dict_file_to_number[file],name_task+" total speak"] = (subject_look_speak.sum())
            results_frame.at[dict_file_to_number[file],name_task+" total listen"] = (subject_look_listen.sum())
            results_frame.at[dict_file_to_number[file],name_task+" total silence"] = (subject_look_silence.sum())
        if dict_file_to_task[file] == "story":
            looking.drop(looking.columns[len(looking.columns)-1], axis=1, inplace=True)
            looking = looking.dropna()
            looking_c = looking.iloc[:,0].astype('bool')
            s_speak_c = looking.iloc[:,1].astype('bool')
            subject_look_speak = looking_c*s_speak_c
            subject_look_silence = looking_c*(~(s_speak_c))
            print(len(subject_look_silence) - subject_look_silence.sum())
            result_dict = dict()
            name_task =dict_file_to_task[file]
            # results_dict[name_task+" Look&Speak_ratio"] = (subject_look_speak.sum()/len(subject_look_speak))
            # results_dict[name_task+" Look&Silence_ratio"] = (subject_look_silence.sum()/len(subject_look_silence))
            results_frame.at[dict_file_to_number[file],name_task+" Look&Speak ratio"] = (subject_look_speak.sum()/len(subject_look_speak))
            results_frame.at[dict_file_to_number[file],name_task+" Look&Silence ratio"] = (subject_look_silence.sum()/len(subject_look_silence))
            results_frame.at[dict_file_to_number[file],name_task+" total speak"] = (subject_look_speak.sum())
            results_frame.at[dict_file_to_number[file],name_task+" total silence"] = (subject_look_silence.sum())
    # metadf.T.to_excel("results.xlsx")
    count = 0
    for file in file_list_emo:
        print(count,"/",len(file_list_emo),file)
        # myopen = read_file_line_to_dict(file)
        count+=1
        # myopen = pd.DataFrame.from_dict(myopen['excel_dict'])
        if not isfile(emo_pickls_folder+"\\"+file)  or 'desktop' in file:
            continue
        pose = pd.read_pickle(emo_pickls_folder+"\\"+file)
        # # myopen.to_pickle(str(count))
        # pose = myopen[POSE].astype(float)
        record_str = file.split(".")[0].split("record")[-1].strip()
        # pose.to_pickle(emo_pickls_folder+"/Pose/"+file+"_Pose")
        # continue
        results_frame.at[record_str,POSE[0]+" std"] = pose.std().at[POSE[0]]
        results_frame.at[record_str,POSE[1]+" std"] = pose.std().at[POSE[1]]
        results_frame.at[record_str,POSE[2]+" std"] = pose.std().at[POSE[2]]
    if save:
        results_frame.to_excel("new results.xlsx")
        results_frame.to_pickle("results_pick")
    if plot:
        results_frame.dropna().plot(x='Type',y='emo Look&Speak ratio',kind = 'scatter')
        results_frame.dropna().plot(x='Type',y='emo Look&Listen ratio',kind = 'scatter')
        plt.show()
    return results_frame
def process_pose(save = True):
    results_frame = pd.DataFrame(index = ASD+ASD_PO+NT+NT_PO,dtype=int)
    # results_frame.columns = columns
    # results_frame.index = numbers_list
    for number in ASD+ASD_PO+NT+NT_PO:
        if int(number) in ASD+ASD_PO:
            results_frame.at[number,"Type"]= "ASD"
        if int(number) in NT+NT_PO:
            results_frame.at[number,"Type"] = "NT"
    emo_pickls_folder = get_folder("Select emo pickle folder")
    if emo_pickls_folder:
        file_list_emo  = listdir(emo_pickls_folder)
    count = 0
    for file in file_list_emo:
        print(count,"/",len(file_list_emo),file)
        # myopen = read_file_line_to_dict(file)
        count+=1
        # myopen = pd.DataFrame.from_dict(myopen['excel_dict'])
        if not isfile(emo_pickls_folder+"\\"+file)  or 'desktop' in file:
            continue
        pose = pd.read_pickle(emo_pickls_folder+"\\"+file)
        # # myopen.to_pickle(str(count))
        # pose = myopen[POSE].astype(float)
        record_str = int(file.split(".")[0].split("record")[-1].strip())
        # pose.to_pickle(emo_pickls_folder+"/Pose/"+file+"_Pose")
        # continue
        results_frame.at[record_str,POSE[0]+" std"] = pose.std().at[POSE[0]]
        results_frame.at[record_str,POSE[1]+" std"] = pose.std().at[POSE[1]]
        results_frame.at[record_str,POSE[2]+" std"] = pose.std().at[POSE[2]]
    if save:
        results_frame.to_excel("POSEnew results.xlsx")
        results_frame.to_pickle("POSE results_pick")
    return results_frame
    # results_frame.dropna().plot(x='Type',y='emo Look&Speak ratio',kind = 'scatter')
    # results_frame.dropna().plot(x='Type',y='emo Look&Listen ratio',kind = 'scatter')
    # plt.show()
def plot_frame():
    results_frame = pd.read_pickle("results_pick")
    # results_frame.drop(results_frame.tail().index,inplace=True) 
    results_frame.dropna().plot(x='Type',y='emo Look&Speak ratio',kind = 'scatter')
    results_frame.dropna().plot(x='Type',y='emo Look&Listen ratio',kind = 'scatter')
    results_frame.dropna().plot(x='Type',y='story Look&Speak ratio',kind = 'scatter')

    # results_frame.plot(x='Type',y=POSE[0]+" std",kind = 'scatter')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(0, len(results_frame)):
        _color =  '#2ca02c'
        row = results_frame.iloc[i]
        if int(row.name) in ASD:
            _color = '#d62728'
        points = list()
        for p in POSE:
            # print(row[p+" std"])
            points.append(row[p+" std"])
        ax.scatter3D(points[0],points[1],points[2],color = _color)
    ax.set_xlabel(POSE[0])
    ax.set_ylabel(POSE[1])
    ax.set_zlabel(POSE[2])
    plt.show()
def pose_plot_frame():
    results_frame = pd.read_pickle("POSE results_pick")

    # results_frame.plot(x='Type',y=POSE[0]+" std",kind = 'scatter')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(0, len(results_frame)):
        _color =  '#2ca02c'
        row = results_frame.iloc[i]
        if int(row.name) in ASD+ASD_PO:
            _color = '#d62728'
        points = list()
        for p in POSE:
            # print(row[p+" std"])
            points.append(row[p+" std"])
        ax.scatter3D(points[0],points[1],points[2],color = _color)
    ax.set_xlabel(POSE[0])
    ax.set_ylabel(POSE[1])
    ax.set_zlabel(POSE[2])
    plt.show()
def combine_all():
    gaze_results = pd.read_pickle(get_file("select gaze results"))
    gaze_results = gaze_results.drop(POSE[0]+" std",1)
    gaze_results = gaze_results.drop(POSE[1]+" std",1)
    gaze_results = gaze_results.drop(POSE[2]+" std",1)

    gaze_results = gaze_results.drop("Type",1) 
    #Story irrlevent for now 
    name_task = "story"
    gaze_results = gaze_results.drop(name_task+" total silence",1)
    gaze_results = gaze_results.drop(name_task+" Look&Speak ratio",1)
    gaze_results = gaze_results.drop(name_task+" Look&Silence ratio",1)
    gaze_results = gaze_results.drop(name_task+" total speak",1)
    gaze_results.index.name= "record"
    gaze_results.index = gaze_results.index.map(str)
    pose_results = pd.read_pickle(get_file("select pose results"))
    pose_results.index.name= "record"
    pose_results.index = pose_results.index.map(str)

    speech_results = pd.read_pickle(get_file("select pose results"))
    speech_results.index.name= "record"
    speech_results.index = speech_results.index.map(str)

    # all_df = pd.merge(gaze_results, pose_results, left_index=True, right_index=True,how="outer")
    all_df = pd.concat([pose_results,gaze_results,speech_results],ignore_index=False)
    all_df = all_df.groupby(["record"]).sum()
    for index,row in pose_results.iterrows():
        all_df.at[index,"Type"] = row["Type"]
    for index,row in speech_results.iterrows():
        all_df.at[index,"Type"] = row["Type"]
    # all_df = pd.merge(all_df, speech_results, left_index=True, right_index=True,how="outer")
    all_df.to_excel("all.xlsx")
    clean_df = all_df[(all_df != 0).all(1)]
    clean_df.to_excel("clean.xlsx")
    print(all_df)
# process_pose()
# pose_plot_frame()
# create_frame()
# plot_frame()
combine_all()