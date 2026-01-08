from re import I
from Files import *
from Excel import *
from Excel_To_Srt import get_all_data_list
import matplotlib.pyplot as plt
import statistics
import pickle
from LittleGui import *
import numpy as np  
colors=['blue','green','red','yellow','black','cyan','magenta']
def millis2decSeconds(time):
    seconds  = time/1000
    return seconds/10
def times_to_pitch_list(speakers_list,pitctAnalysisExcel):
    speakers_pitch_dict = dict()
    speakers_intensity_dict=dict()
    pitch_dict = read_file_line_to_dict(pitctAnalysisExcel)
    for tuple in speakers_list:
        speaker = tuple[0]
        if speaker not in speakers_pitch_dict:
            speakers_pitch_dict[speaker]= list()
            speakers_intensity_dict[speaker]=list()
        start_time = tuple[1]
        end_time = tuple[2]
        times = range(start_time,end_time+100,100)
        for t in times:
            t = t-200 #To "fix" the time it takes to press the button
            t = t/1000
            row = int(t*100+1)
            current_pitch = pitch_dict['excel_dict']['pitch'][row]
            if current_pitch <310 and current_pitch>0:
                speakers_pitch_dict[speaker].append(((pitch_dict['excel_dict']['time'][row]),(pitch_dict['excel_dict']['pitch'][row])))
                speakers_intensity_dict[speaker].append(((pitch_dict['excel_dict']['time'][row]),(pitch_dict['excel_dict']['intensity'][row])))
    return speakers_pitch_dict,speakers_intensity_dict
def plot_pitch_over_time(speaker,lists):
    fig,axs = plt.subplots(len(lists))
    for i,list in enumerate(lists):
        axs[i].scatter(*zip(*list),color=colors[i])
    plt.show()
#not a generel function
def plot_compare_over_time(speaker,data,xlable):
    titles = list(data.keys())
    parameters = list(data[titles[0]].keys())
    fig,axs2d = plt.subplots(len(parameters),len(titles),sharey='row')
    for j,parameter in enumerate(parameters):
        for i,title in enumerate(titles):
            x,y = zip(*normalize_list(data[title][parameter][speaker],True))
            std = statistics.stdev(y)
            axs2d[j,i].set_title(title+"-"+parameter+" , STD : "+str(float("{:.3f}".format(std))))
            axs2d[j,i].set_xlabel(xlable)
            axs2d[j,i].set_ylabel(parameter)
            axs2d[j,i].axhline(0, color=colors[i], linestyle='dashed', linewidth=1,label='mean') # dashed
            axs2d[j,i].annotate('mean', xy=(0, 0), xytext=(3, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))
            axs2d[j,i].yaxis.set_tick_params(which='both', labelbottom=True)
            axs2d[j,i].scatter(*zip(*normalize_list(data[title][parameter][speaker],True)),color=colors[i])
        # for j,parameter in enumerate(parameters):
        #     for i in range(1,len(titles)):
        #         axs2d[j,i].sharey(axs2d[j,0])

        #future reference hspace should be 0.34
    for j,parameter in enumerate(parameters):
        for i,title in enumerate(titles):
            plt.setp(axs2d[i,j].get_yticklabels(), visible=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.34)
    plt.show()
    return None
def plot_compare_over_time2(data,xlable):
    print(data)
    titles = list(data.keys())
    parameters = list(data[titles[0]].keys())
    fig,axs2d = plt.subplots(len(parameters),len(titles),sharey='row')
    for j,parameter in enumerate(parameters):
        for i,title in enumerate(titles):
            x,y = zip(*normalize_list(data[title][parameter],True))
            std = statistics.stdev(y)
            axs2d[j,i].set_title(title+"-"+parameter+" , STD : "+str(float("{:.3f}".format(std))))
            axs2d[j,i].set_xlabel(xlable)
            axs2d[j,i].set_ylabel(parameter)
            axs2d[j,i].axhline(0, color=colors[i], linestyle='dashed', linewidth=1,label='mean') # dashed
            axs2d[j,i].annotate('mean', xy=(0, 0), xytext=(3, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))
            axs2d[j,i].yaxis.set_tick_params(which='both', labelbottom=True)
            axs2d[j,i].scatter(*zip(*normalize_list(data[title][parameter],True)),color=colors[i])
        # for j,parameter in enumerate(parameters):
        #     for i in range(1,len(titles)):
        #         axs2d[j,i].sharey(axs2d[j,0])

        #future reference hspace should be 0.34
    for j,parameter in enumerate(parameters):
        for i,title in enumerate(titles):
            plt.setp(axs2d[i,j].get_yticklabels(), visible=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.34)
    plt.show()
    return None
def plot_sd_over_time(speaker,data,xlabel):
    pitch = data['ASD']['Pitch'][speaker]
    sd_in_sec = list()
    sds_list = list()
    sdss_list = list()
    print(pitch)
    j = 0
    for i,tup in enumerate(pitch):
        sd_in_sec.append(tup[1])
        if(len(sd_in_sec)==10):
            std = statistics.stdev(sd_in_sec)
            sds_list.append((j,std))
            sdss_list.append(std)
            j=j+1
            sd_in_sec = list()
    i = 0
    print(statistics.stdev(sdss_list))
    plt.scatter(*zip(*sds_list),color=colors[i])
    plt.show()
    return None

def normalize_list(data_list,remove_5=False):
    sum=0
    for tupple in data_list:
        sum+=tupple[1]
    mean = sum/len(data_list)
    normalized_list = data_list.copy()
    for i in range(0,len(normalized_list)):
        normalized_list[i]=((normalized_list[i][0],normalized_list[i][1]-mean))
    if remove_5:
        s_ml = sorted(normalized_list, key=lambda tup: abs(tup[1]), reverse=True)
        print(s_ml)
        fivepercent = int(5*len(s_ml)/100)
        for i in range(len(normalized_list)):
            for j in range(fivepercent):
                if normalized_list[i][1]==s_ml[j]:
                    del normalized_list[i]
    return normalized_list
    return normalized_list
def normalize(data,means):
    mean = statistics.mean(data)
    new_data=list()
    for d in data:
        new_data.append(d/means[data.index(d)])
    return new_data
# def time_to_param_list():
def pitch_file_and_times_to_list(praat_dict,times_dict):
    intensity_list=list()
    pitch_list = list()
    for row in range(2,len(praat_dict)+2):
        start_time = times_dict['start'][row]
        end_time = times_dict['end'][row]
        times = range(start_time,end_time+100,100)
        # print(times)
        for t in times:
            t = t-200 #To "fix" the time it takes to press the button
            t = t/1000
            t = int(t*100+1)
            current_pitch = praat_dict['pitch'][t]
            if current_pitch <310 and current_pitch>0:
                intensity_list.append((t,praat_dict['intensity'][t]))
                pitch_list.append((t,current_pitch))
    return intensity_list,pitch_list

def pickle_data():
    praat_xl_file = get_file("Please select the praat excel file")
    time_files = get_files("Please select the times excel files")
    praat_dict = read_file_line_to_dict(praat_xl_file)
    save_folder = get_folder("Please select where to save the pickles")
    record_id  = get_string("please enter record ID ")
    praat_dict = praat_dict['excel_dict']
    print("Omer change file name save ")
    # for time_file in time_files:
    #     time_dict = read_file_line_to_dict(time_file)
    #     time_dict = time_dict['excel_dict']
    #     intensity_list,pitch_list = pitch_file_and_times_to_list(praat_dict,time_dict)
    #     type = time_file.split("/")[-1].split(".")[0]
    #     f = open(save_folder +'/'+record_id+" "+type+" intensity", 'wb')
    #     pickle.dump(intensity_list, f)
    #     f.close()
    #     f = open(save_folder +'/'+record_id+" "+type+" pitch", 'wb')
    #     pickle.dump(pitch_list, f)
    #     f.close()



def run():
    asd_sub_xl_file = get_file("select ASD subject speaking time excel file")
    ast_pitch_xl_file = get_file("Please select the ASD pitch excel file")
    nt_sub_xl_file = get_file("select NT subject speaking time excel file")
    nt_pitch_xl_file = get_file("Please select the NT pitch excel file")

    asd_pitch_dict,asd_intensity_dict = times_to_pitch_list(get_all_data_list(asd_sub_xl_file),ast_pitch_xl_file)
    nt_pitch_dict,nt_intensity_dict = times_to_pitch_list(get_all_data_list(nt_sub_xl_file),nt_pitch_xl_file)
    data = {"ASD" : {"Pitch": asd_pitch_dict, "Intensity" : asd_intensity_dict}, "NT": {"Pitch":nt_pitch_dict,"Intensity":nt_intensity_dict}}
    who='S'
    normalize = False
    # plot_compare_over_time(who,data,"time point (0.1sec)")
    # plot_sd_over_time(who,data,"1 sec")
def hist():
    files = get_files("Select the pickles for the histograms")
    lists = list()
    for file in files:
        with open(file, "rb") as input_file:
            lis = load(input_file)
            lis2 = list()
            for item in lis:
                lis2.append(item[1])
            lists.append(lis2)

    task = files[0].split("/")[-1].split(" ")[0]
    stds = list()
    means = list()
    for lis in lists:
        std = statistics.stdev(lis)
        stds.append(std)
        means.insert(stds.index(std),statistics.mean(lis))
    norm = False
    if norm:
        stds = normalize(stds,means)
    if task.strip() =='intensity':
        plt.hist(stds,range=(0,0.1))
        plt.ylim(ymax=8)
    if task.strip() =='pitch':
        plt.hist(stds,range=(10,70))
        plt.ylim(ymax=15)
    task+=" "+files[0].split("/")[-2]
    task+=" Mean: "+"{:.3f}".format(statistics.mean(stds))  
    plt.title(task)
    plt.show()

def mat():
    files = get_files("Select the pickles for the histograms")
    lists = list()
    data_dict = dict()
    for file in files:
        with open(file, "rb") as input_file:
            task = file.split("/")[-1].split(" ")[1]
            s_num = file.split("/")[-1].split(" ")[2]
            lis = load(input_file)
            lis2 = list()
            if task not in list(data_dict.keys()):
                data_dict[task]=dict()
            if s_num not in list(data_dict[task].keys()):
                data_dict[task][s_num]=list()
            for item in lis:
                data_dict[task][s_num].append(item[1])

    # print(data_dict)
    print(data_dict.keys())
    stds_dict = dict()
    for key in list(data_dict.keys()):
        stds_dict[key]=dict()
    print(stds_dict.keys())
#     param = files[0].split("/")[-1].split(" ")[0]
#     stds = list()
#     means = list()
    for key in list(data_dict.keys()):
        for s in list(data_dict[key].keys()):
            stds_dict[key][s] = statistics.stdev(data_dict[key][s])
        # means.insert(stds.index(std),statistics.mean(lis))
    dif = dict()
    for key in list(data_dict.keys()):
        for key2 in list(data_dict.keys()):
            dif[key,key2]=list()
    for s in list(data_dict[list(data_dict.keys())[0]].keys()):
        for key in list(data_dict.keys()):
            for key2 in list(data_dict.keys()):
                dif[key,key2].append(abs(stds_dict[key][s]-stds_dict[key2][s])) #diff
    mean_dif = dict()
    for key in list(data_dict.keys()):
        for key2 in list(data_dict.keys()):
            mean_dif[key,key2] = statistics.mean(dif[key,key2]) #mean of diff
    print(mean_dif)
    mat = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(len(list(data_dict.keys()))):
        for j in range(len(list(data_dict.keys()))):
            mat[i][j]=mean_dif[list(data_dict.keys())[i],list(data_dict.keys())[j]]
    fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.set_xticklabels(['z']+list(data_dict.keys()))
    ax.set_yticklabels( ['z']+list(data_dict.keys()))
    ax.matshow(mat, cmap='seismic')

    for (i, j), z in np.ndenumerate(mat):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.show()
# run()
# dict = read_file_line_to_dict(get_file())
# dict = dict['excel_dict']
# dict2 = read_file_line_to_dict(get_file())
# dict2 = dict2['excel_dict']
# pickle.dump(dict,str(save_file("Save your pickle",'.pickle')))
# 
# hist()
# mat()
def results_df(save : bool):
    from pickle import load
    import pandas as pd
    # with open(get_file(), "rb") as input_file:
    #     pic = load(input_file)
    # df = pd.DataFrame(pic,columns=["timestamp","pitch"])
    # df =df.set_index("timestamp")
    # print(df)
    # print(type(df.std()["pitch"]))
    NT_files = get_files("Select NT pickles")
    ASD_files = get_files("Select ASD pickles")
    results = pd.DataFrame()
    for file in NT_files:
        with open(file, "rb") as input_file:
            pic = load(input_file)
        file_name = str(file).split("/")[-1]
        param = file_name.split(" ")[0]
        task  = file_name.split(" ")[1].split("_")[-1]
        record_num = int(file_name.split(" ")[-1])
        print(param,task,record_num)
        df = pd.DataFrame(pic,columns=["timestamp",task])
        df =df.set_index("timestamp")
        results.at[record_num,"Type"]="NT"
        results.at[record_num,task+" "+param] = df.std()[task]
    for file in ASD_files:
        with open(file, "rb") as input_file:
            pic = load(input_file)
        file_name = str(file).split("/")[-1]
        param = file_name.split(" ")[0]
        task  = file_name.split(" ")[1].split("_")[-1]
        record_num = int(file_name.split(" ")[-1])
        print(param,task,record_num)
        df = pd.DataFrame(pic,columns=["timestamp",task])
        df =df.set_index("timestamp")
        results.at[record_num,"Type"]="ASD"
        results.at[record_num,task+" "+param] = df.std()[task]
    results.index.name = "record"
    if save:
        results.to_excel("speech results.xlsx")
        results.to_pickle("speech results")
    return results

