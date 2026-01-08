from dataclasses import asdict
from sre_constants import MIN_REPEAT_ONE
import numpy as np
import pandas as pd
from Files import *
from os.path import isfile
JOY = [6,12]
SADNESS = [1,4,15]
SURPRISE = [1,2,5,26]
FEAR = [1,2,4,5,7,20,26]
ANGER = [4,5,7,23]
DISGUST = [9,15] #16 removed, no support from Openface
CONTEMPT = [12,14]

ASD = [4,8,12,13,20,144]
NT = [9,10,11,16,17,21,22,29,31]
ASD_PO = [1,2,3,5,6,27,148]
NT_PO = [28,145,147]


TIME = ["frame"," timestamp"]
REL_AUS =list(dict.fromkeys(JOY+SADNESS+SURPRISE+FEAR+ANGER+DISGUST+CONTEMPT))
REL_AUS.sort()
AUS_C_COLS = list()
AUS_R_COLS = list()
AUS_C_toAUnum = dict()
dict_file_to_number = dict()
for au in REL_AUS:
    au_str = str(au)
    if(au<10):
        au_str = "0"+au_str
    AUS_C_COLS.append(" AU"+au_str+"_c")
    AUS_C_toAUnum[" AU"+au_str+"_c"] = au
for au in REL_AUS:
    au_str = str(au)
    if(au<10):
        au_str = "0"+au_str
    AUS_R_COLS.append(" AU"+au_str+"_r")
print(AUS_C_COLS)
print(AUS_R_COLS)
def prep():
    emo_pickls_folder = get_folder("Select emo pickle folder")
    ouput_dir  =get_folder("Select output dir")
    ouput_dir=ouput_dir+"/"
    if emo_pickls_folder:
        file_list_emo  = listdir(emo_pickls_folder)
    for file in file_list_emo:
        number = file.split("record")[-1]
        number = number.split("/")[0]
        number=number.strip()
        dict_file_to_number[file]=number 
    count = 0
    for emo_pick in file_list_emo:
        print(count,"/",len(file_list_emo),emo_pick)
        count = count+1
        if not isfile(emo_pickls_folder+"\\"+emo_pick)  or 'desktop' in emo_pick:
            continue
    # emo_pick  = get_file("Select emo pickle (DATAFRAME)")
        ausc_df = pd.read_pickle(emo_pickls_folder+"/"+emo_pick)[TIME+AUS_C_COLS]
        ausc_df = ausc_df.reset_index()  # make sure indexes pair with number of rows
        evets_au = pd.DataFrame()
        emo_df = pd.DataFrame()
        MIN_ONE = 5
        MAX_ZERO = 2

        for au in AUS_C_COLS:
            in_event = False
            con_one = 0
            con_zero = 0
            last = -1
            for index,row in ausc_df.iterrows():
                if(float(row[au])==1):
                    con_zero = 0
                    con_one=con_one+1
                    last = 1
                if(float(row[au])==0):
                    con_zero = con_zero+1
                    con_one = 0
                    last = 0
                if not in_event and con_one>=MIN_ONE:
                    in_event = True
                if in_event and con_zero>MAX_ZERO:
                    in_event = False
                if in_event:
                    evets_au.at[row["frame"],au] = 1
                else:
                    evets_au.at[row["frame"],au] = 0
        for index,row in evets_au.iterrows():
            live_au = list()
            for au in AUS_C_COLS:
                if(int(row[au])==1):
                    live_au.append(AUS_C_toAUnum[au])
            for i,live in enumerate(live_au):
                emo_df.at[index,i] = live
        emo_df.to_excel(ouput_dir+str(dict_file_to_number[emo_pick])+" EMO AU.xlsx")
        evets_au.to_excel(ouput_dir+str(dict_file_to_number[emo_pick])+" AU events.xlsx")
        emo_df.to_pickle(ouput_dir+str(dict_file_to_number[emo_pick])+" EMO AU")
        evets_au.to_pickle(ouput_dir+str(dict_file_to_number[emo_pick])+" AU events")
def frame_to_emo():
    emo_pickls_folder = get_folder("Select emo pickle folder")
    ouput_dir  =get_folder("Select output dir")
    ouput_dir=ouput_dir+"/"
    if emo_pickls_folder:
        file_list_emo  = listdir(emo_pickls_folder)
    for file in file_list_emo:
        if not isfile(emo_pickls_folder+"\\"+file)  or 'desktop' in file or "xlsx" in file  or not "EMO AU" in file:
            continue
        number = file.split(".")[0]
        number = number.split("/")[0]
        number=number.strip()
        dict_file_to_number[file]=number 
    count = 0
    for emo_pick in file_list_emo:
        print(count,"/",len(file_list_emo),emo_pick)
        count = count+1
        if not isfile(emo_pickls_folder+"\\"+emo_pick)  or 'desktop' in emo_pick or "xlsx" in emo_pick  or not "EMO AU" in emo_pick:
            continue
        emo_df = pd.read_pickle(emo_pickls_folder+"/"+emo_pick)
        proc_emo_df = pd.DataFrame()
        for index,row in emo_df.iterrows():
            aus = row.tolist()
            emo_in_frame = list()
            if set(JOY).issubset(set(aus)):
                emo_in_frame.append("JOY")
            if set(SADNESS).issubset(set(aus)):
                emo_in_frame.append("SADNESS")
            if set(SURPRISE).issubset(set(aus)):
                emo_in_frame.append("SURPRISE")
            if set(FEAR).issubset(set(aus)):
                emo_in_frame.append("FEAR")
            if set(ANGER).issubset(set(aus)):
                emo_in_frame.append("ANGER")
            if set(DISGUST).issubset(set(aus)):
                emo_in_frame.append("DISGUST")
            if set(CONTEMPT).issubset(set(aus)):
                emo_in_frame.append("CONTEMPT")
            for e,emo_i in enumerate(emo_in_frame):
                    proc_emo_df.at[index,e] = emo_i
        proc_emo_df.to_excel(ouput_dir+str(dict_file_to_number[emo_pick])+"emo framed.xlsx")
        proc_emo_df.to_pickle(ouput_dir+str(dict_file_to_number[emo_pick])+"emo framed")
def frame_to_change():
    pic_folder = get_files("Select AUS files")
    change_df = pd.DataFrame()
    for pic_file in pic_folder:
        if "events" in pic_file:
            print('Skipped '+pic_file)
            continue
        print("Proc "+pic_file)
        record_num = int(pic_file.split(".")[0].split("/")[-1])
        Type = ""
        if record_num in ASD+ASD_PO:
            Type = "ASD"
        if record_num in  NT+NT_PO:
            Type = "NT"
        if Type == "":
            print(record_num," NO Type")
            continue
        prep_df = pd.read_pickle(pic_file)
        last_row_counter = -1
        change_counter = 0
        for index,row in prep_df.iterrows():
            changed = False
            if(last_row_counter>-1):
                changed = not(row.equals(last_row))
            last_row = row
            last_row_counter = last_row_counter+1
            # change_df.at[index,"Change in AU"] = changed
            change_counter = change_counter+int(changed)
        change_df.at[record_num,"Type"] = Type
        change_df.at[record_num,"#Changes"] = change_counter
        change_df.at[record_num,"Change_rate"] = change_counter / len(prep_df)
    change_df.index.name = "record"
    change_df.to_excel("change.xlsx")
    print(change_counter)

# frame_to_emo()
frame_to_change()