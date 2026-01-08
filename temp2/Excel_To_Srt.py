import tkinter as tk
from tkinter import filedialog
from pynput import keyboard
import openpyxl
from Files import *
import os.path
from Excel import *
from operator import itemgetter
def dict_to_list(dict):
    if not dict:
        return None
    data_list = list()
    keys = dict['keys']
    data_dict = dict['excel_dict']
    first__row = dict['first_row']
    last_row = dict['last_row']
    for i in range(first__row,last_row+1):
        data_tuple = ()
        for key in keys:
            data = data_dict[key][i]
            if(key!='speaker'):
                data = int(data)
            data_tuple=data_tuple+(data,)
        data_list.append(data_tuple)
    data_list.sort(key=lambda tup: tup[1])
    return data_list
def millis_to_time_string(time):
    #https://stackoverflow.com/questions/35989666/convert-milliseconds-to-hours-min-and-seconds-python
    millis=time
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))%24    
    millis = int(time) - int(hours)*(1000*60*60)-minutes*(1000*60)-seconds*1000
    return str("%02d:%02d:%02d.%03d" % (hours, minutes, seconds,millis))
    # return str("%02d:%02d:%02d" % (hours, minutes, seconds))

def parse_tuple(parser_dict,tuple):
    speaker = tuple[0]
    start_time = tuple[1]
    end_time = tuple[2]
    line1 =parser_dict[speaker]+":"+"\n"+millis_to_time_string(start_time)+" -> "+millis_to_time_string(end_time)
    return line1
def data_list_to_srt_file(_list):
    parser_dict={'I':'1','S':'2'}
    string_list  = list()
    for data in _list:
        string_list.append(parse_tuple(parser_dict,data))
    write_list_to_file(string_list,get_folder("Please select where to save the srt file")+"/"+now_file_format()+".srt")
    return string_list
def get_all_data_list(subject_excel_file = None, interviewer_excel_file=None, interviewer_flag = False):
    if not subject_excel_file:
        subject_excel_file = get_file("Please select the subject excel file")
    if not interviewer_excel_file and interviewer_flag:
        interviewer_excel_file= get_file("Please select the interviewer excel file")
    subject_excel_dict  = read_file_line_to_dict(subject_excel_file)
    subject_list = dict_to_list(subject_excel_dict)
    all_data = list()
    if interviewer_flag:
        interviewer_excel_dict = read_file_line_to_dict(interviewer_excel_file)
        interviewer_list  = dict_to_list(interviewer_excel_dict)
        all_data = interviewer_list
    all_data = all_data+subject_list
    all_data.sort(key=lambda tup: tup[1])
    return all_data

def run():
    data_list_to_srt_file(get_all_data_list())