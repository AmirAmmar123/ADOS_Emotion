
# importing vlc module
import tkinter
import vlc
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pynput import keyboard
import openpyxl
from Files import *
import os.path
from Excel import *
from datetime import datetime
import pandas as pd

keyq_map = {'1':'Happy','2':'Sad','3':'Scary','4':'Angry','5':'NA','6':'TB','7':'Up','8':'Down','9':'Left','0':'Right','o':'Center'}
class customMP: 
    # creating vlc media player object
    def __init__(self):
            self.video_file = get_file("Please select the video file")
            self.media_player = vlc.MediaPlayer(self.video_file)
            self.interviewer_times=list()
            self.subject_times=list()
            self.I_is_speaking = False
            self.I_current_start_time = 0
            self.S_is_speaking = False
            self.S_current_start_time = 0 
            self.pause_flag = 0
            self.fr = (1/25)
            self.questions_df = pd.DataFrame(columns=['Start','End','Type']).set_index(['Start','End'])
            self.start_time = 0
            self.pressed = False
    def on_press(self,key):
        try:

            if (key.char=='a' or key.char =='A' or key.char=='ש') and not self.I_is_speaking:
                # self.interviewer_times.append(self.media_player.get_time())
                self.I_current_start_time=self.media_player.get_time()
                print(self.I_current_start_time)
                self.I_is_speaking=True
            if (key.char=='d' or key.char =='D' or key.char=='ג') and not self.S_is_speaking:
                # self.interviewer_times.append(self.media_player.get_time())
                self.S_current_start_time=self.media_player.get_time()
                print(self.S_current_start_time)
                self.S_is_speaking=True
            if (key.char in keyq_map.keys() and not self.pressed):
                self.start_time = self.media_player.get_time()
                self.pressed = True
                print(key.char,self.start_time)
        except AttributeError:
            print('special key pressed: {0}'.format(
                key))

    def on_release(self,key):
        if key == keyboard.Key.space:
            self.media_player.pause()
        if key == keyboard.Key.right:
            cur_time = self.media_player.get_time()
            print("cur time",cur_time)
            self.media_player.set_time(cur_time+int(self.fr*1000))
        if key == keyboard.Key.left:
            cur_time = self.media_player.get_time()
            print("cur time",cur_time)
            self.media_player.set_time(cur_time-int(self.fr*1000))
        if key == keyboard.Key.esc:
            return False
        try:
            if key.char=='a'or key.char=='ש':
                self.interviewer_times.append(("I",self.I_current_start_time,self.media_player.get_time()))
                self.I_is_speaking=False
                print("end")
            if key.char=='d'or key.char=='ג':
                cur_time = self.media_player.get_time()
                self.subject_times.append(("S",self.S_current_start_time,cur_time))
                self.S_is_speaking=False
                print("end : ",cur_time)
            if key.char=='t' or key.char =='א':
                print('time is: ',self.media_player.get_time())
            if key.char=='[':
                print('play rate :', self.media_player.get_rate()-0.1)
                self.media_player.set_rate(self.media_player.get_rate()-0.1)
            if key.char==']':
                print('play rate :', self.media_player.get_rate()+0.1)
                self.media_player.set_rate(self.media_player.get_rate()+0.1)
                #     self.interviewer_times.append(self.media_player.get_time())
                    # Stop listener
            if key.char == 'r' or key.char=='ר':
                print("list reset")
                self.interviewer_times=list()
                self.subject_times=list()
            if key.char =='l':
                self.media_player.pause()
                self.media_player.set_time(0)
            if key.char =='z' or key.char=='ז':
                print("last delete")
                if len(self.interviewer_times)>0:
                    del self.interviewer_times[-1]
                if len(self.subject_times)>0:
                    del self.subject_times[-1]
                if len(self.questions_df)>0:
                    self.questions_df = self.questions_df[:-1]
            if key.char in keyq_map.keys():
                self.questions_df.at[(self.start_time,self.media_player.get_time()),'Type' ] = keyq_map[key.char]
                self.pressed = False
        except AttributeError:
            print('special key pressed: {0}'.format(
                key))
    def play_video(self):
        # window = tk.Tk()
        # window.configure(background="White")
        # slider = ttk.Scale(window)    
        # slider['from_'] = 0
        # slider['to']=self.media_player.get_length()
        # slider.grid(row=0, column=0, pady=5)
        self.media_player.set_rate(1)
        self.media_player.play()
        ptime = 0




        # ptime = 0*(1000*60)+0*(1000)
        ptime = 6*60*1000+30*1000
        # ptime = int(14.67776278642615*1000)
# - 5000


        # window.mainloop() [[]]
        self.media_player.set_time(ptime)
        time.sleep(5)
        self.media_player.pause()
        self.media_player.set_time(ptime)
        self.media_player.pause()
        self.fr = 1/self.media_player.get_fps()
        print('fr ',self.fr)
        with keyboard.Listener(
        on_press=self.on_press,
        on_release=self.on_release) as listener:
            listener.join()
    def get_video_file(self):
        return self.video_file

# Collect events until released
#https://stackoverflow.com/questions/47990695/how-to-embed-a-vlc-instance-in-a-tkinter-frame
mp = customMP()
mp.play_video()
name = os.path.basename(mp.video_file)
base  = name.split(".")[0]

base= base+ " "+now_file_format()
xl_file =os.path.dirname(mp.video_file)+"\\"+base
if  mp.interviewer_times or  mp.subject_times:
    print(*mp.interviewer_times, sep='\n')
    print(*mp.subject_times, sep='\n')
    if mp.interviewer_times:
        append_write(xl_file+"_I_"+".xlsx",mp.interviewer_times)
    if mp.subject_times:
        append_write(xl_file+"_S_"+".xlsx",mp.subject_times)
else:
    print("nothing to save")
mp.questions_df.to_csv(xl_file+' Questions.csv')
# printing type of media player variable
