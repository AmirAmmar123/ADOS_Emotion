from genericpath import exists
import pickle 
from SpeechAnalysis import plot_compare_over_time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import Variable, ttk
from os import listdir
from os.path import exists
PICKLE_PATH = 'C:\\Users\\user\\Documents\\לימודים\\תזה\\Code\\Excel parsing\\pickles\\'
colors=['blue','green','red','yellow','black','cyan','magenta']
class Gui : 
    def __init__(self) -> None:
        #Setting up the main window 
        main_image = None
        self.window = tk.Tk()
        self.window.configure(background="White")
        self.window=self.window
        self.window.title('ADOS analysis plots')
        self.window_height = 500
        self.window_width = 900
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        self.x_cordinate = int((self.screen_width/2) - (self.window_width/2))
        self.y_cordinate = int((self.screen_height/2) - (self.window_height/2))

        self.window.geometry("{}x{}+{}+{}".format(self.window_width, self.window_height, self.x_cordinate, self.y_cordinate))
        self.window.attributes('-fullscreen', True)
        pickles = listdir(PICKLE_PATH)
        parsed_pickles = list()
        self.pickles_dict = dict()
        for pickle1 in pickles:
            filename = str(pickle1).split('/')[-1].split(".")[0]
            parsed_pickles.append(filename)
            with open(PICKLE_PATH+pickle1, "rb") as input_file:
                self.pickles_dict[str(filename.strip())]=list(pickle.load(input_file))
        self.var = tk.StringVar()
        self.var.set(parsed_pickles[0])
        self.cb_frame = tk.Frame(self.window)
        self.cb_variables = dict()
        self.to_plot = list()
        for item in parsed_pickles:
            vl = Variable()
            vl.set(0)
            self.cb_variables[item]=vl
            l = tk.Checkbutton(self.cb_frame, text=item, variable=vl,command=self.plot)
            l.pack()
        self.cb_frame.grid(row=0,column=0)
        self.plot_frame = tk.Frame(self.window)
        self.plot_frame.grid(row=0,column=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.window.mainloop()
    def plot(self):
        print(list(self.cb_variables.keys()))
        # print(list(self.pickles_dict.keys()))
        graphs=list()
        graphs.append([(1,1)])
        graphs_dict = dict()
        for graph in list(self.cb_variables.keys()):
            params = graph.split(" ")
            subject_id = params[0]
            task_name = params[1]
            param = params[2]
            # print(self.cb_variables[graph].get())
            if int(self.cb_variables[graph].get())==1:
                graphs.append(self.pickles_dict[graph])
                if subject_id not in graphs_dict:
                    graphs_dict[subject_id]=dict()
                if param not in graphs_dict[subject_id]:
                    graphs_dict[subject_id][param] = dict()

                graphs_dict[subject_id][param]['S'] = self.pickles_dict[graph]
        plot_compare_over_time('S',graphs_dict,'time : 0.1 seconds')
        max = 0 
        # for param in list(graphs_dict.keys()):
        #     for subject in list(graphs_dict[param].keys()):
        #         if len(graphs_dict[param][subject])>max:
        #             max = len(graphs_dict[param][subject])
        # fig,axs = plt.subplots(len(graphs_dict),max)
        # print(graphs)
        # for i,graph in enumerate(graphs):
        #     axs[i].scatter(*zip(*graph),color=colors[i])
        # canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        # canvas.get_tk_widget().grid(row=0,column=2)
        # canvas.draw()
Gui()