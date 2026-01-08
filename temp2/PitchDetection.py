import crepe
from scipy.io import wavfile
import tkinter as tk
from tkinter import filedialog
from os import path
from pydub import AudioSegment
from pathlib import Path
import numpy
def get_file():
    root = tk.Tk()
    root.withdraw()
    xl_file_path = filedialog.askopenfilename(title= "Please select a file:")
    return xl_file_path
def get_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title= "Please select a Directory:")
    return folder_path   
sr, audio = wavfile.read(get_file())
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
print(frequency)
a = numpy.column_stack((time, frequency, confidence))
numpy.savetxt(get_folder()+"/results.csv", a,
              ['%.3f', '%.3f', '%.6f'],
              header='time,frequency,confidence',delimiter=',')