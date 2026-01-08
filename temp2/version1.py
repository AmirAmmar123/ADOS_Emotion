from tkinter.constants import FALSE
import openpyxl
import os.path
import tkinter as tk
from tkinter import filedialog
import statistics
import matplotlib.pyplot as plt
import numpy as np

#Using the examples from https://www.geeksforgeeks.org/working-with-excel-spreadsheets-in-python/
#Helper functions: 
# Backup plan or for later versions
def get_file():
    root = tk.Tk()
    root.withdraw()
    xl_file_path = filedialog.askopenfilename(title= "Please select a file:",filetypes=[('Excel files', "*.xlsx")])
    return xl_file_path
def bell_plot(data,stats):
    y = list()
    for x in data:
        y.append(1/(stats['stdev'] * np.sqrt(2 * np.pi)) * np.exp( - (x - stats['mean'])**2 / (2 * stats['stdev'] **2)))
        plt.style.use('seaborn')
    plt.figure(figsize = (6, 6))
    plt.plot(data, y, color = 'black',
            linestyle = 'dashed')
    plt.scatter( data, y, marker = 'o', s = 25, color = 'red')
    plt.show()
#This can change according to the matlab script that creates the excel file
rows_dict = {"SPEAKER":'A',"START":'B',"END":'C',"minPITCH":'D',"maxPITCH":'E',"avgPITCH":'F',"stdPITCH":'G'} 
xl_file_dir ="C:\\Users\\user\\Desktop\\סרטוני ADOS\\סרטון 1"
xl_file_name = "Video1outputFile.xlsx"
xl_file_path=xl_file_dir+"\\"+xl_file_name
speaker_dict ={"INTERVIEWER":1,"SUBJECT":2} #This is correct only for Vdeo1outputFile!
xl_file_path=None
if xl_file_path is None:
    xl_file_path=get_file()
print("Parsing Excel file: "+xl_file_name)
if not os.path.isfile(xl_file_path):
    print("Error, No such Excel file!")
    exit(1) 
wb_obj = openpyxl.load_workbook(xl_file_path) 
sheet_obj = wb_obj.active 
total_rows = str(len(sheet_obj['A']))
cell_obj = sheet_obj[rows_dict["SPEAKER"]+'2' : rows_dict['END']+total_rows]

countFromTime = 60 #Arbitrary, can be changed...
responsesTimeList=list()
responses=list()
flag=False
last_sentence_end=0
for cell1, cell2, cell3 in cell_obj:
    if cell2.value >=60 and cell1.value==speaker_dict["INTERVIEWER"]:
        #Now we can start the count
        flag=True
        last_sentence_end = cell3.value
    if flag:
        speaker = cell1.value
        start = cell2.value
        if speaker_dict["SUBJECT"]==speaker:
            # print("response: ",last_sentence_end,start)
            responsesTimeList.append(start-last_sentence_end)
            responses.append(start)
            flag = False
stat_funcs = [statistics.mean,statistics.median,statistics.stdev,max,min] 
#Just to test the bell plot
# res = [random.randrange(1, 50, 1) for i in range(50)]
# responsesTimeList=res
stats=dict()
stats_list = list()
stats_func_list = list()
for func in stat_funcs:
    stats[func.__name__]=func(responsesTimeList)
    stats_list.append(func(responsesTimeList))
    stats_func_list.append(func.__name__)
print(stats)
print("max is at time: ",responses[responsesTimeList.index(stats['max'])])
i=0
output_excel_file = get_file()
stats_wb = openpyxl.load_workbook(output_excel_file)
#no titles
stats_sheet = stats_wb.active
stats_func_list.insert(0,"file")
stats_list.insert(0,xl_file_path)
print(stats_sheet.max_row )
if stats_sheet.max_row == 1 and stats_sheet.max_column==1:
    stats_sheet.append(stats_func_list)
    stats_wb.save(output_excel_file)
stats_wb = openpyxl.load_workbook(output_excel_file)
stats_sheet = stats_wb.active
row = stats_sheet.max_row
print(stats_list)
stats_sheet.append(stats_list)
stats_wb.save(output_excel_file)
print("This might be running on a loop, please close the window when you are done!")
while(True):
    i=i+1
    if(i>1000):
        i=0
    
# bell_plot(responsesTimeList,stats)

############## IDEAS
#1. run the matlab script from here , this might help : https://www.pythonpool.com/convert-matlab-to-python/ 
#                                     or Matlab's python engine.
#2. translate the matlab script to python
#3. GUI
#4.Automate the entire process as much as possible