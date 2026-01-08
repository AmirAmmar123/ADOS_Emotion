from LittleGui import load_dataFrame,TestApp
import os 
import os.path
import pandas as pd
RESULTS_DF_DIR = 'results df/'
tables = []
for file in os.listdir(RESULTS_DF_DIR):
    file_path = RESULTS_DF_DIR+file
    df = pd.read_pickle(file_path)
    tables.append((file,df))
TestApp(tables)