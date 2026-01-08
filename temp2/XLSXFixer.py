import os
import pandas as pd
FILES = 'C:/Users/user/Documents/Talking_times_sync-20230108T163532Z-001/Talking_times_sync/'
FILES = 'F:/Synced Videos/OneDrive - Haifa University/Talking_times_sync/'

for file in os.listdir(FILES):
    csv_file = FILES+file
    if csv_file.split('.')[-1]!='xlsx':
        continue
    speech_df = pd.read_excel(csv_file,engine='openpyxl')
    speech_df.columns = ['Type','Start','End']
    speech_df.to_excel(csv_file,engine='openpyxl',index=False)
