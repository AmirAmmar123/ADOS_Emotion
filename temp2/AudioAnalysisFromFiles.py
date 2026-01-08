import Files
import os
import os.path
import pandas as pd
folder = None
folder = 'F:/Synced Videos/OneDrive - Haifa University/Talking_times_sync/'
sf_df = pd.DataFrame(columns=['record','task','wav_file','speak_times']).set_index(['record','task'])
audio_files_path = 'F:/Synced Videos/OneDrive - Haifa University/audio_files/'
audio_fileso = os.listdir(audio_files_path)
audio_files = []
for file in audio_fileso:
    audio_files.append(file.rsplit(".")[0])
print(audio_files)
for file in os.listdir(folder):
    if(file.split('.')[-1] not in ['csv','xlsx']):
        continue
    # print(file)
    subject_n = int(file.split('_')[0])
    task = None
    task2 = None
    if 'TB' in file:
        task = 'TB'
    if 'story' in file.lower() or 'book' in file.lower():
        task = 'Story'
    if 'emo' in file.lower():
        task= 'Emo'
        task2= 'EMO'
    speaking = None
    if file.split('.')[-1]=='csv':
        speaking = 'Marissa'
    if file.split('.')[-1]=='xlsx':
        speaking= 'Subject'
    if speaking != 'Subject':
        continue
    if not task:
        continue
    audio_prefix = str(subject_n)+'_'+task+'_' 
    if task2:
        audio_prefix2 = str(subject_n)+'_'+task2+'_' 
    if audio_prefix+'I' in audio_files:
        # print("V ",subject_n,task,speaking)
        sf_df.at[(subject_n,task),'wav_file'] = audio_files_path+audio_prefix+'I.wav'
        sf_df.at[(subject_n,task),'speak_times'] = folder + file
        continue
    elif audio_prefix2+'I' in audio_files:
        sf_df.at[(subject_n,task),'wav_file'] = audio_files_path+audio_prefix2+'I.wav'
        sf_df.at[(subject_n,task),'speak_times'] = folder + file
        continue   
    if audio_prefix+'S' in audio_files:
        # print("V ",subject_n,task,speaking)
        sf_df.at[(subject_n,task),'wav_file'] = audio_files_path+audio_prefix+'S.wav'
        sf_df.at[(subject_n,task),'speak_times'] = folder + file
        continue
    if audio_prefix+'B' in audio_files:
        # print("V ",subject_n,task,speaking)
        sf_df.at[(subject_n,task),'wav_file'] = audio_files_path+audio_prefix+'B.wav'
        sf_df.at[(subject_n,task),'speak_times'] = folder + file
        continue
    print('ERROR with',subject_n)
print(sf_df)
print(len(sf_df.index))
sf_df.to_csv('subject_files2.csv')
    # print("X ",subject_n,task,speaking)
# import shutil

# import os
# from glob import glob
# PATH = 'G:/.shortcut-targets-by-id/102-cxgpm83UvzAoQ3ad-esMgJD3tL1ea/Bayan/'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.mp4'))]
# for res in result:
#     resl = res.lower()
#     if 'asaf' not in resl and 'faisal' not in resl:
#         continue
#     print(resl)
    # shutil.copy2(res, 'F:/Synced Videos/OneDrive - Haifa University/Faisal/')
