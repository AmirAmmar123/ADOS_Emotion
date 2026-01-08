import os
import glob
from pathlib import Path
audio_dir = 'c:/openface_dired_test/'
csvs = glob.glob(audio_dir+'**/*.csv')
print(csvs)
for file in csvs:
    filepath = Path(file)
    fileName = filepath.name
    splits = fileName.split('_')
    subject_n = int(splits[0])

    task = None
    view = None
    print(splits)
    if 'TB' in splits[1]:
        task = 'TB'
    if 'story' in splits[1].lower() or 'book' in splits[1].lower():
        task = 'Story'
    if 'emo' in splits[1].lower():
        task= 'Emo'
    if str(1) in splits[1]:
        print(subject_n,task)
        input()
        task+='1'
    if str(2) in splits[1]:
        print(subject_n,task)
        input()
        task+='2'
    if 'subject' in splits[2].lower() or 's' == splits[2].lower().split('.')[0]:
        view = 'S'
    if 'marissa' in splits[2].lower() or 'i' == splits[2].lower().split('.')[0]:
        view = 'I'
    if 'both' in splits[2].lower() or 'b' == splits[2].lower().split('.')[0]:
        view = 'B'
    if not task or not view:
        print('ERROR - ',fileName)
        input()
        continue
    # if (str(subject_n)+'_'+task+'_'+view) in used:
    #     # print('used ',str(subject_n)+'_'+task+'_'+view)
    #     continue
    # print(file,str(subject_n)+'_'+task+'_'+view)
    # if os.path.exists(new_file):
    #     print('used ',str(subject_n)+'_'+task+'_'+view)
    #     continue      
    # print(old_file,new_file)
    # continue
    # print(new_file)
    # if os.path.exists(new_file+'.wav'):
    #     continue
    # print(old_file,new_file)
    # if(str(subject_n)+'_'+task+'_'+view+'.wav') in os.listdir(audio_dir):
        # continue
    os.rename(old_file, new_file+'.wav')
    # print(file,str(subject_n)+'_'+task+'_'+view)
# for file in os.listdir(audio_dir):
#     if file.split('.')[-1]!='wav':
#         print(file)
#         old_file = os.path.join(audio_dir,file)
#         new_file = os.path.join(audio_dir, file+'.wav')  
#         os.rename(old_file, new_file)