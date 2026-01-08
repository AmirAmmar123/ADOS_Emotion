import patoolib
import os
from glob import glob
def insert(originalfile,string):
    with open(originalfile,'r') as f:
        with open('newfile.txt','w') as f2: 
            f2.write(string)
            f2.write(f.read())
    os.remove(originalfile)
    os.rename('newfile.txt',originalfile)
PATH = 'C:/Users/user/Desktop/HWRESUB/'
if not os.path.exists(PATH+'Extracted/'):
    os.mkdir(PATH+'Extracted/')
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.rar'))]
for res in result:
    print(res)
    # patoolib.extract_archive(res,outdir=PATH +'Extracted/'+ res.split('\\')[-1].split('.')[0])
    # ID = res.split("HW_AVL_")[-1].split('.')[0].split('_')[-1].split('\\')[0]
    # with open(res,"a") as f:
        # f.write('\n\n long long int get_ID() {return ID;}\n')
    # insert(res,'#define ID '+ID+'\n')
    # print(res)

EXTRACTED = PATH +'Extracted/'
# import pandas as pd
# df = pd.DataFrame(columns=['ID1','ID2','Grade','Notes']).set_index(['ID1','ID2'])
# for dir in os.listdir(EXTRACTED):
#     if (len(dir.split('_'))<4):
#         print(dir.split('_'))
#         continue
#     splits = dir.split('_')
#     df.at[(int(splits[-2]),int(splits[-1])),'Grades']=0
# df.to_csv('AVLHW2023Resub.csv')

# from Files import get_files
# files = get_files()
# for dir in os.listdir(EXTRACTED):
#     if not os.path.isdir(EXTRACTED+dir):
#         continue
#     for file in files:
#         import shutil
#         # if "AVL2023HW.c" in file:
#         shutil.copy2(file,EXTRACTED+dir)

ONLYC = PATH+'cfiles/'
if not os.path.exists(ONLYC):
    os.mkdir(ONLYC)
import shutil
files =  [y for x in os.walk(EXTRACTED) for y in glob(os.path.join(x[0], '*.c'))]
for file in files:
    if "AVL2023HW.c" in file:
        continue
    if 'HW_AVL_' not in file:
        continue
    # print(file.split('HW_AVL_')[-1].split('\\')[0])
    fname = file.split('HW_AVL_')[-1].split('\\')[0]+'.c'
    if not os.path.exists(file):
        continue
    print(fname)
    shutil.copy(file,ONLYC+fname)
# import pandas as pd
# import numpy as np
# df = pd.read_csv('AVLHW2023.csv', encoding='cp1252').set_index(['ID1','ID2'])
# def parse_row(row):
#     parsed = ""
#     for item in row.items():
#         if item[0] in ['Grade','Notes','copy'] or item[1]==np.nan:
#             continue
#         parsed+=item[0]+": "+item[1]+" "
#     return parsed
# # for index,row in df.iterrows():
# #     df.at[index,'Notes'] = parse_row(row)+'\n'+df.at[index,'Notes'] 
# # print(df)
# # df.to_excel('AVLHW2023Grades.xlsx')
# columns = list(df.columns)
# new_columns = []
# for col in columns:
#     new_columns.append(col.strip())
# df.columns = new_columns


# df2 = df.reset_index().drop(['ID1','copy'],axis=1)
# df1 = df.reset_index().drop(['ID2','copy'],axis=1)
# df2=df2.rename(mapper={'ID2':'ID1'},axis=1)
# df2 = pd.concat([df1,df2],ignore_index=True).dropna(axis=0).set_index('ID1')
# print(len(df2))
# print(df2)
# df2.to_csv('testGRADES.csv')
# from Files import get_file
# realdf = pd.read_csv(get_file(),encoding='utf-8').set_index('ID number')
# for index,row in df2.iterrows():
#     realdf.at[index,'Grade'] = row['Grade']
#     realdf.at[index,'Feedback comments'] =parse_row(row)+'\n'+row['Notes']
# print(realdf)
# realdf.to_csv('upload.csv',encoding='utf-8')

        # print(ind)
