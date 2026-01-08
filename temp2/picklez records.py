import pickle
from Files import *
from Excel import *
import pandas as pd
# files = get_files("Please select your csvs")
# print(files)
# csv_list = dict()
# for file in files:
#     if not file:
#         print("File is None")
#     else:
#         print(file)
#         save_pickle_file(read_file_line_to_dict(file),title = "Please save you pickle of "+file)
# POSE  = [' pose_Tx',' pose_Ty',' pose_Tz']
# POSE_STD = [' pose_Tx std',' pose_Ty std',' pose_Tz std']
# files = get_files("Please select your csvs")
# # # file_list_emo = ['G:/האחסון שלי/shared with saleh/gaze pickles/record 4/emo_subject_openface.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 8/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 9/emo subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 10/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 11/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 12/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 16/emo subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 17/emo_subjec.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 20/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 21/emo_subject.pickle', 'G:/האחסון שלי/shared with saleh/gaze pickles/record 22/emo_subject.pickle']
# # # file_list_emo.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 13/emo subject.pickle')
# # # file_list_emo.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 11/emo_subject.pickle')
# # # file_list_emo.append('G:/האחסון שלי/shared with saleh/gaze pickles/record 10/emo_subject.pickle')
# # # files = file_list_emo
# csv_list = dict()
# for file in files:
#     if not file:
#         print("File is None")
#     else:
#         out_file = filedialog.asksaveasfile(mode='wb', defaultextension='.pickle',title="Please save you pickle of "+file)
#         if not out_file:
#             continue
#         myopen = read_file_line_to_dict(file)
#         myopen = pd.DataFrame.from_dict(myopen['excel_dict'])
#         myopen.to_pickle(out_file)
#         # print(file)
#         # pose = myopen[POSE].astype(float)
#         # pose.to_pickle(out_file)
#         continue


POSE  = [' pose_Tx',' pose_Ty',' pose_Tz']
POSE_STD = [' pose_Tx std',' pose_Ty std',' pose_Tz std']
files = get_files("Please select your csvs")
for file in files:
    if not file:
        print("File is None")
    else:
        out_file = filedialog.asksaveasfile(mode='wb', defaultextension='.pickle',title="Please save you pickle of "+file)
        if not out_file:
            continue
        # myopen = read_file_line_to_dict(file)
        myopen = pd.read_pickle(file)
        # myopen.to_pickle(out_file)
        # print(file)
        pose = myopen[POSE].astype(float)
        pose.to_pickle(out_file)
        continue