import pandas as pd
def get_frames_in_range_ms(start,end,df):
    start = start/1000
    end = end/1000
    min_set_flag = False
    min_f = float('inf')
    max_f = -1
    for index,row in df.iterrows():
        if float(row['timestamp']) > end:
            break
        if float(row['timestamp']) < start:
            continue
        if not min_set_flag:
            min_f = index
            min_set_flag = True
        max_f = index
    return (min_f,max_f)
def load_time_calibration_centers(csv_file,openface_df):
    df = pd.read_csv(csv_file)
    df['time'] = df['End'] - df['Start']
    df['time'] = df['time']*(df['Type']=='Center')
    centers = df.copy()
    centers = centers[centers['Type']=='Center']
    centerId = 1
    new_indicies = []
    for index,row in centers.iterrows():
        if index == centers['time'].idxmax():
            new_indicies.append('Center')
            continue
        new_indicies.append('Center'+str(centerId))
        centerId+=1
    centers.reset_index()
    centers['Type'] = new_indicies
    # print(centers)
    row = df.loc[df['time'].idxmax()]
    df = df[df.Type != 'Center']
    # df.loc[len(df.index)] = row
    df = pd.concat([df,centers],axis=0)
    df.drop(['time'],axis=1,inplace=True)
    df.sort_values('Start',inplace=True)
    print(df)
    calibration = []
    for index,row in df.iterrows():
        min_f,max_f = get_frames_in_range_ms(row['Start'],row['End'],openface_df)
        calibration.append((row['Type'],min_f,max_f))
    return calibration

openface_source = 'c:/openface_dired/234/'
S_openFace_File  = openface_source + '234_pen1_subject.csv'
S_openface = pd.read_csv(S_openFace_File).set_index('frame')
for df in [S_openface]:
    cols = df.columns
    new_cols = [col.strip() for col in cols]
    df.columns = new_cols

calibration_file = 'G:/My Drive/ADOS/Videos/Faisal/record234/234_pen1_marissa 25-06-2023 00-13-44 Questions.csv'
calibartion_frames =load_time_calibration_centers(calibration_file,S_openface)
print(calibartion_frames)