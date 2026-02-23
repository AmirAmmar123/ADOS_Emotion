import scipy.io.wavfile
import scipy.signal
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from skimage.morphology import binary_opening
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os, os.path






## given a list of dataframes in which features are saved (for each expression), create a big dataframe of aggregated features
def process_df(df, prefix):
    """Calculate features for a given dataframe and prefix."""
    int_avg = float(np.mean(df[f'int_avg_{prefix}']))
    prom = float(np.mean(df[f'prominence_{prefix}']))
    prom_std = float(np.std(df[f'prominence_{prefix}']))
    area = float(np.mean(df[f'area_{prefix}']))
    dur = float(np.mean(df[f'duration_{prefix}']))
    peaks = list(df[f'peak_lp_{prefix}'])
    count = len(df)
    if count != 0:
        tot_time = df['timestamp'].iloc[0]
        rate = count * 60 / tot_time
    else:
        rate = 0
        tot_time = np.nan
    return int_avg, prom, prom_std, area, dur, peaks, count, rate, tot_time

def create_feature_dict(subjects, activities, cameras, durations, counts, rates, peaks, features, feature_prefix):
    """Create a dictionary of features."""
    names = ['subject', 'task', 'camera', 'duration', f'{feature_prefix}_num', f'{feature_prefix}_rate', f'{feature_prefix}_peaks']
    names.extend([f'{feature_prefix}_{f}' for f in features])
    final_values = [subjects, activities, cameras, durations, counts, rates, peaks]
    final_values.extend(features)
    return dict(zip(names, final_values))

def process_expression(df_path, prefix, au_prefixes):
    """Process dataframes for a given expression prefix."""
    subjects, activities, cameras = [], [], []
    feature_sets = {au: [[] for _ in range(5)] for au in au_prefixes}  # To hold int_avg, prom, prom_std, area, dur for each AU prefix
    durations, rates, peaks, counts = [], [], [], []

    for dirpath, _, filenames in os.walk(df_path):
        for file in filenames:
            file_name = file.split('.')[0]
            sub, act, itw = file_name.split('_')[0], file_name.split('_')[1], file_name.split('_')[-1]
            df = pd.read_csv(os.path.join(dirpath, file), index_col=0)
            df.columns = df.columns.str.strip()
            print(df_path)
            print(f'{prefix.upper()}:', file_name)

            subjects.append(sub)
            activities.append(act)
            cameras.append(itw)

            for au in au_prefixes:
                int_avg, prom, prom_std, area, dur, peak, count, rate, tot_time = process_df(df, f'{prefix}_{au}')
                feature_sets[au][0].append(int_avg)
                feature_sets[au][1].append(prom)
                feature_sets[au][2].append(prom_std)
                feature_sets[au][3].append(area)
                feature_sets[au][4].append(dur)

            if len(df) != 0:
                tot_time = df['timestamp'].iloc[0]
                rate = len(df) * 60 / tot_time
            else:
                rate = 0
                tot_time = np.nan

            peaks.append(peak)
            counts.append(count)
            rates.append(rate)
            durations.append(tot_time)

    features = [item for sublist in feature_sets.values() for item in sublist]
    return create_feature_dict(subjects, activities, cameras, durations, counts, rates, peaks, features, prefix)

def CreateAUsFeaturesDataFrame(list_dfs_path, save_path):
    smile_df, simple_smile_df, sad_df, anger_df, fear_df = None, None, None, None, None

    for df_path in list_dfs_path:
        if 'combosmile' in df_path.lower():
            smile_df = pd.DataFrame(process_expression(df_path, 'combo_smile', ['AU12', 'AU6']))
        elif 'simplesmile' in df_path.lower():
            simple_smile_df = pd.DataFrame(process_expression(df_path, 'simple_smile', ['AU12']))
        elif 'sad' in df_path.lower():
            sad_df = pd.DataFrame(process_expression(df_path, 'combo_sad', ['AU1', 'AU4']))
        elif 'anger' in df_path.lower():
            anger_df = pd.DataFrame(process_expression(df_path, 'combo_anger', ['AU4', 'AU5', 'AU7']))
        elif 'fear' in df_path.lower():
            fear_df = pd.DataFrame(process_expression(df_path, 'combo_fear', ['AU1', 'AU2', 'AU4']))
        else:
            print('Wrong Expression or wrong name!')

    expression_df = smile_df.merge(simple_smile_df, how='left').merge(sad_df, how='left').merge(anger_df, how='left').merge(fear_df, how='left')

    clinic_df = pd.read_excel('/home/galvari/BAYAN/DATA/DataFrames/ClinicalCategoryDataframe.xlsx')
    clinic_df.replace({'Type': {"ASD": 1, 'NT': 0}, 'Gender': {'M': 1, 'F': 0}}, inplace=True)
    clinic_df.rename({'record': 'subject', 'Type': 'ASD', 'Gender': 'male'}, axis=1, inplace=True)
    clinic_df.drop(['Location', 'Notes', 'Exists'], axis=1, inplace=True)
    expression_df['subject'] = expression_df['subject'].astype(int)
    clinic_df['subject'] = clinic_df['subject'].astype(int)
    clinic_df['male'] = clinic_df['male'].astype('category')
    clinic_df['ASD'] = clinic_df['ASD'].astype('category')

    final_df = pd.merge(expression_df, clinic_df, on='subject')
    final_df.to_csv(save_path, index=False)

    return final_df




def CreateExpressionFeaturesDataFrame(list_dfs_path, save_path, handle_cropped=False, merge_clinical=False, clinical_path='/home/galvari/BAYAN/DATA/DataFrames/ClinicalCategoryDataframe.xlsx'):
    all_expression_dfs = []
    for df_path in list_dfs_path:
        subjects = []
        activities = []
        cameras = []
        expression_rates = []
        durations = []
        au_ints = []
        au_proms = []
        au_proms_std = []
        au_areas = []
        au_durs = []
        expression_peaks = []
        expression_nums = []

        unique_subjects = set()
        file_count = 0

        for (dirpath, dirnames, filenames) in os.walk(df_path):
            for file in filenames:
                file_count += 1
                file_name = file.split('.')[0]
                sub = file_name.split('_')[0]
                sub = str(sub)  # Convert to string without stripping leading zeros
                unique_subjects.add(sub)
                act = file_name.split('_')[1]
                itw = file_name.split('_')[-1]

                df = pd.read_csv(os.path.join(dirpath, file), index_col=0)
                df.columns = df.columns.str.strip()
                print(df_path)
                print('{}:'.format(df_path.split('/')[-1]), file_name)
                df.columns = [col.strip('_') for col in df.columns]

                # Determine the correct column suffix by checking if at least one column includes '_combo' or '_simple'
                column_suffix = None
                for col in df.columns:
                    if col.endswith('_combo'):
                        column_suffix = 'combo'
                        break
                    elif col.endswith('_simple'):
                        column_suffix = 'simple'
                        break

                if not column_suffix:
                    print(f"Skipping file {file_name} due to missing required columns.")
                    continue  # Skip files that do not match either 'combo' or 'simple'

                au_int = float(np.mean(df[f'int_avg_{column_suffix}']))
                au_prom = float(np.mean(df[f'prominence_{column_suffix}']))
                au_prom_std = float(np.std(df[f'prominence_{column_suffix}']))
                au_area = float(np.mean(df[f'area_{column_suffix}']))
                au_dur = float(np.mean(df[f'duration_{column_suffix}']))
                expression_peak = list(df[f'peak_lp_{column_suffix}'])
                expression_n = len(df)

                if len(df) != 0:
                    tot_time = df['timestamp'].iloc[0]
                    expression_rate = len(df) * 60 / tot_time
                else:
                    expression_rate = 0
                    tot_time = np.nan

                # Append values to the lists
                subjects.append(sub)
                activities.append(act)
                cameras.append(itw)
                au_ints.append(au_int)
                au_proms.append(au_prom)
                au_areas.append(au_area)
                au_durs.append(au_dur)
                au_proms_std.append(au_prom_std)
                expression_peaks.append(expression_peak)
                expression_nums.append(expression_n)
                expression_rates.append(expression_rate)
                durations.append(tot_time)

        print(f'Unique subjects processed: {list(unique_subjects)}')
        print(f'Total number of files processed: {file_count}')
        
        # Determine expression type
        if 'smiling' in df_path.lower():
            if 'combo' in df_path.lower():
                expression_type = 'combosmile'
            elif 'simple' in df_path.lower():
                expression_type = 'simplesmile'
        elif 'sading' in df_path.lower():
            if 'combo' in df_path.lower():
                expression_type = 'combosad'
            elif 'simple' in df_path.lower():
                expression_type = 'simplesad'
        elif 'fearing' in df_path.lower():
            if 'combo' in df_path.lower():
                expression_type = 'combofear'
            elif 'simple' in df_path.lower():
                expression_type = 'simplefear'
        elif 'angering' in df_path.lower():
            if 'combo' in df_path.lower():
                expression_type = 'comboanger'
            elif 'simple' in df_path.lower():
                expression_type = 'simpleanger'
        else:
            print('Is another expression or the name is WRONG!')
            continue

        expression_dict = {
            'subject': subjects,
            'task': activities,
            'camera': cameras,
            'duration': durations,
            f'{expression_type}_num': expression_nums,
            f'{expression_type}_rate': expression_rates,
            f'{expression_type}_peaks': expression_peaks,
            f'{expression_type}_au_int': au_ints,
            f'{expression_type}_au_prom': au_proms,
            f'{expression_type}_au_prom_std': au_proms_std,
            f'{expression_type}_au_area': au_areas,
            f'{expression_type}_au_dur': au_durs
        }

        expression_df = pd.DataFrame(expression_dict)
        expression_df.update(expression_df.filter(regex='(_rate|_num)$').fillna(0))
        all_expression_dfs.append(expression_df)

        # Debug print statement for each expression dataframe
        print(f"Processed dataframe for {expression_type}:")
        print(expression_df['subject'].unique())

    # Merge all expression dataframes
    if all_expression_dfs:
        combined_df = pd.concat(all_expression_dfs, ignore_index=True)
        combined_df = combined_df.groupby(['subject', 'task', 'camera'], as_index=False).first()

        # Handle cropped versions
        if handle_cropped:
            combined_df['cropped'] = 'no'
            cropped_subjects = []
            for sub in combined_df['subject'].unique():
                cropped_sub = '1000'+ str(int(sub))
                if cropped_sub in combined_df['subject'].values:
                    combined_df.loc[combined_df['subject'] == cropped_sub, 'cropped'] = 'yes'
                    combined_df.loc[combined_df['subject'] == cropped_sub, 'subject'] = sub
                    cropped_subjects.append(cropped_sub)
            combined_df = combined_df[~combined_df['subject'].isin(cropped_subjects)]

        # Debug print before merging with clinical data
        print(f"Combined dataframe subjects before merging with clinical data:")
        print(combined_df['subject'].unique())
        print(f"Total subjects before merge: {combined_df['subject'].nunique()}")

        if merge_clinical:
            clinic_df = pd.read_excel(clinical_path)
            clinic_df.replace({'Type': {"ASD": 1, 'NT': 0}, 'Gender': {'M': 1, 'F': 0}}, inplace=True)
            clinic_df.rename({'record': 'subject', 'Type': 'ASD', 'Gender': 'male'}, axis=1, inplace=True)
            clinic_df.drop(['Location', 'Notes', 'Exists'], axis=1, inplace=True)
            combined_df['subject'] = combined_df['subject'].astype(str)
            clinic_df['subject'] = clinic_df['subject'].astype(str)
            clinic_df['male'] = clinic_df['male'].astype('category')
            clinic_df['ASD'] = clinic_df['ASD'].astype('category')

            # Perform outer merge to identify missing subjects
            final_df = pd.merge(combined_df, clinic_df, on='subject', how='outer', indicator=True)

            # Debug print after merging with clinical data
            print(f"Final dataframe subjects after merging with clinical data:")
            print(final_df['subject'].unique())
            print(f"Total subjects after merge: {final_df['subject'].nunique()}")

            # Identify missing subjects
            missing_subjects = final_df[final_df['_merge'] != 'both']
            print(f"Subjects missing clinical data: {missing_subjects[missing_subjects['_merge'] == 'left_only']['subject'].unique()}")
            print(f"Subjects missing in combined data: {missing_subjects[missing_subjects['_merge'] == 'right_only']['subject'].unique()}")

            # Drop indicator column before saving
            final_df.drop(columns=['_merge'], inplace=True)
        else:
            final_df = combined_df

        final_df.to_csv(save_path, index=False)

        return final_df
    else:
        print("No valid dataframes to merge.")
        return None


    
    
'''   

def CreateExpressionFeaturesDataFrameOld(list_dfs_path, save_path):
    
    for df_path in list_dfs_path:

        subjects = []
        activities = []
        cameras = []

        expression_rates = []
        durations = []

        au_ints = []
        au_proms = []
        au_proms_std = []
        au_areas=[]
        au_durs=[]

        expression_peaks = []
        expression_nums = []

        expression_dict = dict()


        for (dirpath, dirnames, filenames) in os.walk(df_path):

            for file in filenames:
                file_name = file.split('.')[0]

                sub = file_name.split('_')[0]
                act = file_name.split('_')[1]
                itw = file_name.split('_')[-1]



                df = pd.read_csv(dirpath+'/'+file, index_col=0)
                df.columns = df.columns.str.strip()
                print(df_path)
                print('{}:'.format(df_path.split('/')[-1]), file_name)

                au_int = float(np.mean(df['int_avg_combo']))
                au_prom = float(np.mean(df['prominence_combo']))
                au_prom_std = float(np.std(df['prominence_combo']))
                au_area = float(np.mean(df['area_combo']))
                au_dur = float(np.mean(df['duration_combo']))

                expression_peak = list(df['peak_lp_combo'])
                expression_n = len(df)

                if len(df) != 0:
                    tot_time = df['timestamp'].iloc[0]
                    expression_rate = len(df)*60/tot_time
                else:
                    expression_rate = 0 
                    tot_time = np.nan

                # create dataframe

                subjects.append(sub)
                activities.append(act)
                cameras.append(itw)
                au_ints.append(au_int)
                au_proms.append(au_prom)
                au_areas.append(au_area)
                au_durs.append(au_dur)
                au_proms_std.append(au_prom_std)

                expression_peaks.append(expression_peak)
                expression_nums.append(expression_n)
                expression_rates.append(expression_rate)
                durations.append(tot_time)



        expression_final = [subjects,activities,cameras,durations,expression_nums,expression_rates, expression_peaks,
        au_ints,au_proms,au_proms_std,au_areas, au_durs]

        if 'smiling' in df_path.lower():

            smiling_names = ['subject','task','camera','duration','smile_num','smile_rate', 'smile_peaks',
            'smile_au_int','smile_au_prom','smile_au_prom_std','smile_au_area', 'smile_au_dur']

            smiling_dict = dict(zip(smiling_names, expression_final))
            smiling_df = pd.DataFrame(smiling_dict)
            smiling_df.fillna(0, inplace = True)

        elif 'sading' in df_path.lower():

            sading_names = ['subject','task','camera','duration','sad_num','sad_rate', 'sad_peaks',
            'sad_au_int','sad_au_prom','sad_au_prom_std','sad_au_area', 'sad_au_dur']

            sading_dict = dict(zip(sading_names, expression_final))
            sading_df = pd.DataFrame(sading_dict)
            sading_df.fillna(0, inplace = True)

        elif 'fearing' in df_path.lower():

            fearing_names = ['subject','task','camera','duration','fear_num','fear_rate', 'fear_peaks',
            'fear_au_int','fear_au_prom','fear_au_prom_std','fear_au_area', 'fear_au_dur']

            fearing_dict = dict(zip(fearing_names, expression_final))
            fearing_df = pd.DataFrame(fearing_dict)
            fearing_df.fillna(0, inplace = True)


        elif 'angering' in df_path.lower():

            angering_names = ['subject','task','camera','duration','anger_num','anger_rate', 'anger_peaks',
            'anger_au_int','anger_au_prom','anger_au_prom_std','anger_au_area', 'anger_au_dur']

            angering_dict = dict(zip(angering_names, expression_final))
            angering_df = pd.DataFrame(angering_dict)
            angering_df.fillna(0, inplace = True)

        else:
            print('Is another expression or the name is WRONG!')



    ### MERGE THEM
    expression_df = smiling_df.merge(sading_df, how='left').merge(angering_df, how='left').merge(fearing_df, how='left')

    clinic_df = pd.read_excel('/home/galvari/BAYAN/DATA/DataFrames/ClinicalCategoryDataframe.xlsx')
    clinic_df.replace({'Type':{"ASD": 1, 'NT':0},'Gender':{'M':1,'F':0}}, inplace=True)
    clinic_df.rename({'record':'subject','Type':'ASD','Gender':'male'}, axis=1,inplace = True)
    clinic_df.drop(['Location', 'Notes','Exists'], axis=1, inplace = True)
    expression_df['subject']= expression_df['subject'].astype(int)
    clinic_df['subject']= clinic_df['subject'].astype(int)
    clinic_df['male']= clinic_df['male'].astype('category')
    clinic_df['ASD']= clinic_df['ASD'].astype('category')

    final_df = pd.merge(expression_df, clinic_df, on = 'subject')

    final_df.to_csv(save_path, index = False)

    return final_df


'''