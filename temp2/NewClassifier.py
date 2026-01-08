from sklearn.model_selection import LeaveOneOut
from sklearn.covariance import OAS
import numpy as np
import pandas as pd
from Files import get_file, save_pickle_file
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import DataAlgo
import matplotlib.pyplot as plt

PCA_SHIMMER_JITTER2 = False
LIST_PICKLE_FILE = 'list_of_important_tables.pickle'

# Prepare Data
ADOS_SCORES = pd.read_excel(
    'ADOS scores for Omer 21-11-22.xlsx', engine='openpyxl').set_index('task')
AS_TYPE = {'record': 'int32', 'Type': 'string', 'Gender': 'string'}
RECORD2TYPE = pd.read_excel('record2type.xlsx', engine='openpyxl').astype(
    AS_TYPE).set_index('record')
DATA = data = pd.read_excel('New vocal.xlsx', engine='openpyxl').astype({
    'record': 'int32'}).set_index('record')
PCA_SHIMMER_JITTER = False
# data.reset_index(inplace=True)
# data.sort_values(by=['TB_HNR'],inplace=True)
if PCA_SHIMMER_JITTER:
    data_copy = DataAlgo.drop_all_cols_with(data,['shimmer','jitter'])
    # print(DataAlgo.PCA_on_Tuppels([('shimmer',2),('jitter',2)],data))
    pca_list = [data_copy]
    # pca_list = []
    for task in ['tb','story','emo']:
        pca_list.append(DataAlgo.PCA_on_Tuppels([([task,'shimmer'],2),([task,'jitter'],2)],data))
    data = pd.concat(pca_list,axis=1)
print(str(data))
# print(data)
# exit(1)
# data.reset_index(inplace=True)
# data.sort_index(inplace=True)
# data.to_csv('NotSortedData.csv')
# data.sort_values('Type',inplace=True,ascending=False)
# data.to_csv('SortedData.csv')
if 'Type' not in list(data.columns):
    input()
    for index, row in data.iterrows():
        data.at[index, 'Type'] = RECORD2TYPE.at[index, 'Type']
if 'Gender' not in list(data.columns):
    input()
    for index, row in data.iterrows():
        data.at[index, 'Gender'] = RECORD2TYPE.at[index, 'Gender']
scores = ['A2', 'A9', 'A10']
for score in scores:
    for index, row in data.iterrows():
        if int(index) in list(ADOS_SCORES.head()):

            if ADOS_SCORES.at[score, int(index)] not in range(0, 10):
                ADOS_SCORES.at[score, int(index)] = np.nan
            if not pd.isna(ADOS_SCORES.at[score, int(index)]):

                data.at[index, score] = ADOS_SCORES.at[score,
                                                       int(index)]
            else:
                print(index, ' missing')
                data.at[index, score] = np.nan
        else:
            print(index, ' missing')
            data.at[index, score] = np.nan
if PCA_SHIMMER_JITTER2:
    data_copy1 = DataAlgo.drop_all_cols_with(data, ['shimmer', 'jitter'])
    # print(DataAlgo.PCA_on_Tuppels([('shimmer',2),('jitter',2)],data))
    pca_list = []
    for task in ['tb', 'story', 'emo']:
        pca_list.append(DataAlgo.PCA_on_Tuppels(
            [([task, 'shimmer'], 2), ([task, 'jitter'], 2)], data))
    for df in pca_list:
        print(df)
        row_counter = 0
        df.to_csv('df'+str(row_counter)+'.csv')
        for index, row in data.iterrows():
            for col in list(df.columns):
                data_copy1.at[index, col] = df.at[row_counter, col]
            row_counter += 1
    data = data_copy1
LOCATION = True
if LOCATION:
    for index, row in data.iterrows():
        data.at[index, 'Location'] = RECORD2TYPE.at[index, 'Location']
data.at[230,'Location'] = 'New_Lab'
# data.to_csv('POSTDATA.csv')
# input()
CLASSIFY = False
data = data[data['Gender'] == 'M']
data = data.sample(frac = 1)
# print(data['Gender'],data['Location'])
only_diffs = DataAlgo.diffs_by_tasks(data, ['tb', 'story', 'emo'])
data_w_diffs = pd.concat([data,only_diffs], axis=1).dropna()
data_w_diffs.to_csv('data with diff.csv')
class_data = data.drop(scores, axis=1).dropna()
class_data.to_csv('class_data.csv')
print(class_data.groupby('Type').count())
# print(class_data[class_data['Type']=='NT'])
# print(class_data[class_data['Type']=='ASD'])
# print(data[class_data['Type']=='NT'])
# print(data[class_data['Type']=='ASD'])

regress_data = data.drop('Type', axis=1).dropna()

regress_data_w_diffs = pd.concat([regress_data,only_diffs],axis=1).dropna()
X_reg = regress_data.drop(scores+['Gender'],axis=1)
y_reg = regress_data['A2']
X_reg_diff = regress_data_w_diffs.drop(scores+['Gender'],axis=1)
y_reg_diff = regress_data_w_diffs['A2']
total_results = []

X = class_data.drop(['Type','Gender'], axis=1)
X_diffs = data_w_diffs.drop(
    scores+['Type','Gender'], axis=1).dropna()
if LOCATION:
    X.drop('Location',axis=1,inplace=True)
    X_diffs.drop('Location',axis=1,inplace=True)    
    X_reg_diff.drop('Location',axis=1,inplace=True)
    X_reg.drop('Location',axis=1,inplace=True)
X.to_csv('X.csv')
# print(X)
runs = 10
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
# data_w_diffs['Location'] = pd.factorize(data_w_diffs['Location'])[0].astype(np.uint16)
# print(data_w_diffs['Location'])
# best = DataAlgo.statistics_feature_reduction(X_diffs,data_w_diffs['Location'])
NT_mask  = data_w_diffs['A2']<1
print(X_diffs[NT_mask],data_w_diffs[NT_mask]['Location'])
X_diffs = data_w_diffs[NT_mask].drop(scores+['Type','Gender','Location'], axis=1).dropna()
X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'mean')
X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'formant')
X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'fSD')
X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'vtl')
X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'fdisp')
X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'mff')
X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'HNR')
X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'shimmer')
X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'jitter')
X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'delta')
DataAlgo.RandomizationTest(X_diffs,data_w_diffs[NT_mask]["Location"]=="Old_Lab")
exit(1)
# X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'pc')
# X_diffs = DataAlgo.drop_all_cols_with(X_diffs,'pc')
# X_diffs.reset_index(drop=True,inplace=True)
print(data_w_diffs[NT_mask].groupby('Location').count())
print(data_w_diffs[NT_mask]['Location']=='Old_Lab')
# X_diffs = pd.DataFrame(np.random.uniform(0,1,size=(X_diffs.shape[0]*4, 20)), columns=list('ABCDEFGHIJKLMNOPQRST'))
# data_w_diffs['randNumCol'] = np.random.randint(0, 2, data_w_diffs.shape[0])
y =  pd.DataFrame(data = np.random.randint(0, 2, data_w_diffs.shape[0]).T,columns=['Type'])
# y=data_w_diffs['randNumCol']
series =data_w_diffs[NT_mask]["Location"]=="Old_Lab" 
# series=series.append(series)
# series=series.append(series)
# series = data_w_diffs['Type']
# rand_array = np.random.randint(0, 2, X_diffs.shape[0])
# rand_array = np.empty(X_diffs.shape[0])
# rand_array.fill(1)
# rand_array[:int(X_diffs.shape[0]/2)+1].fill(2)
# # print(rand_array)
# np.random.shuffle(rand_array)
# # print(rand_array)
# series = pd.Series(rand_array)
# y = pd.DataFrame()
# y['Class']=series
# print(data_w_diffs['randNumCol'] )
print(y)
# print(str(data_w_diffs))
# print(DataAlgo.LOOcross_val_score(DataAlgo.svm.SVC(),X_diffs,series))
from sklearn.ensemble import RandomForestClassifier
# DataAlgo.append_table_to_pickle(('statstical quality test (diffs)'+' date_time',DataAlgo.test_statistics_reduction_quality(X_diffs[NT_mask],data_w_diffs[NT_mask]['Location']=='Old_Lab',from_last=20,clfs = DataAlgo.LogisticRegression(solver='liblinear'))),LIST_PICKLE_FILE)
# DataAlgo.TestApp([('statstical quality test (diffs)'+' date_time',DataAlgo.test_statistics_reduction_quality(X_diffs[NT_mask],data_w_diffs[NT_mask]['Location']=='Old_Lab',from_last=20,clfs = DataAlgo.LogisticRegression(solver='liblinear')))])
from sklearn.ensemble import RandomForestClassifier
DataAlgo.append_table_to_pickle(('statstical quality test (diffs)'+' date_time RealData realY 5fold',DataAlgo.test_statistics_reduction_quality(X_diffs,series,from_last=7,clfs =[DataAlgo.LogisticRegression()],cv=5)),LIST_PICKLE_FILE)
DataAlgo.append_table_to_pickle(('statstical quality test (diffs)'+' date_time RealData realY LOO',DataAlgo.test_statistics_reduction_quality(X_diffs,series,from_last=7,clfs =[DataAlgo.LogisticRegression()])),LIST_PICKLE_FILE)
# print(str(X_diffs))
# DataAlgo.append_table_to_pickle(('statstical quality test (diffs)'+' date_time',DataAlgo.test_statistics_reduction_quality(X_diffs,data_w_diffs['Location']=='Old_Lab',from_last=20)),LIST_PICKLE_FILE)
print(X_diffs)
exit(1)
# test_permute_importane(X_diffs)
# cols = ['tb_shimmer_pc1','Story_stdevIntensity','emo_shimmer_pc1']
# print(DataAlgo.LOO_multi_classify(X_diffs[cols],data_w_diffs['Type'],runs=10))
# exit(1)
from datetime import datetime
now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y, %H:%M")
date_time = ' ('+date_time+')'
RF_FEATURE_TEST = False
if RF_FEATURE_TEST:
    feature_compare = pd.DataFrame()
    ls = DataAlgo.RF_feature_reduction(X, class_data['Type'], steps=[
                                       *range(X.shape[1], 31, -10)]+[*range(30, 11, -5)]+[*range(10, 0, -1)], compare_table=feature_compare)
    DataAlgo.append_table_to_pickle(
        ('RF feature compare'+date_time, feature_compare), LIST_PICKLE_FILE)
    # feature_compare = pd.DataFrame()
    # ls = DataAlgo.RF_feature_reduction(X_diffs, data_w_diffs['Type'], steps=[
    #                                    *range(X_diffs.shape[1], 31, -10)]+[*range(30, 11, -5)]+[*range(10, 0, -1)], compare_table=feature_compare)
    # DataAlgo.append_table_to_pickle(
    #     ('RF feature compare (&diffs)'+date_time, feature_compare), LIST_PICKLE_FILE)
    exit(1)
wo_diffs_steps = [*range(X.shape[1], 31, -10)]+[*range(30, 11, -5)]+[*range(10, 0, -1)] # hardcoded from tables
w_diffs_steps = [*range(X_diffs.shape[1], 31, -10)]+[*range(30, 11, -5)]+[*range(10, 0, -1)] # hardcoded from tables

HISTOGRAM = False
if HISTOGRAM:
    labels = {'F0':'(Hz)', 'Intensity':'(dB)','HNR':'(dB)'}
    rf_features = DataAlgo.RF_feature_reduction(X_diffs,data_w_diffs['Type'],steps = w_diffs_steps[:-12])
    statistical_features = DataAlgo.statistics_feature_reduction(X_diffs,data_w_diffs['Type'],features=20)
    all_features = list(set(rf_features).union(statistical_features))
    print(all_features)
    for col in all_features:
        all = data_w_diffs
        ASD_mask = all['Type'] == 'ASD'
        NT_mask = all['Type'] == 'NT'
        if col == 'Type' or col =='Gender':
            continue
        # all[NT_mask][col].plot.hist(bins=10,alpha=0.5,label='NT',legend= True)
        # all[ASD_mask][col].plot.hist(bins=10,alpha=0.5,label='ASD',legend= True)
        all.pivot(columns='Type', values=col).plot.hist(bins=10,alpha=0.5)
        counts = all["Type"].value_counts()
        plt.title(col + "(ASD: "+str(counts['ASD'])+", NT: "+str(counts['NT'])+")")
        mean_nt = all[NT_mask][col].mean()
        mean_asd = all[ASD_mask][col].mean()
        plt.text(mean_nt,plt.ylim()[1]/2,'mean NT: '+"{:.3f}".format(mean_nt),rotation=90)
        plt.text(mean_asd,plt.ylim()[1]/2,'mean ASD: '+"{:.3f}".format(mean_asd),rotation=90)
        plt.axvline(mean_nt, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(mean_asd, color='b', linestyle='dotted', linewidth=1)
        ax = plt.gca()
        for key in labels.keys():
            if key in col:
                ax.set_xlabel(labels[key])
        from os.path import exists
        from os import mkdir
        if not exists('Featureshistograms'):
            mkdir('Featureshistograms')
        plt.savefig('Featureshistograms/'+col)
        plt.close()
    exit(1)
rwo_diffs_steps = [*range(X_reg.shape[1], 31, -10)]+[*range(30, 11, -5)]+[*range(10, 0, -1)] # hardcoded from tables
rw_diffs_steps = [*range(X_reg_diff.shape[1], 31, -10)]+[*range(30, 11, -5)]+[*range(10, 0, -1)] # hardcoded from tables
RF_FEATURE_TEST_REG = False
all_results = pd.read_csv('combination compare.csv').set_index('#comb')
best_featrues = DataAlgo.statistics_feature_reduction(X_diffs,data_w_diffs['Type'],features = 30)
where_feature_at = dict()
feature_series = dict()
for bf in best_featrues:
    where_feature_at[bf] = list()
for index,row in all_results.iterrows():
    # print(row)
    # input()
    for col in list(all_results.columns):
        if col == 'MeanAcc':
            continue
        where_feature_at[row[col]].append(int(index))
results_serires = pd.DataFrame()
for best in best_featrues:
    feature_series[best] = all_results.iloc[where_feature_at[best]]['MeanAcc']
    results_serires[best] = all_results.iloc[where_feature_at[best]]['MeanAcc']
print(str(results_serires.mean().sort_values(ascending=False)))
results_serires = results_serires.reindex(results_serires.mean().sort_values(ascending=False).index, axis=1)
results_serires.plot.box()
plt.axhline(results_serires.mean().mean(), color='b', linestyle='--')

plt.show()
exit(1)
results_serires = pd.DataFrame()
for best in best_featrues:
    feature_series[best] = all_results.iloc[where_feature_at[best]]['MeanAcc']
    results_serires[best] = all_results.iloc[where_feature_at[best]]['MeanAcc']
print(results_serires)
    # results_serires.plot.box()DataAlgo.statistical_features_combinations_test(X_diffs,data_w_diffs['Type'])
exit(1)

if RF_FEATURE_TEST_REG:
    print('reg')
    input()
    feature_compare = pd.DataFrame()
    ls = DataAlgo.RF_feature_reduction(X_reg, y_reg, steps=rwo_diffs_steps, compare_table=feature_compare)
    DataAlgo.append_table_to_pickle(
        ('RF feature compare ON A2'+date_time, feature_compare), LIST_PICKLE_FILE)
    feature_compare = pd.DataFrame()
    ls = DataAlgo.RF_feature_reduction(X_reg_diff,y_reg_diff, steps=rw_diffs_steps, compare_table=feature_compare)
    DataAlgo.append_table_to_pickle(
        ('RF feature compare (&diffs) ON A2'+date_time, feature_compare), LIST_PICKLE_FILE)
    # DataAlgo.TestApp([('RF feature compare',feature_compare)])
    exit(1)
REG_QUALITY_TEST = True
if REG_QUALITY_TEST:
    print('reg quality')
    print(y_reg_diff)
    input()
    # DataAlgo.append_table_to_pickle(('A2 statstical quality test (diffs)'+date_time,DataAlgo.test_statistics_reduction_quality(X_reg_diff,y_reg_diff)),LIST_PICKLE_FILE)
    # DataAlgo.append_table_to_pickle(('A2 statstical quality test'+date_time,DataAlgo.test_statistics_reduction_quality(X_reg,y_reg)),LIST_PICKLE_FILE)
    # DataAlgo.append_table_to_pickle(('A2 RF quality test (diffs)X'+date_time,DataAlgo.test_RF_reduction_quality(X_reg_diff,y_reg_diff,steps=rw_diffs_steps,from_last=15)),LIST_PICKLE_FILE)
    # DataAlgo.append_table_to_pickle(('A2>0 statstical quality test (diffs)'+date_time,DataAlgo.test_statistics_reduction_quality(X_reg_diff,y_reg_diff>0)),LIST_PICKLE_FILE)
    # DataAlgo.append_table_to_pickle(('A2>0 statstical quality test'+date_time,DataAlgo.test_statistics_reduction_quality(X_reg,y_reg)),LIST_PICKLE_FILE)
    # DataAlgo.append_table_to_pickle(('A2>0 RF quality test (diffs)X'+date_time,DataAlgo.test_RF_reduction_quality(X_reg_diff,y_reg_diff>0,steps=rw_diffs_steps,from_last=15)),LIST_PICKLE_FILE)
    y_reg_df = pd.DataFrame()
    y_reg_df['A2Type']= y_reg_diff>0
    # y_reg_df = pd.DataFrame(y_reg_more,columns=['Type'])
    # print(y_reg_df)
    DataAlgo.cross_val_feature_check_RF(X_reg_diff,y_reg_df,rw_diffs_steps)

    # DataAlgo.cross_val_feature_check(X_reg_diff,y_reg_df)
    # DataAlgo.append_table_to_pickle(('A2>0 quality pca statistical test diffs'+date_time,DataAlgo.test_statistical_reduction_quality_pca(X_reg_diff,y_reg_diff>0)),LIST_PICKLE_FILE)
    # DataAlgo.append_table_to_pickle(('A2 RF quality testX'+date_time,DataAlgo.test_RF_reduction_quality(X_reg,y_reg,steps =rwo_diffs_steps,from_last=15)),LIST_PICKLE_FILE)
    # DataAlgo.append_table_to_pickle(('A2 quality pca RF test'+date_time,DataAlgo.test_RF_reduction_quality_pca(X,class_data['Type'],from_last = 10)),LIST_PICKLE_FILE)
    # DataAlgo.append_table_to_pickle(('A2 quality pca statistical test'+date_time,DataAlgo.test_statistical_reduction_quality_pca(X,class_data['Type'])),LIST_PICKLE_FILE)
    # DataAlgo.append_table_to_pickle(('A2 quality pca RF test diffs'+date_time,DataAlgo.test_RF_reduction_quality_pca(X_diffs,data_w_diffs['Type'],from_last = 10)),LIST_PICKLE_FILE)
    # DataAlgo.append_table_to_pickle(('A2 quality pca statistical test diffs'+date_time,DataAlgo.test_statistical_reduction_quality_pca(X_diffs,data_w_diffs['Type'])),LIST_PICKLE_FILE)

    exit(1)
stepss=[*range(100, 21, -10)]+[*range(20,
 11, -5)] + \
    [*range(10, 5, -1)] 

if CLASSIFY:
    input()
    # for task, result in DataAlgo.classify_by_task(class_data.drop(['Type',  'Gender'], axis=1), class_data['Type'], ['tb', 'story', 'emo'], DataAlgo.RF_feature_reduction, runsp=5):
    #     DataAlgo.append_table_to_pickle((task, result), LIST_PICKLE_FILE)
    # input()
    for task in ['tb', 'story', 'emo']:
        print(task)
        feature_compare = pd.DataFrame()
        ls = DataAlgo.RF_feature_reduction(X[DataAlgo.get_all_cols_with(X,task)], class_data['Type'], steps=[
                                        *range(100, 21, -10)]+[*range(20, 11, -5)]+[*range(10, 0, -1)], compare_table=feature_compare)
        DataAlgo.append_table_to_pickle(
            ('RF feature compare '+task+" "+date_time, feature_compare), LIST_PICKLE_FILE)
        DataAlgo.test_dataframe_vis()
rf_feature = DataAlgo.RF_feature_reduction(X, class_data['Type'])
if CLASSIFY and False:
    DataAlgo.append_table_to_pickle(('all tasks forest fr', DataAlgo.CV_multi_classify(
        X[rf_feature], class_data['Type'], print_results=False, runs=runs)), LIST_PICKLE_FILE)
# exit(1)
y = data_w_diffs['Type']
print(y)
# exit(1)
# DataAlgo.cross_val_feature_check_RF(X_diffs,data_w_diffs['Type'],stepss)
# DataAlgo.cross_val_feature_check(X_diffs,data_w_diffs['Type'])
# DataAlgo.cross_val_feature_check_RF(X,class_data['Type'],stepss)
# DataAlgo.cross_val_feature_check(X,class_data['Type'])
print('RF reduction for alltask+diffs')
input()
# DataAlgo.test_permute_importane(X=X_diffs,y=y)
# DataAlgo.append_table_to_pickle(('quality RF test'+date_time,DataAlgo.test_RF_reduction_quality(X_diffs,y)),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('statstical quality test (diffs)'+date_time,DataAlgo.test_statistics_reduction_quality(X_diffs,y)),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('statstical quality test'+date_time,DataAlgo.test_statistics_reduction_quality(X,class_data['Type'])),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('RF quality test (diffs)X'+date_time,DataAlgo.test_RF_reduction_quality(X_diffs,y,steps=w_diffs_steps,from_last=15)),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('RF quality testX'+date_time,DataAlgo.test_RF_reduction_quality(X,class_data['Type'],steps =wo_diffs_steps,from_last=15)),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('quality pca RF test'+date_time,DataAlgo.test_RF_reduction_quality_pca(X,class_data['Type'],from_last = 10)),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('quality pca statistical test'+date_time,DataAlgo.test_statistical_reduction_quality_pca(X,class_data['Type'])),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('quality pca RF test diffs'+date_time,DataAlgo.test_RF_reduction_quality_pca(X_diffs,data_w_diffs['Type'],from_last = 10)),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('quality pca statistical test diffs'+date_time,DataAlgo.test_statistical_reduction_quality_pca(X_diffs,data_w_diffs['Type'])),LIST_PICKLE_FILE)
exit(1)
# RF_fl = DataAlgo.RF_Iter_feature_reduction(X,class_data['Type'],steps =wo_diffs_steps)
# RF_fl = RF_fl[-5:]
# feaures = RF_fl[0]
# fig,axes = plt.subplots(len(feaures),1)
# axes = axes.ravel()
# for i,feature in enumerate(feaures):
#     X[feature].plot(kind='kde', ax=axes[i], secondary_y=True)
#     axes[i].set_title(feature)
# plt.show()
# print(RF_fl)

# print(DataAlgo.statistics_feature_reduction(X_diffs, y, features=4))
# exit(1)
# DataAlgo.test_LogisticRegression(X_diffs,y,k=6)
# DataAlgo.test_RF_reduction_quality_pca(X_diffs,y)
# feaures = DataAlgo.statistics_feature_reduction(X_diffs, y, features=6)
# DataAlgo.TestApp([('LDAshrinakge')])
# loo = LeaveOneOut()
# for i in range(0, 10):
#     counter = 0
#     for train, test in loo.split(X_diffs):
#         X_train, X_test = X_diffs[feaures].iloc[train], X_diffs[feaures].iloc[test]
#         y_train, y_test = y.iloc[train], y.iloc[test]
#         # print(y.iloc[train], y.iloc[test])
#         qda = DataAlgo.QuadraticDiscriminantAnalysis()
#         qda.fit(X_train, y_train)
#         y_out = qda.predict(X_test)
#         # print(y_out,y_test)
#         counter += int(y_out == y_test)
#     # print(counter/loo.get_n_splits(X_diffs))

# print(DataAlgo.LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').get_params())
# print(DataAlgo.LinearDiscriminantAnalysis(solver='lsqr').get_params())
# exit(1)
# feaures = ['Story_stdevIntensity', 'tb_shimmer_pc1',  'emo_shimmer_pc1',
#            'tb-story_formant2-SD',  'tb-emo_stdevF0Hz',  'story-emo_pc1']
# feaures = [feaures[0]]+[feaures[3]]
# pd.concat([X_diffs[feaures],y],axis=1).to_csv('test55.csv')
# print(X_diffs[feaures])
# print(y)
# DataAlgo.TestApp(
#     [('6-features', DataAlgo.CV_multi_classify(X_diffs[feaures], y, runs=runs))])
# print(DataAlgo.CV_model_classify(X_diffs[feaures],y=='NT',DataAlgo.LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'),runs=5))
# exit(1)
# pca_n = 6
# pca = PCA(n_components=pca_n)
# Xpca = pca.fit_transform(X=StandardScaler().fit_transform(X_diffs[feaures]))
# features_list2 = []
# for i in range(1,pca_n+1):
#     features_list2.append('p'+str(i))
# Xpca= pd.DataFrame(Xpca,columns = features_list2)
# print(Xpca)
# DataAlgo.append_table_to_pickle(('quality pca RF test'+date_time,DataAlgo.test_RF_reduction_quality_pca(X,class_data['Type'],from_last = 10)),LIST_PICKLE_FILE)
DataAlgo.append_table_to_pickle(('quality pca statistical test'+date_time,DataAlgo.test_statistical_reduction_quality_pca(X,class_data['Type'])),LIST_PICKLE_FILE)

input()
pca_n = 3
pca = PCA(n_components=pca_n)
# X_diffs2 = (X_diffs-X_diffs.mean())/X_diffs.std()
feaures = DataAlgo.statistics_feature_reduction(X,class_data['Type'],features=6)
Xpca = pca.fit_transform(StandardScaler().fit_transform(X[feaures]))
DataAlgo.TestApp([('pca 6',DataAlgo.CV_multi_classify(X[feaures],class_data['Type'],runs=5))])
exit(1)
pca_details = pd.DataFrame(columns=['feature']).set_index(['feature'])
pca_weights = pd.DataFrame(pca.components_.transpose())
for i, feature in enumerate(X_diffs2[feaures].head()):
    pca_details.at[feature, 'p1'] = pca_weights.at[i, 0]
    pca_details.at[feature, 'p2'] = pca_weights.at[i, 1]
# pca_details = pca_details.transpose()
print(pca_details)
pca_details.plot.barh(rot=0, subplots=True)
plt.show()
print(pca.components_, '\n')
exit(1)
df_inv = pd.DataFrame(pca.components_.transpose())
print(df_inv)
result = df_inv.dot(Xpca.transpose()).transpose()
result.columns = feaures
print(result+X_diffs[feaures].mean())
# print(X_diffs[feaures].mean())
normalized_df = (X_diffs-X_diffs.mean())/X_diffs.std()
print(X_diffs[feaures]-result+X_diffs[feaures].mean())
data_original = pd.DataFrame(np.dot(Xpca, pca.components_) + pca.mean_)
data_original.columns = feaures
print()

print(data_original-normalized_df[feaures].reset_index())
print(normalized_df[feaures])
# print(normalized_df[feaures],normalized_df[feaures].shape)
features_list2 = []
for i in range(1, pca_n+1):
    features_list2.append('p'+str(i))
# features_list2 = ['p1','p2','p3']
Xpca = pd.DataFrame(Xpca, columns=features_list2)
# Xpca = (Xpca-Xpca.mean())/Xpca.std()
DataAlgo.TestApp(
    [('pca', DataAlgo.CV_multi_classify(Xpca[features_list2], y, 5))])
# DataAlgo.TestApp([('pca',DataAlgo.CV_model_classify(clf=DataAlgo.LinearDiscriminantAnalysis(solver='lsqr'),X= Xpca[features_list2],y= y,runs=5,cross_val =5))])
# to_show = [('new',DataAlgo.CV_multi_classify(X_diffs[feaures],y,1)),('pca',DataAlgo.CV_multi_classify(Xpca[features_list2],y,1))]
# to_show+=['merged',Xpca.corrwith(X_diffs[feaures],axis=1)]
scatter = plt.scatter(Xpca['p1'], Xpca['p2'], c=(y == 'NT').astype(float))
plt.legend(handles=scatter.legend_elements()[0], labels=['NT', 'ASD'])

ri = 0
for index, row in data_w_diffs.iterrows():
    plt.annotate(index, (Xpca['p1'][ri], Xpca['p2'][ri]))
    ri += 1
plt.show()
corr_pd = pd.DataFrame(pd.concat([Xpca, X_diffs[feaures]], axis=1).corr())
corr_pd[['p1', 'p2']].plot.barh(rot=0, subplots=True)
plt.show()
# DataAlgo.TestApp(to_show)
# X_diffs[feaures].plot.kde()
# feaures = DataAlgo.statistics_feature_reduction(X,data['Type'])
# fig,axes = plt.subplots(len(features_list2),1)
# axes = axes.ravel()
# for i,feature in enumerate(features_list2):
#     Xpca[feature].plot(kind='kde', ax=axes[i], secondary_y=True)
#     axes[i].set_title(feature)
# plt.show()
exit(1)
rf_feature = DataAlgo.RF_feature_reduction(X_diffs, y)
if CLASSIFY:
    total_results.append(('all&diffs tasks', DataAlgo.LOO_multi_classify(
        X_diffs[rf_feature], y, print_results=False, runs=runs)))
rf_feature = DataAlgo.statistics_feature_reduction(X_diffs, y, features=7)
if CLASSIFY:
    total_results.append(('all&diffs2 tasks', DataAlgo.LOO_multi_classify(
        X_diffs[rf_feature], y, print_results=False, runs=runs)))
X_diffs.to_csv('X_diffs.csv')

d2 = data_w_diffs.drop(['Type'], axis=1).dropna()
X_diffs = d2.drop(scores+['Gender'], axis=1)
y = d2['A2']
rf_feature = DataAlgo.RF_feature_reduction(X_diffs, y)
if CLASSIFY:
    total_results.append(('A2 regress all&diffs tasks', DataAlgo.LOO_multi_classify(
        X_diffs[rf_feature], y, print_results=False, runs=runs)))
rf_feature = DataAlgo.RF_feature_reduction(X_diffs, y > 0)
if CLASSIFY:
    total_results.append(('A2>0 all&diffs tasks', DataAlgo.LOO_multi_classify(
        X_diffs[rf_feature], y > 0, print_results=False, runs=runs)))
rf_feature = DataAlgo.statistics_feature_reduction(X_diffs, y)
if CLASSIFY:
    total_results.append(('A2 regress all&diffs tasks2', DataAlgo.LOO_multi_classify(
        X_diffs[rf_feature], y, print_results=False, runs=runs)))
rf_feature = DataAlgo.statistics_feature_reduction(X_diffs, y > 0)
if CLASSIFY:
    total_results.append(('A2>0 all&diffs tasks2', DataAlgo.LOO_multi_classify(
        X_diffs[rf_feature], y > 0, print_results=False, runs=runs)))
# X_diffs.to_csv('X_diffs.csv')
if CLASSIFY:
    merged = pd.DataFrame()
    for task, result in total_results:
        result_copy = result.copy()
        result_copy.columns = result_copy.columns.str.replace(
            'mean', 'mean '+task)
        merged = pd.concat([merged, result_copy['mean '+task]], axis=1)
        print(merged)
    total_results.append(('merged', merged))
    save_pickle_file(total_results, where='list_of_results.pickle')
    for description, result in total_results:
        print(description)
        # DataAlgo.print_class_repot(result)
        # tabloo.show(result)

print(class_data.groupby('Type').count())
