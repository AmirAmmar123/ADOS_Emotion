import numpy as np
import pandas as pd
from Files import get_file,save_pickle_file
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import DataAlgo
import matplotlib.pyplot as plt

PCA_SHIMMER_JITTER = True
LIST_PICKLE_FILE = 'list_of_important_tables.pickle'

### Prepare Data
ADOS_SCORES = pd.read_excel('ADOS scores for Omer 21-11-22.xlsx',engine='openpyxl').set_index('task')
AS_TYPE = {'record':'int32','Type':'string','Gender':'string'}
RECORD2TYPE = pd.read_excel('record2type.xlsx',engine='openpyxl').astype(AS_TYPE).set_index('record')
DATA = data  = pd.read_excel('New vocal.xlsx',engine='openpyxl').astype({'record':'int32','Emo_formant4-SD':'float64'}).set_index('record')
data.reset_index(inplace=True)
dataWoPCA = data.copy()
data.to_csv('NotSortedData.csv')
data.sort_values('Type',inplace=True,ascending=False)
data.to_csv('SortedData.csv')
# print(list(set(list(data.columns)) - set(list(dataWoPCA.columns))))

if 'Type' not in list(data.columns):
    input()
    for index,row in data.iterrows():
        data.at[index,'Type'] = RECORD2TYPE.at[index,'Type']
if 'Gender' not in list(data.columns):
    input()
    for index,row in data.iterrows():
        data.at[index,'Gender'] = RECORD2TYPE.at[index,'Gender']
scores =['A2','A9','A10']
for score in scores:
    for index,row in data.iterrows():
        if int(row['record']) in list(ADOS_SCORES.head()):

            if ADOS_SCORES.at[score,int(row['record'])] not in range(0,10):
                ADOS_SCORES.at[score,int(row['record'])] = np.nan
            if not pd.isna(ADOS_SCORES.at[score,int(row['record'])]):
                
                data.at[index,score] = ADOS_SCORES.at[score,int(row['record'])]
                dataWoPCA.at[index,score] = ADOS_SCORES.at[score,int(row['record'])]

            else:
                print(row['record'],' missing')
                data.at[index,score] =np.nan
                dataWoPCA.at[index,score] =np.nan

        else:
            print(row['record'],' missing')
            data.at[index,score] =np.nan
            dataWoPCA.at[index,score] =np.nan

if PCA_SHIMMER_JITTER:
    data_copy1 = DataAlgo.drop_all_cols_with(data,['shimmer','jitter'])
    # print(DataAlgo.PCA_on_Tuppels([('shimmer',2),('jitter',2)],data))
    pca_list = [data_copy1]
    for task in ['tb','story','emo']:
        pca_list.append(DataAlgo.PCA_on_Tuppels([([task,'shimmer'],2),([task,'jitter'],2)],data))
    for df in pca_list:
        row_counter = 0 
        for index,row in data.iterrows():
            for col in list(df.columns):
                data.at[index,col] = df.at[row_counter,col]
            row_counter+=1

# data.columns = list(dataWoPCA.columns)
print(data)
input()
if PCA_SHIMMER_JITTER:
    data_copy2 = DataAlgo.drop_all_cols_with(dataWoPCA,['shimmer','jitter'])
    # print(DataAlgo.PCA_on_Tuppels([('shimmer',2),('jitter',2)],data))
    pca_list = [data_copy2]
    for task in ['tb','story','emo']:
        pca_list.append(DataAlgo.PCA_on_Tuppels([([task,'shimmer'],2),([task,'jitter'],2)],dataWoPCA))
    for df in pca_list:
        row_counter = 0 
        for index,row in dataWoPCA.iterrows():
            for col in list(df.columns):
                dataWoPCA.at[index,col] = df.at[row_counter,col]
            row_counter+=1

    # dataWoPCA = pd.concat(pca_list,axis=1)
print(data.sort_values(by='record'))
print(dataWoPCA.sort_values(by='record'))
print(list(set(list(data.columns)) - set(list(dataWoPCA.columns))))
for col in list(data_copy1.columns):
    # if col == 'Emo_formant4-SD':
    #     # input()
    #     print(data[col])
    #     print(dataWoPCA[col])
    if col == 'Type':
        continue
    if col == 'Gender':
        continue
    if col in scores:
        continue
    print(data_copy1[col]-data_copy2[col])
input()
data.sort_values(by='record',inplace=True)
dataWoPCA.sort_values(by='record',inplace=True)
for col in list(data.columns):
    # if col == 'Emo_formant4-SD':
    #     # input()
    #     print(data[col])
    #     print(dataWoPCA[col])
    if col == 'Type':
        continue
    if col == 'Gender':
        continue
    if col in scores:
        continue
    print(data[col]-dataWoPCA[col])

# print(data.sort_values(by='record').subtract(dataWoPCA.sort_values(by='record')))
# exit(1)



CLASSIFY  = False
data = data[data['Gender']=='M']
data_w_diffs = pd.concat([data,DataAlgo.diffs_by_tasks(data,['tb','story','emo'])],axis=1).dropna()
data_w_diffs.to_csv('data_w_diffs.csv')
class_data = data.drop(scores,axis=1).dropna()
print(class_data.groupby('Type').count())
regress_data = data.drop('Type',axis=1).dropna()
total_results =[]

X = class_data.drop(['Type','record','Gender'],axis=1)
X.to_csv('X.csv')
# print(X)
runs = 1
X_diffs  = data_w_diffs.drop(scores+['Type','record','Gender'],axis=1).dropna()
# test_permute_importane(X_diffs)
cols = ['tb_shimmer_pc1','Story_stdevIntensity','emo_shimmer_pc1']
# print(DataAlgo.LOO_multi_classify(X_diffs[cols],data_w_diffs['Type'],runs=10))
# exit(1)
RF_FEATURE_TEST = False
if RF_FEATURE_TEST:
    feature_compare = pd.DataFrame()
    ls = DataAlgo.RF_feature_reduction(X,class_data['Type'],steps = [*range(100,21,-10)]+[*range(20,11,-5)]+[*range(10,0,-1)],compare_table=feature_compare)
    DataAlgo.append_table_to_pickle(('RF feature compare',feature_compare),LIST_PICKLE_FILE)
    feature_compare = pd.DataFrame()
    ls = DataAlgo.RF_feature_reduction(X_diffs,data_w_diffs['Type'],steps = [*range(100,21,-10)]+[*range(20,11,-5)]+[*range(10,0,-1)],compare_table=feature_compare)
    DataAlgo.append_table_to_pickle(('RF feature compare (&diffs)',feature_compare),LIST_PICKLE_FILE)
    # DataAlgo.TestApp([('RF feature compare',feature_compare)])

wo_diffs_steps =[*range(100,21,-10)]+[*range(20,11,-5)]+[*range(10,3,-1)] # hardcoded from tables 
w_diffs_steps =[*range(100,21,-10)]+[*range(20,11,-5)]+[*range(10,6,-1)] # hardcoded from tables 


if CLASSIFY:
    for task,result in DataAlgo.classify_by_task(class_data.drop(['Type','record','Gender'],axis=1),class_data['Type'],['tb','story','emo'],DataAlgo.RF_feature_reduction,runsp = runs):
        DataAlgo.append_table_to_pickle((task,result),LIST_PICKLE_FILE)

rf_feature = DataAlgo.RF_feature_reduction(X,class_data['Type'])
if CLASSIFY:
    DataAlgo.append_table_to_pickle(('all tasks forest fr',DataAlgo.LOO_multi_classify(X[rf_feature],class_data['Type'],print_results=False,runs = runs)),LIST_PICKLE_FILE)

y = data_w_diffs['Type']
print(y)
# exit(1)
DataAlgo.cross_val_feature_check(X_diffs,y)
print('RF reduction for alltask+diffs')
# DataAlgo.test_permute_importane(X=X_diffs,y=y)
# DataAlgo.append_table_to_pickle(('quality RF test',DataAlgo.test_RF_reduction_quality(X_diffs,y)))
# DataAlgo.append_table_to_pickle(('statstical quality test (diffs)',DataAlgo.test_statistics_reduction_quality(X_diffs,y)),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('statstical quality test ',DataAlgo.test_statistics_reduction_quality(X,class_data['Type'])),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('RF quality test (diffs)',DataAlgo.test_RF_reduction_quality(X_diffs,y,steps=w_diffs_steps,from_last=6)),LIST_PICKLE_FILE)
# DataAlgo.append_table_to_pickle(('RF quality test ',DataAlgo.test_RF_reduction_quality(X,class_data['Type'],steps =wo_diffs_steps,from_last=4)),LIST_PICKLE_FILE)
# exit(1)

# DataAlgo.test_LogisticRegression(X_diffs,y,k=6)
# DataAlgo.test_RF_reduction_quality_pca(X_diffs,y)
feaures = DataAlgo.statistics_feature_reduction(X_diffs,y,features=6)
# DataAlgo.TestApp([('LDAshrinakge')])
from sklearn.covariance import OAS
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for i in range(0,10):
    counter=0
    for train,test in loo.split(X_diffs):
        X_train, X_test = X_diffs[feaures].iloc[train], X_diffs[feaures].iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        # print(y.iloc[train], y.iloc[test])
        qda = DataAlgo.QuadraticDiscriminantAnalysis()
        qda.fit(X_train,y_train)
        y_out = qda.predict(X_test)
        # print(y_out,y_test)
        counter+= int(y_out==y_test)
    # print(counter/loo.get_n_splits(X_diffs))

# print(DataAlgo.LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').get_params())
# print(DataAlgo.LinearDiscriminantAnalysis(solver='lsqr').get_params())
# exit(1)
feaures = ['Story_stdevIntensity','tb_shimmer_pc1',  'emo_shimmer_pc1',  'tb-story_formant2-SD',  'tb-emo_stdevF0Hz',  'story-emo_pc1']
# feaures = [feaures[0]]+[feaures[3]]
# pd.concat([X_diffs[feaures],y],axis=1).to_csv('test55.csv')
print(X_diffs[feaures])
print(y)
DataAlgo.TestApp([('6-features',DataAlgo.CV_multi_classify(X_diffs[feaures],y,runs=runs))])
# print(DataAlgo.CV_model_classify(X_diffs[feaures],y=='NT',DataAlgo.LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'),runs=5))
exit(1)
# pca_n = 6
# pca = PCA(n_components=pca_n)
# Xpca = pca.fit_transform(X=StandardScaler().fit_transform(X_diffs[feaures]))
# features_list2 = [] 
# for i in range(1,pca_n+1):
#     features_list2.append('p'+str(i))
# Xpca= pd.DataFrame(Xpca,columns = features_list2)
# print(Xpca)
pca_n = 2
pca = PCA(n_components=pca_n)
X_diffs2 = (X_diffs-X_diffs.mean())/X_diffs.std()
Xpca = pca.fit_transform(X_diffs2[feaures])
pca_details = pd.DataFrame(columns = ['feature']).set_index(['feature'])
pca_weights = pd.DataFrame(pca.components_.transpose())
for i,feature in enumerate(X_diffs2[feaures].head()):
    pca_details.at[feature,'p1'] = pca_weights.at[i,0]
    pca_details.at[feature,'p2'] = pca_weights.at[i,1]
# pca_details = pca_details.transpose()
print(pca_details)
pca_details.plot.barh(rot=0, subplots=True)
plt.show()
print(pca.components_,'\n')
exit(1)
df_inv = pd.DataFrame(pca.components_.transpose())
print(df_inv)
result = df_inv.dot(Xpca.transpose()).transpose()
result.columns = feaures
print(result+X_diffs[feaures].mean())
# print(X_diffs[feaures].mean())
normalized_df=(X_diffs-X_diffs.mean())/X_diffs.std()
print(X_diffs[feaures]-result+X_diffs[feaures].mean())
data_original = pd.DataFrame(np.dot(Xpca, pca.components_) + pca.mean_)
data_original.columns = feaures
print()

print(data_original-normalized_df[feaures].reset_index())
print(normalized_df[feaures])
# print(normalized_df[feaures],normalized_df[feaures].shape)
features_list2 = [] 
for i in range(1,pca_n+1):
    features_list2.append('p'+str(i))
# features_list2 = ['p1','p2','p3']
Xpca= pd.DataFrame(Xpca,columns = features_list2)
# Xpca = (Xpca-Xpca.mean())/Xpca.std()
DataAlgo.TestApp([('pca',DataAlgo.CV_multi_classify(Xpca[features_list2],y,5))])
# DataAlgo.TestApp([('pca',DataAlgo.CV_model_classify(clf=DataAlgo.LinearDiscriminantAnalysis(solver='lsqr'),X= Xpca[features_list2],y= y,runs=5,cross_val =5))])
# to_show = [('new',DataAlgo.CV_multi_classify(X_diffs[feaures],y,1)),('pca',DataAlgo.CV_multi_classify(Xpca[features_list2],y,1))]
# to_show+=['merged',Xpca.corrwith(X_diffs[feaures],axis=1)]
scatter = plt.scatter(Xpca['p1'], Xpca['p2'],c=(y=='NT').astype(float))
plt.legend(handles=scatter.legend_elements()[0], labels=['NT','ASD'])

ri = 0 
for index,row  in data_w_diffs.iterrows():
    plt.annotate(row['record'],(Xpca['p1'][ri], Xpca['p2'][ri]))
    ri+=1
plt.show()
corr_pd  =pd.DataFrame(pd.concat([Xpca,X_diffs[feaures]],axis=1).corr())
corr_pd[['p1','p2']].plot.barh(rot=0, subplots=True)
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
rf_feature = DataAlgo.RF_feature_reduction(X_diffs,y)
if CLASSIFY:
    total_results.append(('all&diffs tasks',DataAlgo.LOO_multi_classify(X_diffs[rf_feature],y,print_results=False,runs = runs)))
rf_feature = DataAlgo.statistics_feature_reduction(X_diffs,y,features = 7)
if CLASSIFY:
    total_results.append(('all&diffs2 tasks',DataAlgo.LOO_multi_classify(X_diffs[rf_feature],y,print_results=False,runs = runs)))
X_diffs.to_csv('X_diffs.csv')

d2 = data_w_diffs.drop(['Type'],axis=1).dropna()
X_diffs = d2.drop(scores+['record','Gender'],axis=1)
y = d2['A2']
rf_feature = DataAlgo.RF_feature_reduction(X_diffs,y)
if CLASSIFY:
    total_results.append(('A2 regress all&diffs tasks',DataAlgo.LOO_multi_classify(X_diffs[rf_feature],y,print_results=False,runs = runs)))
rf_feature = DataAlgo.RF_feature_reduction(X_diffs,y>0)
if CLASSIFY:
    total_results.append(('A2>0 all&diffs tasks',DataAlgo.LOO_multi_classify(X_diffs[rf_feature],y>0,print_results=False,runs = runs)))
rf_feature = DataAlgo.statistics_feature_reduction(X_diffs,y)
if CLASSIFY:
    total_results.append(('A2 regress all&diffs tasks2',DataAlgo.LOO_multi_classify(X_diffs[rf_feature],y,print_results=False,runs = runs)))
rf_feature = DataAlgo.statistics_feature_reduction(X_diffs,y>0)
if CLASSIFY:
    total_results.append(('A2>0 all&diffs tasks2',DataAlgo.LOO_multi_classify(X_diffs[rf_feature],y>0,print_results=False,runs = runs)))
# X_diffs.to_csv('X_diffs.csv')
if CLASSIFY:
    merged = pd.DataFrame()
    for task,result in total_results:
        result_copy = result.copy()
        result_copy.columns = result_copy.columns.str.replace('mean', 'mean '+task)
        merged = pd.concat([merged,result_copy['mean '+task]],axis=1)
        print(merged)
    total_results.append(('merged',merged))
    save_pickle_file(total_results,where = 'list_of_results.pickle')
    for description,result in total_results:
        print(description)
        # DataAlgo.print_class_repot(result)
        # tabloo.show(result)

print(class_data.groupby('Type').count())
