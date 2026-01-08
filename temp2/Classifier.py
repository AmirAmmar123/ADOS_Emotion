from Files import get_files
import pandas as pd
import numpy as np
from DataAlgo import *
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
ADOS_SCORES = pd.read_excel('ADOS scores for Omer 21-11-22.xlsx',engine='openpyxl').set_index('task')
AS_TYPE = {'record':'int32','Type':'string','Gender':'string'}
RECORD2TYPE = pd.read_excel('record2type.xlsx',engine='openpyxl').astype(AS_TYPE).set_index('record')
FULL_DATA =  pd.read_excel('New vocal2.xlsx',engine='openpyxl').astype({'record':'int32'}).set_index('record')
RESULTS_DF_DIR = 'results df/'
if not exists(RESULTS_DF_DIR):
    os.mkdir(RESULTS_DF_DIR)
cvs_lst =[]
CVS_data = []
std_devs = []
for task in ['TB','Emo','Story']:
    print(task+'_meanF0Hz' in list(FULL_DATA.columns))
    print(task+'_stdevF0Hz' in list(FULL_DATA.columns))   
    CVS_data.append(FULL_DATA[task+'_stdevF0Hz']/FULL_DATA[task+'_meanF0Hz'])
    cvs_lst.append(task+'_cvF0Hz')
    std_devs.append(task+'_stdevF0Hz')
CVS_data = pd.concat(CVS_data,axis=1)
features_to_keep = ['f0hz','intensity','shimmer','jitter','type','gender']
cols_to_drop = set()
for col in list(FULL_DATA.columns):
    all = True
    for feature in features_to_keep:
        if feature.lower()  in col.lower():
            all = False
    if  all:
        cols_to_drop.add(col)
FULL_DATA = FULL_DATA.drop(cols_to_drop,axis=1)

PCA_SJ = True
if True:
    data_copy  = drop_all_cols_with(FULL_DATA,['shimmer','jitter'])
    # print(DataAlgo.PCA_on_Tuppels([('shimmer',2),('jitter',2)],data))
    pca_list = [data_copy]
    for task in ['tb','story','emo']:
        pca_list.append(PCA_on_Tuppels([([task,'shimmer'],2),([task,'jitter'],2)],FULL_DATA))
    FULL_DATA = pd.concat(pca_list,axis=1)
diffs = diffs_by_tasks(FULL_DATA,['tb','story','emo'])
data = pd.concat([remove_cols_with(FULL_DATA,'mean'),diffs],axis=1).dropna(axis=0)
males = data[data['Gender']=='M']
females = data[data['Gender']=='F']
np.random.seed(0)
#sub sample
# to_remove = np.random.choice(females[females['Type']=='NT'].index,size=(len(females[females['Type']=='NT'])-len(females[females['Type']=='ASD'])),replace=False)


#over sample
# to_add= np.random.choice(females[females['Type']=='ASD'].index,size=(len(females[females['Type']=='NT'])-len(females[females['Type']=='ASD'])),replace=False)
nt_females_count = len(females[females['Type']=='NT'])
asd_females_count = len(females[females['Type']=='ASD'])
if(nt_females_count>2*asd_females_count):
    asd_to_sample = asd_females_count
else:
    asd_to_sample = nt_females_count-asd_females_count
print(asd_to_sample)
sample = females[females['Type']=='ASD'].sample(asd_to_sample,random_state=0)
indices = list(sample.index)
for i in range(len(indices)):
    indices[i]+=1000
sample.index = indices
upsmapled_females = pd.concat([females,sample],axis=0)

# males_data = males.drop(['Gender','Type'],axis=1)
# males_data.values[:]= StandardScaler().fit_transform(males_data)
# z_males = pd.concat([males_data,males['Gender'],males['Type']],axis=1)

# males = pd.concat([z_males,males['Gender'],males['Type']],axis=1)
to_drop = ['Gender','Type']
X = males.drop(to_drop,axis=1)
y = males['Type']
print(X.columns)
# statistics_feature_reduction(X,y,f_classif,10,plot = True)
totalc = total = len(list(X.columns))
steps = [total-2] #minus 2 because Type,Gender
while int(totalc*0.9)>10:
    steps.append(int(totalc*0.9))
    totalc*=0.9
steps.append(10)
steps=list(set(steps))
steps.sort(reverse=True)
ct1 = pd.DataFrame()
ct2 = pd.DataFrame()
ct3 = pd.DataFrame()
# hist_features_by_label(males,'A2',outputdir='FeatureshistogramA2')
# exit(1)
# RF_feature_reduction(X,y,clf = GradientBoostingClassifier(max_depth=100),steps =steps,compare_table=ct3)
# RF_feature_reduction(X,y,clf = RandomForestClassifier(max_depth=100),steps =steps,compare_table=ct1)
# RF_feature_reduction(X,y,clf = DecisionTreeClassifier(max_depth=100),steps =steps,compare_table=ct2)
# for item in [('rf ct',ct1),('dt ct',ct2),('gb ct',ct3)]:
#     append_table_to_pickle(item,'cts pickle')
# TestApp([('rf ct',ct1),('dt ct',ct2),('gb ct',ct3)])
# statistics_feature_reduction(X,y,mutual_info_classif,10,plot = True)

# cross_val_feature_check(X,y,plot=True)
# cross_val_feature_check(X,y,s_func=mutual_info_classif,plot=True)
# cross_val_feature_check_RF(X,y,steps,clf = GradientBoostingClassifier(max_depth=100),filename='CVGB')
# cross_val_feature_check_RF(X,y,steps,clf = DecisionTreeClassifier(max_depth=100),filename='CVDT')
cross_val_feature_check_RF(X,y,steps,clf = RandomForestClassifier(max_depth=100,n_jobs=-1),filename='CVRF')
exit(1)
feature_compare = pd.DataFrame()

from sklearn.model_selection import GridSearchCV
from GridSearchParams import *
from pickle import load,dump

NTASD_DEEP = False
if NTASD_DEEP:
    X_minmax = X.copy()
    X_minmax.values[:] =  MinMaxScaler().fit_transform(X_minmax)
    X_standart = X.copy()
    X_standart.values[:] =  StandardScaler().fit_transform(X_standart)
    rf_f_mm = RF_Iter_feature_reduction(X_minmax,y,RandomForestClassifier(max_depth=10),steps)
    rf_f_sc = RF_Iter_feature_reduction(X_standart,y,RandomForestClassifier(max_depth=10),steps)
    results = {}
    #grid search LDA (minmax,RF,7)
    estimator = LinearDiscriminantAnalysis()
    results['LDA (minmax,RF,7)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_minmax[rf_f_mm[-7]],y)
    estimator = LinearDiscriminantAnalysis()
    results['LDA (minmax,RF,7)svd'] = GridSearchCV(estimator,get_LDAsvd_params(),scoring='accuracy',cv=10).fit(X_minmax[rf_f_mm[-7]],y)
    #grid search SVC (minmax,RF,8)
    estimator = SVC()
    results['SVC (minmax,RF,8)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_minmax[rf_f_mm[-8]],y)
    #grid search RF (minmax,RF,9)
    estimator = RandomForestClassifier()
    results['RF (minmax,RF,9)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_minmax[rf_f_mm[-9]],y)
    #grid search LR (minmax,RF,2)
    estimator = LogisticRegression()
    results['LR (minmax,RF,2)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_minmax[rf_f_mm[-2]],y)

    for key in list(results.keys()):
        print(results[key].best_score_)
    pick_path = 'result pickle'
    print('end phase 1')
    with open (pick_path, 'wb') as pick:
        dump(results, pick)
    #grid search LDA (minmax,f_classif,2) (and print features!!!!)
    estimator = LinearDiscriminantAnalysis()
    results['LDA (minmax,f_classif,2)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_minmax[statistics_feature_reduction(X_minmax,y,features = 2)],y)
    estimator = LinearDiscriminantAnalysis()
    results['LDA (minmax,f_classif,2)svd'] = GridSearchCV(estimator,get_LDAsvd_params(),scoring='accuracy',cv=10).fit(X_minmax[statistics_feature_reduction(X_minmax,y,features = 2)],y)
    #grid search SVC (minmax,f_classif,2)
    estimator = SVC()
    results['SVC (minmax,f_classif,2)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_minmax[statistics_feature_reduction(X_minmax,y,features = 2)],y)
    #grid search QDA (minmax,f_classif,2)
    estimator = QuadraticDiscriminantAnalysis()
    results['QDA (minmax,f_classif,2)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_minmax[statistics_feature_reduction(X_minmax,y,features = 2)],y)

    for key in list(results.keys()):
        print(results[key].best_score_)
    print('end phase 2')
    with open (pick_path, 'wb') as pick:
        dump(results, pick)

    #grid search LDA (standart,RF,7) 
    estimator = LinearDiscriminantAnalysis()
    results['LDA (standart,RF,7)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_standart[rf_f_sc[-7]],y)
    estimator = LinearDiscriminantAnalysis()
    results['LDA (standart,RF,7)svd'] = GridSearchCV(estimator,get_LDAsvd_params(),scoring='accuracy',cv=10).fit(X_standart[rf_f_sc[-7]],y)

    #grid search SVC (standart,RF,6)
    estimator = SVC()
    results['SNC (standart,RF,6)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_standart[rf_f_sc[-6]],y)
    #grid search QDA (standart,RF,13)
    estimator = QuadraticDiscriminantAnalysis()
    results['QDA (standart,RF,13)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_standart[rf_f_sc[-13]],y)
    #grid search RF (standart,RF,7 )
    estimator = RandomForestClassifier()
    results['RF (standart,RF,7)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_standart[rf_f_sc[-7]],y)

    for key in list(results.keys()):
        print(results[key].best_score_)
    pick_path = 'result pickle'
    print('end phase 3')
    with open (pick_path, 'wb') as pick:
        dump(results, pick)

    #grid search LDA (standart,f_classif,7) 
    estimator = LinearDiscriminantAnalysis()
    results['LDA (standart,f_classif,7)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_standart[statistics_feature_reduction(X_standart,y,features = 7)],y)
    estimator = LinearDiscriminantAnalysis()
    results['LDA (standart,f_classif,7)svd'] = GridSearchCV(estimator,get_LDAsvd_params(),scoring='accuracy',cv=10).fit(X_standart[statistics_feature_reduction(X_standart,y,features = 7)],y)
    #grid search SVC (standart,f_classif,6)
    estimator = SVC()
    results['LR (standart,f_classif,6)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_standart[statistics_feature_reduction(X_standart,y,features = 6)],y)
    #grid search QDA (standart,f_classif,9)
    estimator = QuadraticDiscriminantAnalysis()
    results['QDA (standart,f_classif,9)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_standart[statistics_feature_reduction(X_standart,y,features = 9)],y)
    #grid search GBC (standart,f_classif,7 )
    estimator = GradientBoostingClassifier()
    results['GBC (standart,f_classif,7)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_standart[statistics_feature_reduction(X_standart,y,features = 7)],y)
    #grid search LR (standart,f_classif,20 )
    estimator = LogisticRegression()
    results['LR (standart,f_classif,20)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_standart[statistics_feature_reduction(X_standart,y,features = 20)],y)

    for key in list(results.keys()):
        print(results[key].best_score_)
    pick_path = 'result pickle'
    print('end phase 4')
    with open (pick_path, 'wb') as pick:
        dump(results, pick)
pick_path = 'result pickle'
# with open (pick_path, 'rb') as pick:
#     results = load(pick)
# for key in list(results.keys()):
#     print(key,results[key].best_score_)

# exit(1)
# ls = RF_feature_reduction(X,y,steps = steps,compare_table=feature_compare)
# feature_compare.to_pickle(RESULTS_DF_DIR+"  features test"+' ASDNT'+' RF til 1')
# feature_compare.to_csv("  features test"+'ASDNT'+' RF til 1'+'.csv')
# X_n = X.drop(list(diffs.columns),axis=1)
# X_tb = X_n[get_all_cols_with(X_n,'tb')]
# X_story = X_n[get_all_cols_with(X_n,'story')]
# X_emo = X_n[get_all_cols_with(X_n,'emo')]
# # print(X_tb.columns)
# print(X_story.columns)
# print(X_emo.columns)
# my_grid_search(X,y,'males all tasks with diffs')



scores =['A2']

for score in scores:
    for index,row in X.iterrows():
        if int(index) in list(ADOS_SCORES.head()):

            if ADOS_SCORES.at[score,int(index)] not in range(0,10):
                ADOS_SCORES.at[score,int(index)] = np.nan
            if not pd.isna(ADOS_SCORES.at[score,int(index)]):
                
                X.at[index,score] = ADOS_SCORES.at[score,int(index)]
            else:
                print(int(index),' missing')
                X.at[index,score] =np.nan
        else:
            print(int(index),' missing')
            X.at[index,score] =np.nan
y= (X['A2']>0).astype(int)
X = X.drop(['A2'],axis=1)
results = {}
A2DEEP = False
males = pd.concat([X,y],axis=1)
hist_features_by_label(males,'A2',outputdir='FeatureshistogramA2')
exit(1)

if A2DEEP:
    X_minmax = X.copy()
    X_minmax.values[:] =  MinMaxScaler().fit_transform(X_minmax)
    X_standart = X.copy()
    X_standart.values[:] =  StandardScaler().fit_transform(X_standart)
    steps.append(19)
    steps.sort()
    steps=list(set(steps))
    steps.sort(reverse=True)
    print(steps)
    rf_f_mm = RF_Iter_feature_reduction(X_minmax,y,RandomForestClassifier(max_depth=10),steps)
    rf_f_sc = RF_Iter_feature_reduction(X_standart,y,RandomForestClassifier(max_depth=10),steps)
    rf_f_mm_dict = {}
    rf_f_sc_dict ={}
    for fl in rf_f_mm:
        rf_f_mm_dict[len(fl)] = fl
    for fl in rf_f_sc:
        rf_f_sc_dict[len(fl)] = fl
    #grid search LDA A2minmax RF
    estimator = LinearDiscriminantAnalysis()
    results['A2 LDA (minmax,RF,7)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_minmax[rf_f_mm[-7]],y)
    estimator = LinearDiscriminantAnalysis()
    results['A2 LDA (minmax,RF,7)svd'] = GridSearchCV(estimator,get_LDAsvd_params(),scoring='accuracy',cv=10).fit(X_minmax[rf_f_mm[-7]],y)
    pick_path = 'result pickle for A2'
    print('end phase 4')
    with open (pick_path, 'wb') as pick:
        dump(results, pick)
    estimator = RandomForestClassifier()
    results['A2 RF(minmax,f_classif,3)'] = GS_fit(estimator,X_minmax[statistics_feature_reduction(X_minmax,y,features = 3)],y)
    estimator = GradientBoostingClassifier()
    results['A2 GBC(minmax,f_classif,3)'] = GS_fit(estimator,X_minmax[statistics_feature_reduction(X_minmax,y,features = 3)],y)
    estimator = SVC()
    results['A2 SVC(minmax,f_classif,5)'] = GS_fit(estimator,X_minmax[statistics_feature_reduction(X_minmax,y,features = 5)],y)
    estimator = LinearDiscriminantAnalysis()
    results['A2 LDA(minmax,f_classif,10)'] = GS_fit(estimator,X_minmax[statistics_feature_reduction(X_minmax,y,features = 10)],y)
    results['A2 LDA (minmax,f_classif,10)svd'] = GridSearchCV(estimator,get_LDAsvd_params(),scoring='accuracy',cv=10).fit(X_minmax[statistics_feature_reduction(X_minmax,y,features = 10)],y)
    pick_path = 'result pickle for A2'
    print('end phase 4')
    with open (pick_path, 'wb') as pick:
        dump(results, pick)
    estimator = GradientBoostingClassifier()
    results['A2 GBC(minmax,mutal_info,2)'] = GS_fit(estimator,X_minmax[statistics_feature_reduction(X_minmax,y,mutual_info_classif,2)],y)
    pick_path = 'result pickle for A2'
    print('end phase 4')
    with open (pick_path, 'wb') as pick:
        dump(results, pick)
    estimator = LogisticRegression()
    results['A2 LR(standart,RF,19)'] = GS_fit(estimator,X_standart[rf_f_sc_dict[19]],y)
    estimator = RandomForestClassifier()
    results['A2 RF(standart,RF,8)'] = GS_fit(estimator,X_standart[rf_f_sc_dict[8]],y)
    estimator = LinearDiscriminantAnalysis()
    results['A2 LDA(standart,RF,19)'] = GS_fit(estimator,X_standart[rf_f_sc_dict[19]],y)
    results['A2 LDA(standart,RF,19)svd'] = GridSearchCV(estimator,get_LDAsvd_params(),scoring='accuracy',cv=10).fit(X_standart[rf_f_sc_dict[19]],y)
    pick_path = 'result pickle for A2'
    print('end phase 4')
    with open (pick_path, 'wb') as pick:
        dump(results, pick)
    for key in list(results.keys()):
        print(results[key].best_score_)

    #grid search SVC (standart,f_classif,6)
GRID_SEARCH_LDA = False
if GRID_SEARCH_LDA:
    X_minmax = X.copy()
    X_minmax.values[:] =  MinMaxScaler().fit_transform(X_minmax)
    X_standart = X.copy()
    X_standart.values[:] =  StandardScaler().fit_transform(X_standart)
    steps.append(19)
    steps.sort()
    steps=list(set(steps))
    steps.sort(reverse=True)
    print(steps)
    rf_f_mm = RF_Iter_feature_reduction(X_minmax,y,RandomForestClassifier(max_depth=10),steps)
    rf_f_sc = RF_Iter_feature_reduction(X_standart,y,RandomForestClassifier(max_depth=10),steps)
    rf_f_mm_dict = {}
    rf_f_sc_dict ={}
    for fl in rf_f_mm:
        rf_f_mm_dict[len(fl)] = fl
    for fl in rf_f_sc:
        rf_f_sc_dict[len(fl)] = fl
    estimator = LinearDiscriminantAnalysis()
    results['A2 LDA (minmax,RF,7)'] = GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10).fit(X_minmax[rf_f_mm[-7]],y)
    estimator = LinearDiscriminantAnalysis()
    estimator = LinearDiscriminantAnalysis()
    results['A2 LDA(minmax,f_classif,10)'] = GS_fit(estimator,X_minmax[statistics_feature_reduction(X_minmax,y,features = 10)],y)
    results['A2 LDA (minmax,f_classif,10)svd'] = GridSearchCV(estimator,get_LDAsvd_params(),scoring='accuracy',cv=10).fit(X_minmax[statistics_feature_reduction(X_minmax,y,features = 10)],y)
    results['A2 LDA (minmax,RF,7)svd'] = GridSearchCV(estimator,get_LDAsvd_params(),scoring='accuracy',cv=10).fit(X_minmax[rf_f_mm[-7]],y)
    results['A2 LDA(standart,RF,19)'] = GS_fit(estimator,X_standart[rf_f_sc_dict[19]],y)
    results['A2 LDA(standart,RF,19)svd'] = GridSearchCV(estimator,get_LDAsvd_params(),scoring='accuracy',cv=10).fit(X_standart[rf_f_sc_dict[19]],y)
old_results = {}
results_df = pd.DataFrame(columns = ['key','accuracy']).set_index('key')
with open('result pickle','rb') as pick:
    old_results = load(pick,encoding='utf8')
for key in list(old_results.keys()):
    results_df.at[key,'accuracy'] = old_results[key].best_score_
with open('result pickle for A2','rb') as pick:
    old_results = load(pick,encoding='utf8')
for key in list(old_results.keys()):
    results_df.at[key,'accuracy'] = old_results[key].best_score_
for key in list(results.keys()):
    results_df.at[key,'accuracy'] = results[key].best_score_
print(results_df)
results_df.to_csv('grid search results.csv')


# my_grid_search(X,y,'males all tasks with diffs A2-0')
# my_grid_search(X_tb,y,'males tb')
# my_grid_search(X_story,y,'males story')
# my_grid_search(X_emo,y,'males emo')

# feature_compare = pd.DataFrame()
# totalc = total = len(list(X.columns))
# steps = [total-2]
# while int(totalc*0.9)>1:
#     steps.append(int(totalc*0.9))
#     totalc*=0.9
# ls = RF_feature_reduction(X,y,steps = steps,compare_table=feature_compare)
# feature_compare.to_pickle(RESULTS_DF_DIR+"  features test"+'_A20'+' RF til 1')
# feature_compare.to_csv("  features test"+'_A20'+' RF til 1'+'.csv')


exit(1)
# X.values[:] =  MinMaxScaler().fit_transform(X)
# print(cross_val_features_reduction(X,y,20))
for func in [f_classif,mutual_info_classif]:
    print(CV_multi_classify(X[statistics_feature_reduction(X,y,rating_func =func ,features = 7)],y,runs=5))
# print(CV_multi_classify(X[RF_feature_reduction(X,y,steps)],y,runs=5))
exit(1)
z_all = pd.concat([get_z_df(males,to_drop),get_z_df(upsmapled_females,to_drop)],axis=0)
# cross_val_feature_check(z_all.drop(to_drop,axis=1),z_all['Type'])
steps.append(10)
print(steps)
# cross_val_feature_check_RF(z_all.drop(to_drop,axis=1),z_all['Type'],steps)

# print(CV_multi_classify(z_all[statistics_feature_reduction(z_all.drop(to_drop,axis=1),z_all['Type'],7)],z_all['Type']))
final_X = get_z_df(males,to_drop).drop(to_drop,axis=1)

# exit(1)
# reduces = final_X[RF_feature_reduction(final_X,z/['Type'],steps = [40,30,20,15,10,9,8,7])]
# result = test_RF_reduction_quality(z_all.drop(to_drop,axis=1),z_all['Type'],cv = KFold(5))
# result.to_pickle(RESULTS_DF_DIR+'rf reduction on standartadize all balanced oversampled')
# result.to_csv('rf reduction on standartadize all balanced oversampled.csv')
# result = test_statistics_reduction_quality(z_all.drop(to_drop,axis=1),z_all['Type'],cv = KFold(5))
# result.to_pickle(RESULTS_DF_DIR+'stastical reduction on standartadize all balanced oversampled')
# result.to_csv('statisctical reduction on standartadize all balanced oversampled .csv')
# result = test_statistical_reduction_quality_pca(males.drop(to_drop,axis=1),males['Type'],cv = KFold(10))
# result.to_pickle(RESULTS_DF_DIR+'stastical reduction on standartadize males pca (10fold)')
# result.to_csv('statisctical reduction on standartadize males pca (10fold) .csv')
# print(z_all)
exit(1)
# results,clf = CV_model_classify(males[statistics_feature_reduction(males.drop(['Type','Gender'],axis=1),males['Type'])],males['Type'],clf = RandomForestClassifier(max_depth=100),ret_clf = True)
results = CV_multi_classify(males[statistics_feature_reduction(males.drop(['Type','Gender'],axis=1),males['Type'])],males['Type'])

print(males.groupby('Type').count())
print(results)
results,clf = CV_model_classify(females[statistics_feature_reduction(females.drop(['Type','Gender'],axis=1),females['Type'])],females['Type'],clf = RandomForestClassifier(max_depth=100),ret_clf = True)
print(females.groupby('Type').count())
print(results)
# fig = plt.figure()
# _=tree.plot_tree(clf,feature_names=data.columns,class_names=['A','N'],filled=True)
# plt.show()
