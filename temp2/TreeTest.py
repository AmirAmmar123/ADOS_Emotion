from turtle import title
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from Files import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from sklearn.model_selection import cross_val_score,LeaveOneOut,cross_val_predict
from LittleGui import load_dataFrame
# from sklearn import cross_validation
from IPython.display import display
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression,chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
import DataAlgo
kernel = DotProduct() + WhiteKernel()
ADOS_SCORES = pd.read_excel('ADOS scores for Omer 21-11-22.xlsx',engine='openpyxl').set_index('task')
AS_TYPE = {'record':'int32','Type':'string','Gender':'string'}

RECORD2TYPE = pd.read_excel('record2type.xlsx',engine='openpyxl').astype(AS_TYPE).set_index('record')

HANDS = []
HEAD = []
hands_kp = [3,4,6,7]
head_kp = [17,18]
for kp  in hands_kp:
    HANDS.append('TB_'+str(kp)+'-1dist-normed')
    HANDS.append('EMO_'+str(kp)+'-1dist-normed')
for kp  in head_kp:
    HEAD.append('TB_'+str(kp)+'-1dist-normed')
    HEAD.append('EMO_'+str(kp)+'-1dist-normed')
# print(HANDS+HEAD)
clf = DecisionTreeClassifier(random_state=0)
NOT_HEB=[219,221,210]
NOT_HEB = []
EDGE = [36,147]
EDGE= []
EXTRA = [211,212]
EXTRA = []
#subject 150 is very weird.
# without extra - best 'A2' regression
# without NOTHEB best classifcation
# interesting  - all data without extra
# edge ruins both
SUBJECTS2REMOVE = NOT_HEB+EDGE+EXTRA

regression_models = [DecisionTreeRegressor(random_state=0),LinearRegression(),RandomForestRegressor(),SVR(),KNeighborsRegressor(n_neighbors=5)]
regression_models+=[GaussianProcessRegressor(kernel=kernel,random_state=0)]
classification_models = [ DecisionTreeClassifier(max_depth=10),KNeighborsClassifier(n_neighbors=1), svm.SVC(),RandomForestClassifier(max_depth=10, random_state=0)]
classification_models+=[AdaBoostClassifier(),GaussianProcessClassifier(1.0 * RBF(1.0)),    MLPClassifier(alpha=1, max_iter=1000,solver='lbfgs'),    GaussianNB(),    QuadraticDiscriminantAnalysis(),LogisticRegression(solver='liblinear')]


data  = pd.read_excel(get_file(),engine='openpyxl').astype({'record':'int32'}).set_index('record')
# data = pd.read_csv(get_file()).astype({'record':'int32'}).set_index('record')
# print(data)
# print(RECORD2TYPE[10,'Type'])
data.drop(SUBJECTS2REMOVE,axis=0,inplace=True)
if 'Type' not in list(data.columns):
    for index,row in data.iterrows():
        data.at[index,'Type'] = RECORD2TYPE.at[index,'Type']
if 'Gender' not in list(data.columns):
    for index,row in data.iterrows():
        data.at[index,'Gender'] = RECORD2TYPE.at[index,'Gender']
old = len(data)
if 'TB_removed' in list(data.columns):
    data = data[data.TB_removed <=0.1]
if 'EMO_removed' in list(data.columns):
    data = data[data.EMO_removed <=0.1]
print('removed: ',old-len(data))
data.reset_index(inplace=True)
data.sort_values('Type',inplace=True,ascending=False)
scores = ['A2','A9','A10']
REGRESS = False
if REGRESS:
    for score in scores:
        for index,row in data.iterrows():
            # print(index)
            # print(int(row['record']))
            if int(row['record']) in list(ADOS_SCORES.head()):
                if ADOS_SCORES.at[score,int(row['record'])] not in range(0,10):
                    ADOS_SCORES.at[score,int(row['record'])] = np.nan
                if not pd.isna(ADOS_SCORES.at[score,int(row['record'])]):
                    
                    data.at[index,score] = ADOS_SCORES.at[score,int(row['record'])]
                else:
                    print(row['record'],' missing')
                    data.at[index,score] =np.nan
            else:
                print(row['record'],' missing')
                data.at[index,score] =np.nan
data2 = data
data = data[data['Gender']=='M']
# np.random.seed(10)
# drop_indices = np.random.choice(data[data['Gender']=='M'].index,10, replace=False)
# data = data.drop(drop_indices)
if 'task' in list(data.columns):
    data = data[data['task']=='EMO']
    data.drop('task',axis=1,inplace=True)
# for col in list(data.columns):
#     if col not in HANDS+HEAD and col not in ['Type','Gender','record','Score']:
#         data.drop(col,axis=1,inplace=True)
# data=data.join(pd.read_excel(get_file(),engine='openpyxl').astype({'record':'int32'}).set_index('record').drop(['Type','Gender'],axis=1))
# data.to_csv('relevent.csv')
# print(data.columns)
data.dropna(axis=0,inplace=True)
sc = StandardScaler()

# data=data.append(data[data['Type']=='NT'].sample(5)) #oversample
REMOVE_MEANS = True
if REMOVE_MEANS:
    for col in list(data.columns):
        if 'mean' in col.lower():
            data.drop(col,axis=1,inplace=True)
PCAFLAG = False
PCA_SHIMMER_JITTER = True
if PCAFLAG or PCA_SHIMMER_JITTER:
    components = [('shimmer',1),('jitter',1)]
    if PCAFLAG:
        components = []
    pca_data = pd.DataFrame()
    from sklearn.decomposition import PCA
    for component,n in components:
        pca_model = PCA(n_components=n) 
        component_list = []
        for col in list(data.columns):
            if component in col.lower():
                component_list.append(col)
        # print(component_list)
        pc_result = pca_model.fit_transform(sc.fit_transform(data[component_list]))
        pca_cols_names =[]
        for i in range(1,n+1):
            pca_cols_names.append(component+'_pc'+str(i))
        principalDf = pd.DataFrame(data = pc_result,columns=pca_cols_names)
        # print(principalDf)
        pca_data= pd.concat([pca_data,principalDf],axis=1)
    if PCAFLAG:
        pca_data = pd.concat([pca_data,data[['Type','Gender','record']]],axis=1)
        data = pca_data
    if PCA_SHIMMER_JITTER:
        for col in list(data.columns):
            if 'shimmer' in col.lower() or 'jitter' in col.lower():
                data.drop(col,axis=1,inplace=True)
        data = pd.concat([data,pca_data],axis=1)
    # print(pca_data)
# exit(1)
# print(data)
# asds = data[data['Type']=='ASD'].sample(len(data[data['Type']=='NT']))
# data = pd.concat([asds,data[data['Type']=='NT']],axis=0)

# print(data)
X = data.drop('Type', axis=1)
y = data['Type']
orig_y =(data.dropna().groupby('Type').count())

# print(y)
X = X.drop('Gender',axis=1)
X = X.drop('record', axis=1)
if REGRESS:
    X = X.drop(scores,axis=1)
X.dropna(axis=0,inplace=True)
NTS = X
# print(X)
# print(X)
# print(y)
# print(X,y)
# kbest  = SelectKBest(f_classif,k=7).fit(X,data['A9'].dropna())
# X = kbest.transform(X)
# DataAlgo.LOO_multi_classify(StandardScaler().fit_transform(X[DataAlgo.RF_feature_reduction(X,y_scores)]),y_scores)
# DataAlgo.LOO_multi_classify(StandardScaler().fit_transform(X),data['A9'].dropna())
# exit(1)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
# summarize feature importance
sc = StandardScaler()
tree.plot_tree(clf.fit(X,y),
               feature_names=list(X.head()),
               class_names=['ASD','NT'],
               filled = True)
clf = RandomForestClassifier(max_depth=100)
clf.fit(sc.fit_transform(X),y)
importance = clf.feature_importances_
# print(importance)
# importance.sort()
# print(type(importance))
best = list()
for i,v in enumerate(clf.feature_importances_):
    if v>0:
        # print('Feature: '+list(X.head())[i]+', Score: %.5f' % (v))
        best.append(list(X.head())[i])
fig.savefig('new vocal tree3 diff m.png')
from sklearn.model_selection import KFold

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print('FOREST')
model = classification_models[3].fit(sc.fit_transform(X),y)
importances = clf.feature_importances_
sorted_importances = np.flip(np.sort(importances))
# print(X.head())
print('Sorted by importance')
features_to_use =[]
for i,v in enumerate(sorted_importances):
    if(v>0):
        for j in np.where(importances==v)[0]:
            features_to_use.append(list(X.head())[j])
            print('Feature: '+list(X.head())[j]+', Score: %.5f' % (v))
    # best.append(list(X.head())[i])
print('Reduced by tree\n',DataAlgo.RF_Iter_feature_reduction(X=X,y=y))
# exit(1)
classification_scores = pd.DataFrame()
print('END FOREST\n')
# exit(1)
oldX = X.copy()
# kbest  = SelectKBest(f_classif,k=30).fit(X,y)
# X = kbest.transform(X)
X = X[features_to_use[:20]]
clf = RandomForestClassifier()
clf.fit(sc.fit_transform(X),y)
importances = clf.feature_importances_
sorted_importances = np.flip(np.sort(importances))
# print(X.head())
print('Sorted by importance')
features_to_use =[]
for i,v in enumerate(sorted_importances):
    if(v>0):
        for j in np.where(importances==v)[0]:
            features_to_use.append(list(X.head())[j])
            print('Feature: '+list(X.head())[j]+', Score: %.5f' % (v))
X = X[features_to_use[:15]]
print(features_to_use[:15])
    # best.append(list(X.head())[i])
# clf = RandomForestClassifier()
# clf.fit(sc.fit_transform(X),y)
# importances = clf.feature_importances_
# sorted_importances = np.flip(np.sort(importances))
# # print(X.head())
# print('Sorted by importance')
# features_to_use =[]
# for i,v in enumerate(sorted_importances):
#     if(v>0):
#         for j in np.where(importances==v)[0]:
#             features_to_use.append(list(X.head())[j])
#             print('Feature: '+list(X.head())[j]+', Score: %.5f' % (v))
# X = X[features_to_use[:12]]
print(len(list(X.columns)))
# print(X)
# print('\n\nSELECTED: ',kbest.get_feature_names_out(oldX.columns),kbest.get_params())

nos = len(X)
runs = 5
for run in range(0,runs):
    print(run)
    for model in classification_models:
        model_name = type(model).__name__
        if run==0:
            classification_scores.at[model_name,'mean'] = 0
            classification_scores.at[model_name,'std'] =  0
            classification_scores.at[model_name,'time'] =  0
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # print(model_name)
        # print(confusion_matrix(y_test,y_pred))
        # print(classification_report(y_test,y_pred))
        # print(accuracy_score(y_test, y_pred))
        from time import time
        start = time()
        scores = cross_val_score(model, sc.fit_transform(X), y, cv=KFold(n_splits=len(X)))
        end = time()
        
        # model.untrain()
        # print(scores)
        # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print(model_name,scores.mean())
        classification_scores.at[model_name,'mean'] = classification_scores.at[model_name,'mean']+scores.mean()/runs
        classification_scores.at[model_name,'std'] =  classification_scores.at[model_name,'std'] + scores.std()/runs
        classification_scores.at[model_name,'time'] = classification_scores.at[model_name,'time'] + (end-start)/runs

classification_scores.index.name = 'model'

# regressor = DecisionTreeRegressor(random_state=0)
# clf = DecisionTreeClassifier(random_state=0)
# y = data.dropna()['Score']
# # print(y.shape)
# # print('before',data.shape)
# X = data.dropna(axis=0).drop(['Type','Gender','record','Score'],axis=1)
# for col in list(X.columns):
#     if 'normed' not in col:
#         X.drop(col,axis=1,inplace=True)
# # print('after',X.shape)
# X_t = sc.fit_transform(X)
# X_t = SelectKBest(mutual_info_regression, k=3).fit_transform(sc.fit_transform(X), y)
# # X = X[best]
# # X_t = sc.fit_transform(X[best])
# tree.plot_tree(clf.fit(sc.fit_transform(X),y),
#                feature_names=list(X.head()),
#                class_names=['3','2','1','0'],
#                filled = True)
# importance = clf.feature_importances_

# # DecisionBoundaryDisplay.from_estimator(clf.fit(X,y),X,response_method="predict",)
# # plt.show()
# # print(importance)
# # importance.sort()
# # print(type(importance))
# best = list()
# for i,v in enumerate(clf.feature_importances_):
#     if v>0:
#         print('Feature: '+list(X.head())[i]+', Score: %.5f' % (v))
#         best.append(list(X.head())[i])
# fig.savefig('new vocal reg tree3 diff m.png')
# X_t = sc.fit_transform(X)

# X = data.drop('Type', axis=1)
# y = data['Type']
# X = X.drop('Gender',axis=1)
# X = X.drop('record', axis=1)
# X = X.drop('Score',axis=1)
# # print(y)
# # tree.plot_tree(regressor.fit(X_t,y),
# #                feature_names=list(X.head()),
# #                class_names=['0','1','2','3'],
# #                filled = True)

# # plt.show()
# # for k in range(4,6):
# #     regression_models.append(KNeighborsRegressor(n_neighbors = k)) 
# scores = cross_val_score(regressor, X_t, y, cv=5)
# regression_scores = pd.DataFrame()
# for model in regression_models:
#     model_name = type(model).__name__
#     scores = cross_val_score(model, X_t, y, cv=5)
#     predictions = cross_val_predict(model,X_t,y,cv=5)
#     count_true = 0 
#     # print(predictions)
#     for i in range(len(predictions)):
#         predict = predictions[i]
#         predict = round(predict) 
#         if predict == list(y)[i]:
#             count_true+=1
#     regression_scores.at[model_name,'rounded Prediction'] = count_true/len(y)
#     regression_scores.at[model_name,'mean'] = scores.mean()
#     regression_scores.at[model_name,'std'] = scores.std()
# regression_scores.index.name = 'model'
# # print('tree regression')
# # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# # print('linear regression')
# # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# # scores = cross_val_score(RandomForestRegressor(), X_t, y, cv=5)
# # print('rForest regression')
# # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# # scores = cross_val_score(SVR(), X_t, y, cv=5, scoring='r2')
# # print('SV regression')
# # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# # scores = cross_val_score(KNeighborsRegressor(n_neighbors=3),X_t,y,cv=5)
# # print('KN regression')
# # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# # from tabulate import tabulate

# class_reg_scores = pd.DataFrame()
# for model in classification_models:
#     model_name = type(model).__name__
#     scores = cross_val_score(model, sc.fit_transform(X), y, cv=5)
#     class_reg_scores.at[model_name,'mean'] = scores.mean()
#     class_reg_scores.at[model_name,'std'] = scores.std()
# class_reg_scores.index.name = 'model'


# loo_regression_scores = pd.DataFrame()
# loo_class_reg_scores = pd.DataFrame()
# loo_classification_scores = pd.DataFrame()
# loo = LeaveOneOut()
# for model in classification_models:
#     model_name = type(model).__name__
#     scores = [] 
#     for train_index, test_index in loo.split(X_t):
#         fitted = model.fit(X_t[train_index],y[train_index])
#         result = fitted.predict(X_t[test_index])
#         scores.append(int(y[test_index]==result))
#         # scores = cross_val_score(model, X_t, y, cv=loo.split(X_t))
#     print(scores)
#     scores = np.asarray(scores)
#     loo_class_reg_scores.at[model_name,'acc'] = scores.sum()/len(scores)
#     # loo_class_reg_scores.at[model_name,'std'] = scores.std()
#     loo_class_reg_scores.index.name = 'model'

# # print('tree regression')
# # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# # print('linear regression')
# # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# # scores = cross_val_score(RandomForestRegressor(), X_t, y, cv=5)
# # print('rForest regression')
# # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# # scores = cross_val_score(SVR(), X_t, y, cv=5, scoring='r2')
# # print('SV regression')
# # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# # scores = cross_val_score(KNeighborsRegressor(n_neighbors=3),X_t,y,cv=5)
# # print('KN regression')
# # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# # from tabulate import tabulate

# print(nos)
print(tabulate(classification_scores.sort_values(by = ['mean','std'],ascending = False), headers = 'keys', tablefmt = 'psql'))
# load_dataFrame(classification_scores.sort_values(by = ['mean','std'],ascending = False))
# print(tabulate(regression_scores.sort_values(by = ['mean','std'],ascending = False), headers = 'keys', tablefmt = 'psql'))
# print(tabulate(class_reg_scores.sort_values(by = ['mean','std'],ascending = False), headers = 'keys', tablefmt = 'psql'))

# print(data.dropna().groupby('Score').count())
print(orig_y['record'])


# # plt.show()
