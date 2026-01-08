from turtle import title
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from Files import *
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler,FunctionTransformer
from copy import deepcopy
from scipy.spatial import distance
from sklearn.model_selection import cross_val_score,LeaveOneOut,cross_val_predict
from LittleGui import load_dataFrame,TestApp
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
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

import pandastable
import tkinter
from itertools import combinations
import matplotlib.pyplot as plt
np.random.seed(0)
kernel = DotProduct() + WhiteKernel()

regression_models = [DecisionTreeRegressor(random_state=0),LinearRegression(),RandomForestRegressor(),SVR(),KNeighborsRegressor(n_neighbors=5)]
regression_models+=[GaussianProcessRegressor(kernel=kernel,random_state=0)]
classification_models = [ DecisionTreeClassifier(max_depth=100), svm.SVC(),RandomForestClassifier(max_depth=10, random_state=0),RandomForestClassifier(max_depth=20, random_state=0),GradientBoostingClassifier(random_state=0)]
classification_models+=[QuadraticDiscriminantAnalysis(),LogisticRegression(solver='liblinear'),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'),LinearDiscriminantAnalysis(solver='lsqr')]
# classification_models+=[MLPClassifier(alpha=0.1, max_iter=2000,solver='lbfgs')]
tree_models = [DecisionTreeClassifier(max_depth=100), DecisionTreeClassifier(max_depth=10), DecisionTreeClassifier(max_depth=5)]
tree_models += [RandomForestClassifier(max_depth=10, random_state=0),RandomForestClassifier(max_depth=100, random_state=0)]
extra_models = [GaussianProcessClassifier(1.0 * RBF(1.0)),GaussianNB(),AdaBoostClassifier(),KNeighborsClassifier(n_neighbors=5), DecisionTreeClassifier(max_depth=10), DecisionTreeClassifier(max_depth=5)]
def append_table_to_pickle(table,file):
    if not exists(file):
        save_pickle_file([table],where=file)
        return
    ls = load_pickle(file)
    for i,item in enumerate(ls):
        if table[0] == item[0]:
            del ls[i]
    ls.append(table)
    save_pickle_file(ls,where=file)

def sort_acsending(array):
    sorted_array = np.flip(np.sort(array))
    return sorted_array

def statistics_feature_reduction(X,y,rating_func =f_classif,features=10,plot= False):
    kbest  = SelectKBest(rating_func,k=features).fit(X,y)
    mask = kbest.get_support()
    new_features = X.columns[mask]
    if plot:
        k_f = dict(names =list(new_features),scores = list(kbest.scores_[mask]))
        kf_df = pd.DataFrame(k_f)
        kf_df.groupby(['names']).median().sort_values("scores",ascending=False).plot.bar()
        plt.show()
    # print([new_features,kbest.scores_[mask]])
    return(list(new_features))
# def DF_feature_reduction(X,y,clf = DecisionTreeClassifier(),features =10,plot = False):
    
from heapq import nlargest
def RF_feature_reduction(X,y,clf = RandomForestClassifier(max_depth=10,max_features=None),steps = [20,15,10],compare_table = None,plot=False):
    features_to_use =[]
    compare_flag = compare_table is not None
    model = clf
    df = pd.DataFrame(columns=['feature','score']).set_index('feature')
    for step in steps:
        features_to_use = []
        # model = RandomForestClassifier(max_depth=max(10,int(X.shape[1]/2)),max_features=None,n_estimators=50)
        # model = RandomForestClassifier(max_features=None,max_depth=10)

        model.fit(X,y)
        importances = model.feature_importances_
        importances = sort_acsending(importances)

        # .nlargest(2, enumerate(seq), key=lambda x: x[1])
        # largest = nlargest(step, enumerate(clf.feature_importances_), key=lambda x: x[1])
        # for v in largest:
        #     print(list(X.head())[int(v[0])])
        X_head_list = list(X.columns)
        if(step==1):
            print("debug")
        for i,v in enumerate(importances):
            if i >= step:
                break
            for j in np.where(model.feature_importances_==v)[0]:
                features_to_use.append(X_head_list[j])
                if plot:
                    df.at[features_to_use[-1],'score']=v
                if len(features_to_use)>step:
                    break
                if compare_flag:
                    compare_table.at[X_head_list[j],str(steps.index(step))+" ("+str(step)+")"] = v
        features_to_use = features_to_use[0:step]
        X = X.drop(columns=[col for col in X if col not in features_to_use])
        print(X.shape)
        if compare_flag:
            if step > 15:
                compare_table.at['LOO mean_acc',str(steps.index(step))+" ("+str(step)+")"] = 0
                compare_table.at['train_acc',str(steps.index(step))+" ("+str(step)+")"] = 0
                continue
            result = CV_model_classify(X,y,model,runs=5)
            compare_table.at['LOO mean_acc',str(steps.index(step))+" ("+str(step)+")"] = result.at[(type(model).__name__,model.get_params()['max_depth']),'mean']
            compare_table.at['train_acc',str(steps.index(step))+" ("+str(step)+")"] = result.at[(type(model).__name__,model.get_params()['max_depth']),'train_acc']

    # print([features_to_use,importances[0:steps[-1]]]
    # if)
    if compare_flag:
        cols = list(compare_table.columns)
        cols.reverse()
        compare_table.sort_values(by = cols ,ascending=False,inplace=True)
    if plot:
        print(df)
        df.groupby(['feature']).median().sort_values("score",ascending=False).plot.bar()
    return features_to_use
def RF_Iter_feature_reduction(X,y,clf = RandomForestClassifier(max_depth=100),steps = [20,15,10]):
    features_to_use =[]
    fl = []
    for step in steps:
        # print(step)
        features_to_use = []
        clf.fit(X,y)
        importances = sort_acsending(clf.feature_importances_)
        for i,v in enumerate(importances):
            if(v>0):
                for j in np.where(clf.feature_importances_==v)[0]:
                    features_to_use.append(list(X.head())[j])
        features_to_use = features_to_use[0:step]
        # print(features_to_use)
        fl.append(features_to_use.copy())
        X = X.drop(columns=[col for col in X if col not in features_to_use])
    # print(fl)
    return fl
def LOO_model_classify(X,y,clf,runs=1): 
    classification_scores = pd.DataFrame(columns = ['model','params','train_acc','mean','std'])
    classification_scores.set_index(['model','params'],inplace=True)
    sc = StandardScaler()
    for run in range(0,runs):
        print(run)
        model = clf
        model_name = type(model).__name__
        params = ""
        if 'tree' in  model_name.lower() or 'forest' in model_name.lower():
            params= model.get_params()['max_depth']
        loo = LeaveOneOut()
        if run==0:
            classification_scores.at[(model_name,params),'mean'] = 0
            classification_scores.at[(model_name,params),'std'] =  0
            classification_scores.at[(model_name,params),'train_acc'] =  0
        scores = cross_val_score(model, sc.fit_transform(X), y, cv=loo)
        model.fit(sc.fit_transform(X),y)
        ys = model.predict(sc.fit_transform(X))
        acc = np.count_nonzero(ys==y)/len(y)
        classification_scores.at[(model_name,params),'mean'] = classification_scores.at[(model_name,params),'mean']+scores.mean()/runs
        classification_scores.at[(model_name,params),'std'] =  classification_scores.at[(model_name,params),'std'] + scores.std()/runs
        classification_scores.at[(model_name,params),'train_acc'] =  classification_scores.at[(model_name,params),'train_acc'] + acc/runs
    return classification_scores

# def CV_model_classify_pca(X,y,runs=1,cross_val=LeaveOneOut,pca_n = 2):

def CV_model_classify(X,y,clf,runs=1,cross_val=LeaveOneOut(),mode = 'semi',ret_clf = False): 
    classification_scores = pd.DataFrame(columns = ['model','params','train_acc','mean','std'])
    classification_scores.set_index(['model','params'],inplace=True)
    sc = StandardScaler()
    # y = pd.DataFrame(y)
    # y.columns=['Type']
    # data = pd.concat([X,y],axis=1)
    # # print(data)
    # # print(y)
    # # input()
    # data.sort_values(by=['Type'],ascending=False,inplace=True)
    # data = data.sample(frac=1).copy()
    # X = data.drop(['Type'],axis=1)
    # y = data['Type']
    # exit(1)
    for run in range(0,runs):
        print(run)
        model = clf
        model_name = type(model).__name__
        params = ""
        if 'tree' in  model_name.lower() or 'forest' in model_name.lower() or 'gradient' in model_name.lower():
            params= model.get_params()['max_depth']
        if 'Linear' in  model_name:
            parmas = "No Shrinkage"
            if model.get_params()['shrinkage']:
                params= model.get_params()['shrinkage']
        if run==0:
            classification_scores.at[(model_name,params),'mean'] = 0
            classification_scores.at[(model_name,params),'std'] =  0
            classification_scores.at[(model_name,params),'train_acc'] =  0
        if type(cross_val) is not type(LeaveOneOut()):
            scores = cross_val_score(model, X, y, cv=cross_val)
        else:
            scores = LOOcross_val_score(model,X,y)
        # print(scores)
        model.fit(X,y)
        ys = model.predict(X)
        # ys = pd.DataFrame(ys)
        acc = np.count_nonzero(ys==y)/len(y)
        classification_scores.at[(model_name,params),'mean'] = classification_scores.at[(model_name,params),'mean']+scores.mean()/runs
        classification_scores.at[(model_name,params),'std'] =  classification_scores.at[(model_name,params),'std'] + scores.std()/runs
        classification_scores.at[(model_name,params),'train_acc'] =  classification_scores.at[(model_name,params),'train_acc'] + acc/runs
        if ret_clf:
            return classification_scores,model
    return classification_scores
# def cross_val_score2(clf,X,y,cv):
import os
import os.path
def RandomizationTest(X,y):
    clfs = [RandomForestClassifier(max_depth=10),LogisticRegression()]
    clfs = [LogisticRegression()]
    clfs = [RandomForestClassifier(max_depth=10)]
    cvs = [LeaveOneOut(),5]
    rx_ry_df = pd.DataFrame(columns=['catogory','run#'])
    x_ry_df = pd.DataFrame(columns=['catogory','run#'])
    rx_y_df = pd.DataFrame(columns=['catogory','run#'])
    row = 0
    full_results = pd.DataFrame(columns = ['model','X size','cv','data-----','mean','std','train_acc']).set_index(['model','X size','cv','data-----'])
    img_fl = 'RandomExperiment/'
    if not os.path.exists(img_fl):
        os.mkdir(img_fl)
    for k in range(1,2):
        rx_ry_means = list()
        x_ry_means = list()
        rx_y_means = list()
        for n_test in range(0,25):
            print('Running test ',n_test)
            RandomX = pd.DataFrame(np.random.uniform(0,1,size=(X.shape[0]*k, 20)), columns=list('ABCDEFGHIJKLMNOPQRST'))
            rand_array = np.empty(k*X.shape[0])
            rand_array.fill(1)
            rand_array[:int(k*X.shape[0]/2)+1].fill(2)
            np.random.shuffle(rand_array)
            series = pd.Series(rand_array)
            RandomY  = series
            for clf in clfs:
                model_name = type(clf).__name__
                params = ""
                if 'tree' in  model_name.lower() or 'forest' in model_name.lower() or 'gradient' in model_name.lower():
                    params= clf.get_params()['max_depth']
                if 'Linear' in  model_name:
                    parmas = "No Shrinkage"
                    if clf.get_params()['shrinkage']:
                        params= clf.get_params()['shrinkage']
                for cv in cvs:

                    if type(cv) == type(LeaveOneOut()):
                        cv_n = 'l'
                    else:
                        cv_n = str(cv)
                    for n_features in range(1,8):
                        rx_ry_results = CV_model_classify(RandomX[statistics_feature_reduction(RandomX,RandomY,features=n_features)],RandomY,clf,5,cross_val=cv)
                        rx_ry_means.append(rx_ry_results.at[(model_name,params),'mean'])
                        rx_ry_df.at[n_test,str(model_name[0])+'_'+cv_n+' '+str(k)+' '+str(n_features)] = rx_ry_results.at[(model_name,params),'mean']
                        full_results.at[(model_name,RandomX.shape[0],cv_n,'RX&RY'),'mean'] = rx_ry_results.at[(model_name,params),'mean']
                        full_results.at[(model_name,RandomX.shape[0],cv_n,'RX&RY'),'std'] = rx_ry_results.at[(model_name,params),'std']
                        full_results.at[(model_name,RandomX.shape[0],cv_n,'RX&RY'),'train_acc'] = rx_ry_results.at[(model_name,params),'train_acc']
                        if k==1:
                            continue
                            # rx_y_results = CV_model_classify(RandomX[statistics_feature_reduction(RandomX,y,features=n_features)],y,clf,5,cross_val=cv)
                            # rx_y_means.append(rx_y_results.at[(model_name,params),'mean'])
                            # rx_y_df.at[n_test,model_name+'_'+cv_n+' '+str(k)+' '+str(n_features)] = rx_y_results.at[(model_name,params),'mean']
                            # full_results.at[(model_name,RandomX.shape[0],cv_n,'RX&Y'),'mean'] = rx_y_results.at[(model_name,params),'mean']
                            # full_results.at[(model_name,RandomX.shape[0],cv_n,'RX&Y'),'std'] = rx_y_results.at[(model_name,params),'std']
                            # full_results.at[(model_name,RandomX.shape[0],cv_n,'RX&Y'),'train_acc'] = rx_y_results.at[(model_name,params),'train_acc']
                            # x_ry_results = CV_model_classify(X[statistics_feature_reduction(X,RandomY,features=n_features)],RandomY,clf,5,cross_val=cv)
                            # x_ry_means.append(x_ry_results.at[(model_name,params),'mean'])
                            # x_ry_df.at[n_test,model_name+'_'+cv_n+' '+str(k)+' '+str(n_features)] = x_ry_results.at[(model_name,params),'mean']
                            # full_results.at[(model_name,X.shape[0],cv_n,'X&RY'),'mean'] = x_ry_results.at[(model_name,params),'mean']
                            # full_results.at[(model_name,X.shape[0],cv_n,'X&RY'),'std'] = x_ry_results.at[(model_name,params),'std']
                            # full_results.at[(model_name,X.shape[0],cv_n,'X&RY'),'train_acc'] = x_ry_results.at[(model_name,params),'train_acc']
                        # x_y_results  = CV_model_classify(X,y,clf,5,cv)
                        # full_results.at[(model_name,X.shape[0],cv_n,'X&Y'),'mean'] = x_y_results.at[(model_name,params),'mean']
                        # full_results.at[(model_name,X.shape[0],cv_n,'X&Y'),'std'] = x_y_results.at[(model_name,params),'std']
                        # full_results.at[(model_name,X.shape[0],cv_n,'X&Y'),'train_acc'] = x_y_results.at[(model_name,params),'train_acc']    
    rx_ry_df.plot.box()
    plt.savefig(img_fl+'rx_ry_.png')
    plt.tight_layout()
    plt.show()
    plt.clf()
    rx_ry_df.to_csv('rx_ry_df.csv')
    # rx_y_df.plot.box()
    # plt.savefig(img_fl+'rx_y_.png')
    # plt.tight_layout()
    # plt.show()
    # plt.clf()
    # rx_y_df.to_csv('rx_y_df.csv')
    x_ry_df.plot.box()
    plt.savefig(img_fl+'2x_ry_'+str(k)+'.png')
    plt.tight_layout()
    plt.show()
    plt.clf()
    x_ry_df.to_csv('x_ry_df2.csv')
    TestApp([('Random',full_results)])
    append_table_to_pickle(('Random test',full_results),'list_of_important_tables.pickle')








def LOO_multi_classify(X,y,runs = 1,print_results=True):
    classification_scores = pd.DataFrame(columns = ['model','params','mean','std'])
    classification_scores.set_index(['model','params'],inplace=True)
    sc = StandardScaler()
    for run in range(0,runs):
        print(run)
        for model in classification_models:
            model_name = type(model).__name__
            params = ""
            if 'tree' in  model_name.lower() or 'forest' in model_name.lower():
                params= model.get_params()['max_depth']
            if 'Linear' in  model_name:
                parmas = "No Shrinkage"
                if model.get_params()['shrinkage']:
                    params= model.get_params()['shrinkage']
            if run==0:
                classification_scores.at[(model_name,params),'mean'] = 0
                classification_scores.at[(model_name,params),'std'] =  0
            scores = cross_val_score(model, sc.fit_transform(X), y, cv=KFold(n_splits=len(X)))
            classification_scores.at[(model_name,params),'mean'] = classification_scores.at[(model_name,params),'mean']+scores.mean()/runs
            classification_scores.at[(model_name,params),'std'] =  classification_scores.at[(model_name,params),'std'] + scores.std()/runs

    # classification_scores.index.name = 'model'
    if print_results:
        print_class_repot(classification_scores)
    return classification_scores
def CV_multi_classify(X,y,runs = 1,cross_val = LeaveOneOut(),print_results=True):
    classification_scores = pd.DataFrame()
    sc = StandardScaler()
    for model in classification_models:
        classification_scores=pd.concat([classification_scores,CV_model_classify(X,y,model,runs,cross_val)],axis=0)
    print(classification_scores)
    if print_results:
        print_class_repot(classification_scores)
    return classification_scores
                # print('Feature: '+list(X.head())[j]+', Score: %.5f' % (v))


def print_class_repot(classification_scores):
    print(tabulate(classification_scores.sort_values(by = ['mean','std'],ascending = False), headers = 'keys', tablefmt = 'psql'))

def PCA_on_Tuppels(components,data):
    pca_data = pd.DataFrame()
    for component,n in components:
        pca_model = PCA(n_components=n) 
        component_list = []
        for col in list(data.columns):
            if strings_in(component,col):
                component_list.append(col)
        
        # print(component_list)
        sc = StandardScaler()
        pc_result = pca_model.fit_transform(sc.fit_transform(data[component_list]))
        pca_cols_names =[]
        for i in range(1,n+1):
            pca_cols_names.append(unroll_strs(component)+'-pc'+str(i))
        principalDf = pd.DataFrame(pc_result,index=data.index)
        principalDf.columns = pca_cols_names
        pca_data= pd.concat([pca_data,principalDf],axis=1)
    return pca_data
def get_all_cols_with(df,col):
    cols = []
    for column in list(df.columns):
        if strings_in(col,column):
            cols.append(column)
    return cols
def strings_in(strs,lookup):
    if type(strs) is not list:
        strs = [strs]
    for str in strs:
        if str.lower() not in lookup.lower():
            return False
    return True
def drop_all_cols_with(df,cols):
    df_copy = df.copy()
    if type(cols) is not list:
        cols = [cols]
    for col in cols:
        df_copy.drop(get_all_cols_with(df,col),inplace=True,axis=1)

    return df_copy
def unroll_strs(strs):
    ret_str = ""
    if type(strs) is not list:
        strs = [strs]
    for str in strs:
        ret_str+='_'+str
    return ret_str[1:]
from sklearn.model_selection import LeavePOut
# from sklearn import cross_validxation
def LOOcross_val_score(clf,X,y):
    loo = LeaveOneOut()
    # clf = LogisticRegression(solver='liblinear')
    # data = pd.concat([X,y],axis=1)
    correct =0
    
    results = []
    sc = StandardScaler()
    # X = sc.fit_transform(X)
    for train_idx,test_idx in loo.split(X):
        # cX = data.drop(list(y.columns),axis=1)
        # cy = data[list(y.columns)]
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        # print(X_train,y_train)
        clf.fit(X_train,y_train)
        yp = clf.predict(X_test)
        # print(str(yp),str(y_test),yp[0],y_test)
        results.append(np.count_nonzero(yp==y_test))
        # from sklearn.metrics import accuracy_score
        # accuracy = accuracy_score(yp, y_test)
        # input(accuracy)
        # acc = np.count_nonzero(yp==y_test)
        # print(X_train,y_train)
        # print('yp',yp)
        # print('ytest',y_test)
        # print(yp==y_test)
        # correct+=acc
        # input(0)
    # print(correct/len(X))
    results = np.asarray(results,dtype=np.int8)
    # print(results.mean(),results.std())
    return results
   
def balance_df_by_col(df,column):
    groups = df.groupby(column).count()
    grouped_df = list(groups.index)
    dfs = []
    for ind in grouped_df:
        dfs.append(df[df[column]==ind])
    min_len = len(dfs[0])
    for d in dfs:
        if(len(d)<min_len):
            min_len = len(d)
    for i in range(len(dfs)):
        current_df = dfs[i]
        current_len = len(current_df)
        if current_len>min_len:
            dif = current_len-min_len
            to_remove = np.random.choice(current_df.index,size=dif,replace=False)
            dfs[i] = current_df.drop(to_remove).copy()
    balanced_df = pd.concat(dfs,axis=0)
    print(str(balanced_df))
    # input()
    return balanced_df
def cross_val_feature_check(X,y,n_features=10,s_func = f_classif,cross_val = 2,balance_col = 'Type',plot = False,return_rank = False):
    if type(y)==type(pd.Series()):
        y=y.to_frame()
    data = pd.concat([X,y],axis=1)
    data = balance_df_by_col(data,balance_col)
    if 'Type' in  list(y.columns):
        X_NT = data[data['Type']=='NT'].drop(['Type'],axis=1)
        y_NT = data[data['Type']=='NT']['Type']
        X_ASD = data[data['Type']=='ASD'].drop(['Type'],axis=1)
        y_ASD = data[data['Type']=='ASD']['Type']
    else:
        X_NT = data[data['A2Type']==False].drop(['A2Type'],axis=1)
        y_NT = data[data['A2Type']==False]['A2Type']
        X_ASD = data[data['A2Type']==True].drop(['A2Type'],axis=1)
        y_ASD = data[data['A2Type']==True]['A2Type']
    lpo = LeavePOut(2)
    nt_split_ls = []
    asd_split_ls = []
    for train_index, test_index in lpo.split(X_NT):
        nt_split_ls.append((train_index,test_index))
    for train_index, test_index in lpo.split(X_ASD):
        asd_split_ls.append((train_index,test_index))
    counter = pd.DataFrame(columns=['feature','counter']).set_index(['feature'])
    for feature in X.head():
        counter.at[feature,'counter'] =0
    from tqdm import tqdm
    for i in tqdm(range(len(nt_split_ls))):
        (nt_train_idx,nt_test_idx) = nt_split_ls[i]
        (asd_train_idx,asd_test_idx) = asd_split_ls[i]
        X_subset = pd.concat([X_NT.iloc[nt_train_idx],X_ASD.iloc[asd_train_idx]],axis=0)
        y_subset = pd.concat([y_NT.iloc[nt_train_idx],y_ASD.iloc[asd_train_idx]],axis=0)
        features = statistics_feature_reduction(X_subset,y_subset,rating_func=s_func,features=n_features)
        for feature in features:
            counter.at[feature,'counter']+=1
    counter = counter.sort_values(by=['counter'],ascending=False)[counter['counter']>0]
    barplt = counter.plot.bar()
    b_features = statistics_feature_reduction(X,y,rating_func=s_func,features=10)
    if plot:
        plt.show()
    if return_rank:
        return counter
import gc
def cross_val_features_reduction(X,y,n_features=10,func = cross_val_feature_check,scale = False):
    if scale:
        X.values[:] = StandardScaler().fit_transform(X) 
    return list(func((X),y,return_rank = True)[:n_features].index)

def cross_val_feature_check_RF(X,y,steps1,cross_val = 2,clf=RandomForestClassifier(max_depth=100),plot=False,filename = 'cv_fcRF'):
    if type(y)==type(pd.Series()):
        y=y.to_frame()
    data = pd.concat([X,y],axis=1)
    if 'Type' in  list(y.columns):
        X_NT = data[data['Type']=='NT'].drop(['Type'],axis=1)
        y_NT = data[data['Type']=='NT']['Type']
        X_ASD = data[data['Type']=='ASD'].drop(['Type'],axis=1)
        y_ASD = data[data['Type']=='ASD']['Type']
    else:
        X_NT = data[data['A2']==False].drop(['A2'],axis=1)
        y_NT = data[data['A2']==False]['A2']
        X_ASD = data[data['A2']==True].drop(['A2'],axis=1)
        y_ASD = data[data['A2']==True]['A2']
    drop_indices = np.random.choice(X_NT.index, len(X_NT)-len(X_ASD), replace=False)
    X_NT.drop(drop_indices,inplace=True)
    y_NT.drop(drop_indices,inplace=True)
    print(X_ASD.shape)
    print(X_NT.shape)
    # input()
    lpo = LeavePOut(2)
    nt_split_ls = []
    asd_split_ls = []
    for train_index, test_index in lpo.split(X_NT):
        nt_split_ls.append((train_index,test_index))
    for train_index, test_index in lpo.split(X_ASD):
        asd_split_ls.append((train_index,test_index))
    counter = pd.DataFrame(columns=['feature','counter']).set_index(['feature'])
    for feature in X.head():
        counter.at[feature,'counter'] =0
    print(counter,len(nt_split_ls))
    worked = 0 
    # input()
    from tqdm import tqdm

    for i in tqdm(range(len(nt_split_ls))):
        print(i,'/',len(nt_split_ls))
        (nt_train_idx,nt_test_idx) = nt_split_ls[i]
        (asd_train_idx,asd_test_idx) = asd_split_ls[i]
        X_subset = pd.concat([X_NT.iloc[nt_train_idx],X_ASD.iloc[asd_train_idx]],axis=0)
        y_subset = pd.concat([y_NT.iloc[nt_train_idx],y_ASD.iloc[asd_train_idx]],axis=0)
        print('rf')
        print(steps1)
        features = RF_feature_reduction(X_subset,y_subset,clf,steps=steps1)
        if feature == "ERROR":
            continue
        worked+=1
        gc.collect()
        print('rfdone')
        for feature in features:
            counter.at[feature,'counter']+=1

    counter = counter.sort_values(by=['counter'],ascending=False)[counter['counter']>0]
    barplt = counter.plot.bar()
    b_features = RF_feature_reduction(X,y,clf,steps=steps1)
    # print(b_features)
    # print(barplt)
    # print(worked)
    if plot:
        plt.show()
    plt.tight_layout()
    plt.savefig(filename+".png")

from itertools import combinations
def statistical_features_combinations_test(X,y,n_features = 30,n_sample = 4):
    all_results = pd.DataFrame()
    all_results.index.name = '#comb'
    best_featrues = statistics_feature_reduction(X,y,features = n_features)
    i=0
    where_feature_at = dict()
    feature_series = dict()
    for bf in best_featrues:
        where_feature_at[bf] = list()
    for comb in combinations(best_featrues,n_sample):
        result = CV_model_classify(X[list(comb)],y,clf =LogisticRegression(solver='liblinear'),runs = 1)
        for n,c in enumerate(comb):
            all_results.at[i,'Feature#'+str(n+1)] = c
            where_feature_at[c].append(i)
        print(result)
        all_results.at[i,'MeanAcc'] = result.iloc[0,:]['mean']
        i+=1

    results_serires = pd.DataFrame()
    for best in best_featrues:
        feature_series[best] = all_results.iloc[where_feature_at[best]]['MeanAcc']
        results_serires[best] = all_results.iloc[where_feature_at[best]]['MeanAcc']
    print(results_serires)
    results_serires.plot.box()
    plt.tight_layout()
    plt.savefig('featre combination compare.png')
    plt.show()
    
    all_results.to_csv('combination compare.csv')

def classify_by_task(X,y,tasks,fr_func = None,runsp = 1):
    stepss=[*range(100, 21, -10)]+[*range(20,11, -5)] + [*range(10, 2, -1)] 
    results = [] 
    if type(tasks) is not list:
        tasks = [tasks]
    for task in tasks:
        tasks_copy = tasks.copy()
        tasks_copy.remove(task)
        if(len(tasks_copy)>0):
            newX = drop_all_cols_with(X,tasks_copy)
        # print(list(newX.columns))
        if fr_func:
            results.append((task,CV_multi_classify(newX[fr_func(newX,y,steps = stepss)],y,print_results=False,runs= runsp)))
        else:
            results.append((task,CV_multi_classify(newX,y,print_results=False,runs= runsp)))
    return results
def diffs_by_tasks(data,tasks):
    diffs = pd.DataFrame()
    once = True
    created_cols = []
    for task,sub_task in list(combinations(tasks, 2)):
        task_cols = get_all_cols_with(data,task)
        sub_task_cols = get_all_cols_with(data,sub_task)
        for i,col in enumerate(task_cols):
            sub_col = sub_task_cols[i]
                # print(data[col])
                # print(data[sub_col])
            if task+"-"+sub_task+'_'+col.split("_")[-1] in created_cols:
                print('error ', task+"-"+sub_task+'_'+col.split("_")[-1])
                exit(1)
            diffs[task+"-"+sub_task+'_'+col.split("_")[-1]] = data[col]-data[sub_col];
            # if once:
            #     print('data\n',data)
            #     print('col&subcol\n',col,sub_col)
            #     print('col\n',data[col])
            #     print('subcol\n',data[sub_col])
            #     print('diff\n',data[col]-data[sub_col])
            #     print('table\n',diffs)
            #     once = False
    # exit(1)
    return diffs
from sklearn.inspection import permutation_importance

def test_permute_importane(X,y):
    feature_names = RF_feature_reduction(X,y,steps=[30])
    print(feature_names)
    forest = RandomForestClassifier(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X[feature_names], y, stratify=y, random_state=42)
    forest.fit(X_train, y_train)
    start_time = time.time()
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

def test_RF_reduction_quality(X,y,clf = RandomForestClassifier(max_depth=100),steps = None,from_last = 'all',pca_flag = False,beta = 0.9,cv = LeaveOneOut()):
    if not steps:
        start = len(list(X.columns))
        steps =set()
        steps.add(1)
        steps.add(len(list(X.columns)))
        while(start*beta>1): # beta = 0.9 as default
            start = int(start*beta)
            steps.add(start)
        steps = list(steps)
        steps.sort(reverse=True)
        print(steps)
    features_lists = RF_Iter_feature_reduction(X,y,clf,steps)
    if from_last !='all':
        features_lists = features_lists[len(features_lists)-from_last:]
    classification_scores = pd.DataFrame(columns = ['model','params','#features','mean','std'])
    classification_scores.set_index(['model','params','#features'],inplace=True)
    for m,model in enumerate(classification_models):
        params=""
        model_name = type(model).__name__
        print('model: ',m,'/ ',len(classification_models))
        if 'tree' in model_name.lower() or 'forest' in  model_name.lower():
            params= model.get_params()['max_depth']
        if 'Linear' in  model_name:
            parmas = "No Shrinkage"
            if model.get_params()['shrinkage']:
                params= model.get_params()['shrinkage']
        model_name = type(model).__name__
        for f,features_list in enumerate(features_lists):
            if len(features_list)>30:
                continue
            print('features: ',f,'/ ',len(features_lists))

            scores = CV_model_classify(X[features_list],y,model,runs=5,cross_val=cv)
            # print(scores)
            classification_scores.at[(model_name,params,len(features_list)),'mean'] = scores.at[(model_name,params),'mean']
            classification_scores.at[(model_name,params,len(features_list)),'std'] = scores.at[(model_name,params),'std']
            classification_scores.at[(model_name,params,len(features_list)),'train_acc'] = scores.at[(model_name,params),'train_acc']
    return classification_scores.sort_values(by=['mean','std'],ascending=False)

def test_statistics_reduction_quality(X,y,rf_func = f_classif,from_last = 'all',pca_flag = False,clfs = None,cv = LeaveOneOut()):
    classification_scores = pd.DataFrame(columns = ['model','params','#features','mean','std'])
    classification_scores.set_index(['model','params','#features'],inplace=True)
    if not clfs:
        clfs = classification_models
    if type(clfs) is not list:
        clfs = [clfs]
    featurs_range = [*range(len(list(X.columns)),21,-10)]+[*range(20,11,-5)]+[*range(10,0,-1)]
    if from_last !='all':
        featurs_range = featurs_range[-from_last:]
    for f,i in enumerate([*range(len(list(X.columns)),21,-10)]+[*range(20,11,-5)]+[*range(10,0,-1)]):
        print("features : ",f,'/',len([*range(len(X),21,-10)]+[*range(20,11,-5)]+[*range(10,0,-1)]))
        feat = i 
        if i > len(list(X.columns)):
            feat = 'all'
        if i>30: 
            continue
        feature_list = statistics_feature_reduction(X,y,rating_func=rf_func,features=feat)
        # print(feature_list)
        for m,model in enumerate(clfs):
            print('m','/',len(clfs))
            params=""
            model_name = type(model).__name__
            if 'tree' in model_name.lower() or 'forest' in  model_name.lower():
                params= model.get_params()['max_depth']
            if 'Linear' in  model_name:
                parmas = "No Shrinkage"
                if model.get_params()['shrinkage']:
                    params= model.get_params()['shrinkage']
            scores = CV_model_classify(X[feature_list],y,model,runs=5,cross_val = cv)
            classification_scores.at[(model_name,params,len(feature_list)),'mean'] = scores.at[(model_name,params),'mean']
            classification_scores.at[(model_name,params,len(feature_list)),'std'] = scores.at[(model_name,params),'std']
            classification_scores.at[(model_name,params,len(feature_list)),'train_acc'] = scores.at[(model_name,params),'train_acc']
    # print(classification_scores)
    return classification_scores.sort_values(by=['mean','std'],ascending=False)

def test_RF_reduction_quality_pca(X,y,clf = RandomForestClassifier(max_depth=100),steps = [*range(100,19,-10)]+[*range(20,4,-5)],from_last = 'all'):
    classification_scores = pd.DataFrame(columns = ['model','params','#features','#components','mean','std'])
    classification_scores.set_index(['model','params','#features','#components'],inplace=True)
    featurs_range = [*range(X.shape[1], 31, -10)]+[*range(30, 11, -5)]+[*range(10, 0, -1)]
    features_lists = RF_Iter_feature_reduction(X,y,steps=featurs_range)
    if from_last!='all':
        features_lists= features_lists[-(from_last+1):]
    for model in classification_models:
        print(model)
        params=""
        model_name = type(model).__name__
        if 'tree' in model_name.lower() or 'forest' in  model_name.lower():
            params= model.get_params()['max_depth']
        if 'Linear' in  model_name:
            parmas = "No Shrinkage"
            if model.get_params()['shrinkage']:
                params= model.get_params()['shrinkage']
        model_name = type(model).__name__
        for features_list in features_lists:
            if len(features_list) >20:
                pca_n = min(int(len(features_list)/2),X.shape[0])
            else:
                pca_n = min(min(int(len(features_list)),10),X.shape[0])
            pca = PCA(n_components=pca_n)   
            Xpca = pca.fit_transform(X=(X[features_list]))
            features_list2 = [] 
            for i in range(1,pca_n+1):
                features_list2.append('p'+str(i))
            Xpca= pd.DataFrame(Xpca,columns = features_list2)
            for pcan_i in range(pca_n,0,-1):
                features_list3 = []
                for j in range(1,pcan_i+1):
                    features_list3.append('p'+str(j))
                scores = CV_model_classify(Xpca[features_list3],y,model,runs=5)
                classification_scores.at[(model_name,params,len(features_list),pcan_i),'mean'] = scores.at[(model_name,params),'mean']
                classification_scores.at[(model_name,params,len(features_list),pcan_i),'std'] = scores.at[(model_name,params),'std']
                classification_scores.at[(model_name,params,len(features_list),pcan_i),'train_acc'] = scores.at[(model_name,params),'train_acc']
    return classification_scores.sort_values(by=['mean','std'],ascending=False)
def test_statistical_reduction_quality_pca(X,y, cv = LeaveOneOut()):
    classification_scores = pd.DataFrame(columns = ['model','params','#features','#components','mean','std'])
    classification_scores.set_index(['model','params','#features','#components'],inplace=True)
    featurs_range = range(30,1,-1)
    features_lists = []
    for i in featurs_range:
        features_lists.append(statistics_feature_reduction(X,y,features=i))
    for model in tree_models:
        print(model)
        params=""
        model_name = type(model).__name__
        if 'tree' in model_name.lower() or 'forest' in  model_name.lower():
            params= model.get_params()['max_depth']
        if 'Linear' in  model_name:
            parmas = "No Shrinkage"
            if model.get_params()['shrinkage']:
                params= model.get_params()['shrinkage']
        model_name = type(model).__name__
        for features_list in features_lists:
            if len(features_list) >20:
                pca_n = min(int(len(features_list)/2),X.shape[0])
            else:
                pca_n = min(min(int(len(features_list)),10),X.shape[0])
            pca = PCA(n_components=pca_n)   
            Xpca = pca.fit_transform(X=(X[features_list]))
            features_list2 = [] 
            for i in range(1,pca_n+1):
                features_list2.append('p'+str(i))
            Xpca= pd.DataFrame(Xpca,columns = features_list2)
            for pcan_i in range(pca_n,0,-1):
                features_list3 = []
                for j in range(1,pcan_i+1):
                    features_list3.append('p'+str(j))
                scores = CV_model_classify(Xpca[features_list3],y,model,runs=10,cross_val=cv)
                classification_scores.at[(model_name,params,len(features_list),pcan_i),'mean'] = scores.at[(model_name,params),'mean']
                classification_scores.at[(model_name,params,len(features_list),pcan_i),'std'] = scores.at[(model_name,params),'std']
                classification_scores.at[(model_name,params,len(features_list),pcan_i),'train_acc'] = scores.at[(model_name,params),'train_acc']
    return classification_scores.sort_values(by=['mean','std'],ascending=False)

def test_statistics_reduction_quality_pca(X,y,rf_func = f_classif,from_last = 'all'):
    classification_scores = pd.DataFrame(columns = ['model','params','#features','mean','std'])
    classification_scores.set_index(['model','params','#features'],inplace=True)
    featurs_range = [*range(100,21,-10)]+[*range(20,11,-5)]+[*range(10,0,-1)]
    if from_last !='all':
        featurs_range = featurs_range[-from_last:]
    for i in featurs_range:
        print("features : ",i,'\n')
        feat = i 
        if i > len(list(X.columns)):
            feat = 'all'
        feature_list = statistics_feature_reduction(X,y,rating_func=rf_func,features=feat)

        for model in classification_models:
            params=""
            model_name = type(model).__name__
            if 'tree' in model_name.lower() or 'forest' in  model_name.lower():
                params= model.get_params()['max_depth']
            scores = CV_model_classify(X[feature_list],y,model,runs=10)
            classification_scores.at[(model_name,params,len(feature_list)),'mean'] = scores.at[(model_name,params),'mean']
            classification_scores.at[(model_name,params,len(feature_list)),'std'] = scores.at[(model_name,params),'std']
            classification_scores.at[(model_name,params,len(feature_list)),'train_acc'] = scores.at[(model_name,params),'train_acc']
    # print(classification_scores)
    return classification_scores.sort_values(by=['mean','std'],ascending=False)
from scipy.special import expit
def get_z_df(df,to_drop):
    df_data = df.drop(to_drop,axis=1)
    df_data.values[:]= StandardScaler().fit_transform(df_data)
    to_concat = [df_data]
    for col in to_drop:
        to_concat.append(df[col])
    z_df = pd.concat(to_concat,axis=1)
    return z_df
def test_LogisticRegression(X,y,k=6):
    sc = StandardScaler()
    log_reg = LogisticRegression(solver='liblinear')
    features_list = statistics_feature_reduction(X,y,features=k)
    pca = PCA(n_components=3)
    Xpca = pca.fit_transform(X=sc.fit_transform(X[features_list]))
    features_list2 = ['p1','p2','p3']
    Xpca= pd.DataFrame(Xpca,columns = features_list2)
    features_list2 = ['p2']

    log_reg.fit(StandardScaler().fit_transform(X[features_list]),y)
    y_t = log_reg.predict(StandardScaler().fit_transform(X[features_list]))
    print(CV_model_classify(X[features_list],y,LogisticRegression(solver='liblinear'))['mean'],len(features_list))
    print(np.bincount(y_t == y)[1]/len(y))


    log_reg.fit(StandardScaler().fit_transform(Xpca[features_list2]),y)
    y_t = log_reg.predict(StandardScaler().fit_transform(Xpca[features_list2]))
    print(CV_model_classify(Xpca[features_list2],y,LogisticRegression(solver='liblinear'))['mean'],len(features_list))
    print(np.bincount(y_t == y)[1]/len(y))
    features_list = features_list2

    fig,axes = plt.subplots(len(features_list), 1)
    if len(features_list)==1:
        axes = [axes]
    for i,feature in enumerate(features_list):
        
        axes[i].scatter(sc.fit_transform(X)[:,i], y=='NT',color="black")
        axes[i].set_ylabel('ASD')
        axes[i].set_xlabel(feature)
        # print(np.squeeze(log_reg.coef_))
        # exit(1)
        Y_test = log_reg.predict(sc.fit_transform(X[features_list]).reshape(-1,1))
        ys = Y_test == y
        X_test = np.linspace(sc.fit_transform(X)[:,i].min() -10, sc.fit_transform(X)[:,i].max()+10, 300)
        # print(X[feature] * np.squeeze(log_reg.coef_)[i] + log_reg.intercept_)
        loss = expit(X_test * np.squeeze(log_reg.coef_) + log_reg.intercept_).ravel()
        axes[i].plot(X_test, loss, color="red", linewidth=3)
        axes[i].axhline(0.5, color=".5")
        # plt.cm.get_cmap('RdBu')
        axes[i].scatter(x = sc.fit_transform(X)[:,i], y= y=='NT', c=(ys != True).astype(float), marker='d')
        print('train_acc',np.bincount(ys)[1])


    plt.show()
    # print(LOO_model_classify(X[['p1','p2']],y,LogisticRegression(solver='liblinear'))['mean'],len(features_list))
    # print(CV_model_classify(X[['p1','p2']],y,LogisticRegression(solver='liblinear'))['mean'],len(features_list))


    return
def hist_features_by_label(data,label_col,list_of_features = None, outputdir ='Featureshistograms2' ):
    if not list_of_features:
        list_of_features = list(data.columns)
    labels = {'F0':'(Hz)', 'Intensity':'(dB)','HNR':'(dB)'}
    for col in list_of_features:
        if col == label_col:
            continue
        print(col)
        # all = data
        # ASD_mask = all['Type'] == 'ASD'
        # NT_mask = all['Type'] == 'NT'
        # if col == 'Type' or col =='Gender':
        #     continue
        # all[NT_mask][col].plot.hist(bins=10,alpha=0.5,label='NT',legend= True)
        # all[ASD_mask][col].plot.hist(bins=10,alpha=0.5,label='ASD',legend= True)
        types = data[label_col].unique()
        print(types)
        # plt.hist([data.loc[data[label_col] == x,col] for x in types])
        data.pivot(columns=label_col, values=col).plot.hist(bins=10,alpha=0.5)
        counts = data[label_col].value_counts()
        title = col+" ("
        for t in types:
            title +=str(t)+":"+str(counts[t])+", "
        title = title[:-2]
        title+=")"
        plt.title(title)
        colors = ['r','g','b','c','m','y']
        c = 0
        for t in types:
            t_nt = data[data[label_col]==t][col].mean()
            plt.text(t_nt,plt.ylim()[1]/2,'mean '+str(t)+': '+"{:.3f}".format(t_nt),rotation=90)
            plt.axvline(t_nt, color=colors[c], linestyle='dashed', linewidth=1)
            c+=1
        ax = plt.gca()
        for key in labels.keys():
            if key in col:
                ax.set_xlabel(labels[key])
        from os.path import exists
        from os import mkdir
        if not exists(outputdir):
            mkdir(outputdir)
        plt.show()
        plt.savefig(outputdir+col)
        plt.close()

def remove_cols_with(data,word):
    to_remove = []
    for col in list(data.columns):
        if strings_in(word,col):
            to_remove.append(col)
    return data.copy().drop(to_remove,axis=1)

def my_grid_search(X,y,basic_title):
    RESULTS_DF_DIR = 'results df/'
    if not exists(RESULTS_DF_DIR):
        os.mkdir(RESULTS_DF_DIR)
    feature_reduction = [('statistical',statistics_feature_reduction),('rf',RF_feature_reduction)]
    feature_selection = [mutual_info_classif,f_classif]
    sclaling = [('minmax',MinMaxScaler),('standart',StandardScaler)]
    totalc = total = len(list(X.columns))
    steps = [total-2]
    while int(totalc*0.9)>10:
        steps.append(int(totalc*0.9))
        totalc*=0.9
    for scal in sclaling:
        scal_name = scal[0]
        scal_func = scal[1]
        X_scaled = X.copy()
        X_scaled.values[:] =  scal_func().fit_transform(X_scaled)
        result  = test_statistics_reduction_quality(X_scaled,y,f_classif,cv = KFold(10))
        result.to_pickle(RESULTS_DF_DIR+basic_title+'_'+scal_name+' statistical f_classif')
        result.to_csv(basic_title+'_'+scal_name+' statistical f_classif'+'.csv')
        result  = test_statistics_reduction_quality(X_scaled,y,mutual_info_classif,cv = KFold(10))
        result.to_pickle(RESULTS_DF_DIR+basic_title+'_'+scal_name+' statistical mutual_info_classif')
        result.to_csv(basic_title+'_'+scal_name+' statistical mutual_info_classif'+'.csv')
        result  = test_RF_reduction_quality(X_scaled,y,clf=RandomForestClassifier(max_depth=10),cv = KFold(10))
        result.to_pickle(RESULTS_DF_DIR+basic_title+'_'+scal_name+' RF')
        result.to_csv(basic_title+'_'+scal_name+' RF'+'.csv')
                




def test_dataframe_vis():
    # return
    list_of_results = load_pickle('list_of_important_tables.pickle')
    # list_of_results = load_pickle('list_of_results.pickle')

    TestApp(list_of_results)
    # root = tkinter.Tk()
    # current = 0 
    # root.title(list_of_results[0][0])

    # frame = tk.Frame(root)
    # frame.pack(fill='both', expand=True)

    # pt = pandastable .Table(frame,dataframe=list_of_results[0][1],showstatusbar=True,showtoolbar=True)
    # def load_table():
    #     current+=1
    #     pt.model.df = list_of_results[current][1]
    #     pt.parent.title(list_of_results[current][0])
    #     pt.redraw()
    # pt.showIndex()
    # pt.show()
    # button = tk.Button(root, text='Next Table', command=load_table)
    # button.pack()
    # # pt.model.df = list_of_results[0][1]
    # root.mainloop()
    # html = list_of_results[0][1].to_html()
    # text_file = open("index.html", "w")
    # text_file.write(html)
    # text_file.close()
# test_dataframe_vis()
# test = pd.DataFrame()
# test['col1'] = [1,2,4,5,3,6]
# test['col2'] = [10,20,130,40,550,60]
# test['col3'] = [100,2100,300,400,50,600]
# test['col4'] = [1,1,1,2,2,2]
# test['type'] = [0,0,0,1,1,1]
# print('test\n',test)
# print(test['col1']>=4)
# # print(statistics_feature_reduction(test.drop('type',axis=1),test['type'],features=1))
# print(diffs_by_tasks(test,['col1','col2','col3']))
