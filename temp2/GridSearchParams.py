from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression,chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_gridSearch_params(clf):

    if(type(clf)==type(RandomForestClassifier())):
        return get_RF_params()
    if(type(clf)==type(LogisticRegression())):
        return get_LR_params()
    if (type(clf)==type(QuadraticDiscriminantAnalysis())):
        return get_QDA_params()
    if (type(clf)==type(LinearDiscriminantAnalysis())):
        return get_LDA_params()
    if (type(clf)==type(SVC())):
        return get_SVC_params()
    if  (type(clf)==type(GradientBoostingClassifier())):
        return get_GBC_params()
    return None
    

    



def get_RF_params():
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth = [5,10,20]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
            'max_depth': max_depth,
            # 'min_samples_split': min_samples_split,
            # 'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
            }
    return random_grid
def get_LR_params():
    parameters = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
    }
    return parameters
def get_QDA_params():
    parameters_3 = {
    'reg_param': (0.00001, 0.0001, 0.001,0.01, 0.1), 
    'store_covariance': (True, False),
    'tol': (0.0001, 0.001,0.01, 0.1), 
                   }
    return parameters_3
def get_LDA_params():
    parameters_1 = {
    'solver': ['lsqr','eigen'],
    'shrinkage' : ('auto',None)  #note svd does not run with shrinkage and models using it will be tuned separately
                   }
    return parameters_1
def get_LDAsvd_params():
    parameters_2 = {
    'solver' : ['svd'],
    'store_covariance' :(True, False),
                   }
    return parameters_2
def get_SVC_params():
    kernels = ['rbf', 'poly', 'sigmoid']
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': kernels}
    return param_grid
def get_GBC_params():
    parameters = {
    "loss":["log_loss"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 6),
    "min_samples_leaf": np.linspace(0.1, 0.5, 6),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }
    return parameters

def GS_fit(estimator,X,y):
    print(type(estimator))
    return GridSearchCV(estimator,get_gridSearch_params(estimator),scoring='accuracy',cv=10,n_jobs=-1).fit(X,y)
