from scipy.spatial import distance
from Files import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

NOT_HEB = [219,221,210]

data  = pd.read_excel(get_file(),engine='openpyxl')
data = data[data['Gender']=='M']
for n_h in NOT_HEB:
    data = data[(data['record'] !=n_h)]
X = data.drop('Type', axis=1)
y = data['Type']
X = X.drop('Gender',axis=1)
X = X.drop('record', axis=1)
clf = DecisionTreeClassifier(random_state=0)

clf.fit(X,y)
importance = clf.feature_importances_
importance.sort()
print(importance)
print(importance.shape,X.shape,len(list(X.head())))
to_drop = []
for i in range(importance.shape[0]):
    v = importance[i]
    print(i,v)
    if v==0 :
        to_drop.append( list(X.head())[i])
X.drop(to_drop,axis=1,inplace=True)
sc = StandardScaler()
norm_X = sc.fit_transform(X)
print(norm_X.shape,X.shape)
print(sc.mean_,sc.mean_.shape)
for i in range(norm_X.shape[0]):
    row = norm_X[i]
    data.at[data.iloc[[i]].index,'distance']= distance.euclidean(row,sc.mean_)
ASD_mask = data['Type'] == 'ASD'
NT_mask = data['Type'] == 'NT'
data[NT_mask]['distance'].plot.hist(bins=10,alpha=0.5,title='NT')
data[ASD_mask]['distance'].plot.hist(bins=10,alpha=0.5,title='ASD')
plt.legend(['NT','ASD'])
plt.show()
