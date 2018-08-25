import pandas as pd
import numpy as np
from sklearn import preprocessing,cross_validation,neighbors,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_csv('breast-cancer-wisconsin.data.txt')


data = data.drop(['id'],1)
data.replace('?',-9999,inplace=True)
print(data.head())

X = np.array(data.drop(['Class'],1))
y = np.array(data['Class'])


X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)


clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)


acc = clf.score(X_test,y_test)
print(acc)
print(type(X))


