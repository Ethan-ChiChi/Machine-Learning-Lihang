# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:10:01 2019

@author: Dylan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()

#X is data, y is label#
X = iris.data[:100,:2]
y = iris.target[:100]
y = np.array([1 if i == 1 else -1 for i in y])

"""
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

data = np.array(df.iloc[ :100, [0, 1, -1]])
#X,y = data[:,:-1],data[:,-1]
"""

clf = Perceptron(fit_intercept = False, max_iter = 1000, shuffle = False, tol = None)
#print(clf.fit(X,y))
#print(clf.coef_)
#print(clf.intercept_)


x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(X[:50, 0], X[:50, 1], 'bo', color='blue', label='-1')
plt.plot(X[50:100, 0], X[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
