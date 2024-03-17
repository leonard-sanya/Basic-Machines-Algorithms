#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from Linear_regression import LinearRegression
from Logistic_regression import LogisticRegression

#linear regression
#np.random.seed(10)
x = np.linspace(0,1, 10)
y = x + np.random.normal(0, 0.1, (10,))

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


#logistic data
X, y = make_classification(n_features=2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)
def train_test_split(X,y):
  '''
  this function takes as input the sample X and the corresponding features y
  and output the training and test set
  '''
  #np.random.seed(0)
  train_size = 0.8
  n = int(len(X)*train_size)
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  train_idx = indices[: n]
  test_idx = indices[n:]
  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = train_test_split(X,y)

linear = LinearRegression(1000,0.01)
logistic = LogisticRegression(0.01,10000)

def main():
  print("Would you like to run Linear Regression or Logistic Regression ?")
  user = input("Choose 1 for Linear and 2 for Logistic Regression ")
  if user=="1":
    linear.fit(x,y)
  else:
    logistic.fit(X_train, y_train)
    logistic.accuracy(y_train,logistic.predict(X_train))  
    logistic.plot_decision_boundary(X_train,logistic.w,logistic.w[0],y_train)
    logistic.plot_decision_boundary(X_test,logistic.w,logistic.w[0],y_test)
if __name__ == "__main__":
  main()
        

# %%
