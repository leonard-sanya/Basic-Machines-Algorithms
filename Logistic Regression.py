# -*- coding: utf-8 -*-
"""## Dataset

We will use the following randomly generated data from sklearn.
"""

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

#Load the iris dataset from sklearn
X, y = make_classification(n_features=2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)

# vizualize the data
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel('features 1')
plt.ylabel('feature 2')
plt.show()

def train_test_split(X,y):
  '''
  this function takes as input the sample X and the corresponding features y
  and output the training and test set
  '''
  np.random.seed(0) # To demonstrate that if we use the same seed value twice, we will get the same random number twice

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
print(f" the training shape is: {X_train.shape}")
print(f" the test shape is: {X_test.shape}")

"""## Recaps

1. Logistic/sigmoid function:
\begin{equation}
σ(z)= \dfrac{1}{1+ e^{-z}}
\end{equation}
where  $z= x w$.
2. Derivative of Logistic/sigmoid function with respective to $z$:
\begin{equation}
σ'(z)= σ(z)(1-σ(z))
\end{equation}
3. Negative log likelihood or Cross-entropy loss:
\begin{equation}
l(w)= -\dfrac{1}{N}\sum_{i= 1}^{N} \left(y^{(i)}_{true} \times \log y^{(i)}_{pred} + (1-y^{(i)}_{true}) \times \log (1-y^{(i)}_{pred}) \right)
\end{equation}

where:

 $y_{pred}= σ(z)$, $z= xw$.

4. Derivative of Cross-entropy loss with respective to $w$:
\begin{equation}
dl(w)= -\dfrac{1}{N}x^T(y_{true} -y_{ped} )
\end{equation}
5. Apply Batch gradient descent to update $w$.


"""

class LogisticRegression:
  '''
  The goal of this class is to create a LogisticRegression class,
  that we will use as our model to classify data point into a corresponding class
  '''
  def __init__(self,lr,n_epochs):
    self.lr = lr
    self.n_epochs = n_epochs
    self.train_losses = []
    self.w = None
    self.weight = []

  def add_ones(self, x):

    ##### WRITE YOUR CODE HERE #####
    return np.hstack((np.ones((x.shape[0],1)),x))
    #### END CODE ####

  def sigmoid(self, x):
    ##### WRITE YOUR CODE HERE ####

    return  np.divide(1,1 + np.exp(-(x @ self.w)))
    #### END CODE ####

  def cross_entropy(self, x, y_true):
    ##### WRITE YOUR CODE HERE #####
    y_pred = self.sigmoid(x)
    loss = np.divide(-(np.sum(y_true * np.log(y_pred) + (1- y_true) * np.log((1-y_pred)))),y_true.shape[0])
    return loss
    #### END CODE ####

  def predict_proba(self,x):  #This function will use the sigmoid function to compute the probalities
    ##### WRITE YOUR CODE HERE #####
    x = self.add_ones(x)
    proba = self.sigmoid(x)
    return proba
    #### END CODE ####

  def predict(self,x):

    ##### WRITE YOUR CODE HERE #####
    probas = self.predict_proba(x)
    return [1 if p >= 0.5 else 0 for p in probas]#convert the probalities into 0 and 1 by using a treshold=0.5

    #### END CODE ####

  def fit(self,x,y):

    # Add ones to x
    x = self.add_ones (x)

    # reshape y if needed
    y = y.reshape(-1,1)

    # Initialize w to zeros vector >>> (x.shape[1])
    self.w  =  np.zeros((x.shape[1],1))

    for epoch in range(self.n_epochs):
      # make predictions
      y_pred = self.sigmoid(x)

      #compute the gradient
      grads = np.divide(-((x.T)@(y - y_pred)),x.shape[0])


      #update rule
      self.w = self.w - self.lr * grads

      #Compute and append the training loss in a list
      loss = self.cross_entropy(x, y)
      self.train_losses.append(loss)

      if epoch%1000 == 0:
        print(f'loss for epoch {epoch}  : {loss}')

  def accuracy(self,y_true, y_pred):
    ##### WRITE YOUR CODE HERE #####
    correct=0
    for i in range(len(y_true)):
      if y_true[i]==y_pred[i]:
        correct +=1
      else:
        pass
    acc = (np.divide(correct,len(y_true)))*100
    return acc
    #### END CODE ####

model = LogisticRegression(0.01,n_epochs=10000)
model.fit(X_train,y_train)

ypred_train = model.predict(X_train)
acc = model.accuracy(y_train,ypred_train)
print(f"The training accuracy is: {acc}")
print(" ")

ypred_test = model.predict(X_test)
acc = model.accuracy(y_test,ypred_test)
print(f"The test accuracy is: {acc}")

def plot_decision_boundary(X, w, b,y_train):

    # z = w1x1 + w2x2 + w0
    # one can think of the decision boundary as the line x2=mx1+c
    # Solving we find m and c
    x1 = [X[:,0].min(), X[:,0].max()]
    m = -w[1]/w[2]
    c = -b/w[2]
    x2 = m * x1 + c

    # Plotting
    fig = plt.figure(figsize=(10,8))
    plt.scatter(X[:, 0], X[:, 1],c=y_train)
    plt.scatter(X[:, 0], X[:, 1], c=y_train)
    plt.xlim([-2, 3])
    plt.ylim([0, 2.2])
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Decision Boundary')
    plt.plot(x1, x2, 'y-')

plot_decision_boundary(X_train,model.w,model.w[0],y_train)

plot_decision_boundary(X_test,model.w,model.w[0],y_test)

"""## Let’s test out our code for data that is not linearly separable.

"""

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.24)
plt.scatter(X[:,0],X[:,1], c= y)

