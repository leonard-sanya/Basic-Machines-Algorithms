import numpy as np

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
    return np.hstack((np.ones((x.shape[0],1)),x))
    

  def sigmoid(self, x):
    return  np.divide(1,1 + np.exp(-(x @ self.w)))


  def cross_entropy(self, x, y_true):
    y_pred = self.sigmoid(x)
    loss = np.divide(-(np.sum(y_true * np.log(y_pred) + (1- y_true) * np.log((1-y_pred)))),y_true.shape[0])
    return loss
    

  def predict_proba(self,x):  
    x = self.add_ones(x)
    proba = self.sigmoid(x)
    return proba
    

  def predict(self,x):
    probas = self.predict_proba(x)
    return [1 if p >= 0.5 else 0 for p in probas]

  def fit(self,x,y):
    x = self.add_ones (x)
    y = y.reshape(-1,1)
    self.w  =  np.zeros((x.shape[1],1))

    for epoch in range(self.n_epochs):
      y_pred = self.sigmoid(x)
      grads = np.divide(-((x.T)@(y - y_pred)),x.shape[0])
      self.w = self.w - self.lr * grads

      loss = self.cross_entropy(x, y)
      self.train_losses.append(loss)

      if epoch%1000 == 0:
        print(f'loss for epoch {epoch}  : {loss}')

  def accuracy(self,y_true, y_pred):
    correct=0
    for i in range(len(y_true)):
      if y_true[i]==y_pred[i]:
        correct +=1
      else:
        pass
    acc = (np.divide(correct,len(y_true)))*100
    return acc
    