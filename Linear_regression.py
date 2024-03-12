import numpy as np

class LinearRegression:
    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.lr = lr
        self.theta = None

    def initialize_theta(self, x):
        return np.zeros((x.shape[1], 1))

    def linear_fxn(self, x, theta):
        return x @ theta

    def mse(self, x, y, theta):
        return np.sum((self.linear_fxn(x, theta) - y) ** 2) / x.shape[0]

    def gradient(self, x, y, theta):
        return (2 * x.T @ (self.linear_fxn(x, theta) - y)) / x.shape[0]

    def update(self, theta, grads):
        return theta - self.lr * grads

    def fit(self, x, y):
        self.theta = self.initialize_theta(x)
        losses = []

        for i in range(self.epochs):
            y_pred = self.linear_fxn(x, self.theta)
            loss = self.mse(x, y, self.theta)
            grads = self.gradient(x, y, self.theta)
            self.theta = self.update(self.theta, grads)
            losses.append(loss)

        return self.theta, losses