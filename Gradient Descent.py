# -*- coding: utf-8 -*-
"""
In this tutorial, we will look at how to find parameters of a simple linear regression problem using;

* batch gradient descent
* stochastic gradient descent
* mini-batch gradient descent
"""

import numpy as np # imports
import matplotlib.pyplot as plt
np.random.seed(10)

def plot(X, y, theta, epoch, plot_every=5):
  """Plotting function for features and targets"""
  if plot_every is not None and epoch % plot_every == 0:
    xtest = np.linspace(0, 1, 10).reshape(-1,1)
    ypred = linear_function(xtest, theta).reshape(-1,1)
    plt.scatter(X, y, marker="+")
    plt.xlabel("feature")
    plt.ylabel("target")
    plt.plot(xtest, ypred, color="orange")
    plt.show()

# from time import sleep
def plot_loss(losses):
  """Plotting function for losses"""
  plt.plot(losses)
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.title("training curve")

"""Lets begin with some randomly generated data. Plots will allow us visualise the data and functions we approximate along the way"""

np.random.seed(10)
xtrain = np.linspace(0,1, 10)
ytrain = xtrain + np.random.normal(0, 0.1, (10,))

xtrain = xtrain.reshape(-1, 1)
ytrain = ytrain.reshape(-1, 1)

plt.scatter(xtrain, ytrain, marker="+")

"""In class, we learnt that, when developing a machine learning solution to a problem,  we need a few things;
  - Data {features, targets}
  - Hypothesis (based of the relationship we observe between features and targets)
  - A criterion (A function evaluates our hypothesis)
  - A learning algorithm (An algorithm to find the best parameters for our hypothesis)

# Hypothesis
Lets begin with the hypothesis

Here, in our one dimensional data, its easy to observe a linear relationship between our features and targets. So we may settle on the linear function below

$$y = X\theta$$

where, $X \in \mathbb{R}^{N x D}, \theta \in \mathbb{R}^{D}, y \in \mathbb{R}^N$
"""

def linear_function(X, theta):
  """
  Compute the dot product of X$\theta$
  Args:
    X: feature matrix (size - N x D)
    theta: parameters (size - D x 1)

  Returns:
    output y, size N x 1
  """
  assert X.ndim > 1
  assert theta.ndim > 1
  assert X.shape[1]==theta.shape[0],f"columns of X is different from the rows of theta"
  return X @ theta

"""## Criterion/ Loss function
Since we have a continuous label, this problem is essentially a regression one and we can use a mean squared error loss.

This is given as $$L(\theta) = \frac{1}{N}∑(y - {\bar y})^2$$
where $y$ is the targets and $\bar y$ is the output of our hypothesis
"""

def mean_squared_error(ytrue, ypred):
  """
  Computes the mean squared error
  Args:
    ytrue: vector of true labels
    ypred: vector of predicted labels

  Returns:
    mse loss (scalar)
  """

  return np.sum((ytrue-ypred)**2)/len(ytrue)

"""## Gradient descent

Now, our learning algorithm!

We have already seen how to compute closed form solution for our linear regression problem using the maximum likelihood estimation. Here, we use a gradient descent technique. The idea is to take little steps in the direction of minimal loss. The gradients when computed guides us in what direction to take these little steps.

A full training loop using gradient descent algorithm will follow these steps;
- initialise parameters
- Run some number of epochs
  - use parameters to make predictions
  - Compute and store losses
  - Compute gradients of the loss wrt parameters
  - Use gradients to update the parameters
- Do anything else or End!

Lets write a few lines of code that does a part of each of these steps before the main training loop
"""

def initialize_theta(D):
  """Initializes parameter theta
  Args:
    D: size of parameter
  Returns:
    initial parameters of size D x 1
  """
  return np.random.random((D,1))

def batch_gradient(X, y, theta):
  """Computes gradients of loss wrt parameters for a full batch
  Args:
    X: input features of size - N x D
    y: target vector of size - N x 1
    theta: parameters of size - D x 1
  """
  #loss= np.sum((y-X @ theta)**2)/len(y)
  return (2 * X.T @ (linear_function(X, theta) - y))/len(y)

def update_function(theta, grads, step_size):
  """Updates parameters with gradients
  Args:
    theta : parameters of size D x 1
    grads: gradients of size D x 1

  Returns:
    updated parameters of size D x 1
  """
  return theta - step_size * grads #batch_gradient(X, y, theta)

def train_batch_gradient_descent(X, y, num_epochs, step_size=0.1, plot_every=1):
  """
  Trains model with full batch gradient descent
  """
  N, D = X.shape
  theta = initialize_theta(D)
  losses = []
  for epoch in range(num_epochs): # Do some iterations
    ypred = linear_function(X, theta)  # make predictions with current parameters
    loss = mean_squared_error(y, ypred) #batch_gradient(X, y, theta) # Compute mean squared error
    grads = batch_gradient(X, y, theta) # compute gradients of loss wrt parameters
    theta = update_function(theta, grads, step_size) # Update your parameters with the gradients

    losses.append(loss)
    print(f"\nEpoch {epoch}, loss {loss}")
    #plot(X, y, theta, epoch, plot_every)
  return losses

batch_gradient_descent_losses = train_batch_gradient_descent(xtrain, ytrain, 10, plot_every=2)

plot_loss(batch_gradient_descent_losses)

"""Observations:
  1. What do you observe in the plots
  2. You may try different number of epochs
One more thing, it seems the train curve flattens out so quickly, yet, training still progresses regardless.

Can you think of how to avoid the unnecessary training after converging,
Right!...
- You may end the training when we hit a threshold loss value
- Early stopping - stop the training when loss does not change after a certain number of epochs (patience)

## Stochastic Gradient Descent

Stochastic gradient Descent unlike batch gradient descent, pick random sample(s) or subset of samples and updates parameters with their gradients

Below, we will write code that gets a single sample and computes gradients for that sample.

Its also important we shuffle our data
"""

def per_sample_gradient(xi, yi, theta):
  """Computes the gradient for a single sample
  Args:
    xi: vector of sample features, size 1 x D
    yi: sample target, size 1(scalar)
  Returns
  """
  return (2 * xi.T @ ((linear_function(xi,theta))-yi))#/yi.shape[0]

def shuffle_data(X, y):
  """Shuffles the data
  Args:
    X: input features of size - N x D
    y: target vector of size - N x 1

  Returns:
    shuffled data
  """
  N, _ = X.shape
  shuffled_idx = np.random.permutation(N)
  return X[shuffled_idx], y[shuffled_idx]

def per_sample_gradient(xi, yi, theta):
  return (2 * xi.T @ ((linear_function(xi,theta))-yi))
def shuffle_data(X, y):
  N, _ = X.shape
  shuffled_idx = np.random.permutation(N)
  return X[shuffled_idx], y[shuffled_idx]

def train_with_sgd(X, y, num_epochs, step_size, plot_every=1):

  N, D = X.shape
  theta = initialize_theta(D)
  losses = []
  epoch = 0
  loss_tolerance = 0.001
  avg_loss = float("inf")

  while epoch < num_epochs and avg_loss > loss_tolerance:
    running_loss = 0.0
    shuffled_x, shuffled_y = shuffle_data(X, y)

    for idx in range(shuffled_x.shape[0]):
      sample_x = shuffled_x[idx].reshape(-1, D)
      sample_y = shuffled_y[idx].reshape(-1, 1)
      ypred = linear_function(sample_x, theta)
      loss = mean_squared_error(sample_y, ypred)
      running_loss += loss
      grads = batch_gradient(sample_x, sample_y, theta)
      theta = update_function(theta, grads, step_size)

    plot(X, y, theta, epoch, plot_every)
    avg_loss = running_loss/ X.shape[0]
    losses.append(avg_loss)
    print(f"Epoch {epoch}, loss {avg_loss}")

    epoch += 1

  return losses

def train_with_sgd(X, y, num_epochs, step_size, plot_every=1):
  """Train with stochastic gradient descent"""
  N, D = X.shape
  theta = initialize_theta(D)
  losses = []
  epoch = 0
  loss_tolerance = 0.001
  avg_loss = float("inf")

  while epoch < num_epochs and avg_loss > loss_tolerance:
    running_loss = 0.0
    shuffled_x, shuffled_y = shuffle_data(X, y)

    for idx in range(shuffled_x.shape[0]):
      sample_x = shuffled_x[idx].reshape(-1, D)
      sample_y = shuffled_y[idx].reshape(-1, 1)
      ypred = linear_function(sample_x, theta)
      loss = mean_squared_error(sample_y, ypred)
      running_loss += loss
      grads = batch_gradient(sample_x, sample_y, theta)
      theta = update_function(theta, grads, step_size)

   # plot(X, y, theta, epoch, plot_every)
    avg_loss = running_loss/ X.shape[0]
    losses.append(avg_loss)
    print(f"Epoch {epoch}, loss {avg_loss}")

    epoch += 1

  return losses

#Now, lets train!
sgd_losses = train_with_sgd(xtrain, ytrain, num_epochs=10, step_size=0.1, plot_every=2)

plot_loss(sgd_losses)

"""Now, we observe that SGD converges pretty well as well.

Now lets attempt to choose a slightly bigger step_size also known as the learning rate.

What do you observe???
"""

sgd_large_step_size_losses = train_with_sgd(xtrain, ytrain, 10, step_size=2.0, plot_every=10)

plot_loss(sgd_large_step_size_losses)

def get_momentum(momentum, grad, beta):
  return beta * momentum +(1-beta)*grad

def train_sgd_with_momentum(X, y, num_epochs, step_size, beta, plot_every=1):
  """Train with stochastic gradient descent"""
  N, D = X.shape
  theta = initialize_theta(D)
  losses = []
  epoch = 0
  loss_tolerance = 0.001
  avg_loss = float("inf")

  while epoch < num_epochs and avg_loss > loss_tolerance:
    momentum = 0.0
    running_loss = 0.0
    shuffled_x, shuffled_y = shuffle_data(X,y)

    for idx in range(shuffled_x.shape[0]):
      sample_x = shuffled_x[idx].reshape(-1,D)
      sample_y = shuffled_y[idx].reshape(-1,D)
      ypred = linear_function(sample_x, theta)
      loss = mean_squared_error(ypred,sample_y)
      running_loss += loss
      grad = batch_gradient(sample_x, sample_y, theta)
      momentum = get_momentum(momentum, grad, beta)
      theta = update_function(theta,momentum, step_size)


   # plot(X, y, theta, epoch, plot_every)
    avg_loss = running_loss/ X.shape[0]
    losses.append(avg_loss)
    print(f"Epoch {epoch}, loss {avg_loss}")

    epoch += 1

  return losses

sgd_momentum_losses = train_sgd_with_momentum(xtrain, ytrain, 30, 0.1, beta=0.99)

plot_loss(sgd_momentum_losses)

"""Notes
- What do you observe here?
  - Its as though sgd with momentum takes a little longer than without to converge. Here, we check convergence by looking at the training curve. No convergence yet because the curve was still going down when the training ended.
- We can choose to train a little longer or increase the learning rate.

- Now lets try with larger ones, `step_size = 2.0` and `num_epochs=30`
You can compare training without momentum and training with momentum by setting the `beta=0` or `beta=0.99` respectively
"""

# Without momentum
sgd_momentum_losses_large_stepsize = train_sgd_with_momentum(xtrain, ytrain, 10, 2.0, beta=0.0, plot_every=None)
plot_loss(sgd_momentum_losses_large_stepsize)

# With momentum
sgd_momentum_losses_large_stepsize = train_sgd_with_momentum(xtrain, ytrain, 10, 2.0, beta=0.99, plot_every=None)
plot_loss(sgd_momentum_losses_large_stepsize)

"""Notes
- Do you realise that even with the large step size that we used before, our loss function is still smoothen out (with almost no oscillation) ?

## Mini-batch gradient Descent

Now, lets finish off with Mini-batch gradient descent

Here, instead of computing gradients for an entire dataset, we divide the data into batches, each batch has the size `batch_size` which we set ourselves except the last batch. You know why?

There are a few things to note here;

- Since we are taking a batch at a time, when computing loss, we need to do it for the batch, then we average over all sample points.
"""

def minibatch_gradient_descent(X, y, num_epochs, step_size=0.1, batch_size=3, plot_every=1):
  N, D = X.shape
  theta = initialize_theta(D)
  losses = []
  num_batches = N//batch_size
  X, y = shuffle_data(X,y) # shuffle the data

  for epoch in range(num_epochs): # Do some iterations
    running_loss = 0.0

    for batch_idx in range(0, N, batch_size):
      x_batch = X[batch_idx: batch_idx + batch_size] # select a batch of features

      y_batch = y[batch_idx: batch_idx + batch_size] # and a batch of labels

      ypred = linear_function(x_batch,theta) # make predictions with current parameters

      loss = mean_squared_error(ypred,y_batch) # Compute mean squared error

      grads = batch_gradient(x_batch, y_batch, theta)# compute gradients of loss wrt parameters

      theta = update_function(theta,grads, step_size)# Update your parameters with the gradients

      running_loss += (loss * x_batch.shape[0]) # loss is mean for a batch, dividing by N_batch gives
                                                # us a sum for the batch so we can average later by diving
                                                # by the full data size

    avg_loss = running_loss/ N
    losses.append(avg_loss)
    print(f"\nEpoch {epoch}, loss {avg_loss}")
    #plot(X, y, theta, epoch, plot_every)
  return losses

mini_batch_gd_losses = minibatch_gradient_descent(xtrain, ytrain, 10, step_size=0.1, batch_size=3)

plot_loss(mini_batch_gd_losses)

"""Next
 - Early stopping - stopping when training reaches converges before all number of epochs are executed.

 - You can try learning rate scheduling, where instead of using the same learning rate in every single epoch, you can define a number of them at different epochs
"""

