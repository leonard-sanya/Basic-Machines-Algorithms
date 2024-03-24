# Implementation of Linear Regression
We provide a description of the implementation of a simple Linear regression model using gradient descent.

1. Gradient Descent (GD):
   - GD is an optimization algorithm used to minimize the cost function of a machine learning model.
   - GD calculates the gradients of the cost function with respect to all training examples in the dataset and updates the parameters accordingly.
   - The algorithm repeats the process until it converges to a minimum or reaches a predetermined number of iterations.

The implementation of these algorithms often involves the following steps:
1. Initialize the model's parameters randomly.
2. Using the initialized parameters, we make a prediction(y_pred)
3. $$\hat{y} = \sigma\big(\sum_{i=1}^k W_i X_i +b \big)$$

     where $σ(z)= \frac{1}{1+ e^{-z}} \hspace{0.2cm}\text{and}\hspace{0.2cm} z= XW$.
  
4. We then compute the loss (Mean Squared Error)of model prediction(y_pred) compared to the true y value.
5. Using the loss obtained, we compute the gradients of the cost function with respect to the model parameters..
6. Update the model's parameters by taking a step in the direction of the negative gradient.
7. Evaluate the model's performance on a separate validation or test set to monitor its progress.
8. Continue training until convergence or a predefined stopping criterion is met.

# Implementation of Logistic Regression
The Logistic Regression model is designed to classify data points into corresponding classes. This model is implemented in Python using NumPy and Matplotlib libraries.

The implementation of these algorithms involves the following steps:
1. Initialize the model's parameters randomly.
2. Using the initialized parameters, we make a prediction(y_pred) using a sigmoid function.
   $$\hat{y} = \sigma\big(\sum_{i=1}^k W_i X_i +b \big)$$

     where $σ(z)= \frac{1}{1+ e^{-z}} \hspace{0.2cm}\text{and}\hspace{0.2cm} z= XW$.
  
4. We then compute the loss(Negative Log Likelihood Function) of model prediction(y_pred) compared to the true y value.
5. 
   $$\mathcal{L}= -\frac{1}{N}\sum_{i= 1}^{N} \big(y_{true} * log (y_{pred} )+ (1-y_{true})*\log (1-y_{pred}) \big)$$
   
6. Using the loss obtained, we compute the gradients of the cost function with respect to the model parameters.
   $$\theta^{k+1} \leftarrow \theta^{k} - \eta \nabla \mathcal{L} $$
   where $\eta$ is the learning rate.
8. Update the model's parameters by taking a step in the direction of the negative gradient.
9. Evaluate the model's performance on a separate validation or test set to monitor its progress.
10. Continue training until convergence or a predefined stopping criterion is met.

# Model Evaluation

To evaluate the model on our test set, we first use the predict function, which returns the predicted probabilities of belonging to different classes. We then set a threshold of 0.5, implying that predictions with probabilities greater than or equal to 0.5 belong to class 1, and the rest belong to class 0. Next, we compute the model accuracy using the formula:

$$\text{Accuracy}= \frac{TP+TN}{TP+FP+TN+FN}$$
where:

TP = True Positives

TN = True Negatives

FP = False Positives

FN = False Negatives

This metric provides an indication of how well the model is performing on the test set.

# Visualization
We used the Matplotlib libraries to visualize changes in the loss at each epoch or iteration 
