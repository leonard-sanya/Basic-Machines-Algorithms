# Implementation of Gradient-Descent
We provide a description of the implementation of three common variants of gradient descent: Gradient Descent (GD), Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent (Mini-Batch GD) for a simple Linear regression 

1. Gradient Descent (GD):
   - GD is an optimization algorithm used to minimize the cost function of a machine learning model.
   - GD calculates the gradients of the cost function with respect to all training examples in the dataset and updates the parameters accordingly.
   - The algorithm repeats the process until it converges to a minimum or reaches a predetermined number of iterations.

2. Stochastic Gradient Descent (SGD):
   - SGD is a variant of GD that updates the model's parameters based on the gradient of the cost function computed on a single training example at each iteration.
   - Instead of summing the gradients over all training examples, SGD updates the parameters after processing each individual.
   - SGD is computationally efficient as it requires less memory and can process each example in parallel.

3. Mini-Batch Gradient Descent (Mini-Batch GD):
   - Mini-Batch GD is a compromise between GD and SGD, where the algorithm updates the model's parameters based on the gradients computed on a mini-batch of training examples

The implementation of these algorithms often involves the following steps:
1. Initialize the model's parameters randomly.
2. Iterate over the training data for a certain number of epochs (passes through the entire dataset).
3. In each epoch, randomly shuffle the training data.
4. Divide the training data into mini-batches (in the case of Mini-Batch GD) or use a single example (in the case of SGD).
5. Compute the gradients of the cost function with respect to the model's parameters using the mini-batch or single example.
6. Update the model's parameters by taking a step in the direction of the negative gradient.
7. Repeat steps until all mini-batches or examples have been processed.
8. Evaluate the model's performance on a separate validation or test set to monitor its progress.
9. Continue training until convergence or a predefined stopping criterion is met.


