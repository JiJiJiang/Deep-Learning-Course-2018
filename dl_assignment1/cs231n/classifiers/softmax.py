import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  n=X.shape[0]
  #d=W.shape[0] #X.shape[1]
  c=W.shape[1]
  for i in range(n):
    outputs=X[i].dot(W)
    outputs-=np.max(outputs) # enhance numeric stability
    true_class_output=outputs[y[i]]
    # loss
    sum_exp=np.sum(np.exp(outputs))
    loss += (-true_class_output + np.log(sum_exp))
    # dW
    one_hot_vector=np.zeros(c)
    one_hot_vector[y[i]]=1
    for j in range(c):
      dW[:,j] += (np.exp(outputs[j])/sum_exp - one_hot_vector[j]) * X[i]
  # average and regularization
  loss /= n
  loss += 0.5*reg*np.sum(W*W)
  dW /= n
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # d*c

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  n = X.shape[0]
  # d=W.shape[0] #X.shape[1]
  c = W.shape[1]

  outputs = X.dot(W) # n*c (2-D)
  outputs -= np.max(outputs,axis=1,keepdims=True) # n*c-n*1=n*c (2-D)  # enhance numeric stability
  true_class_output = outputs[range(n),y] # n (1-D)
  # loss
  exp_outputs=np.exp(outputs) # n*c (2-D)
  sum_exp_outputs = np.sum(exp_outputs,axis=1,keepdims=True) # n*1 (1-D)
  p=exp_outputs/sum_exp_outputs
  loss += (-np.sum(true_class_output) + np.sum(np.log(sum_exp_outputs)))
  # dW
  one_hot_matrix = np.eye(c)[y] # n*c
  dW += np.dot(X.T,(p - one_hot_matrix)) # (d*n)*(n*c)
  # average and regularization
  loss /= n
  loss += 0.5 * reg * np.sum(W * W)
  dW /= n
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

