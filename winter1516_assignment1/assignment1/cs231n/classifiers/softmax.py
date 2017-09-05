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
  m = X.shape[0]
  k = W.shape[1]
  z = X.dot(W)
  z -= np.max(z)
  
  h = np.exp(z) / np.sum(np.exp(z), axis=1).reshape(m, 1)
  
  y_pred = np.argmax(h, axis=1)
  y_one_hot = np.zeros((m, k))
  y_one_hot[np.arange(m), y_pred] = 1

  #loss = np.sum(-y_one_hot * np.log(h))
  y2 = np.zeros((m, k))
  y2[np.arange(m), y] = 1

  t = np.multiply(y2, h)  
  loss = -np.log(t[t>0]).sum()
  #loss = -np.log(h).sum()

  loss /= m

  loss += (0.5 * reg * np.sum(W * W))

  dW = -X.T.dot((y2 - h)) / m
  dW += reg*W
  #dW[:-1] += reg*W[:-1]

  # num_classes = W.shape[1]
  # num_train = X.shape[0]

  # Compute scores
  # f = np.dot(X, W)

  # # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
  # f -= np.max(f)

  # # Loss: L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}
  # # Compute vector of stacked correct f-scores: [f(x_1)_{y_1}, ..., f(x_N)_{y_N}]
  # # (where N = num_train)
  # f_correct = f[range(num_train), y]
  # loss = -np.mean( np.log(np.exp(f_correct)/np.sum(np.exp(f))) )

  # # Gradient: dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
  # p = np.exp(f)/np.sum(np.exp(f), axis=0)
  # ind = np.zeros(p.shape)
  # ind[range(num_train), y] = 1
  # dW = np.dot(X.T, (p-ind))
  # dW /= num_train

  # # Regularization
  # loss += 0.5 * reg * np.sum(W * W)
  # dW += reg*W
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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss, dW = softmax_loss_naive(W, X, y, reg)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

