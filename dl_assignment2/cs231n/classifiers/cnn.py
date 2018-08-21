import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    self.num_layers = 3
    std = weight_scale
    C, H, W = input_dim
    f = num_filters
    HH = WW = filter_size
    # conv - relu - 2x2 max pool
    self.params['W1'] = np.random.normal(loc=0, scale=std, size=[f, C, HH, WW])
    self.params['b1'] = np.zeros(f)
    # affine - relu
    H_new = 1 + (H-2)/2 # 32/2=16
    self.params['W2'] = np.random.normal(loc=0, scale=std, size=[f*H_new*H_new, hidden_dim])
    self.params['b2'] = np.zeros(hidden_dim)
    # affine
    self.params['W3'] = np.random.normal(loc=0, scale=std, size=[hidden_dim, num_classes])
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out2, cache2 = affine_relu_forward(out1, W2, b2)
    out3, cache3 = affine_forward(out2, W3, b3)
    scores = out3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    # loss
    loss, dScores = softmax_loss(scores, y)
    sum_reg = 0.0
    for i in range(self.num_layers):
      W = self.params['W' + str(i + 1)]
      sum_reg += np.sum(W * W)
    loss += 0.5 * self.reg * sum_reg
    # grad
    dout3, grads['W3'], grads['b3'] = affine_backward(dScores, cache3)
    dout2, grads['W2'], grads['b2'] = affine_relu_backward(dout3, cache2)
    dout1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout2, cache1)
    for i in range(self.num_layers):
      grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass


class FiveLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    {conv - relu - 2x2 max pool}*2 - {affine - relu}*2 - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        self.num_layers = 5
        std = weight_scale
        C, H, W = input_dim
        f = num_filters
        HH = WW = filter_size
        # (conv - relu - 2x2 max pool) * 2
        self.params['W1'] = np.random.normal(loc=0, scale=std, size=[f, C, HH, WW])
        self.params['b1'] = np.zeros(f)
        H_new1 = W_new1 = 1 + (H - 2) / 2  # 32/2=16
        f_new = f*2 # 32*2=64
        self.params['W2'] = np.random.normal(loc=0, scale=std, size=[f_new, f, HH, WW])
        #print self.params['W2'].shape
        self.params['b2'] = np.zeros(f_new)
        H_new2 = W_new2 =  (H_new1-HH+1) / 2
        # affine - relu
        self.params['W3'] = np.random.normal(loc=0, scale=std, size=[f_new * H_new2 * H_new2, hidden_dim])
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['W4'] = np.random.normal(loc=0, scale=std, size=[hidden_dim, hidden_dim])
        self.params['b4'] = np.zeros(hidden_dim)
        # affine
        self.params['W5'] = np.random.normal(loc=0, scale=std, size=[hidden_dim, num_classes])
        self.params['b5'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param1 = {'stride': 1, 'pad': (filter_size - 1) / 2}
        conv_param2 = {'stride': 1, 'pad': 0}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param1 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        pool_param2 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param1, pool_param1)
        out2, cache2 = conv_relu_pool_forward(out1, W2, b2, conv_param2, pool_param2)
        out3, cache3 = affine_relu_forward(out2, W3, b3)
        out4, cache4 = affine_relu_forward(out3, W4, b4)
        out5, cache5 = affine_forward(out4, W5, b5)
        scores = out5

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        # loss
        loss, dScores = softmax_loss(scores, y)
        sum_reg = 0.0
        for i in range(self.num_layers):
            W = self.params['W' + str(i + 1)]
            sum_reg += np.sum(W * W)
        loss += 0.5 * self.reg * sum_reg
        # grad
        dout5, grads['W5'], grads['b5'] = affine_backward(dScores, cache5)
        dout4, grads['W4'], grads['b4'] = affine_relu_backward(dout5, cache4)
        dout3, grads['W3'], grads['b3'] = affine_relu_backward(dout4, cache3)
        dout2, grads['W2'], grads['b2'] = conv_relu_pool_backward(dout3, cache2)
        dout1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout2, cache1)
        for i in range(self.num_layers):
            grads['W' + str(i + 1)] += self.reg * self.params['W' + str(i + 1)]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
