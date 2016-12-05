import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class DZJConvNet(object):
    """
    An amazing convolutional network developed by dzj with the following architecture:

    [[conv - spat_batch_norm - relu]*layer_N - 2x2 max pool]*layer_M - [affine - batch_norm - relu - dropout]*layer_K - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, layer_N = 3, layer_M = 2, layer_K = 2, pool_size = 2, pool_stride = 2, input_dim=(3, 32, 32),
                 num_filters=32, filter_size=3, hidden_dim=200, num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data, H and W should be power of 2
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.

        filter_size, filter_stride, pool_size and pool_stride keep constant among all layers
        """

        self.layer_N = layer_N
        self.layer_M = layer_M
        self.layer_K = layer_K
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size
        self.filter_stride = 1
        # the pad can keep the height and width of input image unchanged during forward pass of conv layer
        self.pad = (filter_size - 1) / 2
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        ############################################################################
        # TODO: Initialize weights and biases                                      #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        ############################################################################
        conv_layers_output_h = input_dim[1]
        conv_layers_output_w = input_dim[2]
        for i in xrange(layer_M):
            for j in xrange(layer_N):
                if i == 0 and j == 0:
                    self.params["CW1"] = np.random.normal(0, weight_scale, (num_filters, input_dim[0], filter_size, filter_size))
                else:
                    self.params["CW" + str(i * layer_N + j + 1)] = np.random.normal(0, weight_scale, (num_filters, num_filters, filter_size, filter_size))
                self.params["Cb" + str(i * layer_N + j + 1)] = np.zeros(num_filters)
            conv_layers_output_h = 1 + (conv_layers_output_h - pool_size) / pool_stride
            conv_layers_output_w = 1 + (conv_layers_output_w - pool_size) / pool_stride
            if type(conv_layers_output_h) != int or type(conv_layers_output_w) != int:
                raise ValueError

        for i in xrange(layer_K):
            if i == 0:
                self.params["FW1"] = np.random.normal(0, weight_scale, (num_filters * conv_layers_output_h * conv_layers_output_w, hidden_dim))
            else:
                self.params["FW" + str(i + 1)] = np.random.normal(0, weight_scale, (hidden_dim, hidden_dim))
            self.params["Fb" + str(i + 1)] = np.zeros(hidden_dim)

        self.params["OW"] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params["Ob"] = np.zeros(num_classes)
        pass
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
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # self.dropout_param['mode'] = mode
        # for bn_param in self.bn_params:
        #     bn_param[mode] = mode

        CW, Cb, FW, Fb = [], [], [], []
        for i in xrange(self.layer_M):
            for j in xrange(self.layer_N):
                CW.append(self.params["CW" + str(i * self.layer_N + j + 1)])
                Cb.append(self.params["Cb" + str(i * self.layer_N + j + 1)])

        for i in xrange(self.layer_K):
            FW.append(self.params["FW" + str(i + 1)])
            Fb.append(self.params["Fb" + str(i + 1)])

        OW = self.params["OW"]
        Ob = self.params["Ob"]

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': self.filter_stride, 'pad': self.pad}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': self.pool_size, 'pool_width': self.pool_size, 'stride': self.pool_stride}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_relu_outs, conv_relu_caches, pool_outs, pool_caches, hidden_fc_relu_outs, hidden_fc_relu_caches = [], [], [], [], [], []
        out_temp, cache_temp = None, None

        for i in xrange(self.layer_M):
            for j in xrange(self.layer_N):
                if i == 0 and j ==0:
                    out_temp = X
                out_temp, cache_temp = conv_relu_forward(out_temp, CW[i * self.layer_N + j], Cb[i * self.layer_N + j], conv_param)
                conv_relu_outs.append(out_temp)
                conv_relu_caches.append(cache_temp)
            out_temp, cache_temp = max_pool_forward_fast(out_temp, pool_param)
            pool_outs.append(out_temp)
            pool_caches.append(cache_temp)

        for i in xrange(self.layer_K):
            out_temp, cache_temp = affine_relu_forward(out_temp, FW[i], Fb[i])
            hidden_fc_relu_outs.append(out_temp)
            hidden_fc_relu_caches.append(cache_temp)

        final_fc_layer_out, final_fc_layer_cache = affine_forward(out_temp, OW, Ob)
        scores = final_fc_layer_out
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if mode == "test":
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * ((np.sum(np.sum(w * w) for w in CW)) + np.sum(np.sum(w * w) for w in FW) + np.sum(OW * OW))

        dout, grads["OW"], grads["Ob"] = affine_backward(dout, final_fc_layer_cache)
        grads["OW"] += self.reg * OW

        for i in xrange(self.layer_K, 0, -1):
            dout, grads["FW" + str(i)], grads["Fb" + str(i)] = affine_relu_backward(dout, hidden_fc_relu_caches[i - 1])
            grads["FW" + str(i)] += self.reg * FW[i - 1]

        for i in xrange(self.layer_M, 0, -1):
            dout = max_pool_backward_fast(dout, pool_caches[i - 1])
            for j in xrange(self.layer_N, 0, -1):
                dout, grads["CW" + str((i - 1) * self.layer_N + j)], grads["Cb" + str((i - 1) * self.layer_N + j)] = \
                    conv_relu_backward(dout, conv_relu_caches[(i - 1) * self.layer_N + j - 1])
                grads["CW" + str((i - 1) * self.layer_N + j)] += self.reg * CW[(i - 1) * self.layer_N + j - 1]
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

pass
