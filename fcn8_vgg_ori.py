from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf

import util.tf

VGG_MEAN = [103.939, 116.779, 123.68]


class FCN8VGG:

    def __init__(self, vgg16_npy_path=None, weight_decay = None):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
        self.wd = weight_decay

    def build(self, rgb, train=False, num_classes=2, random_init_fc8=True,
              debug=False, labels = None):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        # Convert RGB to BGR

        with tf.name_scope('Processing'):
        
            red, green, blue = tf.split(rgb, 3, 3)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], 3)

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)
                if labels is not None:
                    labels = tf.Print(labels, [tf.shape(labels)],
                               message='Shape of input label: ',
                               summarize=4, first_n=1)

        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug, kernel_size = 3, stride = 1)

        self.fc6 = self._fc_layer(self.pool5, "fc6")

        if train:
            self.fc6 = tf.nn.dropout(self.fc6, 0.5)

        self.fc7 = self._fc_layer(self.fc6, "fc7")
        if train:
            self.fc7 = tf.nn.dropout(self.fc7, 0.5)

        self.score_fr = self._score_layer(self.fc7, "score_fr", num_classes)

        self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
                                             num_classes=num_classes)
        self.fuse_pool4 = tf.add(self.score_fr, self.score_pool4)

        self.upscore4 = self._upscore_layer(self.fuse_pool4,
                                            shape=tf.shape(self.pool3),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore4',
                                            ksize=4, stride=2)
        self.score_pool3 = self._score_layer(self.pool3, "score_pool3",
                                             num_classes=num_classes)
        self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)

        self.upscore8 = self._upscore_layer(self.fuse_pool3,
                                             shape=tf.shape(bgr),
                                             num_classes=num_classes,
                                             debug=debug, name='upscore8',
                                             ksize=16, stride=8)
                
        self.pred_up = tf.argmax(self.upscore8, dimension=3)
        self.pred_score = tf.nn.softmax(self.upscore8);
        if train:
            softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.upscore8, labels = labels, name = "softmax_loss")
            softmax_loss = tf.reduce_mean(softmax_loss);
            tf.add_to_collection("losses", softmax_loss)
            self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            tf.summary.scalar('loss', self.loss)

    def _max_pool(self, bottom, name, debug, kernel_size = 2, stride = 2):
        pool = tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _conv_layer(self, bottom, name, dilation = 1):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            
            if dilation == 1:
                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            else:
                conv = tf.nn.atrous_conv2d(bottom, filt, dilation, padding='SAME')
          

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu

    def _score_layer(self, bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name == "score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            elif name == "score_pool3":
                stddev = 0.0001
            # Apply convolution
            weights = self._variable_with_weight_decay(shape, stddev, self.wd)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)

            return bias

    def _variable_with_weight_decay(self, shape, stddev, wd = None):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        if self.wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection("losses", weight_decay)
        _variable_summaries(var)
        return var

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)
            f_shape = [ksize, ksize, num_classes, num_classes]
            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')
            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)], 
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv
    
    
    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear
        
        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name="up_filter", initializer=init, shape=weights.shape, trainable = False)
        return var

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0], dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if self.wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd, name='weight_loss')
            tf.add_to_collection("losses", weight_decay)
        _variable_summaries(var)
        return var

    def get_bias(self, name, num_output=None):
        if name.startswith('fc'):
            bias_wights = 0
            shape = [num_output,]
        else:
            bias_wights = self.data_dict[name][1]
            shape = self.data_dict[name][1].shape
        init = tf.constant_initializer(value=bias_wights, dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        _variable_summaries(var)
        return var

    def get_fc_weight(self, shape):
        num_input = np.prod(shape[0:-1]);
        stddev = (2.0 / num_input)**0.5
        var = self._variable_with_weight_decay(shape, stddev, self.wd);
        return var




    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        var = tf.get_variable(name='biases', shape=shape,
                              initializer=initializer)
        _variable_summaries(var)
        return var

  
    def _fc_layer(self, bottom, name, num_classes=None, relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            num_input = shape[-1];
            if name == 'fc6':
                num_output = 4096;
                kernel_size = 3;
            elif name == 'score_fr':
                num_output = num_classes;
                kernel_size = 1;
            else:
                num_output = 4096;
                kernel_size = 1;

            filt = self.get_fc_weight([1, 1, num_input, num_output])
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_output = num_output)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias



def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    return
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    return
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)
