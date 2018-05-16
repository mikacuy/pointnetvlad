# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Learnable mOdUle for Pooling fEatures (LOUPE)
Contains a collection of models (NetVLAD, NetRVLAD, NetFV and Soft-DBoW)
which enables pooling of a list of features into a single compact 
representation.

Reference:

Learnable pooling method with Context Gating for video classification
Antoine Miech, Ivan Laptev, Josef Sivic

"""

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class PoolingBaseModel(object):
    """Inherit from this class when implementing new models."""

    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
            gating=True,add_batch_norm=True, is_training=True):
        """Initialize a NetVLAD block.

        Args:
        feature_size: Dimensionality of the input features.
        max_samples: The maximum number of samples to pool.
        cluster_size: The number of clusters.
        output_dim: size of the output space after dimension reduction.
        add_batch_norm: (bool) if True, adds batch normalization.
        is_training: (bool) Whether or not the graph is training.
        """

        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        raise NotImplementedError("Models should implement the forward pass.")

    def context_gating(self, input_layer):
        """Context Gating

        Args:
        input_layer: Input layer in the following shape:
        'batch_size' x 'number_of_activation'

        Returns:
        activation: gated layer in the following shape:
        'batch_size' x 'number_of_activation'
        """

        input_dim = input_layer.get_shape().as_list()[1] 
        
        gating_weights = tf.get_variable("gating_weights",
          [input_dim, input_dim],
          initializer = tf.random_normal_initializer(
          stddev=1 / math.sqrt(input_dim)))
        
        gates = tf.matmul(input_layer, gating_weights)
 
        if self.add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [input_dim],
            initializer = tf.random_normal(stddev=1 / math.sqrt(input_dim)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(input_layer,gates)

        return activation

#Edited based on the original version
class NetVLAD(PoolingBaseModel):
    """Creates a NetVLAD class.
    """
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
            gating=True,add_batch_norm=True, is_training=True):
        super(self.__class__, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training)

    def forward(self, reshaped_input):
        """Forward pass of a NetVLAD block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(
              stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)

        # activation = tf.contrib.layers.batch_norm(activation, 
        #         center=True, scale=True, 
        #         is_training=self.is_training,
        #         scope='cluster_bn')

        # activation = slim.batch_norm(
        #       activation,
        #       center=True,
        #       scale=True,
        #       is_training=self.is_training,
        #       scope="cluster_bn")
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn", fused=False)
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [self.cluster_size],
            initializer = tf.random_normal_initializer(
            stddev=1 / math.sqrt(self.feature_size)))
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation,
                [-1, self.max_samples, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(
                stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,
            self.max_samples, self.feature_size])

        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        

        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1, self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        hidden1_weights = tf.get_variable("hidden1_weights",
          [self.cluster_size*self.feature_size, self.output_dim],
          initializer=tf.random_normal_initializer(
          stddev=1 / math.sqrt(self.cluster_size)))
        
        ##Tried using dropout
        #vlad=tf.layers.dropout(vlad,rate=0.5,training=self.is_training)

        vlad = tf.matmul(vlad, hidden1_weights)

        ##Added a batch norm
        vlad = tf.contrib.layers.batch_norm(vlad, 
                                          center=True, scale=True, 
                                          is_training=self.is_training,
                                          scope='bn')

        if self.gating:
            vlad = super(self.__class__, self).context_gating(vlad)

        return vlad
