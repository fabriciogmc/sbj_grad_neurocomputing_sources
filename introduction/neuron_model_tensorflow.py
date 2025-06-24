# Simple neuron model using TensorFlow
#
# Author: Prof. Fabrício Galende M. de Carvalho

import numpy as np
import tensorflow as tf

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # only erros
#tf.config.set_visible_devices([], 'GPU')     # only CPU is used

class Neuron:
    def __init__(self, weights, bias, act_fcn):
        n_features = len(weights)

        # One dense neuron just after the input layer
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(n_features,)),
            tf.keras.layers.Dense(1, activation=act_fcn)
        ])

        # synaptic weights (n_features, 1)   bias: (1,). It must be noted that the activation 
        # potential (pre activation) is given by v = W^T .X^T + bias
        self.weights = np.array(weights, dtype=np.float32).reshape((n_features, 1))
        self.bias    = np.array([bias], dtype=np.float32)

        # Weights and bias that link input layer to dense layer (layers[0] - input layer is not taken into account)
        self.model.layers[0].set_weights([self.weights, self.bias])
        self.pre_act = np.array([0], dtype=np.float32)

    def output(self, x):
        self.pre_act = tf.matmul(x, self.weights) + self.bias
        return self.model(x)

