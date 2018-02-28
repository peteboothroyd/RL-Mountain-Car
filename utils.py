import tensorflow as tf
import numpy as np
import random

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def build_mlp(
        input_placeholder,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
        var_collection=None
    ):

    with tf.variable_scope(scope):
        output = input_placeholder
        for i in range(n_layers):
            layer = tf.layers.Dense(units=size, activation=activation,
                                    kernel_initializer=tf.glorot_normal_initializer(),
                                    kernel_regularizer=kernel_regularizer,
                                    name="dense_{}".format(i))
            output = layer(output)

            if var_collection is not None:
                weights = layer.trainable_weights
                for weight in weights:
                    tf.add_to_collection(var_collection, weight)


    return output
    