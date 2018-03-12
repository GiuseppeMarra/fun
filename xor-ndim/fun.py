import tensorflow as tf
import math

eps = 10e-12

def fun_activation(x, k=10, kernel=tf.abs, lambda_alphas = 0.):

    batch_size, size = tf.shape(x)[0], x.get_shape()[1].value

    # get (or create) center variables for kernel functions
    centers = tf.get_variable("fun_centers", shape=[k,size], initializer=tf.truncated_normal_initializer)

    # compute kernel on input
    x = tf.tile(tf.expand_dims(x,1), [1, k, 1])
    mu  = tf.tile(tf.expand_dims(centers, 0), [batch_size,1,1])
    diff = x- mu
    G = kernel(diff)

    # get (or create) alphas and weight kernel functions
    alphas = tf.get_variable("fun_alphas", shape=[k, size], initializer=tf.truncated_normal_initializer)

    a_G = tf.multiply(alphas, G)

    '''Regularization terms'''

    # minimize number of activeted alphas

    tf.add_to_collection("losses",
                       lambda_alphas * tf.reduce_sum(tf.abs(alphas)))

    return tf.reduce_sum(a_G, axis=1)


class FunRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, k, kernel=tf.abs, reuse=None):
        super(FunRNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self.k = k
        self.kernel = kernel


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):

        new_input = tf.concat([inputs, state], 1)
        new_input_size = new_input.get_shape()[1].value

        with tf.variable_scope("FUNCell"):
            w1 = tf.get_variable("w1", shape=[new_input_size, self._num_units],
                                 initializer=tf.ones_initializer)
            b1 = tf.get_variable("b1", shape=[self._num_units], initializer=tf.random_uniform_initializer)

            a1 = tf.matmul(new_input, w1) + b1

            with tf.variable_scope("state"):
                state = fun_activation(a1, k=self.k, kernel=self.kernel)

        return state, state