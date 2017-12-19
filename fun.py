import tensorflow as tf
import math

def gaussian_kernel(sigma):

    def __internal__(x):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * tf.exp(-0.5 * x / sigma)

    return __internal__



def fun_activation(x, k=10, kernel=gaussian_kernel(1.)):

    batch_size, size = tf.shape(x)[0], x.get_shape()[1].value
    centers = tf.get_variable("fun_centers", shape=[k,size], initializer=tf.truncated_normal_initializer)

    x = tf.tile(tf.expand_dims(x,1), [1, k, 1])
    mu  = tf.tile(tf.expand_dims(centers, 0), [batch_size,1,1])

    diff = x- mu
    G = kernel(diff)

    alphas = tf.get_variable("alphas", shape=[k,size], initializer=tf.truncated_normal_initializer)
    a_G = tf.multiply(alphas,G)
    return tf.reduce_sum(a_G, axis=1)
