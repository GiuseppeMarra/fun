import tensorflow as tf
import fun
import os
import multiprocessing
from threading import Thread
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np




learning_rate = 0.01
stop_at = 100


class FunXORConfiguration(object):

    input_size = 2
    output_size = 1
    k = 100
    sigma = 0.2
    kernel = tf.abs
    # kernel = fun.gaussian_kernel(config.sigma)





class FunXOR():

    def __init__(self, input, labels, config, reuse=False):
        self.global_step = tf.train.get_or_create_global_step()

        #First Layer
        with tf.variable_scope("XOR", reuse=reuse):
            # w1 = tf.get_variable("w1", shape=[config.input_size, config.output_size], initializer=tf.random_uniform_initializer)
            # b1 = tf.get_variable("b1", shape=[config.output_size], initializer=tf.random_uniform_initializer)
            w1 = tf.ones([config.input_size, config.output_size])
            b1 = tf.zeros([config.output_size])
            a1 = tf.matmul(input,w1) + b1
            y = fun.fun_activation(a1, k=config.k, kernel=tf.abs)
            # y = tf.nn.relu(a1)

        self.output = y

        #Cost Function
        self.loss = tf.reduce_mean(tf.square(y - labels))
        self.loss += tf.add_n(tf.get_collection('losses'), name='total_loss')
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(a1),
        #                                                                    labels=tf.cast(labels, tf.float32)))

        #Evaluation
        correct_pred = tf.abs(labels - y) < 0.5
        cast = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(cast)



def train():

    #Reading Data
    inputs = tf.constant([[0.,0], [0,1], [1,0], [1,1]])
    labels = tf.constant([[0.], [1], [1], [0]])

    #Instantiating the configuration object
    config = FunXORConfiguration()


    #Instantiating the model and the desired summaries
    with tf.variable_scope("Model") as scope:
        model = FunXOR(input=inputs, labels=labels, config=config)


    #Optimization
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'loss': model.loss,
                 'accuracy': model.accuracy,
                 'h1': tf.squeeze(model.output)},
        every_n_iter=100)



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            _, acc = sess.run((train_op, model.accuracy))
            if i%100==0: print(acc)

        with tf.variable_scope("Model/XOR", reuse=True):
            X = np.reshape(np.arange(start=-5, stop=5, step=0.1, dtype=np.float32), [-1,1])
            x_ = tf.convert_to_tensor(X)
            y_ = fun.fun_activation(x_, k=config.k, kernel=tf.abs)
            y_ = sess.run(y_)
            plt.scatter(X,np.reshape(y_, [-1,1]))
            plt.show()






def main(_):
    train()



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
    # p1 = multiprocessing.Process(target=train(log_dir))
    # p2 = multiprocessing.Process(target=eval(log_dir))
    #
    # p1.start()
    # p2.start()
    #
    # print("started")
    # p1.join()
    # p2.join()


