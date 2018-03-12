# EXTERNAL LIBRARIES
import tensorflow as tf
import numpy as np
import fun
import sys, select
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# DATA SETTING
sequence_size= 30
input_size = 1
hidden_size = 1
output_size= 1

# LEARNING PARAMETERS
batch_size = 500

EPOCHS = 10000
learning_rate = 1e-3

# VISUALIZATION OPTIONS
plot_each = 1

#################################################################################################
# Utils

def heardEnter():

    ''' Listen for the user pressing ENTER '''

    i,o,e = select.select([sys.stdin],[],[],0.0001)

    for s in i:

        if s == sys.stdin:
            input = sys.stdin.readline()
            return True

    return False

##################################################################################################
# Generate dataset

def generateSequenceSum(sequence_size, dataset_size=300, max_n=10):

    # a sequence of the type
    # (INPUT):  2 0 0 5 0 0 9 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 ...
    # (OUTPUT): 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 ...

    m = (sequence_size // max_n)

    X = np.zeros((dataset_size, sequence_size))
    Y = np.zeros((dataset_size, sequence_size))

    for d in range(dataset_size):

        x = np.zeros([sequence_size], dtype=np.float32)
        y = np.zeros([sequence_size], dtype=np.float32)

        for i in range(m):

            idx = np.random.randint(0,sequence_size)
            n = np.random.randint(1, max_n+1)
            x[idx] = n

        ones = 0
        for i in range(sequence_size):
            ones += x[i]
            if ones:
                y[i] = 1
                ones -=1

        X[d] = x
        Y[d] = y

    return X,Y

##################################################################################################

def cell(model_name="RNN", hidden_size = 1, k=3):
    if model_name == "FUN":
        return fun.FunRNNCell(
                            hidden_size,
                            k=k,
                            reuse=tf.get_variable_scope().reuse)
    elif model_name == "LSTM":
        return tf.contrib.rnn.LSTMCell(
                            hidden_size,
                            activation=tf.sigmoid,
                            reuse=tf.get_variable_scope().reuse)
    elif model_name == "RNN":
        return tf.contrib.rnn.BasicRNNCell(
                            hidden_size,
                            activation=tf.sigmoid,
                            reuse=tf.get_variable_scope().reuse)

class STATIC_RNN():

    def __init__(self, model_name = "RNN", hidden_size=1, k=3):

        self.x = tf.placeholder(shape=[None, sequence_size, input_size], dtype=tf.float32)
        self.y = tf.placeholder(shape= [None, sequence_size, output_size], dtype=tf.float32)

        inputs = tf.unstack(self.x, num=sequence_size, axis=1)
        outputs, _ = tf.contrib.rnn.static_rnn(
            cell(model_name=model_name, hidden_size=hidden_size, k=k), inputs, dtype=tf.float32)

        # compute cost
        hcost = tf.stack(outputs, axis=1)
        if model_name=="FUN":
            hcost = tf.sigmoid(hcost) # TODO: non funziona con rnn lstm

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - hcost), reduction_indices=[1]))

        # evaluation
        thresh = tf.where(hcost > 0.5, tf.ones_like(hcost), tf.zeros_like(hcost))
        self.accuracy = 100 - tf.reduce_mean(tf.abs(self.y - thresh)) * 100

        # optimization
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

def train():

    X_test, Y_test = generateSequenceSum(sequence_size, 25000)
    X_test = np.reshape(X_test, [25000, sequence_size, 1])
    Y_test = np.reshape(Y_test, [25000, sequence_size, 1])

    for model_name in ("FUN","LSTM", "RNN"):

        for k in (20,):

            max_accuracy = 0.



            print("\n\nUsing %s model" % model_name)

            # reset the default graph to build an (eventually) new one
            tf.reset_default_graph()

            plot_epoch = []
            plot_acc = []

            m = STATIC_RNN(model_name=model_name, hidden_size=hidden_size, k=k)

            saver = tf.train.Saver()
            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())



                for e in range(EPOCHS):

                    X, Y = generateSequenceSum(sequence_size, batch_size)
                    batch_X = np.reshape(X, [batch_size, sequence_size, 1])
                    batch_Y = np.reshape(Y, [batch_size, sequence_size, 1])

                    sess.run(m.train_op, feed_dict={m.x: batch_X, m.y: batch_Y})

                    acc, loss = sess.run((m.accuracy, m.loss) , feed_dict={m.x: X_test, m.y: Y_test})

                    print(e, acc, loss)
                    if acc > max_accuracy: max_accuracy = acc

                    if heardEnter(): break


                saver.save(sess, "savings/model.ckpt")
                A_START, A_STOP = -10, 10
                A_STEP = abs(A_STOP - A_START) / 1000.
                A = np.reshape(np.arange(start=A_START, stop=A_STOP, step=A_STEP, dtype=np.float32),
                               [-1, 1])
                # A = np.tile(A, [1, 20])
                A_tensor = tf.convert_to_tensor(A)

                with tf.variable_scope("rnn/fun_rnn_cell/FUNCell/state", reuse=True):
                    y_ = fun.fun_activation(A_tensor,
                                            k=k)

                    y_ = sess.run(y_)


                plt.plot(A, y_)
                # ax[i // 4][i % 4].set_ylim([-5 ,5])
                # ax[i // 4][i % 4].axes.xaxis.set_ticklabels([])
                # ax[i // 4][i % 4].axes.yaxis.set_ticklabels([])

                plt.show()
                plt.close()





###################################################################################################
''' Main '''

def main(_):

    train()

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()



