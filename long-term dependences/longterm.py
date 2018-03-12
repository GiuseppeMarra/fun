# EXTERNAL LIBRARIES

import numpy as np
import fun
import sys, select
import itertools
import os
import tensorflow as tf

# CONFIGURATION

class config():

    # One bit a time

    input_size = 1
    output_size = 1

    # All models parameters

    hidden_size = 20

    # FuN parameters

    k = 100 # Number of centers
    lambda_alphas = 0.00 # Regularization term for the alphas

    # Training setup

    EPOCHS = 10000 # Maximum number of epochs

    dataset_size = 5000 # A total of 100000 iterations
    batch_size = 500

    # visualization
    plot_each = 10

#################################################################################################
# Utilities

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

def xor_dataset(ndim=2):

    ''' Create dataset for n-dimensional XOR '''

    inputs, labels = [], []

    for item in ["".join(seq) for seq in itertools.product("01", repeat=ndim)]:

        single_input = []
        for i in list(item):
            single_input.append(float(i))

        inputs.append(single_input)
        labels.append([float(sum(single_input) % 2)])

    return inputs, labels

def generateBaseProblem(n=2):

    ''' Randomly generate a problem. Half of the labels is one, half is zero. '''

    x,y = xor_dataset(n)

    # Generate IFF problem
    y = np.array([1,0,0,1])

    return x,y

def generateBatch(x, y, sequence_size, batch_size=300):

    ''' Given a problem x (input) and y (labels), generate a batch of examples.
        Noise is added to the sequence, until it reaches the size "sequence_size". '''

    first_L = np.shape(x)[1]

    X = np.random.randint(0, 2, (batch_size, sequence_size))
    Y = np.abs(X[:, 0] - X[:, 1])

    for i in range(batch_size):
        r = np.random.randint(0, len(x))
        X[i, 0:first_L] = x[r]
        Y[i] = y[r]

    return X,Y


##################################################################################################

def cell(conf, model_name="RNN"):

    if model_name == "FUN":
        return fun.FunRNNCell(
                            conf.hidden_size,
                            k = conf.k,
                            lambda_alphas = conf.lambda_alphas,
                            reuse = tf.get_variable_scope().reuse)

    elif model_name == "LSTM":
        return tf.contrib.rnn.LSTMCell(
                            conf.hidden_size,
                            activation = tf.sigmoid,
                            reuse = tf.get_variable_scope().reuse)

    elif model_name == "RNN":
        return tf.contrib.rnn.BasicRNNCell(
                            conf.hidden_size,
                            activation = tf.sigmoid,
                            reuse = tf.get_variable_scope().reuse)


class STATIC_RNN():

    def __init__(self, conf, learning_rate, sequence_size, model_name = "RNN"):


        self.x = tf.placeholder(shape=[None, sequence_size, conf.input_size], dtype=tf.float32)
        self.y = tf.placeholder(shape= [None, conf.output_size], dtype=tf.float32)

        inputs = tf.unstack(self.x, num=sequence_size, axis=1)

        outputs, state = tf.contrib.rnn.static_rnn(
                            cell(conf, model_name=model_name),
                            inputs,
                            dtype=tf.float32)

        # compute cost
        h = outputs[-1]

        w = tf.get_variable("W",
                            shape=[conf.hidden_size, conf.output_size],
                            initializer=tf.truncated_normal_initializer)

        b = tf.get_variable("B",
                            shape=[conf.output_size],
                            initializer=tf.zeros_initializer)

        hcost= tf.matmul(h, w)+b

        if model_name=="FUN":
            hcost = tf.sigmoid(hcost)

        self.loss = tf.reduce_mean(tf.square(self.y -hcost))

        #evaluation
        thresh = tf.where(hcost > 0.5, tf.ones_like(hcost), tf.zeros_like(hcost))
        self.accuracy = 100 - tf.reduce_mean(tf.abs(self.y - thresh)) * 100

        # optimization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

def train():

    conf = config()

    f = open("LOG_LONGTERM_IFF_2BIT.txt", 'w')

    f.write("Configuration:\n")
    f.write("input_size = %d, output_size = %d, hidden_size = %d\n" % (conf.input_size, conf.output_size, conf.hidden_size))
    f.write("k = %d, lambda_alphas = %f\n" % (conf.k, conf.lambda_alphas))
    f.write("Max. epochs = %d, dataset_size = %d, batch_size = %d\n" % (conf.EPOCHS, conf.dataset_size, conf.batch_size))
    f.write("Adam opt. algorithm\n")

    f.flush()
    os.fsync(f.fileno())


    for n in (2,):

        # Models to be compared
        d = { "LSTM": 0, "FUN": 0}

        # Generate a (binary problem) problem
        x, y = generateBaseProblem(n)
        print "GENERATED PROBLEM:"


        for example in range(len(x)):
            print x[example], y[example]
            f.write(str(x[example]) + " ==> " + str(y[example]) + "\n")

        f.write("\n\n n , sequence_size , model_name , tentativo , accuracy, epochs, loss, learning_rate\n")

        for sequence_size in np.arange(5,51,5):

            for model_name in d.keys():

                acc = 0

                lr_list = (1e-3,)

                for learning_rate in lr_list:

                    for tentativo in range(5):

                        print("n=%d | sequence_size=%d | model_name=%s | tentativo=%d | learning_rate=%f"
                                % (n, sequence_size, model_name, tentativo, learning_rate))


                        # reset the default graph to build an (eventually) new one
                        tf.reset_default_graph()

                        m = STATIC_RNN(conf,
                                           learning_rate = learning_rate,
                                           sequence_size = sequence_size,
                                           model_name = model_name)

                        with tf.Session() as sess:

                            sess.run(tf.global_variables_initializer())

                            for e in range(conf.EPOCHS):

                                acc, loss = 0, 0
                                for k in range(conf.dataset_size // conf.batch_size):

                                    # generate new batch
                                    batch_X, batch_Y = generateBatch(x,y,sequence_size, conf.batch_size)
                                    batch_X = np.reshape(batch_X, [conf.batch_size, sequence_size, 1])
                                    batch_Y = np.reshape(batch_Y, [conf.batch_size, 1])

                                    _, acc_, loss_ =sess.run((m.train_op, m.accuracy, m.loss), feed_dict={m.x: batch_X, m.y: batch_Y})

                                    acc, loss = acc+acc_, loss+loss_

                                acc = acc/(conf.dataset_size // conf.batch_size)
                                loss = loss/(conf.dataset_size // conf.batch_size)

                                if (e == 0) or (e%conf.plot_each == conf.plot_each-1):
                                    print "Epoch =", e+1, "\t| Loss =", loss, "\t| Accuracy (train) =", acc

                                if heardEnter(): break

                                if np.isnan(loss): break

                                if acc > 99.: break

                            # generate a test set
                            X, Y = generateBatch(x,y,sequence_size, conf.dataset_size)
                            X = np.reshape(X, [conf.dataset_size, sequence_size, 1])
                            Y = np.reshape(Y, [conf.dataset_size, 1])

                            acc = sess.run(m.accuracy, feed_dict={m.x: X, m.y: Y})
                            print "Accuracy on test =", acc, " (model ", model_name, ")"

                            f.write("%d, %d, %s, %d, %.3f, %d, %.3f, %f\n" % (n, sequence_size, model_name, tentativo, acc, e, loss, learning_rate))
                            f.flush()
                            os.fsync(f.fileno())

    f.close()


###################################################################################################
''' Main '''

def main(_):

    train()

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()



