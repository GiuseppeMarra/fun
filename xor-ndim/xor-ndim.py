import tensorflow as tf
import fun
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

learning_rate = 0.01

import itertools
import sys
import select
import seaborn as sns


###################################################################################################
''' Some tools '''

def heardEnter():

    ''' Listen for the user pressing ENTER '''

    i,o,e = select.select([sys.stdin],[],[],0.0001)

    for s in i:

        if s == sys.stdin:
            input = sys.stdin.readline()
            return True

    return False

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

###################################################################################################
''' Model tools '''

class FunXORConfiguration(object):

    ''' Model configuration '''

    def __init__(self, ndim):

        # input specs
        self.input_size = ndim
        self.output_size = 1

        # number of kernel-funcs for each neuron
        self.k = 1

        # absolute-value kernel
        self.kernel = tf.abs
        self.lambda_alphas = 0.

class FunXOR():

    ''' Create the model's variables and operations '''

    def __init__(self, input, labels, config, reuse=False):
        self.global_step = tf.train.get_or_create_global_step()

        #First Layer
        with tf.variable_scope("XOR", reuse=reuse):

            w1 = tf.get_variable("w1",
                                 shape=[config.input_size, config.output_size],
                                 initializer=tf.truncated_normal_initializer)

            b1 = tf.get_variable("b1",
                                 shape=[config.output_size],
                                 initializer=tf.truncated_normal_initializer)

            a1 = tf.matmul(input,w1) + b1

            y = fun.fun_activation(a1,
                                   k=config.k,
                                   kernel=config.kernel,
                                   lambda_alphas = config.lambda_alphas)

        self.activations = a1

        self.output = y

        #Cost Function
        self.loss = tf.reduce_mean(tf.square(y - labels))
        self.loss += tf.add_n(tf.get_collection('losses'))

        #Evaluation
        correct_pred = tf.abs(labels - y) < 0.5
        cast = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(cast)

###################################################################################################
''' Model training (and visualization) '''

def train(config):

    #Reading Data
    inputs, labels = xor_dataset(ndim=config.input_size)

    # Try different settings
    lambda_alphas = (0.000, 0.00075)
    kappas = (50,100,300)

    # creaate a figure, for visualization purpose
    fig, ax = plt.subplots(2, len(kappas))

    for iter_reg, config.lambda_alphas in enumerate(lambda_alphas):

        for iter, config.k in enumerate(kappas):

            tf.reset_default_graph()

            #Instantiating the model and the desired summaries
            model_name = "Model_"+str(kappas[iter])+"_"+str(lambda_alphas[iter_reg])

            with tf.variable_scope(model_name) as scope:
                model = FunXOR(input=inputs, labels=labels, config=config)


            #Optimization
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)

            saver = tf.train.Saver()
            # Training session
            with tf.Session() as sess:
                # initialize all variables
                sess.run(tf.global_variables_initializer())

                # FOR-cycle on the training epochs
                EPOCHS = 10**6
                for i in range(EPOCHS):

                    # training operation
                    _, acc, loss = sess.run((train_op, model.accuracy, model.loss))

                    # Print training information
                    if i == 0 or i%100 == 99:
                        print "Epoch =", i+1, "\t| Loss =", round(loss, 3), "\t| Accuracy =", round(acc, 3)

                        # if acc == 1: break

                    # If user press ENTER, break the FOR-cycle
                    if heardEnter(): break

                saver.save(sess, model_name+"/model.ckpt")



                with tf.variable_scope(model_name+"/XOR", reuse=True):

                    # find specific range of action of the FuN node
                    all_activations = sess.run(model.activations)
                    A_START, A_STOP = all_activations.min(), all_activations.max()
                    delta = abs(A_STOP-A_START)/5.
                    A_START, A_STOP = A_START - delta, A_STOP + delta
                    A_STEP = abs(A_STOP-A_START)/1000.
                    A = np.reshape(np.arange(start=A_START, stop=A_STOP, step=A_STEP, dtype=np.float32), [-1,1])
                    A_tensor = tf.convert_to_tensor(A)

                    y_ = fun.fun_activation(A_tensor,
                                            k=config.k,
                                            kernel=config.kernel,
                                            lambda_alphas=config.lambda_alphas)
                    y_ = sess.run(y_)

                    output = sess.run(model.output)

                    ax[iter_reg, iter].plot(A, np.reshape(y_, [-1,1]))
                    ax[iter_reg, iter].scatter(all_activations, output)

                    ax[iter_reg, iter].set_ylim([-.2, 1.2])

                with tf.variable_scope(model_name+"/XOR", reuse=True):

                    w1 = tf.get_variable("w1")
                    b1 = tf.get_variable("b1")
                    centers = tf.get_variable("fun_centers")
                    alphas = tf.get_variable("fun_alphas")

                    w,b, c, a = sess.run((w1, b1, centers, alphas)
                                         )
                    print(w)
                    print(b)
                    if config.lambda_alphas==0.00075 and config.k == 300:
                        plt.savefig("funzioni.eps")
                        plt.close()

                        sns.kdeplot(np.squeeze(a), bw=0.5, )
                        plt.xlabel("$\chi$")
                        # plt.hist(np.squeeze(a), bins=30)
                        plt.savefig("distribuzione_chi.eps")
                        plt.close()

                        plt.scatter(c, np.squeeze(a))
                        plt.xlabel("$c$")
                        plt.ylabel("$\chi$")
                        plt.savefig("distribuzione_chi.eps")
                        plt.close()

    plt.show()

###################################################################################################
''' Main '''

def main(_):

    for ndim in (2,4,):

            config = FunXORConfiguration(ndim=ndim)
            train(config)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


