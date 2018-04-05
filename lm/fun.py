import tensorflow as tf

eps = 1e-12

class FunActivation(tf.layers.Layer):
    def __init__(self, k=10, kernel=tf.nn.relu, lambda_alphas_r=0.):
        super(FunActivation, self).__init__()
        self.k = k
        self.kernel = kernel
        self.lambda_alphas_r = lambda_alphas_r

    def build(self, input_shape):
        size = input_shape[1].value

        # get (or create) center variables for kernel functions
        self.centers = tf.get_variable("fun_centers", shape=[self.k, size], initializer=tf.truncated_normal_initializer)
        self.alphas = tf.get_variable("fun_alphas", shape=[self.k, size], initializer=tf.zeros_initializer)

        self.built = True

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        self.x = tf.tile(tf.expand_dims(inputs, 1), [1, self.k, 1])
        mu = tf.tile(tf.expand_dims(self.centers, 0), [batch_size, 1, 1])
        diff = self.x - mu
        self.G = self.kernel(diff)

        a_G = tf.multiply(self.alphas, self.G)

        '''Regularization terms'''

        # minimize number of active alphas
        tf.add_to_collection("alphas_regularization",
                             self.lambda_alphas_r * tf.reduce_sum(tf.abs(self.alphas)))

        return tf.reduce_sum(a_G, axis=1)


def fun_activation(x, k=10, kernel=tf.nn.relu, lambda_alphas_r=0.):

    batch_size, size = tf.shape(x)[0], x.get_shape()[1].value

    # get (or create) center variables for kernel functions
    centers = tf.get_variable("fun_centers", shape=[k, size], initializer=tf.truncated_normal_initializer)
    tf.add_to_collection("centers", centers)

    # compute kernel on input
    x = tf.tile(tf.expand_dims(x,1), [1, k, 1])
    mu = tf.tile(tf.expand_dims(centers, 0), [batch_size,1,1])
    diff = x - mu
    G = kernel(diff)

    # get (or create) alphas and weight kernel functions
    alphas = tf.get_variable("fun_alphas", shape=[k, size], initializer=tf.zeros_initializer)
    tf.add_to_collection("alphas", alphas)


    a_G = tf.multiply(alphas, G)

    '''Regularization terms'''

    # minimize number of active alphas
    tf.add_to_collection("alphas_regularization",
                         lambda_alphas_r * tf.reduce_sum(tf.abs(alphas)))

    return tf.reduce_sum(a_G, axis=1)


class FunRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, k, lambda_alphas_r=0., lambda_weights_r=0., kernel=tf.nn.relu, is_training=True, reuse=None):
        super(FunRNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self.k = k
        self.kernel = kernel
        self.lambda_alphas_r = lambda_alphas_r
        self.lambda_weights_r = lambda_weights_r
        self.is_training = is_training


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
                                 initializer=tf.truncated_normal_initializer)
            b1 = tf.get_variable("b1", shape=[self._num_units], initializer=tf.zeros_initializer)

            a1 = tf.matmul(new_input, w1) + b1

            # a1 = tf.contrib.layers.batch_norm(a1,center=True, scale=True, scope='bn', is_training=self.is_training)
            #a1 = tf.contrib.layers.batch_norm(a1,center=True, scale=True, scope='bn')

            '''Regularization terms'''

            tf.add_to_collection("weights_regularization",
                                 self.lambda_weights_r * tf.reduce_sum(tf.square(w1)))

            with tf.variable_scope("state"):
                # state = fun_activation(a1, k=self.k, lambda_alphas_r=self.lambda_alphas_r, kernel=self.kernel)
                self.fun = FunActivation(k=self.k, lambda_alphas_r=self.lambda_alphas_r, kernel=self.kernel)
                state = self.fun.apply(a1)



        return state, state