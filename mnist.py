import tensorflow as tf
import fun
import os
import multiprocessing
from threading import Thread
import time

mode = "eval"
log_dir ="tmp/mnist"
batch_size = 256
learning_rate = 0.0006
num_steps = 20000


class FunConfiguration(object):

    input_size = 784
    hidden_size = 300
    output_size = 10
    k = 10
    sigma = 4



class FunClassifier():

    def __init__(self, input, labels, config, reuse=False):
        self.global_step = tf.train.get_or_create_global_step()

        #First Layer
        with tf.variable_scope("Layer1", reuse=reuse):
            w1 = tf.get_variable("w1", shape=[config.input_size, config.hidden_size], initializer=tf.truncated_normal_initializer)
            b1 = tf.get_variable("b1", shape=[config.hidden_size], initializer=tf.zeros_initializer)
            a1 = tf.matmul(input,w1) + b1
            h1 = fun.fun_activation(a1, k=config.k, kernel= fun.gaussian_kernel(config.sigma))

        #Second Layer
        with tf.variable_scope("Layer2", reuse=reuse):
            w2 = tf.get_variable("w2", shape=[config.hidden_size, config.output_size], initializer=tf.truncated_normal_initializer)
            b2 = tf.get_variable("b2", shape=[config.output_size], initializer=tf.zeros_initializer)
            a2 = self.logits = tf.matmul(h1,w2) + b2

        self.output = y = tf.nn.softmax(a2)

        #Cost Function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                       labels=labels))

        #Evaluation
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def train(log_root):

    #Reading Data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(
        (mnist.train.images, mnist.train.labels))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()

    #Instantiating the configuration object
    config = FunConfiguration()


    #Instantiating the model and the desired summaries
    with tf.variable_scope("Model") as scope:
        model = FunClassifier(input=images, labels=labels, config=config)
        tf.summary.scalar("Loss", model.loss)
        tf.summary.scalar("Accuracy", model.accuracy)


    #Optimization
    lrn_rate = tf.constant(learning_rate, tf.float32)
    train_op = tf.train.AdamOptimizer(lrn_rate).minimize(model.loss, model.global_step)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=os.path.join(log_root,"train"),
        summary_op=tf.summary.merge_all())

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss': model.loss,
                 'accuracy': model.accuracy},
        every_n_iter=500)

    stop_hook = tf.train.StopAtStepHook(num_steps=num_steps)

    #Run Operations just before starting the learning
    scaffold = tf.train.Scaffold(
        local_init_op=iterator.initializer)

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=log_root, # save automatically the model.ckpt every 60 sec
            save_checkpoint_secs=2,
            hooks=[logging_hook],
            chief_only_hooks=[summary_hook, stop_hook],
            save_summaries_steps=0,
            scaffold=scaffold,
            config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)

def eval(log_root):
    tf.reset_default_graph()

    # Reading Data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(
        (mnist.test.images, mnist.test.labels))
    dataset = dataset.repeat()
    dataset = dataset.batch(5000)
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()

    # Instantiating the configuration object
    config = FunConfiguration()

    # Instantiating the model and the desired summaries
    with tf.variable_scope("Model") as scope:
        model = FunClassifier(input=images, labels=labels, config=config)
        tf.summary.scalar("Loss", model.loss)
        tf.summary.scalar("Accuracy", model.accuracy)


    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0},allow_soft_placement=True))
    sess.run(iterator.initializer)
    summary_writer = tf.summary.FileWriter(os.path.join(log_root,"eval"))
    saver = tf.train.Saver()

    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            return
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', log_root)
            return
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        loss, accuracy, step = sess.run((model.loss, model.accuracy, model.global_step))

        loss_summary = tf.Summary()
        loss_summary.value.add(tag="Model/Loss", simple_value=loss)
        summary_writer.add_summary(loss_summary, step)

        accuracy_summary = tf.Summary()
        accuracy_summary.value.add(tag="Model/Accuracy", simple_value=accuracy)
        summary_writer.add_summary(accuracy_summary, step)


        summary_writer.flush()
        if step==num_steps:
            break
        time.sleep(2)



def main(_):

    if mode=="train":
        train(log_dir)
    else:
        eval(log_dir)


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


