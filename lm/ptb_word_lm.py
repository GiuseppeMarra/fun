# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import select
import sys
import os
import reader
import util

from tensorflow.python.client import device_lib

def heardEnter():

    ''' Listen for the user pressing ENTER '''

    i,o,e = select.select([sys.stdin],[],[],0.0001)

    for s in i:

        if s == sys.stdin:
            input = sys.stdin.readline()
            return True

    return False

data_path = "data/simple-examples/data/"

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph_lstm(inputs, config, is_training)

    softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(config.learning_rate, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op_gd = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())


    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)


    self._train_op_adam = tf.train.AdamOptimizer(0.001).minimize(self._cost)



  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    outputs, state = tf.nn.static_rnn(cell, inputs,
                                      initial_state=self._initial_state)

    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  # def export_ops(self, name):
  #   """Exports ops to collections."""
  #   self._name = name
  #   ops = {util.with_prefix(self._name, "cost"): self._cost}
  #   if self._is_training:
  #     ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
  #     if self._rnn_params:
  #       ops.update(rnn_params=self._rnn_params)
  #   for name, op in ops.items():
  #     tf.add_to_collection(name, op)
  #   self._initial_state_name = util.with_prefix(self._name, "initial")
  #   self._final_state_name = util.with_prefix(self._name, "final")
  #   util.export_state_tuples(self._initial_state, self._initial_state_name)
  #   util.export_state_tuples(self._final_state, self._final_state_name)

  # def import_ops(self):
  #   """Imports ops from collections."""
  #   if self._is_training:
  #     self._train_op = tf.get_collection_ref("train_op")[0]
  #     self._lr = tf.get_collection_ref("lr")[0]
  #     self._new_lr = tf.get_collection_ref("new_lr")[0]
  #     self._lr_update = tf.get_collection_ref("lr_update")[0]
  #     rnn_params = tf.get_collection_ref("rnn_params")
  #     if self._cell and rnn_params:
  #       params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
  #           self._cell,
  #           self._cell.params_to_canonical,
  #           self._cell.canonical_to_params,
  #           rnn_params,
  #           base_variable_scope="Model/RNN")
  #       tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
  #   self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
  #   num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
  #   self._initial_state = util.import_state_tuples(
  #       self._initial_state, self._initial_state_name, num_replicas)
  #   self._final_state = util.import_state_tuples(
  #       self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

class Config(object):
  def __init__(self):
    self.init_scale = 0.05
    self.learning_rate = 0.001
    self.max_grad_norm = 10
    self.num_layers = 2
    self.num_steps = 35
    self.hidden_size = 300
    self.max_epoch = 6
    self.max_max_epoch = 50
    self.keep_prob = 0.5
    self.lr_decay = 1.
    self.batch_size = 64
    self.vocab_size = 10000

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class Config(object):
  def __init__(self):
    self.init_scale = 0.05
    self.learning_rate = 0.001
    self.max_grad_norm = 10
    self.num_layers = 2
    self.num_steps = 35
    self.hidden_size = 300
    self.max_epoch = 6
    self.max_max_epoch = 30
    self.keep_prob = 0.5
    self.lr_decay = 1.
    self.batch_size = 64
    self.vocab_size = 10000

    self.lambda_alphas = 0.


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, st in enumerate(model.initial_state):
      feed_dict[st] = state[i]

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))

  return np.exp(costs / iters)



def train(config, save_path):
  os.mkdir(save_path)
  tf.reset_default_graph()
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data


  initializer = tf.random_uniform_initializer(-config.init_scale,
                                              config.init_scale)

  with tf.name_scope("Train"):
    train_input = PTBInput(config=config, data=train_data, name="TrainInput")
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config, input_=train_input)
    tf.summary.scalar("Training Loss", m.cost)
    tf.summary.scalar("Learning Rate", m.lr)

  with tf.name_scope("Valid"):
    valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
    tf.summary.scalar("Validation Loss", mvalid.cost)

  with tf.name_scope("Test"):
    test_input = PTBInput(
        config=config, data=test_data, name="TestInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mtest = PTBModel(is_training=False, config=config,
                       input_=test_input)


  min_val_perplexity = None
  saver = tf.train.Saver()
  f = open(os.path.join(save_path, "log.txt"), "w")

  coord = tf.train.Coordinator()
  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=session, coord=coord)

    for i in range(config.max_max_epoch):

      # lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
      # m.assign_lr(session, config.learning_rate * config.lr_decay)

      t_op  = m._train_op_adam #if i < 25 else m._train_op_gd

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      f.write("Epoch: %d Learning rate: %.3f \n" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, eval_op=t_op,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      f.write("Epoch: %d Train Perplexity: %.3f \n" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid)
      if min_val_perplexity is None:
        min_val_perplexity = valid_perplexity
        saver.save(session, save_path=os.path.join(save_path, "model.ckpt"))

      if valid_perplexity <min_val_perplexity:
        min_val_perplexity = valid_perplexity
        saver.save(session, save_path=os.path.join(save_path, "model.ckpt"))
        print("Saving best validation model")
        f.write("Saving best validation model\n")

      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
      f.write("Epoch: %d Valid Perplexity: %.3f \n" % (i + 1, valid_perplexity))
      f.flush()
      os.fsync(f.fileno())

      if heardEnter():
        break

    print("Restoring best validation model")
    f.write("Restoring best validation model\n")
    saver.restore(session, tf.train.latest_checkpoint(save_path))
    test_perplexity = run_epoch(session, mtest)
    print("Test Perplexity: %.3f" % test_perplexity)
    f.write("Test Perplexity: %.3f" % test_perplexity)

    coord.request_stop()
    coord.join()
    return train_perplexity, valid_perplexity, test_perplexity

def config_generator(config, dict):

  def _recursive_call(items, attr_id):
    attr_name = items[attr_id][0]
    attr_values = items[attr_id][1]
    for attr_value in attr_values:
      setattr(config,attr_name, attr_value)
      if attr_id==(len(items)-1): # base case[
          yield config
      else:
        for i in _recursive_call(items, attr_id+1):
          yield i


  items = dict.items()
  for i in _recursive_call(items, 0):
    yield i


if __name__ == "__main__":

  save_path = "savings"

  with open(os.path.join(save_path,"results.csv"), "w") as f:

    dict = {
      "num_layers" : [1, 2],
      "hidden_size" : [200, 600, 1000],
      "keep_prob" : [0.35, 0.5, 0.8]
    }
    config = Config()
    f.write(",".join([str(attr[0]) for attr in vars(config).items()])) # csv header
    f.write(",train_acc,valid_acc,test_acc")
    f.write("\n")

    for config in config_generator(config, dict):
      name = "run_"+str(config.num_layers) + "_" + str(config.hidden_size) +  "_" + str(config.keep_prob).replace(".", "") + "_" + str(config.batch_size)
      tr, vl, ts = train(config, os.path.join(save_path, name))
      f.write(",".join([str(attr[1]) for attr in vars(config).items()]))
      f.write(",%f,%f,%f" % (tr, vl, ts))
      f.write("\n")
      f.flush()
      os.fsync(f.fileno())