# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains and evaluates EMNIST."""

import collections
import functools
from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from dp_ftrl import dp_fedavg
from dp_ftrl import optimizer_utils
from dp_ftrl import training_loop

TEST_BATCH_SIZE = 1024

IRRELEVANT_FLAGS = frozenset(iter(flags.FLAGS))

flags.DEFINE_string(
    'experiment_name', 'emnist', 'The name of this experiment. Will be'
    'append to  --root_output_dir to separate experiment results.')
flags.DEFINE_string('root_output_dir', '/tmp/dpftrl/emnist',
                    'Root directory for writing experiment output.')
flags.DEFINE_integer('rounds_per_checkpoint', 100,
                     'How often to checkpoint the global model.')
flags.DEFINE_integer(
    'rounds_per_eval', 20,
    'How often to evaluate the global model on the validation dataset.')
flags.DEFINE_integer('clients_per_thread', 1, 'TFF executor configuration.')

# Training
flags.DEFINE_integer('clients_per_round', 100,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('client_batch_size', 16, 'Batch size used on the client.')
flags.DEFINE_integer('total_rounds', 10, 'Number of total training rounds.')
flags.DEFINE_integer(
    'total_epochs', None,
    'If not None, use shuffling of clients instead of random sampling.')
flags.DEFINE_enum('client_optimizer', 'sgd', ['sgd'], 'Client optimzier')
flags.DEFINE_enum('server_optimizer', 'sgd', [
    'sgd', 'ftrlprox', 'dpftrlprox', 'dpftrl', 'dpsgd', 'dpsgdm', 'dpftrlm',
    'dpftrlproxd'
], 'Server optimizer')
flags.DEFINE_float('client_lr', 0.02, 'Client learning rate.')
flags.DEFINE_float('server_lr', 1.0, 'Server learning rate.')
# optimizer specific
flags.DEFINE_float('server_momentum', 0.9, 'Server momentum.')
flags.DEFINE_float('decay_rate', 0.5,
                   'Power decay rate for proximal terms in FTRL.')

# Differential privacy
flags.DEFINE_float('clip_norm', 1.0, 'Clip L2 norm.')
flags.DEFINE_float('noise_multiplier', 0.01,
                   'Noise multiplier for DP algorithm.')

# EMNIST
flags.DEFINE_boolean('only_digits', True,
                     'If True, use the 10 digits version of EMNIST.')

HPARAM_FLAGS = [f for f in flags.FLAGS if f not in IRRELEVANT_FLAGS]
FLAGS = flags.FLAGS


def _get_emnist_dataset(
    only_digits: bool,
    client_epochs_per_round: int,
    client_batch_size: int,
):
  """Loads and preprocesses the EMNIST dataset.

  Args:
    only_digits: If True, load EMNIST with 10 digits. If False, load EMNIST with
      62 characters.
    client_epochs_per_round: client local epochs for training.
    client_batch_size: client batch size for training.

  Returns:
    A `(emnist_train, emnist_test)` tuple where `emnist_train` is a
    `tff.simulation.datasets.ClientData` object representing the training data
    and `emnist_test` is a single `tf.data.Dataset` representing the test data
    of all clients.
  """
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=only_digits)

  def element_fn(element):
    return collections.OrderedDict(
        x=tf.expand_dims(element['pixels'], -1), y=element['label'])

  def preprocess_train_dataset(dataset):
    # Use buffer_size same as the maximum client dataset size,
    # 418 for Federated EMNIST
    return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
        count=client_epochs_per_round).batch(
            client_batch_size, drop_remainder=False)

  def preprocess_test_dataset(dataset):
    return dataset.map(element_fn).batch(TEST_BATCH_SIZE, drop_remainder=False)

  emnist_train = emnist_train.preprocess(preprocess_train_dataset)
  emnist_test = preprocess_test_dataset(
      emnist_test.create_tf_dataset_from_all_clients())
  return emnist_train, emnist_test


def _server_optimizer_fn(model_weights, name, learning_rate, noise_std):
  """Returns server optimizer."""
  model_weight_shape = tf.nest.map_structure(tf.shape, model_weights)
  if name == 'sgd':
    return optimizer_utils.SGDServerOptimizer(learning_rate)
  elif name == 'sgdm':
    return optimizer_utils.DPSGDMServerOptimizer(
        learning_rate,
        momentum=FLAGS.server_momentum,
        noise_std=0,
        model_weight_shape=model_weight_shape)
  elif name == 'dpftrl':
    return optimizer_utils.DPFTRLMServerOptimizer(
        learning_rate,
        momentum=0,
        noise_std=noise_std,
        model_weight_shape=model_weight_shape)
  elif name == 'dpsgd':
    return optimizer_utils.DPSGDMServerOptimizer(
        learning_rate,
        momentum=0,
        noise_std=noise_std,
        model_weight_shape=model_weight_shape)
  elif name == 'dpsgdm':
    return optimizer_utils.DPSGDMServerOptimizer(
        learning_rate,
        momentum=FLAGS.server_momentum,
        noise_std=noise_std,
        model_weight_shape=model_weight_shape)
  elif name == 'dpftrlm':
    return optimizer_utils.DPFTRLMServerOptimizer(
        learning_rate,
        momentum=FLAGS.server_momentum,
        noise_std=noise_std,
        model_weight_shape=model_weight_shape)
  else:
    raise ValueError('Unknown server optimizer name {}'.format(name))


def _client_optimizer_fn(name, learning_rate):
  if name == 'sgd':
    return tf.keras.optimizers.SGD(learning_rate)
  else:
    raise ValueError('Unknown client optimizer name {}'.format(name))


def _create_original_fedavg_cnn_model(only_digits):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  This function is duplicated from research/optimization/emnist/models.py to
  make this example completely stand-alone.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10 if only_digits else 62),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model


def _get_client_datasets_fn(train_data):
  """Returns function for client datasets per round."""
  if FLAGS.total_epochs is None:

    def client_datasets_fn(round_num: int, epoch: int):
      del round_num
      sampled_clients = np.random.choice(
          train_data.client_ids, size=FLAGS.clients_per_round, replace=False)
      return [
          train_data.create_tf_dataset_for_client(client)
          for client in sampled_clients
      ], epoch

    logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
  else:
    client_shuffer = training_loop.ClientIDShuffler(FLAGS.clients_per_round,
                                                    train_data)

    def client_datasets_fn(round_num: int, epoch: int):
      sampled_clients, epoch = client_shuffer.sample_client_ids(
          round_num, epoch)
      return [
          train_data.create_tf_dataset_for_client(client)
          for client in sampled_clients
      ], epoch

    logging.info('Shuffle clients for max %d epochs and %d rounds',
                 FLAGS.total_epochs, FLAGS.total_rounds)
  return client_datasets_fn


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_devices = tf.config.list_logical_devices('GPU')
  server_device = tf.config.list_logical_devices('CPU')[0]
  tff.backends.native.set_local_execution_context(
      max_fanout=2 * FLAGS.clients_per_round,
      server_tf_device=server_device,
      client_tf_devices=client_devices,
      clients_per_thread=FLAGS.clients_per_thread)

  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)

  train_data, test_data = _get_emnist_dataset(
      FLAGS.only_digits,
      FLAGS.client_epochs_per_round,
      FLAGS.client_batch_size,
  )

  def tff_model_fn():
    keras_model = _create_original_fedavg_cnn_model(FLAGS.only_digits)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    return dp_fedavg.KerasModelWrapper(keras_model, test_data.element_spec,
                                       loss)

  noise_std = FLAGS.clip_norm * FLAGS.noise_multiplier / float(
      FLAGS.clients_per_round)
  server_optimizer_fn = functools.partial(
      _server_optimizer_fn,
      name=FLAGS.server_optimizer,
      learning_rate=FLAGS.server_lr,
      noise_std=noise_std)
  client_optimizer_fn = functools.partial(
      _client_optimizer_fn,
      name=FLAGS.client_optimizer,
      learning_rate=FLAGS.client_lr)
  iterative_process = dp_fedavg.build_federated_averaging_process(
      tff_model_fn,
      dp_clip_norm=FLAGS.clip_norm,
      server_optimizer_fn=server_optimizer_fn,
      client_optimizer_fn=client_optimizer_fn)

  keras_metics = [tf.keras.metrics.SparseCategoricalAccuracy()]
  model = tff_model_fn()

  def evaluate_fn(model_weights, dataset):
    model.from_weights(model_weights)
    metrics = dp_fedavg.keras_evaluate(model.keras_model, dataset, keras_metics)
    return collections.OrderedDict(
        (metric.name, metric.result().numpy()) for metric in metrics)

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in HPARAM_FLAGS
  ])
  total_epochs = 0 if FLAGS.total_epochs is None else FLAGS.total_epochs
  training_loop.run(
      iterative_process,
      client_datasets_fn=_get_client_datasets_fn(train_data),
      validation_fn=functools.partial(evaluate_fn, dataset=test_data),
      total_rounds=FLAGS.total_rounds,
      total_epochs=total_epochs,
      experiment_name=FLAGS.experiment_name,
      train_eval_fn=None,
      test_fn=functools.partial(evaluate_fn, dataset=test_data),
      root_output_dir=FLAGS.root_output_dir,
      hparam_dict=hparam_dict,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
      rounds_per_train_eval=2000)


if __name__ == '__main__':
  app.run(main)
