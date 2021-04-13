# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains and evaluates an EMNIST classification model."""

import functools
import math

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from distributed_discrete_gaussian import fl_utils
from utils import training_loop
from utils import utils_impl
from utils.datasets import emnist_dataset
from utils.models import emnist_models

with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_enum('model', '1m_cnn', ['cnn', '2nn', '1m_cnn'],
                    'Which model to use.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 100, 'Number of clients per round.')
  flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of client (inner optimizer) epochs per federated round.')
  flags.DEFINE_boolean(
      'uniform_weighting', True,
      'Whether to weigh clients uniformly. If false, clients '
      'are weighted by the number of samples.')
  flags.DEFINE_boolean('only_digits', False,
                       'Whether to use only digits for the EMNIST dataset.')
  flags.DEFINE_integer('client_datasets_random_seed', 42,
                       'Random seed for client sampling.')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server')
  utils_impl.define_optimizer_flags('client')

with utils_impl.record_new_flags() as training_loop_flags:
  flags.DEFINE_integer('total_rounds', 1500, 'Number of total training rounds.')
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/ddg_emnist/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as compression_flags:
  flags.DEFINE_integer('num_bits', 16, 'Number of bits for quantization.')
  flags.DEFINE_float('beta', math.exp(-0.5), 'Beta for stochastic rounding.')
  flags.DEFINE_integer('k_stddevs', 4,
                       'Number of stddevs to bound the signal range.')

with utils_impl.record_hparam_flags() as dp_flags:
  flags.DEFINE_float(
      'epsilon', 10.0, 'Epsilon for the DP mechanism. '
      'No DP used if this is None.')
  flags.DEFINE_float('delta', None, 'Delta for the DP mechanism. ')
  flags.DEFINE_float('l2_norm_clip', 0.03, 'Initial L2 norm clip.')

  dp_mechanisms = ['gaussian', 'ddgauss']
  flags.DEFINE_enum('dp_mechanism', 'ddgauss', dp_mechanisms,
                    'Which DP mechanism to use.')

FLAGS = flags.FLAGS


def create_1m_cnn_model(only_digits=True):
  """A ConvNet slightly smaller than emnist_models.create_conv_dropout_model.

  A model for the EMNIST character recognition task that is slightly under 2^20
  parameters so we don't pad too much zeros after randomized Hadamard transform.

  Args:
    only_digits: Model uses 10 output units if set to True; otherwise use 62.

  Returns:
    A Keras model for the EMNIST task.
  """
  data_format = 'channels_last'

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Conv2D(
          64, kernel_size=(3, 3), activation='relu', data_format=data_format),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          10 if only_digits else 62, activation=tf.nn.softmax),
  ])

  return model


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  emnist_task = 'digit_recognition'
  emnist_train, _ = tff.simulation.datasets.emnist.load_data(only_digits=False)
  _, emnist_test = emnist_dataset.get_centralized_datasets(
      only_digits=False, emnist_task=emnist_task)

  train_preprocess_fn = emnist_dataset.create_preprocess_fn(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      emnist_task=emnist_task)

  input_spec = train_preprocess_fn.type_signature.result.element

  if FLAGS.model == 'cnn':
    model_builder = functools.partial(
        emnist_models.create_conv_dropout_model, only_digits=FLAGS.only_digits)
  elif FLAGS.model == '2nn':
    model_builder = functools.partial(
        emnist_models.create_two_hidden_layer_model,
        only_digits=FLAGS.only_digits)
  elif FLAGS.model == '1m_cnn':
    model_builder = functools.partial(
        create_1m_cnn_model, only_digits=FLAGS.only_digits)
  else:
    raise ValueError('Cannot handle model flag [{!s}].'.format(FLAGS.model))

  logging.info('Training model:')
  logging.info(model_builder().summary())

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  compression_dict = utils_impl.lookup_flag_values(compression_flags)
  dp_dict = utils_impl.lookup_flag_values(dp_flags)

  # Most logic for deciding what baseline to run is here.
  aggregation_factory = fl_utils.build_aggregator(
      compression_flags=compression_dict,
      dp_flags=dp_dict,
      num_clients=len(emnist_train.client_ids),
      num_clients_per_round=FLAGS.clients_per_round,
      num_rounds=FLAGS.total_rounds,
      client_template=model_builder().trainable_variables)

  def tff_model_fn():
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        loss=loss_builder(),
        input_spec=input_spec,
        metrics=metrics_builder())

  server_optimizer_fn = lambda: utils_impl.create_optimizer_from_flags('server')
  client_optimizer_fn = lambda: utils_impl.create_optimizer_from_flags('client')

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn=tff_model_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=tff.learning.ClientWeighting.UNIFORM,
      client_optimizer_fn=client_optimizer_fn,
      model_update_aggregation_factory=aggregation_factory)

  @tff.tf_computation(tf.string)
  def build_train_dataset_from_client_id(client_id):
    client_dataset = emnist_train.dataset_computation(client_id)
    return train_preprocess_fn(client_dataset)

  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, iterative_process)
  training_process.get_model_weights = iterative_process.get_model_weights

  client_ids_fn = tff.simulation.build_uniform_sampling_fn(
      emnist_train.client_ids,
      size=FLAGS.clients_per_round,
      replace=False,
      random_seed=FLAGS.client_datasets_random_seed)

  # We convert the output to a list (instead of an np.ndarray) so that it can
  # be used as input to the iterative process.
  client_sampling_fn = lambda x: list(client_ids_fn(x))

  evaluate_fn = tff.learning.build_federated_evaluation(tff_model_fn)

  def test_fn(state):
    return evaluate_fn(
        iterative_process.get_model_weights(state), [emnist_test])

  def validation_fn(state, round_num):
    del round_num
    return evaluate_fn(
        iterative_process.get_model_weights(state), [emnist_test])

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_sampling_fn,
      validation_fn=validation_fn,
      test_fn=test_fn,
      total_rounds=FLAGS.total_rounds,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)


if __name__ == '__main__':
  app.run(main)
