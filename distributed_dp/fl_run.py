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
"""Runs federated training tasks."""

import functools
import math
import os
import pprint
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from distributed_dp import fl_utils
from utils import task_utils
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

# Hard-code the total number of clients for the datasets. Task names are defined
# in `utiis/task_utils`. We limit to a subset of the tasks.
_SUPPORTED_TASKS_NUM_CLIENTS = {
    'emnist_character': 3400,
    'stackoverflow_tag': 342477,
    'stackoverflow_word': 342477
}
_SUPPORTED_TASKS = list(_SUPPORTED_TASKS_NUM_CLIENTS.keys())

with utils_impl.record_hparam_flags() as optimizer_flags:
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters.
  flags.DEFINE_integer('clients_per_thread', 1, 'TFF config.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', None, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 100,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 42,
                       'Random seed for client sampling.')
  flags.DEFINE_integer(
      'max_elements_per_client', None, 'Maximum number of '
      'elements for each training client. If set to None, all '
      'available examples are used.')

  # Training loop configuration
  flags.DEFINE_integer('total_rounds', 1500, 'Number of total training rounds.')
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/ddp_fl/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer(
      'num_validation_examples', -1, 'The number of validation'
      'examples to use. If set to -1, all available examples '
      'are used.')
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
  flags.DEFINE_float('l2_norm_clip', 2.0, 'Initial L2 norm clip.')

  dp_mechanisms = ['gaussian', 'ddgauss']
  flags.DEFINE_enum('dp_mechanism', 'ddgauss', dp_mechanisms,
                    'Which DP mechanism to use.')

with utils_impl.record_hparam_flags() as task_flags:
  # Defines "--task" (options from `task_utils`) and "--<task>_<arg>" flags
  # aligned with input args at `tff.simulation.baselines.*` tasks.
  task_utils.define_task_flags()

FLAGS = flags.FLAGS


def create_if_not_exists(path):
  """Creates a directory if it does not already exist."""
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def write_hparams(params_dict):
  results_dir = os.path.join(FLAGS.root_output_dir, 'results',
                             FLAGS.experiment_name)
  hparam_file = os.path.join(results_dir, 'hparams.txt')
  create_if_not_exists(results_dir)
  with tf.io.gfile.GFile(hparam_file, 'w') as f:
    # Also add meta info.
    params_dict['_expname'] = FLAGS.experiment_name
    params_dict['_rootdir'] = FLAGS.root_output_dir
    pprint.pprint(params_dict, stream=f)


def create_1m_cnn_model(only_digits: bool = False, seed: Optional[int] = 0):
  """A CNN model with slightly under 2^20 (roughly 1 million) params.

  A simple CNN model for the EMNIST character recognition task that is very
  similar to the default recommended model from `create_conv_dropout_model`
  but has slightly under 2^20 parameters. This is useful if the downstream task
  involves randomized Hadamard transform, which requires the model weights /
  gradients / deltas concatednated as a single vector to be padded to the
  nearest power-of-2 dimensions.

  This model is used in https://arxiv.org/abs/2102.06387.

  When `only_digits=False`, the returned model has 1,018,174 trainable
  parameters. For `only_digits=True`, the last dense layer is slightly smaller.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    seed: A random seed governing the model initialization and layer randomness.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  initializer = tf.keras.initializers.GlorotUniform(seed=seed)

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1),
          kernel_initializer=initializer),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Conv2D(
          64,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          kernel_initializer=initializer),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          128, activation='relu', kernel_initializer=initializer),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=initializer),
  ])

  return model


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  compression_dict = utils_impl.lookup_flag_values(compression_flags)
  dp_dict = utils_impl.lookup_flag_values(dp_flags)

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      max_elements=FLAGS.max_elements_per_client)

  if FLAGS.task == 'emnist_character':
    # Since we use a custom model for EMNIST, we need to manually construct the
    # TFF datasets and the TFF `Task` object.
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        only_digits=False)
    eval_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=64, shuffle_buffer_size=1)  # No shuffling.

    emnist_preprocessing = tff.simulation.baselines.emnist.emnist_preprocessing
    train_preprocess_fn = emnist_preprocessing.create_preprocess_fn(
        train_client_spec, emnist_task='character_recognition')
    eval_preprocess_fn = emnist_preprocessing.create_preprocess_fn(
        eval_client_spec, emnist_task='character_recognition')

    task_datasets = tff.simulation.baselines.task_data.BaselineTaskDatasets(
        train_data=emnist_train,
        test_data=emnist_test,
        validation_data=None,
        train_preprocess_fn=train_preprocess_fn,
        eval_preprocess_fn=eval_preprocess_fn)

    def emnist_model_fn():
      return tff.learning.from_keras_model(
          keras_model=create_1m_cnn_model(),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          input_spec=task_datasets.element_type_structure,
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    task = tff.simulation.baselines.baseline_task.BaselineTask(
        task_datasets, emnist_model_fn)

  elif FLAGS.task in _SUPPORTED_TASKS:
    task = task_utils.create_task_from_flags(train_client_spec)
  else:
    raise ValueError(f'Unsupported task "{FLAGS.task}". Must be one of '
                     f'{_SUPPORTED_TASKS}.')

  model_trainable_variables = task.model_fn().trainable_variables

  # The aggregator encapsulates the DDP algorithm.
  aggregation_factory, params_dict = fl_utils.build_aggregator(
      compression_flags=compression_dict,
      dp_flags=dp_dict,
      num_clients=_SUPPORTED_TASKS_NUM_CLIENTS[FLAGS.task],
      num_clients_per_round=FLAGS.clients_per_round,
      num_rounds=FLAGS.total_rounds,
      client_template=model_trainable_variables)
  write_hparams(params_dict)

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn=task.model_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=tff.learning.ClientWeighting.UNIFORM,
      client_optimizer_fn=client_optimizer_fn,
      model_update_aggregation_factory=aggregation_factory,
      use_experimental_simulation_loop=True)

  train_data = task.datasets.train_data.preprocess(
      task.datasets.train_preprocess_fn)
  training_process = (
      tff.simulation.compose_dataset_computation_with_iterative_process(
          train_data.dataset_computation, iterative_process))

  client_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          train_data.client_ids, random_seed=FLAGS.client_datasets_random_seed),
      size=FLAGS.clients_per_round)

  program_state_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name)

  test_data = task.datasets.get_centralized_test_data()
  validation_data = test_data.take(FLAGS.num_validation_examples)
  federated_eval = tff.learning.build_federated_evaluation(task.model_fn)

  # TODO(b/210890827): Use a polymorphic computation if possible
  @tff.federated_computation(training_process.initialize.type_signature.result,
                             federated_eval.type_signature.parameter[1])
  def evaluation_fn(state, evaluation_data):
    return federated_eval(state.model, evaluation_data)

  evaluation_selection_fn = lambda _: [validation_data]
  state = tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=client_selection_fn,
      total_rounds=FLAGS.total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=FLAGS.rounds_per_eval,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=FLAGS.rounds_per_checkpoint,
      metrics_managers=metrics_managers)

  test_fn = training_utils.create_test_fn(task)
  test_metrics = test_fn(state.model)

  for metrics_manager in metrics_managers:
    metrics_manager.release(test_metrics, FLAGS.total_rounds + 1)


if __name__ == '__main__':
  app.run(main)
