# Copyright 2022, Google LLC.
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
"""Runs global training on EMNIST with varying levels of data paucity."""

import functools
from typing import Callable

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import pseudo_client_data
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

with utils_impl.record_hparam_flags() as training_flags:
  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/data_poor_fl/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 100, 'Number of total training rounds.')

  # Train client configuration
  flags.DEFINE_integer('clients_per_train_round', 10,
                       'How many clients to sample at each training round.')
  flags.DEFINE_integer('examples_per_pseudo_client', 100,
                       'Maximum number of examples per pseudo-client.')
  flags.DEFINE_integer(
      'train_epochs', 1,
      'Number of epochs performed by a client during a round of training.')
  flags.DEFINE_integer('train_batch_size', 10, 'Batch size on train clients.')

  # Training algorithm configuration
  flags.DEFINE_enum('train_algorithm', 'fedavg', ['fedavg', 'fedsgd'],
                    'Which training algorithm to use.')
  flags.DEFINE_bool(
      'example_weighting', True, 'Whether to use example weighting when '
      'aggregating client updates (True) or uniform weighting (False).')
  flags.DEFINE_bool('clipping', True, 'Whether to use adaptive clipping.')
  flags.DEFINE_bool('zeroing', True, 'Whether to use adaptive zeroing.')

  # Random seeds for reproducibility
  flags.DEFINE_integer(
      'base_random_seed', 0, 'An integer random seed governing'
      ' the randomness in the simulation.')

  # Debugging flags
  flags.DEFINE_bool(
      'use_aggregator_debug_measurements', True, 'Whether to compute debugging '
      'measurements for the model update aggregator.')
  flags.DEFINE_bool(
      'use_synthetic_data', False, 'Whether to use synthetic data. This should '
      'only be set to True for debugging purposes.')

with utils_impl.record_hparam_flags() as optimizer_flags:
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

FLAGS = flags.FLAGS

# Change constant to a flag if needs to be configured.
_ROUNDS_PER_EVALUATION = 10
_ROUNDS_PER_CHECKPOINT = 50
_EMNIST_MAX_ELEMENTS_PER_CLIENT = 418


def _validate_flags():
  """Validates that flag specifications are compatible."""
  if FLAGS.train_algorithm == 'fedsgd' and FLAGS.client_optimizer is not None:
    raise ValueError(
        'Found flag --client_optimizer={!s}, but --train_algorithm=fedsgd. '
        'If using fedsgd, please do not set --client_optimizer'.format(
            FLAGS.client_optimizer))


def _write_hparams():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(training_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  if FLAGS.train_algorithm != 'fedsgd':
    opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Write the updated hyperparameters to a file.
  training_utils.write_hparams_to_csv(hparam_dict, FLAGS.root_output_dir,
                                      FLAGS.experiment_name)


def _create_train_algorithm(
    model_fn: Callable[[], tff.learning.Model]
) -> tff.learning.templates.LearningProcess:
  """Creates a learning process for client training."""
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  model_aggregator = tff.learning.robust_aggregator(
      zeroing=FLAGS.zeroing,
      clipping=FLAGS.clipping,
      add_debug_measurements=FLAGS.use_aggregator_debug_measurements)

  if FLAGS.train_algorithm == 'fedsgd':
    return tff.learning.algorithms.build_fed_sgd(
        model_fn=model_fn,
        server_optimizer_fn=server_optimizer_fn,
        model_aggregator=model_aggregator)
  elif FLAGS.train_algorithm == 'fedavg':
    client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags(
        'client')
    if FLAGS.example_weighting:
      client_weighting = tff.learning.ClientWeighting.NUM_EXAMPLES
    else:
      client_weighting = tff.learning.ClientWeighting.UNIFORM
    return tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_weighting=client_weighting,
        model_aggregator=model_aggregator)
  else:
    raise ValueError(f'Found unexpected value {FLAGS.train_algorithm} for '
                     'FLAGS.train_algorithm')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))
  if not FLAGS.experiment_name:
    raise ValueError('FLAGS.experiment_name must be set.')

  # Configuring the base EMNIST task
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.train_epochs,
      batch_size=FLAGS.train_batch_size,
      shuffle_buffer_size=_EMNIST_MAX_ELEMENTS_PER_CLIENT)
  task = tff.simulation.baselines.emnist.create_character_recognition_task(
      train_client_spec,
      model_id='cnn',
      only_digits=False,
      use_synthetic_data=FLAGS.use_synthetic_data)
  train_preprocess_fn = task.datasets.train_preprocess_fn
  test_data = task.datasets.get_centralized_test_data()

  # Creating pseudo-clients
  base_train_data = task.datasets.train_data
  logging.info('Creating pseudo-clients. This can take some time, generally '
               'around 5 minutes.')
  extended_train_data = pseudo_client_data.create_pseudo_client_data(
      base_train_data,
      examples_per_pseudo_client=FLAGS.examples_per_pseudo_client)
  training_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          extended_train_data.client_ids, random_seed=FLAGS.base_random_seed),
      size=FLAGS.clients_per_train_round)

  # Creating the training process (and wiring in a dataset computation)
  @tff.tf_computation(tf.string)
  def build_train_dataset_from_client_id(client_id):
    raw_client_data = extended_train_data.dataset_computation(client_id)
    return train_preprocess_fn(raw_client_data)

  learning_process = _create_train_algorithm(task.model_fn)
  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, learning_process)
  training_process.get_model_weights = learning_process.get_model_weights

  # Defining eval artifacts
  federated_eval = tff.learning.build_federated_evaluation(task.model_fn)

  def evaluation_fn(state, evaluation_data):
    return federated_eval(
        training_process.get_model_weights(state), evaluation_data)

  # Configuring release managers and performing training/eval
  program_state_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name)
  _write_hparams()
  tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=training_selection_fn,
      total_rounds=FLAGS.total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=lambda round_num: [test_data],
      rounds_per_evaluation=_ROUNDS_PER_EVALUATION,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=_ROUNDS_PER_CHECKPOINT,
      metrics_managers=metrics_managers)

if __name__ == '__main__':
  app.run(main)
