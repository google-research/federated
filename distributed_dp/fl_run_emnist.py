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

import math
import os.path
import pprint
from typing import Callable, List, Tuple

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from distributed_dp import fl_utils
from optimization.tasks import emnist
from optimization.tasks import training_specs
from utils import utils_impl
from utils.optimizers import optimizer_utils

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
  optimizer_utils.define_optimizer_flags('server')
  optimizer_utils.define_optimizer_flags('client')

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


def _configure_managers() -> Tuple[tff.simulation.FileCheckpointManager,
                                   List[tff.simulation.MetricsManager]]:
  """Configures checkpoint and metrics managers from flags."""
  root_output_dir = FLAGS.root_output_dir
  experiment_name = FLAGS.experiment_name
  tf.io.gfile.makedirs(root_output_dir)

  checkpoint_dir = os.path.join(root_output_dir, 'checkpoints', experiment_name)
  tf.io.gfile.makedirs(checkpoint_dir)
  checkpoint_manager = tff.simulation.FileCheckpointManager(
      checkpoint_dir, step=FLAGS.rounds_per_checkpoint)

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  tf.io.gfile.makedirs(results_dir)
  csv_file = os.path.join(results_dir, 'experiment.metrics.csv')
  csv_manager = tff.simulation.CSVMetricsManager(csv_file)

  summary_dir = os.path.join(root_output_dir, 'logdir', experiment_name)
  tensorboard_manager = tff.simulation.TensorBoardManager(summary_dir)

  logging.info('Writing...')
  logging.info('    checkpoints to: %s', checkpoint_dir)
  logging.info('    CSV metrics to: %s', csv_file)
  logging.info('    TensorBoard summaries to: %s', summary_dir)

  return checkpoint_manager, [csv_manager, tensorboard_manager]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  compression_dict = utils_impl.lookup_flag_values(compression_flags)
  dp_dict = utils_impl.lookup_flag_values(dp_flags)

  def iterative_process_builder(
      model_fn: Callable[[], tff.learning.Model],
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`."""

    model_trainable_variables = model_fn().trainable_variables

    # Most logic for deciding what to run is here.
    aggregation_factory = fl_utils.build_aggregator(
        compression_flags=compression_dict,
        dp_flags=dp_dict,
        num_clients=3400,
        num_clients_per_round=FLAGS.clients_per_round,
        num_rounds=FLAGS.total_rounds,
        client_template=model_trainable_variables)

    return tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_weighting=tff.learning.ClientWeighting.UNIFORM,
        client_optimizer_fn=client_optimizer_fn,
        model_update_aggregation_factory=aggregation_factory)

  task_spec = training_specs.TaskSpec(
      iterative_process_builder=iterative_process_builder,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      client_batch_size=FLAGS.client_batch_size,
      clients_per_round=FLAGS.clients_per_round,
      client_datasets_random_seed=FLAGS.client_datasets_random_seed)

  runner_spec = emnist.configure_training(task_spec, FLAGS.model)

  def round_end_evaluation_fn(state, round_num):
    if round_num % FLAGS.rounds_per_eval == 0:
      validation_metrics = runner_spec.validation_fn(state, round_num)
    else:
      validation_metrics = {}
    return validation_metrics

  checkpoint_manager, metrics_managers = _configure_managers()

  state = tff.simulation.run_simulation(
      process=runner_spec.iterative_process,
      client_selection_fn=runner_spec.client_datasets_fn,
      total_rounds=FLAGS.total_rounds,
      validation_fn=round_end_evaluation_fn,
      file_checkpoint_manager=checkpoint_manager,
      metrics_managers=metrics_managers)

  test_metrics = runner_spec.test_fn(state)

  logging.info('Test metrics:\n %s', pprint.pformat(test_metrics))

  for metrics_manager in metrics_managers:
    metrics_manager.save_metrics(test_metrics, FLAGS.total_rounds + 1)


if __name__ == '__main__':
  app.run(main)
