# Copyright 2020, Google LLC.
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
"""Runs federated training on various tasks using a generalized form of FedAvg.

Specifically, we create (according to flags) an iterative processes that adapts
the client and server learning rate according to the history of loss values
encountered throughout training. For more details on the learning rate decay,
see `callbacks.py` and `adaptive_fed_avg.py`.
"""

import os.path
import pprint
from typing import Callable, List, Tuple

from absl import app
from absl import flags
from absl import logging
import tensorflow_federated as tff

from adaptive_lr_decay import adaptive_fed_avg
from adaptive_lr_decay import callbacks
from optimization.tasks import cifar100
from optimization.tasks import emnist
from optimization.tasks import emnist_ae
from optimization.tasks import shakespeare
from optimization.tasks import stackoverflow_nwp
from optimization.tasks import stackoverflow_tp
from optimization.tasks import training_specs
from utils import utils_impl
from utils.optimizers import optimizer_utils

_SUPPORTED_TASKS = [
    'cifar100', 'emnist_cr', 'emnist_ae', 'shakespeare', 'stackoverflow_nwp',
    'stackoverflow_lr'
]

with utils_impl.record_hparam_flags() as optimizer_flags:
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as callback_flags:
  flags.DEFINE_float(
      'client_decay_factor', 0.1, 'Amount to decay the client learning rate '
      'upon reaching a plateau.')
  flags.DEFINE_float(
      'server_decay_factor', 0.9, 'Amount to decay the server learning rate '
      'upon reaching a plateau.')
  flags.DEFINE_float(
      'min_delta', 1e-4, 'Minimum delta for improvement in the learning rate '
      'callbacks.')
  flags.DEFINE_integer(
      'window_size', 100, 'Number of rounds to take a moving average over when '
      'estimating the training loss in learning rate callbacks.')
  flags.DEFINE_integer(
      'patience', 100, 'Number of rounds of non-improvement before decaying the'
      'learning rate.')
  flags.DEFINE_float('min_lr', 0.0, 'The minimum learning rate.')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', -1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/adaptive_lr_decay/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as task_flags:
  # Task specification
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

  # CIFAR-100 flags
  flags.DEFINE_integer('cifar100_crop_size', 24, 'The height and width of '
                       'images after preprocessing.')

  # EMNIST CR flags
  flags.DEFINE_enum(
      'emnist_cr_model', 'cnn', ['cnn', '2nn'], 'Which model to '
      'use. This can be a convolutional model (cnn) or a two '
      'hidden-layer densely connected network (2nn).')

  # Shakespeare flags
  flags.DEFINE_integer(
      'shakespeare_sequence_length', 80,
      'Length of character sequences to use for the RNN model.')

  # Stack Overflow NWP flags
  flags.DEFINE_integer('so_nwp_vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('so_nwp_num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('so_nwp_sequence_length', 20,
                       'Max sequence length to use.')
  flags.DEFINE_integer('so_nwp_max_elements_per_user', 1000, 'Max number of '
                       'training sentences to use per user.')
  flags.DEFINE_integer(
      'so_nwp_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')

  # Stack Overflow LR flags
  flags.DEFINE_integer('so_tp_vocab_tokens_size', 10000,
                       'Vocab tokens size used.')
  flags.DEFINE_integer('so_tp_vocab_tags_size', 500, 'Vocab tags size used.')
  flags.DEFINE_integer(
      'so_tp_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')
  flags.DEFINE_integer('so_tp_max_elements_per_user', 1000,
                       'Max number of training '
                       'sentences to use per user.')

FLAGS = flags.FLAGS


def _configure_managers() -> Tuple[tff.simulation.FileCheckpointManager,
                                   List[tff.simulation.MetricsManager]]:
  """Configures checkpoint and metrics managers from flags."""
  root_output_dir = FLAGS.root_output_dir
  experiment_name = FLAGS.experiment_name
  utils_impl.create_directory_if_not_exists(root_output_dir)

  checkpoint_dir = os.path.join(root_output_dir, 'checkpoints', experiment_name)
  utils_impl.create_directory_if_not_exists(checkpoint_dir)
  checkpoint_manager = tff.simulation.FileCheckpointManager(
      checkpoint_dir, step=FLAGS.rounds_per_checkpoint)

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  utils_impl.create_directory_if_not_exists(results_dir)
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

  client_lr_callback = callbacks.create_reduce_lr_on_plateau(
      learning_rate=FLAGS.client_learning_rate,
      decay_factor=FLAGS.client_decay_factor,
      min_delta=FLAGS.min_delta,
      min_lr=FLAGS.min_lr,
      window_size=FLAGS.window_size,
      patience=FLAGS.patience)

  server_lr_callback = callbacks.create_reduce_lr_on_plateau(
      learning_rate=FLAGS.server_learning_rate,
      decay_factor=FLAGS.server_decay_factor,
      min_delta=FLAGS.min_delta,
      min_lr=FLAGS.min_lr,
      window_size=FLAGS.window_size,
      patience=FLAGS.patience)

  def iterative_process_builder(
      model_fn: Callable[[], tff.learning.Model],
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    return adaptive_fed_avg.build_fed_avg_process(
        model_fn,
        client_lr_callback,
        server_lr_callback,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn)

  task_spec = training_specs.TaskSpec(
      iterative_process_builder=iterative_process_builder,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      client_batch_size=FLAGS.client_batch_size,
      clients_per_round=FLAGS.clients_per_round,
      client_datasets_random_seed=FLAGS.client_datasets_random_seed)

  if FLAGS.task == 'cifar100':
    runner_spec = cifar100.configure_training(
        task_spec, crop_size=FLAGS.cifar100_crop_size)
  elif FLAGS.task == 'emnist_cr':
    runner_spec = emnist.configure_training(
        task_spec, model=FLAGS.emnist_cr_model)
  elif FLAGS.task == 'emnist_ae':
    runner_spec = emnist_ae.configure_training(task_spec)
  elif FLAGS.task == 'shakespeare':
    runner_spec = shakespeare.configure_training(
        task_spec, sequence_length=FLAGS.shakespeare_sequence_length)
  elif FLAGS.task == 'stackoverflow_nwp':
    runner_spec = stackoverflow_nwp.configure_training(
        task_spec,
        vocab_size=FLAGS.so_nwp_vocab_size,
        num_oov_buckets=FLAGS.so_nwp_num_oov_buckets,
        sequence_length=FLAGS.so_nwp_sequence_length,
        max_elements_per_user=FLAGS.so_nwp_max_elements_per_user,
        num_validation_examples=FLAGS.so_nwp_num_validation_examples)
  elif FLAGS.task == 'stackoverflow_tp':
    runner_spec = stackoverflow_tp.configure_training(
        task_spec,
        vocab_tokens_size=FLAGS.so_tp_vocab_tokens_size,
        vocab_tags_size=FLAGS.so_tp_vocab_tags_size,
        max_elements_per_user=FLAGS.so_tp_max_elements_per_user,
        num_validation_examples=FLAGS.so_tp_num_validation_examples)
  else:
    raise ValueError(
        '--task flag {} is not supported, must be one of {}.'.format(
            FLAGS.task, _SUPPORTED_TASKS))

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
