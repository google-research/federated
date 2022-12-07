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

import asyncio
import functools

from absl import app
from absl import flags
import tensorflow_federated as tff

from adaptive_lr_decay import adaptive_fed_avg
from adaptive_lr_decay import callbacks
from utils import task_utils
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

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
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer(
      'max_elements_per_client', None, 'Maximum number of '
      'elements for each training client. If set to None, all '
      'available examples are used.')
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
  flags.DEFINE_integer(
      'num_validation_examples', -1, 'The number of validation'
      'examples to use. If set to -1, all available examples '
      'are used.')

with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

FLAGS = flags.FLAGS


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

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      max_elements=FLAGS.max_elements_per_client)
  task = task_utils.create_task_from_flags(train_client_spec)

  iterative_process = adaptive_fed_avg.build_fed_avg_process(
      task.model_fn,
      client_lr_callback,
      server_lr_callback,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn)
  train_data = task.datasets.train_data.preprocess(
      task.datasets.train_preprocess_fn)
  training_process = (
      tff.simulation.compose_dataset_computation_with_iterative_process(
          train_data.dataset_computation, iterative_process))

  training_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          train_data.client_ids, random_seed=FLAGS.client_datasets_random_seed),
      size=FLAGS.clients_per_round)

  test_data = task.datasets.get_centralized_test_data()
  validation_data = test_data.take(FLAGS.num_validation_examples)
  federated_eval = tff.learning.build_federated_evaluation(task.model_fn)
  evaluation_selection_fn = lambda round_num: [validation_data]

  def evaluation_fn(state, evaluation_data):
    return federated_eval(state.model, evaluation_data)

  program_state_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name)
  state = tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=training_selection_fn,
      total_rounds=FLAGS.total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=FLAGS.rounds_per_eval,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=FLAGS.rounds_per_checkpoint,
      metrics_managers=metrics_managers)

  loop = asyncio.get_event_loop()

  async def write_final_metrics(metrics, round_num):
    await asyncio.gather(*[
        manager.release(value=metrics, key=round_num)  # pytype: disable=missing-parameter
        for manager in metrics_managers
    ])

  test_metrics = federated_eval(state.model, [test_data])
  loop.run_until_complete(
      write_final_metrics(test_metrics, FLAGS.total_rounds + 1))


if __name__ == '__main__':
  app.run(main)
