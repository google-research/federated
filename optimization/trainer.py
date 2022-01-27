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

Specifically, we create (according to flags) an iterative processes that allows
for client learning rate schedules, as well as various client and server
optimization methods.
"""

import functools

from absl import app
from absl import flags
import tensorflow_federated as tff

from utils import task_utils
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils


with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')
  optimizer_utils.define_lr_schedule_flags('client')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')
  flags.DEFINE_integer(
      'max_elements_per_client', None, 'Maximum number of '
      'elements for each training client. If set to None, all '
      'available examples are used.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer(
      'num_validation_examples', -1, 'The number of validation'
      'examples to use. If set to -1, all available examples '
      'are used.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

  flags.DEFINE_bool(
      'use_synthetic_data', False, 'If set to True, this uses '
      'synthetic data, and is only suitable for debugging.')

with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

FLAGS = flags.FLAGS


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task flags
  task_flag_dict = utils_impl.lookup_flag_values(task_flags)
  hparam_dict.update(task_flag_dict)
  training_utils.write_hparams_to_csv(hparam_dict, FLAGS.root_output_dir,
                                      FLAGS.experiment_name)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  client_learning_rate_fn = optimizer_utils.create_lr_schedule_from_flags(
      'client')

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      max_elements=FLAGS.max_elements_per_client)
  task = task_utils.create_task_from_flags(
      train_client_spec, use_synthetic_data=FLAGS.use_synthetic_data)
  learning_process = tff.learning.algorithms.build_weighted_fed_avg_with_optimizer_schedule(
      model_fn=task.model_fn,
      client_learning_rate_fn=client_learning_rate_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn)
  train_data = task.datasets.train_data.preprocess(
      task.datasets.train_preprocess_fn)
  training_process = (
      tff.simulation.compose_dataset_computation_with_iterative_process(
          train_data.dataset_computation, learning_process))
  training_process.get_model_weights = learning_process.get_model_weights

  training_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          train_data.client_ids, random_seed=FLAGS.client_datasets_random_seed),
      size=FLAGS.clients_per_round)

  test_data = task.datasets.get_centralized_test_data()
  validation_data = test_data.take(FLAGS.num_validation_examples)
  federated_eval = tff.learning.build_federated_evaluation(task.model_fn)
  evaluation_selection_fn = lambda round_num: [validation_data]

  def evaluation_fn(state, evaluation_data):
    return federated_eval(
        training_process.get_model_weights(state), evaluation_data)

  program_state_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name)
  _write_hparam_flags()
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

  test_metrics = evaluation_fn(state, [test_data])
  for metrics_manager in metrics_managers:
    metrics_manager.release(test_metrics, FLAGS.total_rounds + 1)


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment_name')
  flags.mark_flag_as_required('task')
  app.run(main)
