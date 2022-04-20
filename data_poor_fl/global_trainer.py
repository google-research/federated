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
"""Runs global training/evaluation via HypCluster."""

import functools
from typing import Callable

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import hypcluster_eval
from data_poor_fl import hypcluster_train
from data_poor_fl import optimizer_flag_utils
from data_poor_fl.pseudo_client_tasks import emnist_pseudo_client
from utils import task_utils
from utils import training_utils
from utils import utils_impl

with utils_impl.record_hparam_flags() as training_flags:
  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/data_poor_fl/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 100, 'Number of total training rounds.')

  # HypCluster configuration
  flags.DEFINE_integer('num_clusters', 1, 'How many clusters to use.')

  # Train configuration
  flags.DEFINE_integer('clients_per_train_round', 10,
                       'How many clients to sample at each training round.')
  flags.DEFINE_integer(
      'train_epochs', 1,
      'Number of epochs performed by a client during a round of training.')
  flags.DEFINE_integer('train_batch_size', 10, 'Batch size on train clients.')
  flags.DEFINE_integer(
      'max_train_elements', None, 'Max number of examples used by each client '
      'when training. If set to `None`, all examples are used. This should only'
      ' be set to a positive value for Stack Overflow tasks.')

  # Pseudo-client configuration
  flags.DEFINE_bool('use_pseudo_clients', False, 'Whether to split the data '
                    'into pseudo-clients.')
  flags.DEFINE_integer(
      'examples_per_pseudo_client', None,
      'Maximum number of examples per pseudo-client. This '
      'should only be set if the use_pseudo_clients flag is '
      'set to True.')

  # Evaluation configuration
  flags.DEFINE_integer('clients_per_eval_round', None,
                       'How many clients to sample at each evaluation round.')
  flags.DEFINE_integer('eval_batch_size', 100, 'Batch size for evaluation.')

  # Random seeds for reproducibility
  flags.DEFINE_integer(
      'base_random_seed', 0, 'An integer random seed governing'
      ' the randomness in the simulation.')

  # Debugging flags
  flags.DEFINE_bool(
      'use_synthetic_data', False, 'Whether to use synthetic data. This should '
      'only be set to True for debugging purposes.')

with utils_impl.record_hparam_flags() as optimizer_flags:
  optimizer_flag_utils.define_optimizer_flags('client')
  optimizer_flag_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

FLAGS = flags.FLAGS

# Change constant to a flag if needs to be configured.
_ROUNDS_PER_EVALUATION = 50
_ROUNDS_PER_CHECKPOINT = 50


def _write_hparams():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(training_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_flag_utils.remove_unused_flags(
      'client', opt_flag_dict)
  opt_flag_dict = optimizer_flag_utils.remove_unused_flags(
      'server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task flags
  task_flag_dict = utils_impl.lookup_flag_values(task_flags)
  hparam_dict.update(task_flag_dict)

  # Write the updated hyperparameters to a file.
  training_utils.write_hparams_to_csv(hparam_dict, FLAGS.root_output_dir,
                                      FLAGS.experiment_name)


def _create_train_computation(
    model_fn: Callable[[], tff.learning.Model],
    num_clusters: int = 1) -> tff.learning.templates.LearningProcess:
  """Creates a learning process for client training."""
  client_optimizer = optimizer_flag_utils.create_optimizer_from_flags('client')
  server_optimizer = optimizer_flag_utils.create_optimizer_from_flags('server')
  model_aggregator = tff.aggregators.MeanFactory(no_nan_division=True)

  return hypcluster_train.build_hypcluster_train(
      model_fn=model_fn,
      num_clusters=num_clusters,
      client_optimizer=client_optimizer,
      server_optimizer=server_optimizer,
      model_aggregator=model_aggregator)


def _create_task() -> tff.simulation.baselines.BaselineTask:
  """Creates a task for performing federated training and evaluation."""
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.train_epochs,
      batch_size=FLAGS.train_batch_size,
      max_elements=FLAGS.max_train_elements)
  eval_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=1, batch_size=FLAGS.eval_batch_size)

  task = task_utils.create_task_from_flags(
      train_client_spec,
      eval_client_spec=eval_client_spec,
      use_synthetic_data=FLAGS.use_synthetic_data)
  if FLAGS.task == 'emnist_character' and FLAGS.use_pseudo_clients:
    task = emnist_pseudo_client.build_task(
        base_task=task,
        examples_per_pseudo_client=FLAGS.examples_per_pseudo_client)
  return task


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))
  if not FLAGS.experiment_name:
    raise ValueError('FLAGS.experiment_name must be set.')

  task = _create_task()

  # Building train artifacts
  train_data = task.datasets.train_data
  train_preprocess_fn = task.datasets.train_preprocess_fn
  train_sampling_seed = int('{}{}'.format(FLAGS.clients_per_train_round,
                                          FLAGS.base_random_seed))
  training_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          train_data.client_ids, random_seed=train_sampling_seed),
      size=FLAGS.clients_per_train_round)

  @tff.tf_computation(tf.string)
  def build_train_dataset_from_client_id(client_id):
    raw_client_data = train_data.dataset_computation(client_id)
    return train_preprocess_fn(raw_client_data)

  learning_process = _create_train_computation(task.model_fn,
                                               FLAGS.num_clusters)
  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, learning_process)
  training_process.get_model_weights = learning_process.get_model_weights

  # Building eval artifacts
  eval_data = task.datasets.test_data
  eval_preprocess_fn = task.datasets.eval_preprocess_fn
  if FLAGS.clients_per_eval_round:
    eval_sampling_seed = int('{}{}'.format(FLAGS.clients_per_eval_round,
                                           FLAGS.base_random_seed))
    evaluation_selection_fn = functools.partial(
        tff.simulation.build_uniform_sampling_fn(
            eval_data.client_ids, random_seed=eval_sampling_seed),
        size=FLAGS.clients_per_eval_round)
  else:
    evaluation_selection_fn = lambda x: eval_data.client_ids

  @tff.tf_computation(tf.string)
  def build_eval_dataset_from_client_id(client_id):
    raw_client_data = eval_data.dataset_computation(client_id)
    return eval_preprocess_fn(raw_client_data)

  base_eval_computation = hypcluster_eval.build_hypcluster_eval(
      task.model_fn, FLAGS.num_clusters)
  build_dataset_and_eval = tff.simulation.compose_dataset_computation_with_computation(
      build_eval_dataset_from_client_id, base_eval_computation)

  def evaluation_fn(state, evaluation_data):
    return build_dataset_and_eval(
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
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=_ROUNDS_PER_EVALUATION,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=_ROUNDS_PER_CHECKPOINT,
      metrics_managers=metrics_managers)

if __name__ == '__main__':
  app.run(main)
