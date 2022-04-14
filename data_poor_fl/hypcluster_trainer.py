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
"""Runs HypCluster on EMNIST with varying levels of data paucity."""

import collections
import functools
import math
from typing import Callable, List, Tuple

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import hypcluster_eval
from data_poor_fl import hypcluster_train
from data_poor_fl import optimizer_flag_utils
from data_poor_fl import personalization_utils
from data_poor_fl.pseudo_client_tasks import emnist_pseudo_client
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

  # Train client configuration
  flags.DEFINE_integer('clients_per_train_round', 10,
                       'How many clients to sample at each training round.')
  flags.DEFINE_integer(
      'train_epochs', 1,
      'Number of epochs performed by a client during a round of training.')
  flags.DEFINE_integer('train_batch_size', 10, 'Batch size on train clients.')

  # Pseudo-client configuration
  flags.DEFINE_bool('use_pseudo_clients', False, 'Whether to split the data '
                    'into pseudo-clients.')
  flags.DEFINE_integer(
      'examples_per_pseudo_client', None,
      'Maximum number of examples per pseudo-client. This '
      'should only be set if the use_pseudo_clients flag is '
      'set to True.')

  # Training algorithm configuration
  flags.DEFINE_bool('warmstart_hypcluster', False,
                    'Whether to warm-start HypCluster.')
  flags.DEFINE_string(
      'warmstart_root_dir', '',
      'Directory to load checkpoints from previous FedAvg training. Only used '
      'when `warmstart_hypcluster` is True.')
  flags.DEFINE_integer('num_clusters', 2,
                       'Number of clusters used in HypCluster.')

  # Eval algorithm configuration
  flags.DEFINE_integer(
      'clients_per_evaluation', 100, 'Number of clients sampled to perform '
      'federated evaluation.')
  flags.DEFINE_float(
      'eval_client_fraction', 0.25, 'The fraction of clients partitioned into '
      'the eval group, as opposed to the training group.')

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

FLAGS = flags.FLAGS

# Change constant to a flag if needs to be configured.
_ROUNDS_PER_EVALUATION = 10
_ROUNDS_PER_CHECKPOINT = 50
_EMNIST_MAX_ELEMENTS_PER_CLIENT = 418
# EMNIST has 3400 clients, we use the training data from 2500 clients for
# training, and the training data from the rest 900 clients for evaluation.


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

  # Write the updated hyperparameters to a file.
  training_utils.write_hparams_to_csv(hparam_dict, FLAGS.root_output_dir,
                                      FLAGS.experiment_name)


def _load_init_model_weights(
    model_fn: Callable[[],
                       tff.learning.Model]) -> List[tff.learning.ModelWeights]:
  """Load model weights to warm-start HypCluster."""
  state_manager = tff.program.FileProgramStateManager(FLAGS.warmstart_root_dir)
  learning_process_for_metedata = tff.learning.algorithms.build_weighted_fed_avg(
      model_fn=model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
      server_optimizer_fn=lambda: tf.keras.optimizers.Adam(1.0),
      client_weighting=tff.learning.ClientWeighting.NUM_EXAMPLES,
      model_aggregator=tff.learning.robust_aggregator(
          zeroing=True, clipping=True, add_debug_measurements=True))
  init_state = learning_process_for_metedata.initialize()
  loaded_models = []
  versions_saved = state_manager.versions()
  if FLAGS.num_clusters >= len(versions_saved):
    raise ValueError(
        f'The checkpoint directory {FLAGS.warmstart_root_dir} only has '
        f'{len(versions_saved)-1} checkpoints, but expected to load '
        f'{FLAGS.num_clusters} models. Please use a smaller value for '
        'FLAGS.num_clusters, or use a different checkpoint directory.')
  for i in range(1, FLAGS.num_clusters + 1):
    version = versions_saved[-i]
    state = state_manager.load(version=version, structure=init_state)
    loaded_models.append(learning_process_for_metedata.get_model_weights(state))
  return loaded_models


def _create_train_algorithm(
    model_fn: Callable[[], tff.learning.Model]
) -> tff.learning.templates.LearningProcess:
  """Creates a learning process for HypCluster training."""
  server_optimizer = optimizer_flag_utils.create_optimizer_from_flags('server')
  # Need to set `no_nan_division=True` to avoid NaNs in the learned model, which
  # can happen when a model is not selected by any client in a round.
  model_aggregator = tff.aggregators.MeanFactory(no_nan_division=True)
  client_optimizer = optimizer_flag_utils.create_optimizer_from_flags('client')
  initial_model_weights_list = None
  if FLAGS.warmstart_hypcluster:
    if not FLAGS.warmstart_root_dir:
      raise ValueError('Must provide a `warmstart_root_dir` when '
                       '`warmstart_hypcluster` is True.')
    initial_model_weights_list = _load_init_model_weights(model_fn)
  return hypcluster_train.build_hypcluster_train(
      model_fn=model_fn,
      num_clusters=FLAGS.num_clusters,
      client_optimizer=client_optimizer,
      server_optimizer=server_optimizer,
      model_aggregator=model_aggregator,
      initial_model_weights_list=initial_model_weights_list)


def _split_client_ids(client_ids: List[str]) -> Tuple[List[str], List[str]]:
  """Splits the client ids into training and evaluation."""
  num_eval_clients = math.floor(FLAGS.eval_client_fraction * len(client_ids))
  random_state = np.random.RandomState(seed=FLAGS.base_random_seed)
  shuffled_client_ids = random_state.permutation(client_ids)
  train_client_ids = shuffled_client_ids[num_eval_clients:]
  eval_client_ids = shuffled_client_ids[:num_eval_clients]
  return train_client_ids, eval_client_ids


def _create_task() -> tff.simulation.baselines.BaselineTask:
  """Creates a task for performing federated training and evaluation."""
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.train_epochs,
      batch_size=FLAGS.train_batch_size,
      shuffle_buffer_size=_EMNIST_MAX_ELEMENTS_PER_CLIENT)
  task = tff.simulation.baselines.emnist.create_character_recognition_task(
      train_client_spec,
      model_id='cnn',
      only_digits=False,
      use_synthetic_data=FLAGS.use_synthetic_data)

  if FLAGS.use_pseudo_clients:
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
  train_preprocess_fn = task.datasets.train_preprocess_fn
  eval_preprocess_fn = task.datasets.eval_preprocess_fn
  client_data = task.datasets.train_data
  if FLAGS.use_synthetic_data:
    # Synthetic data may not have sufficiently many clients to split
    train_client_ids = client_data.client_ids
    eval_client_ids = client_data.client_ids
  else:
    train_client_ids, eval_client_ids = _split_client_ids(
        client_data.client_ids)

  training_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          train_client_ids, random_seed=FLAGS.base_random_seed),
      size=FLAGS.clients_per_train_round)

  # Creating the training process (and wiring in a dataset computation)
  @tff.tf_computation(tf.string)
  def build_train_dataset_from_client_id(client_id):
    raw_client_data = client_data.dataset_computation(client_id)
    return train_preprocess_fn(raw_client_data)

  learning_process = _create_train_algorithm(task.model_fn)
  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, learning_process)
  training_process.get_model_weights = learning_process.get_model_weights

  @tff.tf_computation(tf.string)
  def build_eval_datasets_from_client_id(client_id):
    raw_data = client_data.dataset_computation(client_id)
    # Unbatching before splitting the data into half. This allows splitting at
    # the example level instead of at the batch level.
    reshaped_data = eval_preprocess_fn(raw_data).unbatch()
    selection_data, test_data = personalization_utils.split_half(reshaped_data)
    return collections.OrderedDict([
        (hypcluster_eval.SELECTION_DATA_KEY,
         selection_data.batch(FLAGS.train_batch_size)),
        (hypcluster_eval.TEST_DATA_KEY, test_data.batch(FLAGS.train_batch_size))
    ])

  hypcluster_eval_comp = hypcluster_eval.build_hypcluster_eval_with_dataset_split(
      model_fn=task.model_fn, num_clusters=FLAGS.num_clusters)
  # Compose the dataset computation with the hypcluster eval computation. Note
  # that `tff.simulation.compose_dataset_computation_with_computation` does not
  # work when the dataset computation returns a dict of two datasets.
  model_weights_at_server_type = tff.types.at_server(
      training_process.get_model_weights.type_signature.result)

  @tff.federated_computation(model_weights_at_server_type,
                             tff.types.at_clients(tf.string))
  def composed_dataset_comp_with_hypcluster_eval(model_weights, client_ids):
    processed_datasets = tff.federated_map(build_eval_datasets_from_client_id,
                                           client_ids)
    return hypcluster_eval_comp(model_weights, processed_datasets)

  def evaluation_fn(state, federated_data):
    return composed_dataset_comp_with_hypcluster_eval(
        training_process.get_model_weights(state), federated_data)

  evaluation_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          eval_client_ids, random_seed=FLAGS.base_random_seed),
      size=FLAGS.clients_per_evaluation)

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
