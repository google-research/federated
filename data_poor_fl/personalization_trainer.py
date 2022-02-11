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
"""Runs FedAvg + Finetuning on EMNIST with varying levels of data paucity."""

import collections
import functools
import math
from typing import Callable, List, Tuple

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff


from data_poor_fl import personalization_utils
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
  flags.DEFINE_enum('train_algorithm', 'fedopt', ['fedopt', 'fedsgd'],
                    'Which training algorithm to use.')
  flags.DEFINE_bool(
      'example_weighting', True, 'Whether to use example weighting when '
      'aggregating client updates (True) or uniform weighting (False).')
  flags.DEFINE_bool('clipping', True, 'Whether to use adaptive clipping.')
  flags.DEFINE_bool('zeroing', True, 'Whether to use adaptive zeroing.')

  # Finetuning evaluation configuration
  flags.DEFINE_integer(
      'finetuning_max_epochs', 20, 'Maximum number of finetuning epochs. The '
      'best epoch will be in [0, finetuning_max_epochs], which is identified '
      'by post-processing the evaluation metrics.')
  flags.DEFINE_integer(
      'clients_per_evaluation', 100, 'Number of clients sampled to perform '
      'finetuning evaluation.')

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
  optimizer_utils.define_optimizer_flags('finetuning')

FLAGS = flags.FLAGS

# Change constant to a flag if needs to be configured.
_ROUNDS_PER_EVALUATION = 10
_ROUNDS_PER_CHECKPOINT = 50
_EMNIST_MAX_ELEMENTS_PER_CLIENT = 418
# EMNIST has 3400 clients, we use the training data from 2500 clients for
# training, and the training data from the rest 900 clients for evaluation.
_NUM_RAW_EVAL_CLIENTS = 900


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
  opt_flag_dict = optimizer_utils.remove_unused_flags('finetuning',
                                                      opt_flag_dict)
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
  elif FLAGS.train_algorithm == 'fedopt':
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


def _get_pseudo_client_ids(examples_per_pseudo_clients: int,
                           base_client_examples_df: pd.DataFrame,
                           separator: str = '-') -> List[str]:
  """Generates a list of pseudo-client ids."""
  pseudo_client_ids = []
  for _, row in base_client_examples_df.iterrows():
    num_pseudo_clients = math.ceil(row.num_examples /
                                   examples_per_pseudo_clients)
    client_id = row.client_id
    expanded_client_ids = [
        client_id + separator + str(i) for i in range(num_pseudo_clients)
    ]
    pseudo_client_ids += expanded_client_ids
  return pseudo_client_ids


def _split_pseudo_client_ids(
    raw_client_ids: List[str],
    pseudo_client_ids: List[str],
    separator: str = '-') -> Tuple[List[str], List[str]]:
  """Splits the pseudo-client ids into training and evaluation."""
  random_state = np.random.RandomState(seed=FLAGS.base_random_seed)
  shuffled_raw_client_ids = random_state.permutation(raw_client_ids)
  raw_eval_client_ids = shuffled_raw_client_ids[:_NUM_RAW_EVAL_CLIENTS]
  pseudo_train_client_ids = []
  pseudo_eval_client_ids = []
  for pseudo_client_id in pseudo_client_ids:
    raw_id, _ = pseudo_client_id.split(separator)
    if raw_id in raw_eval_client_ids:
      pseudo_eval_client_ids.append(pseudo_client_id)
    else:
      pseudo_train_client_ids.append(pseudo_client_id)
  return pseudo_train_client_ids, pseudo_eval_client_ids


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
  eval_preprocess_fn = task.datasets.eval_preprocess_fn

  # Creating pseudo-clients
  if not FLAGS.use_synthetic_data:
    csv_file_path = 'data_poor_fl/emnist_train_num_examples.csv'
    with open(csv_file_path) as csv_file:
      train_client_example_counts = pd.read_csv(csv_file)
    separator = '-'
    pseudo_client_ids = _get_pseudo_client_ids(FLAGS.examples_per_pseudo_client,
                                               train_client_example_counts,
                                               separator)
    pseudo_train_client_ids, pseudo_eval_client_ids = _split_pseudo_client_ids(
        task.datasets.train_data.client_ids, pseudo_client_ids, separator)
  else:
    pseudo_train_client_ids, pseudo_eval_client_ids = None, None

  extended_train_data = pseudo_client_data.create_pseudo_client_data(
      base_client_data=task.datasets.train_data,
      examples_per_pseudo_client=FLAGS.examples_per_pseudo_client,
      pseudo_client_ids=pseudo_train_client_ids)
  training_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          extended_train_data.client_ids, random_seed=FLAGS.base_random_seed),
      size=FLAGS.clients_per_train_round)
  extended_eval_data = pseudo_client_data.create_pseudo_client_data(
      base_client_data=task.datasets.train_data,
      examples_per_pseudo_client=FLAGS.examples_per_pseudo_client,
      pseudo_client_ids=pseudo_eval_client_ids)

  # Creating the training process (and wiring in a dataset computation)
  @tff.tf_computation(tf.string)
  def build_train_dataset_from_client_id(client_id):
    raw_client_data = extended_train_data.dataset_computation(client_id)
    return train_preprocess_fn(raw_client_data)

  learning_process = _create_train_algorithm(task.model_fn)
  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, learning_process)
  training_process.get_model_weights = learning_process.get_model_weights

  @tff.tf_computation(tf.string)
  def build_finetuning_datasets_from_client_id(client_id):
    raw_data = extended_eval_data.dataset_computation(client_id)
    # `tff.learning.build_personalization_eval` expects *unbatched* client-side
    # datasets. Batching is part of user-supplied personalization function.
    reshaped_data = eval_preprocess_fn(raw_data).unbatch()
    train_data, test_data = personalization_utils.split_half(reshaped_data)
    return collections.OrderedDict(train_data=train_data, test_data=test_data)

  def personalize_fn_builder():
    return personalization_utils.build_personalize_fn(
        optimizer_fn=optimizer_utils.create_optimizer_fn_from_flags(
            'finetuning'),
        batch_size=FLAGS.train_batch_size,
        max_num_epochs=FLAGS.finetuning_max_epochs)

  # Creating the evaluation computation (and wiring in a dataset computation).
  personalize_fn_name = 'finetuning'
  finetuning_eval = tff.learning.build_personalization_eval(
      model_fn=task.model_fn,
      personalize_fn_dict=collections.OrderedDict([(personalize_fn_name,
                                                    personalize_fn_builder)]),
      baseline_evaluate_fn=personalization_utils.baseline_evaluate_fn,
      max_num_clients=FLAGS.clients_per_evaluation)
  # Compose the dataset computation with the finetuning eval computation. Note
  # that `tff.simulation.compose_dataset_computation_with_computation` does not
  # work when the dataset computation returns a dict of two datasets.
  @tff.federated_computation(finetuning_eval.type_signature.parameter[0],
                             tff.types.at_clients(tf.string))
  def composed_dataset_comp_with_finetuning_eval(model_weights, client_ids):
    processed_datasets = tff.federated_map(
        build_finetuning_datasets_from_client_id, client_ids)
    return finetuning_eval(model_weights, processed_datasets)

  def evaluation_fn(state, federated_data):
    raw_metrics = composed_dataset_comp_with_finetuning_eval(
        training_process.get_model_weights(state), federated_data)
    return personalization_utils.postprocess_finetuning_metrics(
        raw_metrics,
        accuracy_name='sparse_categorical_accuracy',
        personalize_fn_name=personalize_fn_name)

  evaluation_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          extended_eval_data.client_ids, random_seed=FLAGS.base_random_seed),
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
