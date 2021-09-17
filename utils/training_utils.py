# Copyright 2021, Google LLC.
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
"""Utilities for performing federated learning training simulations via TFF."""

import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from utils import utils_impl


def configure_managers(
    root_output_dir: str,
    experiment_name: str,
    rounds_per_checkpoint: int = 50,
    csv_metrics_manager_save_mode: tff.simulation.SaveMode = tff.simulation
    .SaveMode.APPEND
) -> Tuple[tff.simulation.FileCheckpointManager,
           List[tff.simulation.MetricsManager]]:
  """Configures checkpoint and metrics managers.

  DEPRECATED: This is deprecated due to its usage of the deprecated
  `tff.simulation.FileCheckpointManager`. Please use `configure_output_managers`
  instead.

  Args:
    root_output_dir: A string representing the root output directory for the
      training simulation. All metrics and checkpoints will be logged to
      subdirectories of this directory.
    experiment_name: A unique identifier for the current training simulation,
      used to create appropriate subdirectories of `root_output_dir`.
    rounds_per_checkpoint: How often to write checkpoints.
    csv_metrics_manager_save_mode: A SaveMode specifying the save mode for
      CSVMetricsManager.

  Returns:
    A `tff.simulation.FileCheckpointManager`, and a list of
    `tff.simulation.MetricsManager` instances.
  """
  warnings.warn('`configure_managers` is deprecated, please use '
                '`configure_output_managers` instead.')
  checkpoint_dir = os.path.join(root_output_dir, 'checkpoints', experiment_name)
  checkpoint_manager = tff.simulation.FileCheckpointManager(
      checkpoint_dir, step=rounds_per_checkpoint)

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  csv_file = os.path.join(results_dir, 'experiment.metrics.csv')
  csv_manager = tff.simulation.CSVMetricsManager(
      csv_file, save_mode=csv_metrics_manager_save_mode)

  summary_dir = os.path.join(root_output_dir, 'logdir', experiment_name)
  tensorboard_manager = tff.simulation.TensorBoardManager(summary_dir)

  logging.info('Writing...')
  logging.info('    checkpoints to: %s', checkpoint_dir)
  logging.info('    CSV metrics to: %s', csv_file)
  logging.info('    TensorBoard summaries to: %s', summary_dir)
  return checkpoint_manager, [csv_manager, tensorboard_manager]


def configure_output_managers(
    root_output_dir: str,
    experiment_name: str,
    csv_metrics_manager_save_mode: tff.simulation.SaveMode = tff.simulation
    .SaveMode.APPEND
) -> Tuple[tff.simulation.FileCheckpointManager,
           List[tff.simulation.MetricsManager]]:
  """Configures a file program state manager and metrics managers.

  Args:
    root_output_dir: A string representing the root output directory for the
      training simulation. All metrics and checkpoints will be logged to
      subdirectories of this directory.
    experiment_name: A unique identifier for the current training simulation,
      used to create appropriate subdirectories of `root_output_dir`.
    csv_metrics_manager_save_mode: A SaveMode specifying the save mode for
      CSVMetricsManager.

  Returns:
    A `tff.program.FileProgramStateManager`, and a list of
    `tff.simulation.MetricsManager` instances.
  """
  program_state_dir = os.path.join(root_output_dir, 'program_states',
                                   experiment_name)
  program_state_manager = tff.program.FileProgramStateManager(
      root_dir=program_state_dir)
  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  csv_file = os.path.join(results_dir, 'experiment.metrics.csv')
  csv_manager = tff.simulation.CSVMetricsManager(
      csv_file, save_mode=csv_metrics_manager_save_mode)

  summary_dir = os.path.join(root_output_dir, 'logdir', experiment_name)
  tensorboard_manager = tff.simulation.TensorBoardManager(summary_dir)

  logging.info('Writing...')
  logging.info('    program states to: %s', program_state_dir)
  logging.info('    CSV metrics to: %s', csv_file)
  logging.info('    TensorBoard summaries to: %s', summary_dir)
  return program_state_manager, [csv_manager, tensorboard_manager]


def write_hparams_to_csv(hparam_dict: Dict[str, Any], root_output_dir: str,
                         experiment_name: str) -> None:
  """Writes a dictionary of hyperparameters to a CSV file.

  All hyperparameters are written atomically to
  `{root_output_dir}/results/{experiment_name}/hparams.csv`.

  Args:
    hparam_dict: A dictionary mapping string values to keys.
    root_output_dir: root_output_dir: A string representing the root output
      directory for the training simulation.
    experiment_name: A unique identifier for the current training simulation.
  """
  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  tf.io.gfile.makedirs(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_series_to_csv(hparam_dict, hparam_file)


def create_validation_fn(
    task: tff.simulation.baselines.BaselineTask,
    validation_frequency: int,
    num_validation_examples: Optional[int] = None
) -> Callable[[tff.learning.ModelWeights, int], Any]:
  """Creates a function for validating performance of a `tff.learning.Model`."""
  if task.datasets.validation_data is not None:
    validation_set = task.datasets.validation_data
  else:
    validation_set = task.datasets.test_data
  validation_set = validation_set.create_tf_dataset_from_all_clients()
  if num_validation_examples is not None:
    validation_set = validation_set.take(num_validation_examples)
  validation_set = task.datasets.eval_preprocess_fn(validation_set)

  evaluate_fn = tff.learning.build_federated_evaluation(task.model_fn)

  def validation_fn(model_weights, round_num):
    if round_num % validation_frequency == 0:
      return evaluate_fn(model_weights, [validation_set])
    else:
      return {}

  return validation_fn


def create_test_fn(
    task: tff.simulation.baselines.BaselineTask
) -> Callable[[tff.learning.ModelWeights], Any]:
  """Creates a function for testing performance of a `tff.learning.Model`."""
  test_set = task.datasets.get_centralized_test_data()
  evaluate_fn = tff.learning.build_federated_evaluation(task.model_fn)

  def test_fn(model_weights):
    return evaluate_fn(model_weights, [test_set])

  return test_fn


def create_client_selection_fn(
    task: tff.simulation.baselines.BaselineTask,
    clients_per_round: int,
    random_seed: Optional[int] = None
) -> Callable[[int], List[tf.data.Dataset]]:
  """Creates a random sampling function over training client datasets."""
  train_data = task.datasets.train_data.preprocess(
      task.datasets.train_preprocess_fn)
  client_id_sampling_fn = tff.simulation.build_uniform_sampling_fn(
      task.datasets.train_data.client_ids, random_seed=random_seed)

  def client_selection_fn(round_num):
    client_ids = client_id_sampling_fn(round_num, clients_per_round)
    return [train_data.create_tf_dataset_for_client(x) for x in client_ids]

  return client_selection_fn
