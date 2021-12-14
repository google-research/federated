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

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from utils import utils_impl


def create_managers(
    root_dir: str,
    experiment_name: str,
    csv_save_mode: tff.program.CSVSaveMode = tff.program.CSVSaveMode.APPEND
) -> Tuple[tff.program.FileProgramStateManager,
           List[tff.program.ReleaseManager]]:
  """Creates a set of managers for running a simulation.

  The managers that are created and how they are configured are indended to be
  used with `tff.simulation.run_training_process` to run a simulation.

  Args:
    root_dir: A string representing the root output directory for the
      simulation.
    experiment_name: A unique identifier for the simulation, used to create
      appropriate subdirectories in `root_dir`.
    csv_save_mode: A `tff.program.CSVSaveMode` specifying the save mode for the
      `tff.program.CSVFileReleaseManager`.

  Returns:
    A `tff.program.FileProgramStateManager`, and a list of
    `tff.program.ReleaseManager`s consisting of a
    `tff.program.LoggingReleaseManager`, a `tff.program.CSVFileReleaseManager`,
    and a `tff.program.TensorboardReleaseManager`.
  """
  program_state_dir = os.path.join(root_dir, 'checkpoints', experiment_name)
  program_state_manager = tff.program.FileProgramStateManager(
      root_dir=program_state_dir)

  logging_release_manager = tff.program.LoggingReleaseManager()

  csv_file_path = os.path.join(root_dir, 'results', experiment_name,
                               'experiment.metrics.csv')
  csv_file_release_manager = tff.program.CSVFileReleaseManager(
      file_path=csv_file_path, save_mode=csv_save_mode)

  summary_dir = os.path.join(root_dir, 'logdir', experiment_name)
  tensorboard_release_manager = tff.program.TensorboardReleaseManager(
      summary_dir=summary_dir)

  logging.info('Writing...')
  logging.info('    program state to: %s', program_state_dir)
  logging.info('    CSV metrics to: %s', csv_file_path)
  logging.info('    TensorBoard summaries to: %s', summary_dir)
  return program_state_manager, [
      logging_release_manager,
      csv_file_release_manager,
      tensorboard_release_manager,
  ]


def configure_managers(
    root_output_dir: str,
    experiment_name: str,
    rounds_per_checkpoint: int = 50,
    csv_metrics_manager_save_mode: tff.simulation.SaveMode = tff.simulation
    .SaveMode.APPEND
) -> Tuple[tff.simulation.FileCheckpointManager,
           List[tff.simulation.MetricsManager]]:
  """Configures checkpoint and metrics managers.

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
