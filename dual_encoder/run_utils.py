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
"""TFF utils for Federated Learning projects."""

import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


def _configure_managers(
    root_output_dir: str, experiment_name: str
) -> Tuple[tff.program.FileProgramStateManager,
           List[tff.program.ReleaseManager]]:
  """Configures program state and metrics managers."""

  program_state_dir = os.path.join(root_output_dir, 'checkpoints',
                                   experiment_name)
  program_state_manager = tff.program.FileProgramStateManager(program_state_dir)

  logging_release_manager = tff.program.LoggingReleaseManager()

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  csv_file = os.path.join(results_dir, 'experiment.metrics.csv')
  csv_manager = tff.program.CSVFileReleaseManager(
      csv_file, save_mode=tff.program.CSVSaveMode.WRITE)

  summary_dir = os.path.join(root_output_dir, 'logdir', experiment_name)
  tensorboard_manager = tff.program.TensorboardReleaseManager(summary_dir)

  logging.info('Writing...')
  logging.info('    program state to: %s', program_state_dir)
  logging.info('    CSV metrics to: %s', csv_file)
  logging.info('    TensorBoard summaries to: %s', summary_dir)

  return program_state_manager, [
      logging_release_manager,
      csv_manager,
      tensorboard_manager,
  ]


def client_datasets_fn_from_tf_datasets(
    tf_datasets: List[tf.data.Dataset],
    clients_per_round: int,
) -> Callable[[int], List[tf.data.Dataset]]:
  """Produces a sampling function for train/val/test from a list of datasets."""

  def client_datasets_fn(_):
    sampled_clients = np.random.choice(
        list(range(len(tf_datasets))),
        size=clients_per_round,
        replace=False).tolist()
    return [tf_datasets[client_id] for client_id in sampled_clients]

  return client_datasets_fn


def train_and_eval(
    trainer: tff.templates.IterativeProcess,
    evaluator: tff.Computation,
    num_rounds: int,
    train_datasets: List[tf.data.Dataset],
    test_datasets: List[tf.data.Dataset],
    num_clients_per_round: int,
    num_clients_per_round_eval: Optional[int] = None,
    experiment_name: Optional[str] = 'federated_ml_mf',
    root_output_dir: Optional[str] = '/tmp/fed_recon',
    rounds_per_checkpoint: int = 50,
) -> tff.learning.framework.ServerState:
  """Trains, and evaluates a federated learning model.

  Args:
    trainer: A `tff.templates.IterativeProcess`.
    evaluator: A `tff.Computation` returned by build_federated_evaluation.
    num_rounds: The number of rounds to train for.
    train_datasets: A list of tensorflow training Datasets, one for each
      client.
    test_datasets: A list of tensorflow test Datasets, one for each client.
    num_clients_per_round: The number of clients to select for each training
      round. This is also used as the number of clients per eval round if
      `num_clients_per_round_eval` is None.
    num_clients_per_round_eval: The number of clients to select for each eval
      round. If this is None, `num_clients_per_round` is used instead for both
      train and eval.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    rounds_per_checkpoint: How often to checkpoint the iterative process state.
      If you expect the job to restart frequently, this should be small. If no
      interruptions are expected, this can be made larger.

  Returns:
    state: The final `state` of the iterative process after training.
  """
  # Default to num_clients_per_round if num_clients_per_round_eval is not
  # provided.
  if num_clients_per_round_eval is None:
    num_clients_per_round_eval = num_clients_per_round

  # Create client sampling functions for each of train and val.
  train_client_datasets_fn = client_datasets_fn_from_tf_datasets(
      train_datasets, clients_per_round=num_clients_per_round)
  val_client_datasets_fn = client_datasets_fn_from_tf_datasets(
      test_datasets, clients_per_round=num_clients_per_round_eval)

  def evaluation_fn(state: Any, sampled_data) -> Dict[str, float]:
    model = trainer.get_model_weights(state)
    return evaluator(model, sampled_data)

  program_state_manager, metrics_managers = _configure_managers(
      root_output_dir, experiment_name)

  state = tff.simulation.run_training_process(
      training_process=trainer,
      training_selection_fn=train_client_datasets_fn,
      total_rounds=num_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=val_client_datasets_fn,
      rounds_per_evaluation=1,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=rounds_per_checkpoint,
      metrics_managers=metrics_managers)

  return state
