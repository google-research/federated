# Copyright 2019, Google LLC.
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
"""Internal dispatcher for training loops."""

import os.path
import pprint
import time
from typing import Any, Callable, Dict, List, Optional

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff


def create_if_not_exists(path):
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _setup_outputs(root_output_dir, experiment_name):
  """Set up directories for experiment loops, write hyperparameters to disk."""

  if not experiment_name:
    raise ValueError('experiment_name must be specified.')

  create_if_not_exists(root_output_dir)

  checkpoint_dir = os.path.join(root_output_dir, 'checkpoints', experiment_name)
  create_if_not_exists(checkpoint_dir)
  checkpoint_mngr = tff.simulation.FileCheckpointManager(checkpoint_dir)

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  create_if_not_exists(results_dir)
  csv_file = os.path.join(results_dir, 'experiment.metrics.csv')
  metrics_mngr = tff.simulation.CSVMetricsManager(csv_file)

  summary_logdir = os.path.join(root_output_dir, 'logdir', experiment_name)
  tb_mngr = tff.simulation.TensorBoardManager(summary_dir=summary_logdir)

  logging.info('Writing...')
  logging.info('    checkpoints to: %s', checkpoint_dir)
  logging.info('    metrics csv to: %s', metrics_mngr.metrics_filename)
  logging.info('    summaries to: %s', summary_logdir)

  return checkpoint_mngr, metrics_mngr, tb_mngr


def _write_metrics(metrics_mngr, tb_mngr, metrics, round_num):
  """Atomic metrics writer which inlines logic from MetricsHook class."""
  if not isinstance(metrics, dict):
    raise TypeError('metrics should be type `dict`.')
  if not isinstance(round_num, int):
    raise TypeError('round_num should be type `int`.')
  logging.info('Metrics at round {:d}:\n{!s}'.format(round_num,
                                                     pprint.pformat(metrics)))

  metrics_mngr.save_metrics(metrics, round_num)
  tb_mngr.save_metrics(metrics, round_num)


def run(iterative_process: tff.templates.IterativeProcess,
        client_datasets_fn: Callable[[int], List[tf.data.Dataset]],
        validation_fn: Callable[[Any, int], Dict[str, float]],
        total_rounds: int,
        experiment_name: str,
        test_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
        root_output_dir: Optional[str] = '/tmp/fed_opt',
        rounds_per_eval: Optional[int] = 1,
        rounds_per_checkpoint: Optional[int] = 50):
  """Runs federated training for a given `tff.templates.IterativeProcess`.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  Args:
    iterative_process: A `tff.templates.IterativeProcess` instance to run.
    client_datasets_fn: Function accepting an integer argument (the round
      number) and returning a list of client datasets to use as federated data
      for that round.
    validation_fn: A callable accepting the current state of `iterative_process`
      and the current round number, and returning a dict of evaluation metrics.
      Used to compute validation metrics throughout the training process.
    total_rounds: The number of federated training rounds to perform.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    test_fn: An optional callable accepting the current state of
      `iterative_process` and returning a dict of test set metrics. Used to
      compute test metrics at the end of the training process.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    rounds_per_eval: How often to compute validation metrics.
    rounds_per_checkpoint: How often to checkpoint the iterative process state.
      If you expect the job to restart frequently, this should be small. If no
      interruptions are expected, this can be made larger.

  Returns:
    The final `state` of the iterative process after training.
  """
  if not isinstance(iterative_process, tff.templates.IterativeProcess):
    raise TypeError(
        'iterative_process must be a `tff.templates.IterativeProcess`.')
  if not callable(client_datasets_fn):
    raise TypeError('client_datasets_fn should be callable.')
  if not callable(validation_fn):
    raise TypeError('validation_fn should be callable.')
  if test_fn is not None and not callable(test_fn):
    raise TypeError('test_fn should be callable.')

  logging.info('Starting iterative_process training loop...')
  initial_state = iterative_process.initialize()

  checkpoint_mngr, metrics_mngr, tb_mngr = _setup_outputs(
      root_output_dir, experiment_name)

  logging.info('Asking checkpoint manager to load checkpoint.')
  state, round_num = checkpoint_mngr.load_latest_checkpoint(initial_state)

  if state is None:
    logging.info('Initializing experiment from scratch.')
    state = initial_state
    round_num = 0
  else:
    logging.info('Restarted from checkpoint round %d', round_num)
    round_num += 1  # Increment to avoid overwriting current checkpoint
  metrics_mngr.clear_metrics(round_num)

  loop_start_time = time.time()
  loop_start_round = round_num
  while round_num < total_rounds:
    data_prep_start_time = time.time()
    federated_train_data = client_datasets_fn(round_num)
    train_metrics = {
        'prepare_datasets_secs': time.time() - data_prep_start_time
    }

    training_start_time = time.time()
    state, round_metrics = iterative_process.next(state, federated_train_data)
    train_metrics['training_secs'] = time.time() - training_start_time
    train_metrics.update(round_metrics)

    loop_time = time.time() - loop_start_time
    loop_rounds = (round_num - loop_start_round + 1)
    logging.info('Round {:2d}, {:.2f}s per round in average.'.format(
        round_num, loop_time / loop_rounds))

    if (round_num % rounds_per_checkpoint == 0 or
        round_num == total_rounds - 1):
      save_checkpoint_start_time = time.time()
      checkpoint_mngr.save_checkpoint(state, round_num)
      train_metrics['save_checkpoint_secs'] = (
          time.time() - save_checkpoint_start_time)

    metrics = {'train': train_metrics}

    if round_num % rounds_per_eval == 0:
      # Compute validation metrics
      evaluate_start_time = time.time()
      validation_metrics = validation_fn(state, round_num)
      validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
      metrics['eval'] = validation_metrics

    _write_metrics(metrics_mngr, tb_mngr, metrics, round_num)
    round_num += 1

  # Final metrics evaluation once the training has completed
  metrics = {}

  # Validation metrics
  evaluate_start_time = time.time()
  validation_metrics = validation_fn(state, round_num)
  validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
  metrics['eval'] = validation_metrics

  # Test set metrics
  if test_fn:
    test_start_time = time.time()
    test_metrics = test_fn(state)
    test_metrics['evaluate_secs'] = time.time() - test_start_time
    metrics['test'] = test_metrics
  _write_metrics(metrics_mngr, tb_mngr, metrics, total_rounds)

  return state
