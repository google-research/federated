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
"""Training loops for DP-FTRL."""

import os.path
import pprint
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from dp_ftrl import dp_fedavg
from utils import utils_impl
from tensorboard.plugins.hparams import api as hp


def _create_if_not_exists(path):
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _setup_outputs(root_output_dir: str, experiment_name: str,
                   hparam_dict: Dict[str, Any]):
  """Set up directories for experiment loops, write hyperparameters to disk."""

  if not experiment_name:
    raise ValueError('experiment_name must be specified.')

  _create_if_not_exists(root_output_dir)

  checkpoint_dir = os.path.join(root_output_dir, 'checkpoints', experiment_name)
  _create_if_not_exists(checkpoint_dir)
  checkpoint_mngr = tff.simulation.FileCheckpointManager(checkpoint_dir)

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  _create_if_not_exists(results_dir)
  csv_file = os.path.join(results_dir, 'experiment.metrics.csv')
  metrics_mngr = tff.simulation.CSVMetricsManager(csv_file)

  summary_logdir = os.path.join(root_output_dir, 'logdir', experiment_name)
  _create_if_not_exists(summary_logdir)
  tensorboard_mngr = tff.simulation.TensorBoardManager(summary_logdir)

  if hparam_dict:
    summary_writer = tf.summary.create_file_writer(summary_logdir)
    hparam_dict['metrics_file'] = metrics_mngr.metrics_filename
    hparams_file = os.path.join(results_dir, 'hparams.csv')
    utils_impl.atomic_write_series_to_csv(hparam_dict, hparams_file)
    with summary_writer.as_default():
      hp.hparams({k: v for k, v in hparam_dict.items() if v is not None})

  logging.info('Writing...')
  logging.info('    checkpoints to: %s', checkpoint_dir)
  logging.info('    metrics csv to: %s', metrics_mngr.metrics_filename)
  logging.info('    summaries to: %s', summary_logdir)

  return checkpoint_mngr, metrics_mngr, tensorboard_mngr


def _write_metrics(metrics_mngr, tensorboard_mngr, metrics, round_num):
  """Atomic metrics writer which inlines logic from MetricsHook class."""
  if not isinstance(metrics, dict):
    raise TypeError('metrics should be type `dict`.')
  if not isinstance(round_num, int):
    raise TypeError('round_num should be type `int`.')
  logging.info('Metrics at round {:d}:\n{!s}'.format(round_num,
                                                     pprint.pformat(metrics)))
  metrics_mngr.save_metrics(metrics, round_num)
  tensorboard_mngr.save_metrics(metrics, round_num)


def _compute_numpy_l2_difference(model, previous_model):
  squared_norms = tf.nest.map_structure(lambda x, y: tf.linalg.norm(x - y)**2,
                                        model, previous_model)
  l2_total_tensor = tf.reduce_sum(tf.nest.flatten(squared_norms))**0.5
  return l2_total_tensor.numpy()


def run(
    iterative_process: tff.templates.IterativeProcess,
    client_datasets_fn: Callable[[int, int], Tuple[List, int]],  # pylint: disable=g-bare-generic
    validation_fn: Callable[[Any], Dict[str, float]],
    total_epochs: int,
    total_rounds: int,
    experiment_name: str,
    train_eval_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
    test_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
    root_output_dir: Optional[str] = '/tmp/fed_opt',
    hparam_dict: Optional[Dict[str, Any]] = None,
    rounds_per_eval: Optional[int] = 1,
    rounds_per_checkpoint: Optional[int] = 50,
    rounds_per_train_eval: Optional[int] = 100,
    server_state_epoch_update_fn: Optional[Callable[
        [dp_fedavg.ServerState], dp_fedavg.ServerState]] = None):
  """Runs federated training for a given `tff.templates.IterativeProcess`.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  Args:
    iterative_process: A `tff.templates.IterativeProcess` instance to run.
    client_datasets_fn: Function accepts integer arguments (the round number and
      the epoch) and returns a tuple of a list of client datasets to use as data
      data for that round, and the updated epoch index.
    validation_fn: A callable accepting the `model` attribute of the iterative
      process state and returning a dict of evaluation metrics. Used to compute
      validation metrics throughout the training process.
    total_epochs: Nubmer of total epochs if using `ClientIDShuffler` to shuffle
      clients. Use 0 when sampling clients and control by `total_rounds`.
    total_rounds: The number of federated training rounds to perform. If
      `ClientIDShuffler` is used for `client_datasets_fn`, the total rounds will
      take the minimum of `total_rounds` and rounds_per_epoch*`total_epochs`.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    train_eval_fn: An optional callable accepting the `model` attribute of the
      iterative process state and returning a dict of evaluation metrics. Used
      to compute training metrics over the entire training dataset throughout
      the course of the iterative process. If set to `None`, no such evaluation
      is done.
    test_fn: An optional callable accepting the `model` attribute of the
      iterative process state and returning a dict of test metrics. Used to
      compute test metrics at the end of the training process.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    hparam_dict: An optional dictionary specifying hyperparameters of the
      experiment. If provided, the hyperparameters will be written to CSV.
    rounds_per_eval: How often to compute validation metrics.
    rounds_per_checkpoint: How often to checkpoint the iterative process state.
      If you expect the job to restart frequently, this should be small. If no
      interruptions are expected, this can be made larger.
    rounds_per_train_eval: How often to compute metrics over the entire training
      dataset. Note that this is only done if a `train_eval_fn` argument is
      supplied.
    server_state_epoch_update_fn: A function to update the `SeverState` outside
      of TFF iterative process. It is called at the beginning of each epoch
      traversing all the clients. Used to restart tree for FTRL algorithm.

  Returns:
    The final `state` of the iterative process after training.
  """
  if not isinstance(iterative_process, tff.templates.IterativeProcess):
    raise TypeError('iterative_process should be type '
                    '`tff.templates.IterativeProcess`.')
  if not callable(client_datasets_fn):
    raise TypeError('client_datasets_fn should be callable.')
  if not callable(validation_fn):
    raise TypeError('validation_fn should be callable.')
  if train_eval_fn is not None and not callable(train_eval_fn):
    raise TypeError('train_eval_fn should be callable.')
  if test_fn is not None and not callable(test_fn):
    raise TypeError('test_fn should be callable.')

  logging.info('Starting iterative_process training loop...')
  initial_state = iterative_process.initialize()

  checkpoint_mngr, metrics_mngr, tensorboard_mngr = _setup_outputs(
      root_output_dir, experiment_name, hparam_dict)

  logging.info('Asking checkpoint manager to load checkpoint.')
  state, round_num = checkpoint_mngr.load_latest_checkpoint(initial_state)

  # TODO(b/172867399): we disable restarting from checkpoint when shuffling
  # client IDs by epochs. Non-trivial amount of change has to be made to make
  # sure disjoint clients are used cross rounds when restarts. A better design
  # of client dataset generator with random seed instead of `client_datasets_fn`
  # accepting `epoch` as argument, can help.
  epoch = 0 if total_epochs > 0 else -1
  if state is None or total_epochs > 0:
    state = initial_state
    round_num = 0
    logging.info('Initializing experiment from scratch at round %d.', round_num)
  else:
    logging.info('Restarted from checkpoint round %d', round_num)
    round_num += 1  # Increment to avoid overwriting current checkpoint
  metrics_mngr.clear_metrics(round_num)

  loop_start_time = time.time()
  while epoch < total_epochs and round_num < total_rounds:
    data_prep_start_time = time.time()
    prev_epoch = epoch
    federated_train_data, epoch = client_datasets_fn(round_num, epoch)
    # Server state is updated outside of TFF iterative process, which is used
    # to restart the tree in DP-FTRL.
    if server_state_epoch_update_fn is not None and epoch == prev_epoch + 1:
      logging.info('External server state update at epoch %d', epoch)
      state = server_state_epoch_update_fn(state)

    train_metrics = {
        'prepare_datasets_secs': time.time() - data_prep_start_time
    }

    training_start_time = time.time()
    prev_model = state.model
    state, loss = iterative_process.next(state, federated_train_data)

    train_metrics['training_secs'] = time.time() - training_start_time
    train_metrics['model_delta_l2_norm'] = _compute_numpy_l2_difference(
        state.model, prev_model)
    train_metrics['loss'] = loss

    logging.info('Round {:2d}, {:.2f}s per round in average.'.format(
        round_num, (time.time() - loop_start_time) / (round_num + 1)))

    if (round_num % rounds_per_checkpoint == 0 or
        round_num == total_rounds - 1):
      save_checkpoint_start_time = time.time()
      try:
        checkpoint_mngr.save_checkpoint(state, round_num)
      except Exception:  # pylint: disable=broad-except
        logging.info('Checkpoint saving exception: %s', Exception)
      train_metrics['save_checkpoint_secs'] = (
          time.time() - save_checkpoint_start_time)

    metrics = {'train': train_metrics}

    if train_eval_fn and round_num % rounds_per_train_eval == 0:
      # Compute metrics over the entire training dataset
      train_eval_start = time.time()
      train_eval_metrics = train_eval_fn(state.model)
      train_eval_metrics['evaluate_secs'] = time.time() - train_eval_start
      metrics['train_eval'] = train_eval_metrics

    if round_num % rounds_per_eval == 0:
      # Compute validation metrics
      evaluate_start_time = time.time()
      validation_metrics = validation_fn(state.model)
      validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
      metrics['eval'] = validation_metrics
      _write_metrics(metrics_mngr, tensorboard_mngr, metrics, round_num)

    round_num += 1

  # Final metrics evaluation once the training has completed
  metrics = {}

  # Validation metrics
  evaluate_start_time = time.time()
  validation_metrics = validation_fn(state.model)
  validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
  metrics['eval'] = validation_metrics

  # Training set metrics
  if train_eval_fn:
    train_eval_start = time.time()
    train_eval_metrics = train_eval_fn(state.model)
    train_eval_metrics['evaluate_secs'] = time.time() - train_eval_start
    metrics['train_eval'] = train_eval_metrics

  # Test set metrics
  if test_fn:
    test_start_time = time.time()
    test_metrics = test_fn(state.model)
    test_metrics['evaluate_secs'] = time.time() - test_start_time
    metrics['test'] = test_metrics
  _write_metrics(metrics_mngr, tensorboard_mngr, metrics, round_num)

  return state


class ClientIDShuffler(object):
  """Shuffling clients in federated learning for DP-FTRL."""

  def __init__(self,
               clients_per_round: int,
               client_data: tff.simulation.datasets.ClientData,
               drop_remainder: bool = True):
    self._client_ids = list(client_data.client_ids)
    self._clients_per_round = clients_per_round
    self._drop_remainder = drop_remainder
    self._epoch = 0
    self._start_index = 0

  def _shuffle_client_ids(self):
    random.shuffle(self._client_ids)
    self._start_index = 0
    self._epoch += 1

  def sample_client_ids(self, round_num: int, epoch: int) -> Tuple[List, int]:  # pylint: disable=g-bare-generic
    """Returns sampled client IDs and the updated epoch index.

    This function can be used as `client_datasets_fn` in `training_loop.run`.

    Args:
      round_num: the current round index.
      epoch: the current epoch index.
    """
    if epoch != self._epoch:
      raise ValueError(
          'Epoch index for client shuffling does not match: {} vs {}'.format(
              epoch, self._epoch))
    end_index = min(self._start_index + self._clients_per_round,
                    len(self._client_ids))
    sampled_ids = self._client_ids[self._start_index:end_index]
    skip_remainder_flag = (
        self._drop_remainder and
        (end_index + self._clients_per_round) > len(self._client_ids))
    if skip_remainder_flag or end_index >= len(self._client_ids):
      logging.info(
          'shuffling clients at epoch %d, round %d, client start index %d',
          epoch, round_num, self._start_index)
      self._shuffle_client_ids()
    else:
      self._start_index = end_index
    return sampled_ids, self._epoch
