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
"""Training loops for DP-FTRL."""

import asyncio
import os.path
import pprint
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import logging
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from tensorboard.plugins.hparams import api as hp


def _setup_outputs(root_output_dir: str, experiment_name: str,
                   hparam_dict: Dict[str, Any]):
  """Set up directories for experiment loops, write hyperparameters to disk."""

  if not experiment_name:
    raise ValueError('experiment_name must be specified.')

  program_state_dir = os.path.join(root_output_dir, 'checkpoints',
                                   experiment_name)
  program_state_mngr = tff.program.FileProgramStateManager(program_state_dir)

  logging_mngr = tff.program.LoggingReleaseManager()

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  csv_file = os.path.join(results_dir, 'experiment.metrics.csv')
  metrics_mngr = tff.program.CSVFileReleaseManager(
      file_path=csv_file, key_fieldname='round_num')

  summary_logdir = os.path.join(root_output_dir, 'logdir', experiment_name)
  tensorboard_mngr = tff.program.TensorBoardReleaseManager(summary_logdir)

  if hparam_dict:
    summary_writer = tf.summary.create_file_writer(summary_logdir)
    hparam_dict['metrics_file'] = csv_file
    hparams_file = os.path.join(results_dir, 'hparams.csv')
    pd.Series(hparam_dict).to_csv(hparams_file)
    with summary_writer.as_default():
      hp.hparams({k: v for k, v in hparam_dict.items() if v is not None})

  logging.info('Writing...')
  logging.info('    program state to: %s', program_state_dir)
  logging.info('    metrics csv to: %s', csv_file)
  logging.info('    summaries to: %s', summary_logdir)

  return program_state_mngr, [logging_mngr, metrics_mngr, tensorboard_mngr]


def _write_metrics(metrics_mngrs, metrics, round_num):
  """Atomic metrics writer which inlines logic from MetricsHook class."""
  loop = asyncio.get_event_loop()

  if not isinstance(metrics, dict):
    raise TypeError('metrics should be type `dict`.')
  if not isinstance(round_num, int):
    raise TypeError('round_num should be type `int`.')
  logging.info('Metrics at round {:d}:\n{!s}'.format(round_num,
                                                     pprint.pformat(metrics)))
  loop.run_until_complete(
      asyncio.gather(*[m.release(metrics, round_num) for m in metrics_mngrs]))


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
    clients_seed: Optional[int] = None,
):
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
    total_epochs: Number of total epochs if using `ClientIDShuffler` to shuffle
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
    clients_seed: An optional seed to use for the client shuffling function.

  Returns:
    The final `state` of the iterative process after training.
  """
  loop = asyncio.get_event_loop()

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

  if clients_seed is None:
    clients_seed = random.getrandbits(32)

  program_state_mngr, metrics_mngrs = _setup_outputs(root_output_dir,
                                                     experiment_name,
                                                     hparam_dict)

  logging.info('Asking checkpoint manager to load checkpoint.')
  restored_state, round_num = loop.run_until_complete(
      program_state_mngr.load_latest(
          (initial_state, clients_seed, total_epochs)))

  if restored_state is None:
    state = initial_state
    # This condition ensures that in the case of client sampling, the loop below
    # iterates until the round condition is reached.
    epoch = 0 if total_epochs > 0 else -1
    round_num = 0
    logging.info('Initializing experiment from scratch at round %d.', round_num)
  else:
    state = restored_state[0]
    clients_seed = restored_state[1]
    epoch = restored_state[2]
    logging.info('Restarted from checkpoint round %d', round_num)
    round_num += 1  # Increment to avoid overwriting current checkpoint

  loop_start_time = time.time()
  while epoch < total_epochs and round_num < total_rounds:
    data_prep_start_time = time.time()
    federated_train_data, epoch = client_datasets_fn(round_num, epoch)

    train_metrics = {
        'prepare_datasets_secs': time.time() - data_prep_start_time
    }
    training_start_time = time.time()
    state, metrics = tff.structure.from_container(
        iterative_process.next(state, federated_train_data))
    train_metrics['training_secs'] = time.time() - training_start_time

    logging.info('Round {:2d}, {:.2f}s per round in average.'.format(
        round_num, (time.time() - loop_start_time) / (round_num + 1)))

    if (round_num % rounds_per_checkpoint == 0 or
        round_num == total_rounds - 1):
      save_checkpoint_start_time = time.time()
      try:
        loop.run_until_complete(
            program_state_mngr.save((state, clients_seed, epoch), round_num))
      except Exception:  # pylint: disable=broad-except
        logging.info('Checkpoint saving exception: %s', Exception)
      train_metrics['save_checkpoint_secs'] = (
          time.time() - save_checkpoint_start_time)

    metrics = {'train': train_metrics}
    if hasattr(iterative_process, 'get_model_weights'):
      model_weights = iterative_process.get_model_weights(state)
    else:
      model_weights = state.model

    if train_eval_fn and round_num % rounds_per_train_eval == 0:
      # Compute metrics over the entire training dataset
      train_eval_start = time.time()
      train_eval_metrics = train_eval_fn(model_weights)
      train_eval_metrics['evaluate_secs'] = time.time() - train_eval_start
      metrics['train_eval'] = train_eval_metrics

    if round_num % rounds_per_eval == 0:
      # Compute validation metrics
      evaluate_start_time = time.time()
      validation_metrics = validation_fn(model_weights)
      validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
      metrics['eval'] = validation_metrics
      _write_metrics(metrics_mngrs, metrics, round_num)

    round_num += 1

  # Final metrics evaluation once the training has completed
  metrics = {}
  if hasattr(iterative_process, 'get_model_weights'):
    model_weights = iterative_process.get_model_weights(state)
  else:
    model_weights = state.model

  # Validation metrics
  evaluate_start_time = time.time()
  validation_metrics = validation_fn(model_weights)
  validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
  metrics['eval'] = validation_metrics

  # Training set metrics
  if train_eval_fn:
    train_eval_start = time.time()
    train_eval_metrics = train_eval_fn(model_weights)
    train_eval_metrics['evaluate_secs'] = time.time() - train_eval_start
    metrics['train_eval'] = train_eval_metrics

  # Test set metrics
  if test_fn:
    test_start_time = time.time()
    test_metrics = test_fn(model_weights)
    test_metrics['evaluate_secs'] = time.time() - test_start_time
    metrics['test'] = test_metrics
  _write_metrics(metrics_mngrs, metrics, round_num)

  return state


def _compute_largest_multiple_less_than(a, b):
  """Returns largest multiple of a which is less than or equal to b."""
  return int(b / a) * a


class ClientIDShuffler(object):
  """Shuffling clients in federated learning for DP-FTRL."""

  def __init__(
      self,
      clients_per_round: int,
      client_data: tff.simulation.datasets.ClientData,
      drop_remainder: bool = True,
      seed: Optional[int] = None,
  ):
    self._client_ids = list(client_data.client_ids)
    if clients_per_round <= 0:
      raise ValueError(
          f'ClientIDShuffler requires at least 1 client per roudn to be sampled. Initialized to sample {clients_per_round} clients clients per round.'
      )
    self._clients_per_round = clients_per_round
    # We will start every sample at index a multiple of self._clients_per_round.
    # If we need to ensure that the remainder is dropped (so that every sample
    # has the same number of clients), we may do this by modding these multiples
    # by the largest multiple of self._clients_per_round which is less than the
    # length of self._client_ids.
    if drop_remainder:
      self._start_index_modulus = _compute_largest_multiple_less_than(
          self._clients_per_round, len(self._client_ids))
    else:
      # Otherwise, we will simply mod by the length of client_ids.
      self._start_index_modulus = len(self._client_ids)
    self._epoch = 0
    if seed is None:
      seed = random.getrandbits(32)
    self._seed = seed
    # Initialize the client IDs to a shuffled list for epoch 0.
    self._shuffle_client_ids(self._epoch)

  def _shuffle_client_ids(self, epoch):
    random.Random(self._seed + epoch).shuffle(self._client_ids)

  def sample_client_ids(self, round_num: int, epoch: int) -> Tuple[List, int]:  # pylint: disable=g-bare-generic
    """Returns sampled client IDs and the updated epoch index.

    This function can be used as `client_datasets_fn` in `training_loop.run`.

    Args:
      round_num: the current round index.
      epoch: the current epoch index.
    """
    if epoch < self._epoch:
      raise ValueError('To ensure the epochs tracked by an algorithm are '
                       'appropriate, ClientIDShuffler does not support setting '
                       '`epoch` smaller than a previous value.')
    elif epoch > self._epoch:
      # This may indicate restarting from a checkpoint; shuffle for this epoch,
      # and make sure that the internal epoch tracker is set correctly.
      logging.info('shuffling clients for epoch %d, selecting for round %d',
                   epoch, round_num)
      self._shuffle_client_ids(epoch)
      self._epoch = epoch

    raw_start_index = round_num * self._clients_per_round
    # We make this calculation assuming that round is also 0-indexed.
    minimal_epoch = int(raw_start_index / len(self._client_ids))
    # Epoch is 0-indexed.
    if minimal_epoch > epoch:
      raise ValueError('Mismatch between round number and epoch. To '
                       f'sample for round {round_num} from a dataset with '
                       f'{len(self._client_ids)} clients with '
                       f'{self._clients_per_round} clients per round, the '
                       f'epoch must be at least {minimal_epoch}. Found epoch '
                       f'{epoch}.')
    start_index = raw_start_index % self._start_index_modulus
    end_index = min(start_index + self._clients_per_round,
                    len(self._client_ids))
    sampled_ids = self._client_ids[start_index:end_index]
    if end_index >= self._start_index_modulus:
      # We've reached the end of the client IDs, and at the next sample we will
      # need to wrap around. We need to increment the epoch
      # and reshuffle.
      logging.info('shuffling clients at epoch %d, round %d', epoch, round_num)
      self._epoch += 1
      self._shuffle_client_ids(self._epoch)
    return sampled_ids, self._epoch
