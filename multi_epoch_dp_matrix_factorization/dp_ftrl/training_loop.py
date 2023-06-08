# Copyright 2023, Google LLC.
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
from collections.abc import Callable
import os.path
import pprint
import random
import time
from typing import Any, Optional

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from utils import utils_impl as experiment_utils
from tensorboard.plugins.hparams import api as hp


def _setup_outputs(
    root_output_dir: str, run_name: str, hparam_dict: dict[str, Any]
):
  """Set up directories for experiment loops, write hyperparameters to disk."""

  if not run_name:
    raise ValueError('run_name must be specified.')

  program_state_dir = os.path.join(root_output_dir, 'checkpoints', run_name)
  program_state_mngr = tff.program.FileProgramStateManager(program_state_dir)

  logging_mngr = tff.program.LoggingReleaseManager()

  results_dir = os.path.join(root_output_dir, 'results', run_name)
  csv_file = os.path.join(results_dir, 'experiment.metrics.csv')
  metrics_mngr = tff.program.CSVFileReleaseManager(
      file_path=csv_file,
      save_mode=tff.program.CSVSaveMode.WRITE,
      key_fieldname='round_num',
  )

  summary_logdir = os.path.join(root_output_dir, 'logdir', run_name)
  tensorboard_mngr = tff.program.TensorBoardReleaseManager(summary_logdir)

  if hparam_dict:
    summary_writer = tf.summary.create_file_writer(summary_logdir)
    hparam_dict['metrics_file'] = csv_file
    hparams_file = os.path.join(results_dir, 'hparams.csv')
    experiment_utils.atomic_write_series_to_csv(hparam_dict, hparams_file)
    with summary_writer.as_default():
      hp.hparams({k: v for k, v in hparam_dict.items() if v is not None})

  logging.info('Writing...')
  logging.info('    program state to: %s', program_state_dir)
  logging.info('    metrics csv to: %s', csv_file)
  logging.info('    summaries to: %s', summary_logdir)

  metric_managers = [logging_mngr, metrics_mngr, tensorboard_mngr]
  return program_state_mngr, metric_managers


def _write_metrics(metrics_mngrs, metrics, round_num):
  """Atomic metrics writer which inlines logic from MetricsHook class."""
  loop = asyncio.get_event_loop()

  if not isinstance(metrics, dict):
    raise TypeError('metrics should be type `dict`.')
  if not isinstance(round_num, int):
    raise TypeError('round_num should be type `int`.')
  logging.info(
      'Metrics at round {:d}:\n{!s}'.format(round_num, pprint.pformat(metrics))
  )
  metrics_type = tff.types.infer_unplaced_type(metrics)
  loop.run_until_complete(
      asyncio.gather(
          *[m.release(metrics, metrics_type, round_num) for m in metrics_mngrs]
      )
  )


def run(
    iterative_process: tff.templates.IterativeProcess,
    client_datasets_fn: Callable[[int], tuple[list, int]],  # pylint: disable=g-bare-generic
    validation_fn: Callable[[Any], dict[str, float]],
    total_epochs: int,
    total_rounds: int,
    run_name: str,
    train_eval_fn: Optional[Callable[[Any], dict[str, float]]] = None,
    test_fn: Optional[Callable[[Any], dict[str, float]]] = None,
    root_output_dir: Optional[str] = '/tmp/fed_opt',
    hparam_dict: Optional[dict[str, Any]] = None,
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
    client_datasets_fn: Function accepts an integer argument (the round number)
      and returns a tuple (list_of_client_datasets, epoch_of_current_round). If
      `compose_dataset_computation_with_learning_process` is used, then this
      function should return a list of client_ids instead of datasets. The
      epoch_of_current_round may always be zero if epoch-tracking is not
      supported.
    validation_fn: A callable accepting the `model` attribute of the iterative
      process state and returning a dict of evaluation metrics. Used to compute
      validation metrics throughout the training process.
    total_epochs: Number of total epochs to run if client_datasets_fn correctly
      computes epochs. Will be ignored (and can be set to zero) otherwise, e.g.
      if client_datasets_fn always returns 0 for the current epoch.
    total_rounds: The number of federated training rounds to perform. If
      `ClientIDShuffler` is used for `client_datasets_fn`, the total rounds will
      take the minimum of `total_rounds` and rounds_per_epoch*`total_epochs`.
    run_name: The name of this run. This will be appended to the
      `root_output_dir` for purposes of writing outputs.
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
    raise TypeError(
        'iterative_process should be type `tff.templates.IterativeProcess`.'
    )
  if not callable(client_datasets_fn):
    raise TypeError('client_datasets_fn should be callable.')
  if not callable(validation_fn):
    raise TypeError('validation_fn should be callable.')
  if train_eval_fn is not None and not callable(train_eval_fn):
    raise TypeError('train_eval_fn should be callable.')
  if test_fn is not None and not callable(test_fn):
    raise TypeError('test_fn should be callable.')

  logging.info('Starting iterative_process training loop...')
  state = iterative_process.initialize()

  if clients_seed is None:
    clients_seed = random.getrandbits(32)

  program_state_mngr, metrics_mngrs = _setup_outputs(
      root_output_dir, run_name, hparam_dict
  )

  def get_program_state():
    return (state, clients_seed)

  logging.info('Asking checkpoint manager to load checkpoint.')
  restored_state, round_num = loop.run_until_complete(
      program_state_mngr.load_latest(get_program_state())
  )

  if restored_state is None:
    # state remains the initial state
    round_num = 0
    logging.info('Initializing experiment from scratch at round %d.', round_num)
  else:
    state = restored_state[0]
    clients_seed = restored_state[1]
    logging.info('Restarted from checkpoint round %d', round_num)
    round_num += 1  # Increment to avoid overwriting current checkpoint

  loop_start_time = time.time()
  first_round_this_loop = round_num
  while round_num < total_rounds:
    data_prep_start_time = time.time()
    federated_train_data, epoch = client_datasets_fn(round_num)
    if total_epochs > 0 & epoch >= total_epochs:
      logging.info('Terminating training after reaching epoch number %s', epoch)
      # Done training, we have completed total_epochs.
      break

    data_prep_secs = time.time() - data_prep_start_time
    train_metrics = {'prepare_datasets_secs': data_prep_secs, 'epoch': epoch}
    round_start_time = time.time()
    logging.info(
        'Calling iterative_process.next(...), data prep took %s secs',
        data_prep_secs,
    )
    state, train_process_metrics = tff.structure.from_container(
        iterative_process.next(state, federated_train_data)
    )
    train_metrics['training_secs'] = time.time() - round_start_time
    train_metrics.update(train_process_metrics)

    logging.info(
        'Round {:2d}, {:.2f}s per round in average.'.format(
            round_num,
            (time.time() - loop_start_time)
            / (round_num - first_round_this_loop + 1),
        )
    )

    if round_num % rounds_per_checkpoint == 0 or round_num == total_rounds - 1:
      save_checkpoint_start_time = time.time()
      try:
        loop.run_until_complete(
            program_state_mngr.save(get_program_state(), round_num)
        )
      except Exception:  # pylint: disable=broad-except
        logging.info('Checkpoint saving exception: %s', Exception)
      train_metrics['save_checkpoint_secs'] = (
          time.time() - save_checkpoint_start_time
      )

    if hasattr(iterative_process, 'get_model_weights'):
      model_weights = iterative_process.get_model_weights(state)
    else:
      model_weights = state.model

    train_metrics['model_l2_norm'] = tf.linalg.global_norm(
        tf.nest.flatten(model_weights)
    )
    metrics = {'train': train_metrics}

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

    # Write metrics on every round.
    _write_metrics(metrics_mngrs, metrics, round_num)

    round_num += 1

  # Final metrics evaluation once the training has completed
  #
  # Rounds are zero-indexed so the metrics and checkpoint associated with round
  # zero correspond to the state *after* the first round of training. However,
  # to avoid releasing metrics for the same round_num twice, we keep the
  # final increment of round_num here. So, if total_rounds = 1000, the
  # 'train' and other metrics computed above for the final round will be
  # associated with round_num 999, but the test metrics (and other metrics
  # computed below) based on the same model state will be associated with
  # round_num 1000.
  logging.info('Training completed.')
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
    logging.info('Computing train-set metrics.')
    train_eval_start = time.time()
    train_eval_metrics = train_eval_fn(model_weights)
    train_eval_metrics['evaluate_secs'] = time.time() - train_eval_start
    metrics['train_eval'] = train_eval_metrics

  # Test set metrics
  if test_fn:
    logging.info('Computing test-set metrics.')
    test_start_time = time.time()
    test_metrics = test_fn(model_weights)
    test_metrics['evaluate_secs'] = time.time() - test_start_time
    metrics['test'] = test_metrics
  _write_metrics(metrics_mngrs, metrics, round_num)

  return state


class ClientIDShuffler(object):
  """Shuffling clients in federated learning for DP-FTRL."""

  def __init__(
      self,
      clients_per_round: int,
      client_ids: list[Any],
      seed: int,
      reshuffle_each_epoch: bool = True,
  ):
    """Creates shuffler.

    For a round in an arbitrary epoch, we define the round_index (the index
    into the current epoch) by
    ```
    rounds_per_epoch = len(client_ids) // clients_per_round
    crnt_epoch = round // rounds_per_epoch
    round_index = round % rounds_per_epoch
    ```
    The set of clients that participate in a given round is determined
    uniquely by the round_index, crnt_epoch, seed.  Each single client
    participates in at most one round per epoch, with a total of
    rounds_per_epoch * clients_per_round participating in each epoch
    (if this is less than len(client_ids), then some clients will be randomly
    excluded using the seed).

    When reshuffle_each_epoch=False, then crnt_epoch no longer influences
    the clients selected on a given round, it only depends on the round_index
    and the seed. This ensures that for any client that participates,
    they always participate in rounds that are exactly rounds_per_epoch apart.

    Args:
      clients_per_round: Return exactly this many clients for each round.
      client_ids: A list of client_ids to select from (the population).
      seed: Random seed.
      reshuffle_each_epoch: If true, a different shuffling of the clients is
        used inside each epoch.
    """
    self._seed = seed
    # Keep a copy of unshuffled client_ids; we always initialize from this
    # when shuffling, which ensures the order of clients on a given epoch
    # doesn't depend on how they were shuffled on previous epochs.
    # This should ensure a consistent order even when restarting from
    # a checkpoint.
    self._unshuffled_client_ids = client_ids
    self._client_ids = None  # Set by _shuffle_clients()

    num_clients = len(client_ids)
    if clients_per_round <= 0:
      raise ValueError(
          'ClientIDShuffler requires at least 1 client per round to be sampled.'
          f'Initialized to sample {clients_per_round} clients per round.'
      )
    elif clients_per_round > num_clients:
      raise ValueError(
          f'Only {num_clients} clients available, cannot ensure '
          f'{clients_per_round} clients per round.'
      )

    self._reshuffle_each_epoch = reshuffle_each_epoch
    self._clients_per_round = clients_per_round
    self.rounds_per_epoch = num_clients // clients_per_round
    assert self.rounds_per_epoch >= 1
    self._shuffle_clients(epoch=0)
    logging.info(
        (
            'ClientIDShuffler providing %s clients_per_round '
            'with %s rounds_per_epoch (%s clients in the dataset).'
        ),
        self._clients_per_round,
        self.rounds_per_epoch,
        num_clients,
    )

  def _shuffle_clients(self, epoch):
    logging.info('shuffling clients for epoch %d', epoch)
    self._client_ids = list(self._unshuffled_client_ids)
    random.Random(self._seed + epoch).shuffle(self._client_ids)
    self._shuffled_for_epoch = epoch

  def __call__(self, round_num: int) -> tuple[list[Any], int]:
    """Returns sampled client IDs and the updated epoch index.

    This function can be used as `client_datasets_fn` in `training_loop.run`.

    Args:
      round_num: the current round index.

    Returns:
      A tuple (client_id_list, current_epoch) where
      client_id_list are the clients that should participate in the current
      round, and current_epoch gives the epoch to which this round belongs.
    """
    epoch = round_num // self.rounds_per_epoch

    if self._reshuffle_each_epoch and self._shuffled_for_epoch != epoch:
      if epoch < self._shuffled_for_epoch:
        raise ValueError('No known valid usecase for decreasing epochs.')
      self._shuffle_clients(epoch)

    round_index = round_num % self.rounds_per_epoch
    # Compute index into self._client_ids based on the round_index,
    # (the round number inside the current epoch,
    # from 0 to rounds_per_epoch - 1).
    start = round_index * self._clients_per_round
    end = start + self._clients_per_round
    assert end <= len(self._client_ids)
    sampled_ids = self._client_ids[start:end]
    return sampled_ids, epoch
