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
"""Training loops for periodic distribution shift simulations."""

# TODO(b/193904908): add unit tests.

import asyncio
import collections
import pprint
import time
from typing import Any, Callable, List, MutableMapping, Optional, Tuple

from absl import logging
import numpy as np
import tensorflow_federated as tff

from periodic_distribution_shift import fedavg_temporal_kmeans


MetricsType = MutableMapping[str, Any]
FileCheckpointManager = tff.program.FileProgramStateManager
MetricsManager = tff.program.ReleaseManager
ValidationFnType = Callable[[Any, int], MetricsType]

TRAIN_STEP_TIME_KEY = 'train_step_time_in_seconds'
TRAIN_STEPS_PER_HOUR_KEY = 'train_steps_per_hour'
VALIDATION_METRICS_PREFIX = 'validation/'
VALIDATION_TIME_KEY = 'validation/validation_time_in_seconds'


def _load_initial_checkpoint(
    template_state: Any,
    file_checkpoint_manager: FileCheckpointManager) -> Tuple[Any, int]:
  """Loads a server state and starting round number from a checkpoint manager.

  This method loads a starting state for the iterative process and a starting
  round number indicating the first round to begin the entire training
  process. If a checkpoint is found, the starting state is set to the checkpoint
  state, and the next round to run is set to the round directly after the
  checkpoint round.

  If no checkpoint is found, the starting state is set to `template_state` and
  the starting round is set to `0`.

  Args:
    template_state: A nested structure to use as a template when reconstructing
      a checkpoint.
    file_checkpoint_manager: A `tff.program.FileProgramStateManager` used to
      load a checkpoint.

  Returns:
    A tuple of `(state, start_round)`, where `state` matches the Python
    structure in `initial_state`, and `start_round` is a nonnegative integer
    indicating the round at which training starts.
  """
  loop = asyncio.get_event_loop()
  ckpt_state, ckpt_round = loop.run_until_complete(
      file_checkpoint_manager.load_latest(template_state))
  if ckpt_state is None:
    start_state = template_state
    start_round = 0
  else:
    start_state = ckpt_state
    start_round = ckpt_round + 1
  return start_state, start_round


def _compute_validation_metrics(state: Any, round_num: int,
                                validation_fn: ValidationFnType) -> MetricsType:
  """Computes validation metrics for a given server state and round number.

  Specifically, this will return an ordered dictionary of metrics. The keys in
  the output of `validation_fn` will be prefixed with
  `tff.simulation.VALIDATION_METRICS_PREFIX`. Additionally, the dictionary will
  contain a metric representing the number of seconds required to compute the
  validation metrics, with key `tff.simulation.VALIDATION_TIME_KEY`.

  Args:
    state: The current state of a simulation.
    round_num: An integer indicating the current round number.
    validation_fn: A callable accepting `state` and `round_num`, and returning a
      mapping of metrics with string-valued keys.

  Returns:
    A mapping of validation metrics, where each key has been prefixed by
    `tff.simulation.VALIDATION_METRICS_PREFIX`.
  """
  validation_start_time = time.time()
  validation_metrics = validation_fn(state, round_num)
  validation_time = time.time() - validation_start_time
  prefixed_validation_metrics = collections.OrderedDict()
  prefixed_validation_metrics[VALIDATION_TIME_KEY] = validation_time
  for key, value in validation_metrics.items():
    prefixed_validation_metrics[VALIDATION_METRICS_PREFIX + key] = value
  return prefixed_validation_metrics


def _create_on_loop_start_fn(
    file_checkpoint_manager: Optional[FileCheckpointManager] = None,
    metrics_managers: Optional[List[MetricsManager]] = None,
    validation_fn: Optional[ValidationFnType] = None):
  """Creates a pre-loop callback function.

  This pre-loop callback performs a number of tasks depending on its input
  arguments. In its full generality, the callback will attempt to load a
  starting state and round number from a checkpoint, and clear all metrics saved
  after that starting round.

  If no checkpoint is available, we assume that no training has occurred, in
  which case we perform pre-training tasks. These include (in order, depending
  on the input arguments) computing validation, metrics on the starting state,
  saving those validation metrics via metrics managers, and saving an initial
  checkpoint. Note that if the validation and metrics writing occurs, we use a
  round number of `0`, which is reserved for pre-training tasks.

  Once the tasks above (or some subset of the tasks, depending on which
  arguments are supplied) are completed, the pre-loop callback returns the
  starting state and round number for training.

  Args:
    file_checkpoint_manager: An optional `tff.program.FileProgramStateManager`
      used to load an initial checkpoint, and save an initial checkpoint if no
      such checkpoint is found.
    metrics_managers: An optional list of ` tff.program.CSVFileReleaseManager`
      instances used to save initial validation metrics. Note that this occurs
      only if `validation_fn` is not `None.
    validation_fn: A callable accepting the training state and a nonnegative
      integer round number, and returning a python mapping of metrics with
      string-valued keys.

  Returns:
    A callable that accepts the initial state of an iterative process. The
    callable performs the tasks descreibed above, and returns a starting state
    and a positive integer round number at which the training loop should start.
  """
  if metrics_managers is None:
    metrics_managers = []

  def on_loop_start(initial_state):
    """Attempts to load a checkpoint before resuming training."""
    loop = asyncio.get_event_loop()

    if file_checkpoint_manager is not None:
      start_state, start_round = _load_initial_checkpoint(
          initial_state, file_checkpoint_manager)
    else:
      start_state = initial_state
      start_round = 0

    if start_round == 0:
      # Perform pre-training actions, including computing initial validation
      # metrics and saving an initial checkpoint.
      if validation_fn is not None:
        validation_metrics = _compute_validation_metrics(
            start_state, 0, validation_fn)
        loop.run_until_complete(
            asyncio.gather(
                *[m.release(validation_metrics, 0) for m in metrics_managers]))

      if file_checkpoint_manager is not None:
        loop.run_until_complete(file_checkpoint_manager.save(start_state, 0))
      start_round = 1

    return start_state, start_round

  return on_loop_start


def _create_on_round_end_fn(
    file_checkpoint_manager: Optional[FileCheckpointManager] = None,
    metrics_managers: Optional[List[MetricsManager]] = None,
    validation_fn: Optional[ValidationFnType] = None):
  """Creates a on-round-end callback function.

  In its full generality, this on-round-end callback computes validation metrics
  on the state of an iterative process at a given round number, updates an
  input mapping of metrics with these validation metrics, saves the metrics
  via ` tff.program.CSVFileReleaseManager` objects, and saves a checkpoint via a
  `tff.program.FileProgramStateManager`.

  Args:
    file_checkpoint_manager: An optional `tff.program.FileProgramStateManager`
      used to save a checkpoint. If `None`, no checkpoint saving occurs.
    metrics_managers: An optional list of ` tff.program.CSVFileReleaseManager`
      instances used to save metrics.
    validation_fn: An optional callable accepting the training state and a
      nonnegative integer round number, and returning a python mapping of
      metrics with string-valued keys.

  Returns:
    A callable accepting the state of an iterative process an integer round
    number, and a mapping of metrics with key-valued strings. The callable
    performs the tasks listed above, and returns the same state and a
    mapping of metrics with key-valued strings, potentially updated to include
    validation metrics.
  """
  loop = asyncio.get_event_loop()

  if metrics_managers is None:
    metrics_managers = []

  def on_round_end(state: Any, round_num: int,
                   round_metrics: MetricsType) -> Tuple[Any, MetricsType]:
    if validation_fn is not None:
      validation_metrics = _compute_validation_metrics(state, round_num,
                                                       validation_fn)
      round_metrics.update(validation_metrics)

    loop.run_until_complete(
        asyncio.gather(
            *[m.release(round_metrics, round_num) for m in metrics_managers]))

    if file_checkpoint_manager is not None:
      loop.run_until_complete(file_checkpoint_manager.save(state, round_num))

    return state, round_metrics

  return on_round_end


def run_simulation_with_kmeans(
    train_process,
    period: int,
    client_selection_fn,
    total_rounds: int,
    file_checkpoint_manager: Optional[FileCheckpointManager] = None,
    metrics_managers: Optional[List[MetricsManager]] = None,
    validation_fn: Optional[ValidationFnType] = None,
    test_fn=None,
    test_freq: int = 32,
    num_tests: int = 3,
    server_optimizer_fn=None,
    server_lr_schedule=None,
    model_fn=None,
    aggregated_kmeans: bool = False,
    geo_lr: float = 0.2,
    prior_fn: str = 'linear',
    zero_mid: bool = False,
):
  """Runs a federated training simulation for a given iterative process.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> state)`.
    *   `next`: `<state, client_data> -> <state, metrics>` where state matches
        the output type of `initialize`, and `metrics` has member that is a
        python mapping with string-valued keys.

  This method performs up to `total_rounds` updates to the `state` of `process`.
  At each `round_num`, this update occurs by applying `process.next` to
  `state` and the output of ``client_selection_fn(round_num)`. We refer to this
  as a single "training step".

  This method also records how long it takes (in seconds) to call
  `client_selection_fn` and `process.next` at each round and add this to the
  round metrics with key `tff.simulation.TRAIN_STEP_TIME_KEY`. We also record
  how many training steps would occur per hour, which has key
  `tff.simulation.TRAIN_STEPS_PER_HOUR_KEY`.

  In full generality, after each round, we compute validation metrics via
  `validation_fn` (if not `None`), add these to the metrics created by
  `process.next` (prefixing with `tff.simulation.VALIDATION_METRICS_KEY`), save
  the combined metrics using the `metrics_managers` (if not `None`), and save a
  checkpoint via `file_checkpoint_manager` (if not `None`).

  Args:
    train_process: A `tff.templates.IterativeProcess` instance to run.
    period: Number of rounds for the period of distribution shifts.
    client_selection_fn: Callable accepting an integer round number, and
      returning a list of client data to use as federated data for that round.
    total_rounds: The number of federated training rounds to perform.
    file_checkpoint_manager: An optional `tff.program.FileProgramStateManager`
      used to periodically save checkpoints of the iterative process state.
    metrics_managers: An optional list of ` tff.program.CSVFileReleaseManager`
      objects used to save training metrics throughout the simulation.
    validation_fn: An optional callable accepting the current state of the
      iterative process (ie. the first output argument of
      `iterative_process.next`) and the current round number, and returning a
      mapping of validation metrics.
    test_fn: The test function to evaluate on the test set.
    test_freq: Calls the test function every `test_freq` rounds.
    num_tests: Number of tests, towards the end of training.
    server_optimizer_fn: Function for building the optimizer on the server.
    server_lr_schedule: The learning rate scheduler on the server.
    model_fn: Function for building the model.
    aggregated_kmeans: Whether we are using the aggregated k-means.
    geo_lr: Step size for the geometric updates.
    prior_fn: Name of the prior function, can be either `linear` or `cosine`.
    zero_mid: Whether to set the step size of the geometric update to 0 in
      the middle of each period.

  Returns:
    The `state` of the iterative process after training.
  """
  on_loop_start = _create_on_loop_start_fn(file_checkpoint_manager,
                                           metrics_managers, validation_fn)
  on_round_end = _create_on_round_end_fn(file_checkpoint_manager,
                                         metrics_managers, validation_fn)
  if aggregated_kmeans:
    return _run_simulation_with_aggregated_kmeans_and_callbacks(
        train_process,
        period,
        client_selection_fn=client_selection_fn,
        total_rounds=total_rounds,
        on_loop_start=on_loop_start,
        on_round_end=on_round_end,
        test_fn=test_fn,
        test_freq=test_freq,
        test_num=num_tests,
        server_optimizer_fn=server_optimizer_fn,
        server_lr_schedule=server_lr_schedule,
        model_fn=model_fn,
        geo_lr=geo_lr,
        prior_fn=prior_fn,
        zero_mid=zero_mid,
    )
  else:
    return _run_simulation_with_callbacks(
        train_process,
        client_selection_fn,
        total_rounds,
        on_loop_start,
        on_round_end,
        test_fn,
        test_freq=test_freq,
        test_num=num_tests,
        server_optimizer_fn=server_optimizer_fn,
        server_lr_schedule=server_lr_schedule,
        model_fn=model_fn,
    )


def _run_simulation_with_callbacks(
    train_process,
    client_selection_fn,
    total_rounds: int,
    on_loop_start: Optional[Callable[[Any], Tuple[Any, int]]] = None,
    on_round_end: Optional[Callable[[Any, int, MetricsType],
                                    Tuple[Any, MetricsType]]] = None,
    test_fn=None,
    test_freq: int = 32,
    test_num: int = 4,
    server_optimizer_fn=None,
    server_lr_schedule=None,
    model_fn=None):
  """Run the updates without kmeans.

  Args:
    train_process: A `tff.templates.IterativeProcess` instance to run. Must meet
      the type signature requirements documented above.
    client_selection_fn: Callable accepting an integer round number, and
      returning a list of client data to use as federated data for that round.
    total_rounds: The number of federated training rounds to perform.
    on_loop_start: An optional callable accepting the initial `state` of the
      iterative process, and returning a (potentially updated) `state` and an
      integer `round_num` used to determine where to resume the simulation loop.
    on_round_end: An optional callable accepting the `state` of the iterative
      process, an integer round number, and a mapping of metrics. The callable
      returns a (potentially updated) `state` of the same type, and a
      (potentially updated) mapping of metrics.
    test_fn: The test function.
    test_freq: Call the test function every `test_freq` steps.
    test_num: Number of tests at the end of training.
    server_optimizer_fn: Optimizer of the server.
    server_lr_schedule: Learning rate schedule on the server.
    model_fn: Keras model wrapper.

  Returns:
    The `state` of the iterative process after training.
  """
  initial_state = train_process.initialize()

  if on_loop_start is not None:
    state, start_round = on_loop_start(initial_state)
  else:
    state = initial_state
    start_round = 1

  model = model_fn()

  server_lr = server_lr_schedule(1)
  server_optimizer = server_optimizer_fn(server_lr)
  # We initialize the server optimizer variables to avoid creating them
  # within the scope of the tf.function server_update.
  fedavg_temporal_kmeans.initialize_optimizer_vars(model, server_optimizer)

  for round_num in range(start_round, total_rounds + 1):
    round_metrics = collections.OrderedDict(round_num=round_num)

    train_start_time = time.time()
    federated_train_data = client_selection_fn(round_num)

    _, metrics, model_delta = train_process.next(state, federated_train_data)

    state = fedavg_temporal_kmeans.server_update(model, server_optimizer, state,
                                                 model_delta,
                                                 state.kmeans_centers)

    train_time = time.time() - train_start_time
    round_metrics[TRAIN_STEP_TIME_KEY] = train_time
    if train_time == 0.0:
      round_metrics[TRAIN_STEPS_PER_HOUR_KEY] = None
    else:
      round_metrics[TRAIN_STEPS_PER_HOUR_KEY] = 1 / train_time * 60.0 * 60.0
    round_metrics.update(metrics)

    if test_fn is not None:
      if (round_num % test_freq == 0 and
          round_num >= total_rounds - test_freq * test_num):
        test_metrics = test_fn(state.model)
        round_metrics.update(test_metrics)

    if on_round_end is not None:
      state, round_metrics = on_round_end(state, round_num, round_metrics)
    logging.info('Output metrics at round {:d}:\n{!s}'.format(
        round_num, pprint.pformat(round_metrics)))

  return state


def _run_simulation_with_aggregated_kmeans_and_callbacks(
    process,
    period: int,
    client_selection_fn,
    total_rounds: int,
    on_loop_start: Optional[Callable[[Any], Tuple[Any, int]]] = None,
    on_round_end: Optional[Callable[[Any, int, MetricsType],
                                    Tuple[Any, MetricsType]]] = None,
    test_fn=None,
    test_freq: int = 32,
    test_num: int = 4,
    server_optimizer_fn=None,
    server_lr_schedule=None,
    model_fn=None,
    geo_lr=0.2,
    prior_fn='linear',
    zero_mid: bool = False):
  """Training loop of FedTKM for periodic distribution shift.

  Args:
    process: A `tff.templates.IterativeProcess` instance to run. Must meet the
      type signature requirements documented above.
    period: Estimated period of the distribution shift.
    client_selection_fn: Callable accepting an integer round number, and
      returning a list of client data to use as federated data for that round.
    total_rounds: The number of federated training rounds to perform.
    on_loop_start: An optional callable accepting the initial `state` of the
      iterative process, and returning a (potentially updated) `state` and an
      integer `round_num` used to determine where to resume the simulation loop.
    on_round_end: An optional callable accepting the `state` of the iterative
      process, an integer round number, and a mapping of metrics. The callable
      returns a (potentially updated) `state` of the same type, and a
      (potentially updated) mapping of metrics.
    test_fn: The test function.
    test_freq: Call the test function every `test_freq` steps.
    test_num: Number of tests at the end of training.
    server_optimizer_fn: Optimizer of the server.
    server_lr_schedule: Learning rate schedule on the server.
    model_fn: Keras model wrapper.
    geo_lr: Step size of the geometric update.
    prior_fn: Function type of the temporal prior.
    zero_mid: Whether to use the schedule on geo_lr to reduce it to 0 in the
      middle.

  Returns:
    The `state` of the iterative process after training.
  """
  initial_state = process.initialize()

  if on_loop_start is not None:
    state, start_round = on_loop_start(initial_state)
  else:
    state = initial_state
    start_round = 1

  model = model_fn()

  server_lr = server_lr_schedule(1)
  server_optimizer = server_optimizer_fn(server_lr)
  # We initialize the server optimizer variables to avoid creating them
  # within the scope of the tf.function server_update.
  fedavg_temporal_kmeans.initialize_optimizer_vars(model, server_optimizer)

  dist_scalar = state.dist_scalar
  for round_num in range(start_round, total_rounds + 1):
    round_metrics = collections.OrderedDict(round_num=round_num)

    train_start_time = time.time()
    federated_train_data = client_selection_fn(round_num)

    (_, metrics, model_delta, kmeans_delta_sum, kmeans_n_samples,
     c1_ratio) = process.next(state, federated_train_data)
    del federated_train_data

    # TODO(b/193904908): abstract following updates out of the training loop.
    kmeans_delta = kmeans_delta_sum / np.maximum(kmeans_n_samples, 1e-7)
    # rescale dist ratio according to cluster1_ratios
    dist_scalar = fedavg_temporal_kmeans.geometric_scalar_from_g1_ratio_fn(
        c1_ratio, round_num, period, dist_scalar, geo_lr, prior_fn, zero_mid)

    # finally, update the model!
    updated_centers = kmeans_delta + state.kmeans_centers

    server_lr = server_lr_schedule(state.round_num)
    server_optimizer.lr.assign(server_lr)
    state = fedavg_temporal_kmeans.server_update(model, server_optimizer, state,
                                                 model_delta, updated_centers,
                                                 dist_scalar)

    round_metrics['dist_scalar'] = dist_scalar
    round_metrics['c1_ratio'] = c1_ratio

    train_time = time.time() - train_start_time
    round_metrics[TRAIN_STEP_TIME_KEY] = train_time
    if train_time == 0.0:
      round_metrics[TRAIN_STEPS_PER_HOUR_KEY] = None
    else:
      round_metrics[TRAIN_STEPS_PER_HOUR_KEY] = 1 / train_time * 60.0 * 60.0
    round_metrics.update(metrics)

    if test_fn is not None:
      if (round_num % test_freq == 0 and
          round_num >= total_rounds - test_freq * test_num):
        test_metrics = test_fn(state.model, state.kmeans_centers,
                               state.dist_scalar)
        round_metrics.update(test_metrics)

    if on_round_end is not None:
      state, round_metrics = on_round_end(state, round_num, round_metrics)
    logging.info('Output metrics at round {:d}:\n{!s}'.format(
        round_num, pprint.pformat(round_metrics)))

  return state
