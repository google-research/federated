# Copyright 2019, The TensorFlow Federated Authors.
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
"""Training loops for iterative process simulations."""

import collections
import pprint
import time
from typing import Any, Callable, List, MutableMapping, Optional, Tuple

from absl import logging
import tensorflow_federated as tff

from generalization.utils import metric_utils

MetricsType = MutableMapping[str, Any]
FileCheckpointManager = tff.simulation.FileCheckpointManager
MetricsManager = tff.simulation.MetricsManager
EvalFnType = Callable[[Any, int], MetricsType]


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
    file_checkpoint_manager: A `tff.simulation.FileCheckpointManager` used to
      load a checkpoint.

  Returns:
    A tuple of `(state, start_round)`, where `state` matches the Python
    structure in `initial_state`, and `start_round` is a nonnegative integer
    indicating the round at which training starts.
  """
  ckpt_state, ckpt_round = file_checkpoint_manager.load_latest_checkpoint(
      template_state)
  if ckpt_state is None:
    start_state = template_state
    start_round = 0
  else:
    start_state = ckpt_state
    start_round = ckpt_round + 1
  return start_state, start_round


def _compute_eval_metrics(state: Any, round_num: int, eval_fn: EvalFnType,
                          prefix: str) -> MetricsType:
  """Computes evaluation metrics for a given server state and round number.

  Specifically, this will return an ordered dictionary of metrics. The keys in
  the output of `eval_fn` will be prefixed with `prefix`. Additionally, the
  dictionary will contain a metric representing the number of seconds required
  to compute the eval metrics, with key `prefix + metric_utils.TIME_KEY`.

  Args:
    state: The current state of a simulation.
    round_num: An integer indicating the current round number.
    eval_fn: A callable accepting `state` and `round_num`, and returning a
      mapping of metrics with string-valued keys.
    prefix: A str to be prefixed to evaluation metrics.

  Returns:
    A mapping of evaluation metrics, where each key has been prefixed by
    `prefix`.
  """
  eval_start_time = time.time()
  eval_metrics = eval_fn(state, round_num)
  eval_time = time.time() - eval_start_time
  prefixed_eval_metrics = collections.OrderedDict()
  prefixed_eval_metrics[prefix + metric_utils.TIME_KEY] = eval_time
  for key, value in eval_metrics.items():
    prefixed_eval_metrics[prefix + key] = value
  return prefixed_eval_metrics


def _create_on_loop_start_fn(
    file_checkpoint_manager: Optional[FileCheckpointManager] = None,
    metrics_managers: Optional[List[MetricsManager]] = None,
    part_train_eval_fn: Optional[EvalFnType] = None,
    part_val_fn: Optional[EvalFnType] = None,
    unpart_fn: Optional[EvalFnType] = None):
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
    file_checkpoint_manager: An optional `tff.simulation.FileCheckpointManager`
      used to load an initial checkpoint, and save an initial checkpoint if no
      such checkpoint is found.
    metrics_managers: An optional list of `tff.simulation.MetricsManager`
      instances used to save initial validation metrics. Note that this occurs
      only if `unpart_fn` is not `None.
    part_train_eval_fn: An optional callable accepting the current state of the
      iterative process (ie. the first output argument of
      `iterative_process.next`) and the current round number, and returning a
      mapping of evaluation metrics on training chunk of training clients.
    part_val_fn: An optional callable accepting the current state of the
      iterative process (ie. the first output argument of
      `iterative_process.next`) and the current round number, and returning a
      mapping of evaluation metrics on validation chunk of training clients.
    unpart_fn: An optional callable accepting the current state of the iterative
      process (ie. the first output argument of `iterative_process.next`) and
      the current round number, and returning a mapping of validation metrics.

  Returns:
    A callable that accepts the initial state of an iterative process. The
    callable performs the tasks descreibed above, and returns a starting state
    and a positive integer round number at which the training loop should start.
  """
  if metrics_managers is None:
    metrics_managers = []

  def on_loop_start(initial_state):
    """Attempts to load a checkpoint before resuming training."""

    if file_checkpoint_manager is not None:
      start_state, start_round = _load_initial_checkpoint(
          initial_state, file_checkpoint_manager)
    else:
      start_state = initial_state
      start_round = 0

    for metrics_mngr in metrics_managers:
      metrics_mngr.clear_metrics(start_round)

    if start_round == 0:
      # Perform pre-training actions, including computing initial validation
      # metrics and saving an initial checkpoint.
      metrics = collections.OrderedDict()

      for eval_fn, prefix in ((part_train_eval_fn,
                               metric_utils.PART_TRAIN_EVAL_METRICS_PREFIX),
                              (part_val_fn,
                               metric_utils.PART_VAL_METRICS_PREFIX),
                              (unpart_fn, metric_utils.UNPART_METRICS_PREFIX)):
        if eval_fn is not None:
          metrics.update(_compute_eval_metrics(start_state, 0, eval_fn, prefix))

      if metrics:
        for metrics_mngr in metrics_managers:
          metrics_mngr.save_metrics(metrics, 0)

      if file_checkpoint_manager is not None:
        file_checkpoint_manager.save_checkpoint(start_state, round_num=0)
      start_round = 1

    return start_state, start_round

  return on_loop_start


def _create_on_round_end_fn(
    file_checkpoint_manager: Optional[FileCheckpointManager] = None,
    metrics_managers: Optional[List[MetricsManager]] = None,
    part_train_eval_fn: Optional[EvalFnType] = None,
    part_val_fn: Optional[EvalFnType] = None,
    unpart_fn: Optional[EvalFnType] = None):
  """Creates a on-round-end callback function.

  In its full generality, this on-round-end callback computes validation metrics
  on the state of an iterative process at a given round number, updates an
  input mapping of metrics with these validation metrics, saves the metrics
  via `tff.simulation.MetricsManager` objects, and saves a checkpoint via a
  `tff.simulation.FileCheckpointManager`.

  Args:
    file_checkpoint_manager: An optional `tff.simulation.FileCheckpointManager`
      used to save a checkpoint. If `None`, no checkpoint saving occurs.
    metrics_managers: An optional list of `tff.simulation.MetricsManager`
      instances used to save metrics.
    part_train_eval_fn: An optional callable accepting the current state of the
      iterative process (ie. the first output argument of
      `iterative_process.next`) and the current round number, and returning a
      mapping of evaluation metrics on training chunk of training clients.
    part_val_fn: An optional callable accepting the current state of the
      iterative process (ie. the first output argument of
      `iterative_process.next`) and the current round number, and returning a
      mapping of evaluation metrics on validation chunk of training clients.
    unpart_fn: An optional callable accepting the current state of the iterative
      process (ie. the first output argument of `iterative_process.next`) and
      the current round number, and returning a mapping of validation metrics.

  Returns:
    A callable accepting the state of an iterative process an integer round
    number, and a mapping of metrics with key-valued strings. The callable
    performs the tasks listed above, and returns the same state and a
    mapping of metrics with key-valued strings, potentially updated to include
    validation metrics.
  """
  if metrics_managers is None:
    metrics_managers = []

  def on_round_end(state: Any, round_num: int,
                   round_metrics: MetricsType) -> Tuple[Any, MetricsType]:

    for eval_fn, prefix in ((part_train_eval_fn,
                             metric_utils.PART_TRAIN_EVAL_METRICS_PREFIX),
                            (part_val_fn, metric_utils.PART_VAL_METRICS_PREFIX),
                            (unpart_fn, metric_utils.UNPART_METRICS_PREFIX)):
      if eval_fn is not None:
        round_metrics.update(
            _compute_eval_metrics(state, round_num, eval_fn, prefix))

    for metrics_mngr in metrics_managers:
      metrics_mngr.save_metrics(round_metrics, round_num)

    if file_checkpoint_manager is not None:
      file_checkpoint_manager.save_checkpoint(state, round_num)

    return state, round_metrics

  return on_round_end


def _record_test_metrics(
    final_state: tff.learning.framework.ServerState,
    total_rounds: int,
    test_fn: Optional[EvalFnType],
    metrics_managers: Optional[List[MetricsManager]],
) -> None:
  """Record test metrics at the end of training.

  Args:
    final_state: The `state` of the iterative process after training.
    total_rounds: The number of federated training rounds performed.
    test_fn: An optional callable accepting the current state of the iterative
      process (ie. the first output argument of `iterative_process.next`), and
      returning a mapping of test metrics.
    metrics_managers: An optional list of `tff.simulation.MetricsManager`
      objects used to save training metrics throughout the simulation.
  """

  if metrics_managers is None:
    metrics_managers = []

  if test_fn is not None:

    test_final_metrics = _compute_eval_metrics(final_state, total_rounds + 1,
                                               test_fn,
                                               metric_utils.TEST_METRICS_PREFIX)
    logging.info('Final test metrics:\n %s', pprint.pformat(test_final_metrics))

    for metrics_manager in metrics_managers:
      metrics_manager.save_metrics(test_final_metrics, total_rounds + 1)


def run_simulation(
    process: tff.templates.IterativeProcess,
    client_selection_fn: Callable[[int], Any],
    total_rounds: int,
    *,  # Caller passes below args by name.
    part_train_eval_fn: Optional[EvalFnType] = None,
    part_val_fn: Optional[EvalFnType] = None,
    unpart_fn: Optional[EvalFnType] = None,
    test_fn: Optional[EvalFnType] = None,
    file_checkpoint_manager: Optional[FileCheckpointManager] = None,
    metrics_managers: Optional[List[MetricsManager]] = None,
) -> tff.learning.framework.ServerState:
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
  round metrics with key `tff.simulation.ROUND_TIME_KEY`. We also record
  how many training steps would occur per hour, which has key
  `tff.simulation.ROUND_PER_HOUR_KEY`.

  In full generality, after each round, we compute validation metrics via
  `unpart_fn` (if not `None`), add these to the metrics created by
  `process.next` (prefixing with `tff.simulation.VALIDATION_METRICS_KEY`), save
  the combined metrics using the `metrics_managers` (if not `None`), and save a
  checkpoint via `file_checkpoint_manager` (if not `None`).

  Args:
    process: A `tff.templates.IterativeProcess` instance to run.
    client_selection_fn: Callable accepting an integer round number, and
      returning a list of client data to use as federated data for that round.
    total_rounds: The number of federated training rounds to perform.
    part_train_eval_fn: An optional callable accepting the current state of the
      iterative process (ie. the first output argument of
      `iterative_process.next`) and the current round number, and returning a
      mapping of evaluation metrics on training chunk of training clients.
    part_val_fn: An optional callable accepting the current state of the
      iterative process (ie. the first output argument of
      `iterative_process.next`) and the current round number, and returning a
      mapping of evaluation metrics on validation chunk of training clients.
    unpart_fn: An optional callable accepting the current state of the iterative
      process (ie. the first output argument of `iterative_process.next`) and
      the current round number, and returning a mapping of validation metrics.
    test_fn: An optional callable accepting the current state of the iterative
      process (ie. the first output argument of `iterative_process.next`) and
      the current round number, and returning a mapping of test metrics.
    file_checkpoint_manager: An optional `tff.simulation.FileCheckpointManager`
      used to periodically save checkpoints of the iterative process state.
    metrics_managers: An optional list of `tff.simulation.MetricsManager`
      objects used to save training metrics throughout the simulation.

  Returns:
    The `state` of the iterative process after training.
  """
  on_loop_start = _create_on_loop_start_fn(
      file_checkpoint_manager,
      metrics_managers,
      part_train_eval_fn,
      part_val_fn,
      unpart_fn,
  )
  on_round_end = _create_on_round_end_fn(file_checkpoint_manager,
                                         metrics_managers, part_train_eval_fn,
                                         part_val_fn, unpart_fn)
  final_state = tff.simulation.run_simulation_with_callbacks(
      process, client_selection_fn, total_rounds, on_loop_start, on_round_end)

  _record_test_metrics(final_state, total_rounds, test_fn, metrics_managers)
  return final_state
