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

import asyncio
import collections
import pprint
import time
from typing import Any, Callable, List, MutableMapping, Optional, Tuple

from absl import logging
import tensorflow_federated as tff

from generalization.utils import metric_utils

MetricsType = MutableMapping[str, Any]
EvalFnType = Callable[[Any, int], MetricsType]


def _load_initial_program_state(
    template_state: Any,
    program_state_manager: tff.program.ProgramStateManager) -> Tuple[Any, int]:
  """Loads a server state and starting round number from a program state manager.

  This method loads a starting state for the iterative process and a starting
  round number indicating the first round to begin the entire training
  process. If program state is found, the starting state is set to the program
  state, and the next round to run is set to the round directly after the round.

  If no program state is found, the starting state is set to `template_state`
  and the starting round is set to `0`.

  Args:
    template_state: A nested structure to use as a template when loading program
      state.
    program_state_manager: A `tff.program.ProgramStateManager` used to load
      program state.

  Returns:
    A tuple of `(state, start_round)`, where `state` matches the Python
    structure in `initial_state`, and `start_round` is a nonnegative integer
    indicating the round at which training starts.
  """
  loop = asyncio.get_event_loop()

  ckpt_state, ckpt_round = loop.run_until_complete(
      program_state_manager.load_latest(template_state))
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
    program_state_manager: Optional[tff.program.ProgramStateManager] = None,
    metrics_managers: Optional[List[tff.program.ReleaseManager]] = None,
    part_train_eval_fn: Optional[EvalFnType] = None,
    part_val_fn: Optional[EvalFnType] = None,
    unpart_fn: Optional[EvalFnType] = None):
  """Creates a pre-loop callback function.

  This pre-loop callback performs a number of tasks depending on its input
  arguments. In its full generality, the callback will attempt to load the
  program state and round number.

  If no program state is available, we assume that no training has occurred, in
  which case we perform pre-training tasks. These include (in order, depending
  on the input arguments) computing validation, metrics on the starting state,
  saving those validation metrics via metrics managers, and saving an initial
  program state. Note that if the validation and metrics writing occurs, we use
  a round number of `0`, which is reserved for pre-training tasks.

  Once the tasks above (or some subset of the tasks, depending on which
  arguments are supplied) are completed, the pre-loop callback returns the
  starting state and round number for training.

  Args:
    program_state_manager: An optional `tff.program.ProgramStateManager`
      used to load an initial program state, and save an initial program state
      if no such program state is found.
    metrics_managers: An optional list of `tff.program.ReleaseManager`
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
  loop = asyncio.get_event_loop()

  if metrics_managers is None:
    metrics_managers = []

  def on_loop_start(initial_state):
    """Attempts to load program state before resuming training."""

    if program_state_manager is not None:
      start_state, start_round = _load_initial_program_state(
          initial_state, program_state_manager)
    else:
      start_state = initial_state
      start_round = 0

    if start_round == 0:
      # Perform pre-training actions, including computing initial validation
      # metrics and saving an initial program state.
      metrics = collections.OrderedDict()

      for eval_fn, prefix in ((part_train_eval_fn,
                               metric_utils.PART_TRAIN_EVAL_METRICS_PREFIX),
                              (part_val_fn,
                               metric_utils.PART_VAL_METRICS_PREFIX),
                              (unpart_fn, metric_utils.UNPART_METRICS_PREFIX)):
        if eval_fn is not None:
          metrics.update(_compute_eval_metrics(start_state, 0, eval_fn, prefix))

      if metrics:
        loop.run_until_complete(
            asyncio.gather(*[m.release(metrics, 0) for m in metrics_managers]))

      if program_state_manager is not None:
        loop.run_until_complete(program_state_manager.save(start_state, 0))
      start_round = 1

    return start_state, start_round

  return on_loop_start


def _create_on_round_end_fn(
    program_state_manager: Optional[tff.program.ProgramStateManager] = None,
    rounds_per_saving_program_state: int = 1,
    metrics_managers: Optional[List[tff.program.ReleaseManager]] = None,
    part_train_eval_fn: Optional[EvalFnType] = None,
    part_val_fn: Optional[EvalFnType] = None,
    unpart_fn: Optional[EvalFnType] = None):
  """Creates a on-round-end callback function.

  In its full generality, this on-round-end callback computes validation metrics
  on the state of an iterative process at a given round number, updates an
  input mapping of metrics with these validation metrics, saves the metrics
  via `tff.program.ReleaseManager` objects, and saves program state via a
  `tff.program.ProgramStateManager`.

  Args:
    program_state_manager: An optional `tff.program.ProgramStateManager`
      used to save a program state. If `None`, no saving occurs.
    rounds_per_saving_program_state: The number of training rounds to run
      between saving program state.
    metrics_managers: An optional list of `tff.program.ReleaseManager`
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
  loop = asyncio.get_event_loop()

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

    loop.run_until_complete(
        asyncio.gather(
            *[m.release(round_metrics, round_num) for m in metrics_managers]))

    if program_state_manager is not None:
      if round_num % rounds_per_saving_program_state == 0:
        loop.run_until_complete(program_state_manager.save(state, round_num))

    return state, round_metrics

  return on_round_end


def _record_test_metrics(
    final_state: tff.learning.framework.ServerState, total_rounds: int,
    test_fn: Optional[EvalFnType],
    metrics_managers: Optional[List[tff.program.ReleaseManager]]):
  """Record test metrics at the end of training.

  Args:
    final_state: The `state` of the iterative process after training.
    total_rounds: The number of federated training rounds performed.
    test_fn: An optional callable accepting the current state of the iterative
      process (ie. the first output argument of `iterative_process.next`), and
      returning a mapping of test metrics.
    metrics_managers: An optional list of `tff.program.ReleaseManager` objects
      used to save training metrics throughout the simulation.
  """
  loop = asyncio.get_event_loop()

  if metrics_managers is None:
    metrics_managers = []

  if test_fn is not None:

    test_final_metrics = _compute_eval_metrics(final_state, total_rounds + 1,
                                               test_fn,
                                               metric_utils.TEST_METRICS_PREFIX)
    logging.info('Final test metrics:\n %s', pprint.pformat(test_final_metrics))

    loop.run_until_complete(
        asyncio.gather(*[
            m.release(test_final_metrics, total_rounds + 1)
            for m in metrics_managers
        ]))


def _run_simulation_with_callbacks(
    process: tff.templates.IterativeProcess,
    client_selection_fn: Callable[[int], Any],
    total_rounds: int,
    on_loop_start: Optional[Callable[[Any], Tuple[Any, int]]] = None,
    on_round_end: Optional[Callable[[Any, int, MetricsType],
                                    Tuple[Any, MetricsType]]] = None):
  """Runs federated training for a given `tff.templates.IterativeProcess`.

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
  round metrics with key `tff.simulation.ROUND_TIME_KEY`.

  This method uses up to two callbacks. The first, `on_loop_start`, accepts the
  initial state of `process`, and returns a starting `state` and `round_num` for
  the training loop. The callback can be used for things such as loading program
  states.

  The second callback, `on_round_end` is called after each training step. It
  accepts the output state and metrics of `process.next`, and the current round
  number, and returns a new state and metrics mapping. This can be used for
  computing and saving additional metrics.

  WARNING: These callbacks can access and mutate state and are intended for more
  advanced simulations where the state can be mutated outside of calling
  `process.next`. For example, the `on_round_end` callback can be used to
  mutate state according to the training metrics, enabling various kinds of
  adaptive simulations. If your simulation does not require such mutation, we
  recommend `tff.simulation.run_training_process` instead.

  Args:
    process: A `tff.templates.IterativeProcess` instance to run. Must meet the
      type signature requirements documented above.
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

  Returns:
    The `state` of the iterative process after training.
  """
  logging.info('Initializing simulation process')
  initial_state = process.initialize()

  if on_loop_start is not None:
    logging.info('Running on loop start callback')
    state, start_round = on_loop_start(initial_state)
  else:
    state = initial_state
    start_round = 1

  for round_num in range(start_round, total_rounds + 1):
    logging.info('Executing round %d', round_num)
    round_metrics = collections.OrderedDict(round_num=round_num)

    train_start_time = time.time()
    federated_train_data = client_selection_fn(round_num)

    state, metrics = process.next(state, federated_train_data)
    train_time = time.time() - train_start_time
    round_metrics[tff.simulation.ROUND_TIME_KEY] = train_time
    round_metrics.update(metrics)

    if on_round_end is not None:
      logging.info('running round end callback')
      state, round_metrics = on_round_end(state, round_num, round_metrics)

  return state


def run_simulation(
    process: tff.templates.IterativeProcess,
    client_selection_fn: Callable[[int], Any],
    total_rounds: int,
    *,  # Caller passes below args by name.
    part_train_eval_fn: Optional[EvalFnType] = None,
    part_val_fn: Optional[EvalFnType] = None,
    unpart_fn: Optional[EvalFnType] = None,
    test_fn: Optional[EvalFnType] = None,
    program_state_manager: Optional[tff.program.ProgramStateManager] = None,
    rounds_per_saving_program_state: int = 1,
    metrics_managers: Optional[List[tff.program.ReleaseManager]] = None
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
  program state via `program_state_manager` (if not `None`).

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
    program_state_manager: An optional `tff.program.ProgramStateManager`
      used to periodically save program state of the iterative process state.
    rounds_per_saving_program_state: The number of training rounds to run
      between saving program state.
    metrics_managers: An optional list of `tff.program.ReleaseManager`
      objects used to save training metrics throughout the simulation.

  Returns:
    The `state` of the iterative process after training.
  """
  on_loop_start = _create_on_loop_start_fn(
      program_state_manager,
      metrics_managers,
      part_train_eval_fn,
      part_val_fn,
      unpart_fn,
  )
  on_round_end = _create_on_round_end_fn(program_state_manager,
                                         rounds_per_saving_program_state,
                                         metrics_managers, part_train_eval_fn,
                                         part_val_fn, unpart_fn)
  final_state = _run_simulation_with_callbacks(process, client_selection_fn,
                                               total_rounds, on_loop_start,
                                               on_round_end)

  _record_test_metrics(final_state, total_rounds, test_fn, metrics_managers)
  return final_state
