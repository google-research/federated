# Copyright 2020, Google LLC.
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
"""Training loop for fedopt-guide experiments."""

import os.path
import pprint
import time
from typing import Any, Callable, Dict, List, Optional

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from utils import utils_impl
from tensorboard.plugins.hparams import api as hp


class IterativeProcessCompatibilityError(TypeError):
  pass


def create_if_not_exists(path):
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _setup_outputs(root_output_dir, experiment_name, hparam_dict):
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
  create_if_not_exists(summary_logdir)
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


def _check_iterative_process_compatibility(iterative_process):
  """Checks the compatibility of an iterative process with the training loop."""
  error_message = (
      'The iterative_process argument must be of '
      'type`tff.templates.IterativeProcess`, and must have an '
      'attribute `get_model_weights`, which must be a `tff.Computation`. This '
      'computation must accept as input the state of `iterative_process`, and '
      'its output must be a nested structure of tensors matching the expected '
      'shape of the first input argument to `evaluation_fn`.')
  compatibility_error = IterativeProcessCompatibilityError(error_message)

  if not isinstance(iterative_process, tff.templates.IterativeProcess):
    raise compatibility_error
  if not hasattr(iterative_process, 'get_model_weights'):
    raise compatibility_error
  elif not callable(iterative_process.get_model_weights):
    raise compatibility_error
  get_model_weights_fn = iterative_process.get_model_weights

  if not isinstance(get_model_weights_fn, tff.Computation):
    raise compatibility_error
  input_type = get_model_weights_fn.type_signature.parameter
  server_state_type = iterative_process.state_type.member
  server_state_type.is_assignable_from(input_type)
  # TODO(b/174268978): Once we enforce federated evaluations, we can check
  # compatibility with `validation_fn` without actually running the function.


def run(iterative_process: tff.templates.IterativeProcess,
        train_client_datasets_fn: Callable[[int], List[tf.data.Dataset]],
        evaluation_fn: Callable[[Any, int], Dict[str, float]],
        total_rounds: int,
        experiment_name: str,
        test_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
        root_output_dir: Optional[str] = '/tmp/fedopt_guide',
        hparam_dict: Optional[Dict[str, Any]] = None,
        rounds_per_eval: Optional[int] = 10,
        rounds_per_checkpoint: Optional[int] = 50):
  """Runs federated training for a given `tff.templates.IterativeProcess`.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  The iterative process must also have a callable attribute `get_model_weights`
  that takes as input the state of the iterative process, and returns a
  nested structure of tensors that can be input to the `evaluation_fn` and
  `test_fn` (if provided).

  Args:
    iterative_process: A `tff.templates.IterativeProcess` instance to run.
    train_client_datasets_fn: Function accepting an integer argument (the round
      number) and returning a list of train client datasets to use as federated
      data for that training round.
    evaluation_fn: A callable accepting the output of the `get_model_weights`
      attribute of the iterative process and a `round_num`, and returning a
      dictionary of evaluation metrics. Used to compute evaluation metrics
      throughout the training process.
    total_rounds: The number of federated training rounds to perform.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    test_fn: A callable accepting the output of the `get_model_weights`
      attribute of the iterative process and returning a dictionary of test
      metrics. Used to compute test metrics at the end of the training process.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    hparam_dict: An optional dictionary specifying hyperparameters of the
      experiment. If provided, the hyperparameters will be written to CSV.
    rounds_per_eval: How often to compute validation metrics.
    rounds_per_checkpoint: How often to checkpoint the iterative process state.
      If you expect the job to restart frequently, this should be small. If no
      interruptions are expected, this can be made larger.

  Returns:
    The final `state` of the iterative process after training.
  """
  _check_iterative_process_compatibility(iterative_process)
  if not callable(train_client_datasets_fn):
    raise TypeError('train_client_datasets_fn should be callable.')
  if not callable(evaluation_fn):
    raise TypeError('evaluation_fn should be callable.')
  if test_fn is not None and not callable(test_fn):
    raise TypeError('test_fn should be callable.')

  logging.info('Starting iterative_process training loop...')
  initial_state = iterative_process.initialize()

  if not hasattr(initial_state, 'model'):
    raise TypeError('The server state must have a model attribute.')

  checkpoint_mngr, metrics_mngr, tensorboard_mngr = _setup_outputs(
      root_output_dir, experiment_name, hparam_dict)

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
  current_model = iterative_process.get_model_weights(state)

  loop_start_time = time.time()
  while round_num < total_rounds:
    data_prep_start_time = time.time()
    federated_train_data = train_client_datasets_fn(round_num)
    train_metrics = {
        'prepare_datasets_secs': time.time() - data_prep_start_time
    }

    training_start_time = time.time()
    prev_model = iterative_process.get_model_weights(state)
    state, round_metrics = iterative_process.next(state, federated_train_data)
    current_model = iterative_process.get_model_weights(state)

    train_metrics['training_secs'] = time.time() - training_start_time
    train_metrics['model_delta_l2_norm'] = _compute_numpy_l2_difference(
        current_model, prev_model)
    train_metrics.update(round_metrics)

    logging.info('Round {:2d}, {:.2f}s per round in average.'.format(
        round_num, (time.time() - loop_start_time) / (round_num + 1)))

    if (round_num % rounds_per_checkpoint == 0 or
        round_num == total_rounds - 1):
      save_checkpoint_start_time = time.time()
      checkpoint_mngr.save_checkpoint(state, round_num)
      train_metrics['save_checkpoint_secs'] = (
          time.time() - save_checkpoint_start_time)

    metrics = {'train': train_metrics}

    if round_num % rounds_per_eval == 0:
      # Compute evaluation metrics
      evaluate_start_time = time.time()
      validation_metrics = evaluation_fn(current_model, round_num)
      validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
      metrics['eval'] = validation_metrics

    _write_metrics(metrics_mngr, tensorboard_mngr, metrics, round_num)
    round_num += 1

  # Final metrics evaluation once the training has completed
  metrics = {}

  # Evaluation metrics
  evaluate_start_time = time.time()
  validation_metrics = evaluation_fn(current_model, round_num)
  validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
  metrics['eval'] = validation_metrics

  # Test set metrics
  if test_fn:
    test_start_time = time.time()
    test_metrics = test_fn(current_model)
    test_metrics['evaluate_secs'] = time.time() - test_start_time
    metrics['test'] = test_metrics
  _write_metrics(metrics_mngr, tensorboard_mngr, metrics, total_rounds)

  return state
