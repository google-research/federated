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
"""Utility functions and classes for logging hyperparameters and metrics."""

import collections
import os
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

from absl import logging
from clu import metric_writers
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import tree

from utils import utils_impl
from tensorboard.plugins.hparams import api as hp

MetricsType = MutableMapping[str, Any]
Callback = tf.keras.callbacks.Callback

EPOCH_TIME_KEY = 'epoch_time_in_seconds'

PART_TRAIN_EVAL_METRICS_PREFIX = 'part_train_eval_'
PART_VAL_METRICS_PREFIX = 'part_val_'
UNPART_METRICS_PREFIX = 'unpart_'
TEST_METRICS_PREFIX = 'test_'

TIME_KEY = 'time_in_seconds'


def _flatten_nested_dict(struct: Mapping[str, Any]) -> Dict[str, Any]:
  """Flattens a given nested structure of tensors, sorting by flattened keys.

  For example, if we have the nested dictionary {'d':3, 'a': {'b': 1, 'c':2}, },
  this will produce the (ordered) dictionary {'a/b': 1, 'a/c': 2, 'd': 3}. This
  will unpack lists, so that {'a': [3, 4, 5]} will be flattened to the ordered
  dictionary {'a/0': 3, 'a/1': 4, 'a/2': 5}. The values of the resulting
  flattened dictionary will be the tensors at the corresponding leaf nodes
  of the original struct.

  Args:
    struct: A nested dictionary.

  Returns:
    A `collections.OrderedDict` representing a flattened version of `struct`.
  """
  flat_struct = tree.flatten_with_path(struct)
  flat_struct = [('/'.join(map(str, path)), item) for path, item in flat_struct]
  return collections.OrderedDict(sorted(flat_struct))


class AtomicCSVLoggerCallback(tf.keras.callbacks.Callback):
  """A callback that writes per-epoch values to a CSV file."""

  def __init__(self, path: str):
    self._path = path

  def on_epoch_end(self, epoch: int, logs: Optional[Dict[Any, Any]] = None):
    results_path = os.path.join(self._path, 'experiment.metrics.csv')
    if tf.io.gfile.exists(results_path):
      # Read the results until now.
      results_df = utils_impl.atomic_read_from_csv(results_path)
      # Slice off results after the current epoch, this indicates the job
      # restarted.
      results_df = results_df[:epoch]
      # Add the new epoch.
      results_df = results_df.append(logs, ignore_index=True)
    else:
      results_df = pd.DataFrame(logs, index=[epoch])
    utils_impl.atomic_write_to_csv(results_df, results_path)


class EpochTimerCallback(tf.keras.callbacks.Callback):
  """A callback that records time used for training for each epoch."""

  def __init__(self):
    self.time_start = None

  def on_epoch_begin(self, epoch: int, logs=None):
    self.time_start = time.time()

  def on_epoch_end(self, epoch: int, logs=None):
    time_end = time.time()
    elapsed_time = time_end - self.time_start
    logs[EPOCH_TIME_KEY] = elapsed_time


class MetricWriterManager(tff.simulation.MetricsManager):
  """A `tff.simulation.MetricsManager` that wraps a `MetricWriter` instance."""

  def __init__(self, metric_writer: metric_writers.MetricWriter):
    self._writer = metric_writer

  def save_metrics(self, metrics: Mapping[str, Any], round_num: int):
    self._writer.write_scalars(
        step=round_num, scalars=_flatten_nested_dict(metrics))


class MetricWriterCallback(tf.keras.callbacks.Callback):
  """A keras callback that wraps a clu `MetricWriter` instance."""

  def __init__(self, metric_writer: metric_writers.MetricWriter):
    self._writer = metric_writer

  def on_epoch_end(self, epoch: int, logs=None):
    self._writer.write_scalars(step=epoch, scalars=logs)


def _make_output_dirs(root_output_dir, experiment_name):
  """Get directories for outputs. Create if not exist."""
  tf.io.gfile.makedirs(root_output_dir)

  checkpoint_dir = os.path.join(root_output_dir, 'checkpoints', experiment_name)
  tf.io.gfile.makedirs(checkpoint_dir)

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  tf.io.gfile.makedirs(results_dir)

  summary_dir = os.path.join(root_output_dir, 'logdir', experiment_name)
  tf.io.gfile.makedirs(summary_dir)

  return checkpoint_dir, results_dir, summary_dir


def write_hparams(hparam_dict: Dict[str, Any], root_output_dir: str,
                  experiment_name: str) -> None:
  """Writes a dictionary of hyperparameters to CSV and Tensorboard HParam dashboard.

  All hyperparameters are written atomically to
  `{root_output_dir}/results/{experiment_name}/hparams.csv` as csv
  `{root_output_dir}/log_dir/{experiment_name}` as a tensorboard summary.

  Args:
    hparam_dict: A dictionary mapping string values to keys.
    root_output_dir: root_output_dir: A string representing the root output
      directory for the training simulation.
    experiment_name: A unique identifier for the current training simulation.
  """
  _, results_dir, summary_dir = _make_output_dirs(root_output_dir,
                                                  experiment_name)

  # Use a subdirectory to keep consistency with keras callback structure.
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_series_to_csv(hparam_dict, hparam_file)

  summary_writer = tf.summary.create_file_writer(summary_dir)
  with summary_writer.as_default():
    to_str_none = lambda v: 'None' if v is None else v
    hp.hparams({k: to_str_none(v) for k, v in hparam_dict.items()})

  logging.info('Writing...')
  logging.info('    hparameters CSV to: %s', hparam_file)
  logging.info('    TensorBoard summaries to: %s', summary_dir)


def configure_default_managers(
    root_output_dir: str,
    experiment_name: str,
    rounds_per_checkpoint: int,
) -> Tuple[tff.simulation.FileCheckpointManager,
           List[tff.simulation.MetricsManager]]:
  """Configures checkpoint and metrics managers for federated experiments.

  Args:
    root_output_dir: A string representing the root output directory for the
      training simulation. All metrics and checkpoints will be logged to
      subdirectories of this directory.
    experiment_name: A unique identifier for the current training simulation,
      used to create appropriate subdirectories of `root_output_dir`.
    rounds_per_checkpoint: How often to write checkpoints.

  Returns:
    A `tff.simulation.FileCheckpointManager`, and a list of
    `tff.simulation.MetricsManager` instances.
  """
  checkpoint_dir, results_dir, summary_dir = _make_output_dirs(
      root_output_dir, experiment_name)

  checkpoint_manager = tff.simulation.FileCheckpointManager(
      checkpoint_dir, step=rounds_per_checkpoint)

  csv_file = os.path.join(results_dir, 'experiment.metrics.csv')

  metric_managers = [tff.simulation.CSVMetricsManager(csv_file)]

  metric_managers.append(
      MetricWriterManager(
          metric_writers.create_default_writer(logdir=summary_dir)))
  logging.info('Writing...')
  logging.info('    checkpoints to: %s', checkpoint_dir)
  logging.info('    CSV metrics to: %s', csv_file)
  logging.info('    TensorBoard summaries to: %s', summary_dir)

  return checkpoint_manager, metric_managers


def configure_default_callbacks(
    root_output_dir: str,
    experiment_name: str,
    epochs_per_checkpoint: int,
) -> Tuple[Callback, List[Callback]]:
  """Configure checkpoint, backup and metric callbacks for centralized experiments.

  Args:
    root_output_dir: A string representing the root output directory for the
      training simulation. All metrics, checkpoints and backups will be logged
      to subdirectories of this directory.
    experiment_name: A unique identifier for the current training simulation,
      used to create appropriate subdirectories of `root_output_dir`.
    epochs_per_checkpoint: How often to write checkpoints.

  Returns:
    A callback for checkpointing, and a list of callbacks for metrics logging.
  """
  checkpoint_dir, results_dir, summary_dir = _make_output_dirs(
      root_output_dir, experiment_name)

  backup_dir = os.path.join(root_output_dir, 'backup', experiment_name)
  tf.io.gfile.makedirs(backup_dir)

  # `checkpoint_callback` is intended for manual inspection.
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_dir,
      save_freq='epoch',
      period=epochs_per_checkpoint,
      write_graph=False)

  metric_callbacks = [AtomicCSVLoggerCallback(results_dir)]
  metric_callbacks.append(
      MetricWriterCallback(
          metric_writers.create_default_writer(logdir=summary_dir)))
  logging.info('Writing...')
  logging.info('    checkpoints to: %s', checkpoint_dir)
  logging.info('    CSV metrics to: %s', results_dir)
  logging.info('    TensorBoard summaries to: %s', summary_dir)

  return checkpoint_callback, metric_callbacks
