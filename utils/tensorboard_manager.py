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
"""Utility class for logging metrics and hyperparameters to TensorBoard."""

import collections
from typing import Any, Dict

from absl import logging
import tensorflow as tf
import tree

from utils import metrics_manager
from tensorboard.plugins.hparams import api as hp


def _create_if_not_exists(path):
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _flatten_nested_dict(struct: Dict[str, Any]) -> Dict[str, Any]:
  """Flattens a given nested dictionary, sorting by flattened key value.

  For example, if we have the nested dictionary {'d':3, 'a': {'b': 1, 'c':2}, },
  this will produce the (ordered) dictionary {'a/b': 1, 'a/c': 2, 'd': 3}.

  Args:
    struct: A nested dictionary.

  Returns:
    A `collections.OrderedDict` representing a flattened version of `struct`.
    Compared with the input `struct`, this data is flattened, with the key
    names equal to the path in the nested structure. The `OrderedDict` is
    sorted by the flattened keys.
  """
  flat_struct = tree.flatten_with_path(struct)
  flat_struct = [('/'.join(map(str, path)), item) for path, item in flat_struct]
  return collections.OrderedDict(sorted(flat_struct))


class TensorBoardManager(metrics_manager.MetricsManager):
  """Utility class for saving metrics using `tf.summary`.

  This class is intended to log scalar metrics and hyperparameters so that they
  can be used with TensorBoard.
  """

  def __init__(self, summary_dir: str = '/tmp/logdir'):
    """Returns an initialized `SummaryWriterManager`.

    This class will write metrics to `summary_dir` using a
    `tf.summary.SummaryWriter`, created via `tf.summary.create_file_writer`.

    Args:
      summary_dir: A path on the filesystem containing all outputs of the
        associated summary writer.

    Raises:
      ValueError: If `root_metrics_dir` is an empty string.
      ValueError: If `summary_dir` is an empty string.
    """
    super().__init__()
    if not summary_dir:
      raise ValueError('Empty string passed for summary_dir argument.')

    self._logdir = summary_dir
    _create_if_not_exists(self._logdir)
    self._summary_writer = tf.summary.create_file_writer(self._logdir)
    self._latest_round_num = None

  def update_metrics(self, round_num: int,
                     metrics_to_append: Dict[str, Any]) -> Dict[str, Any]:
    """Updates the stored metrics data with metrics for a specific round.

    The specified `round_num` must be later than the latest round number
    previously used with `update_metrics`. Note that we do not check whether
    the underlying summary writer has previously written any metrics with the
    given `round_num`. Thus, if the `TensorboardManager` is created from a
    directory containing previously written metrics, it may overwrite them. This
    is intended usage, allowing one to restart and resume experiments from
    previous rounds.

    Args:
      round_num: Communication round at which `metrics_to_append` was collected.
      metrics_to_append: A nested structure of metrics collected during
        `round_num`. The nesting will be flattened for purposes of writing to
        TensorBoard.

    Returns:
      A `collections.OrderedDict` of the metrics used to update the manager.
      Compared with the input `metrics_to_append`, this data is flattened,
      with the key names equal to the path in the nested structure, and
      `round_num` has been added as an additional key (overwriting the value
      if already present in the input `metrics_to_append`). The `OrderedDict` is
      sorted by the flattened keys.

    Raises:
      ValueError: If the provided round number is negative.
      ValueError: If the provided round number is less than or equal to the
        last round number used with `update_metrics`.
    """
    if not isinstance(round_num, int) or round_num < 0:
      raise ValueError(
          f'round_num must be a nonnegative integer, received {round_num}.')
    if self._latest_round_num and round_num <= self._latest_round_num:
      raise ValueError(f'Attempting to append metrics for round {round_num}, '
                       'but metrics already exist through round '
                       f'{self._latest_round_num}.')

    metrics_to_append['round_num'] = round_num
    flat_metrics = _flatten_nested_dict(metrics_to_append)
    with self._summary_writer.as_default():
      for name, val in flat_metrics.items():
        tf.summary.scalar(name, val, step=round_num)

    self._latest_round_num = round_num
    return flat_metrics

  def update_hparams(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
    """Records a dictionary of hyperparameters.

    Args:
      hparams: A nested structure of hyperparameters The nesting will be
        flattened for writing to TensorBoard (with the new keys equal to the
        paths in the nested structure).

    Returns:
      A `collections.OrderedDict` of the hyperparameters written to TensorBoard.
      Compared with the input `hparams`, this data is flattened, with the key
      names equal to the path in the nested structure. The `OrderedDict` is
      sorted by the flattened keys.
    """
    flat_hparams = _flatten_nested_dict(hparams)
    with self._summary_writer.as_default():
      hp.hparams(flat_hparams)
    return flat_hparams
