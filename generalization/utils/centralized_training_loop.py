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
"""Dispatcher for centralized training loops."""

import collections
import pprint
import time
from typing import Callable, List, Mapping, MutableMapping, Optional

from absl import logging
import tensorflow as tf

from generalization.utils import metric_utils

MetricsType = MutableMapping[str, float]

EvalFnType = Optional[Callable[[tf.keras.Model], Mapping[str, float]]]


def _compute_eval_metrics(model: tf.keras.Model, eval_fn: EvalFnType,
                          prefix: str) -> MetricsType:
  """Computes evaluation metrics for a given keras model.

  Specifically, this will return an ordered dictionary of metrics. The keys in
  the output of `eval_fn` will be prefixed with `prefix`. Additionally, the
  dictionary will contain a metric representing the number of seconds required
  to compute the eval metrics, with key `prefix + metric_utils.TIME_KEY`.

  Args:
    model: The current keras model.
    eval_fn: A callable accepting keras model, and returning a mapping of
      metrics with string-valued keys.
    prefix: A str to be prefixed to evaluation metrics.

  Returns:
    A mapping of evaluation metrics, where each key has been prefixed by
    `prefix`.
  """
  eval_start_time = time.time()
  eval_metrics = eval_fn(model)
  eval_time = time.time() - eval_start_time
  prefixed_eval_metrics = collections.OrderedDict()
  prefixed_eval_metrics[prefix + metric_utils.TIME_KEY] = eval_time
  for key, value in eval_metrics.items():
    prefixed_eval_metrics[prefix + key] = value
  return prefixed_eval_metrics


class EvalCallback(tf.keras.callbacks.Callback):
  """A callback that executes eval_fn at the end of every epoch.

  Specifically, at the end of every epoch, this callback will compute an ordered
  dictionary of metrics and update to `logs`. The keys in the output of
  `eval_fn` will be prefixed with `prefix`. Additionally, the dictionary will
  contain a metric representing the number of seconds required o compute the
  eval metrics, with key `prefix + metric_utils.TIME_KEY`.
  """

  def __init__(self, eval_fn, prefix):
    """Initialize the evaluation callback.

    Args:
      eval_fn: A callable accepting a keras model, and returning a mapping of
        metrics with string-valued keys.
      prefix: A str to be prefixed to evaluation metrics.
    """
    self._eval_fn = eval_fn
    self._prefix = prefix

  def on_epoch_end(self, epoch: int, logs=None):
    """Execute `eval_fn`, attach prefix and write into logs."""
    logging.info('Starting evaluating %s metrics for epoch %d', self._prefix,
                 epoch)

    prefixed_eval_metrics = _compute_eval_metrics(
        model=self.model, eval_fn=self._eval_fn, prefix=self._prefix)

    logging.info('Finished evaluating %s metrics for epoch %d. Results:',
                 self._prefix, epoch)
    logging.info('  %s', str(prefixed_eval_metrics))

    logs.update(prefixed_eval_metrics)


def _record_test_metrics(
    final_model: tf.keras.Model, total_epochs: int,
    test_fn: Optional[EvalFnType],
    metrics_callbacks: Optional[List[tf.keras.callbacks.Callback]]):
  """Record test metrics at the end of training."""

  if metrics_callbacks is None:
    metrics_callbacks = []

  if test_fn is not None:
    test_final_metrics = _compute_eval_metrics(
        model=final_model,
        eval_fn=test_fn,
        prefix=metric_utils.TEST_METRICS_PREFIX)

    logging.info('Final test metrics:\n %s', pprint.pformat(test_final_metrics))

    for metrics_callback in metrics_callbacks:
      metrics_callback.on_epoch_end(
          epoch=total_epochs + 1, logs=test_final_metrics)


def run(
    keras_model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    num_epochs: int,
    *,  # Caller passes below args by name.
    steps_per_epoch: Optional[int] = None,
    decay_epochs: Optional[int] = None,
    lr_decay: Optional[float] = None,
    part_train_eval_fn: Optional[EvalFnType] = None,
    part_val_fn: Optional[EvalFnType] = None,
    unpart_fn: Optional[EvalFnType] = None,
    test_fn: Optional[EvalFnType] = None,
    checkpoint_callback: Optional[tf.keras.callbacks.Callback] = None,
    metrics_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
) -> tf.keras.callbacks.History:
  """Run centralized training for a given compiled `tf.keras.Model`.

  Args:
    keras_model: A compiled `tf.keras.Model`.
    train_dataset: The `tf.data.Dataset` to be used for training.
    num_epochs: How many training epochs to perform.
    steps_per_epoch: An optional integer specifying the total number of steps
      (batches of samples) before declaring one epoch finished and starting the
      next epoch. If not provided, the epoch will run until the input dataset is
      exhausted.
    decay_epochs: Number of training epochs before decaying the learning rate.
    lr_decay: How much to decay the learning rate by every `decay_epochs`.
    part_train_eval_fn: An optional callable that accepts a `tf.keras.Model` and
      emits a mapping of evaluation metrics on training chunk of training
      clients.
    part_val_fn: An optional callable that accepts a `tf.keras.Model` and emits
      a mapping of evaluation metrics on validation chunk of training clients.
    unpart_fn: An optional callable that accepts a `tf.keras.Model` and emits a
      mapping of metrics.
    test_fn: An optional callable that accepts a `tf.keras.Model` and emits a
      mapping of test metrics, used after training.
    checkpoint_callback: An optional callback for checkpointing.
    metrics_callbacks: An optional list of callbacks for metrics logging.

  Returns:
    A `tf.keras.callbacks.History` object.
  """
  callbacks = []

  # Epoch timer should only record training time.
  callbacks.append(metric_utils.EpochTimerCallback())

  if part_train_eval_fn is not None:
    callbacks.append(
        EvalCallback(
            eval_fn=part_train_eval_fn,
            prefix=metric_utils.PART_TRAIN_EVAL_METRICS_PREFIX))

  if part_val_fn is not None:
    callbacks.append(
        EvalCallback(
            eval_fn=part_val_fn, prefix=metric_utils.PART_VAL_METRICS_PREFIX))

  if unpart_fn is not None:
    callbacks.append(
        EvalCallback(
            eval_fn=unpart_fn, prefix=metric_utils.UNPART_METRICS_PREFIX))

  if decay_epochs is not None and decay_epochs > 0:
    # Reduce the learning rate after a fixed number of epochs.
    def decay_lr(epoch, learning_rate):
      if epoch != 0 and epoch % decay_epochs == 0:
        return learning_rate * lr_decay
      else:
        return learning_rate

    lr_callback = tf.keras.callbacks.LearningRateScheduler(decay_lr, verbose=1)
    callbacks.append(lr_callback)

  callbacks.append(tf.keras.callbacks.ProgbarLogger(count_mode='steps'))

  if checkpoint_callback is not None:
    callbacks.append(checkpoint_callback)

  if metrics_callbacks is not None:
    callbacks.extend(metrics_callbacks)

  logging.info('Training model:')
  logging.info(keras_model.summary())

  history = keras_model.fit(
      train_dataset,
      epochs=num_epochs,
      steps_per_epoch=steps_per_epoch,
      callbacks=callbacks,
      verbose=1)

  _record_test_metrics(
      final_model=keras_model,
      total_epochs=num_epochs,
      test_fn=test_fn,
      metrics_callbacks=metrics_callbacks)

  return history
