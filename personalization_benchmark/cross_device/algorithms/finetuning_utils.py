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
"""Utilities for finetuning-based personalization."""

import collections
from typing import Any, Callable, Optional, OrderedDict
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

_NUM_TEST_EXAMPLES = 'num_examples'
_NUM_FINETUNE_EXAMPLES = 'num_train_examples'
_BASELINE_METRICS = 'baseline_metrics'
_RAW_METRICS_BEFORE_PROCESS = 'raw_metrics'
_OptimizerFnType = Callable[[], tf.keras.optimizers.Optimizer]
_MetricsType = OrderedDict[str, Any]
_FinetuneEvalFnType = Callable[
    [tff.learning.Model, tf.data.Dataset, tf.data.Dataset, Any], _MetricsType]


def build_finetune_eval_fn(optimizer_fn: _OptimizerFnType, batch_size: int,
                           num_finetuning_epochs: int,
                           finetune_last_layer: bool) -> _FinetuneEvalFnType:
  """Builds a `tf.function` that finetunes the model and evaluates on test data.

  The returned `tf.function` represents the logic to run on each client. It
  takes a `tff.learning.Model` (with weights already initialized to the desired
  initial model weights), an unbatched finetuning dataset, an unbatched test
  dataset, and an optional `context` (e.g., extra dataset) as input, finetunes
  the model on the training dataset, and returns the metrics evaluated on the
  test dataset (the evaluation is done after *every* finetuning epoch).

  Args:
    optimizer_fn: A no-argument function that returns a
      `tf.keras.optimizers.Optimizer`.
    batch_size: An `int` specifying the batch size used in finetuning
    num_finetuning_epochs: An `int` specifying the number of finetuning epochs.
    finetune_last_layer: If True, only finetune the last layer, otherwise,
      finetune all layers. Note that this only works for models built via
      `tff.learning.from_keras_model`.

  Returns:
    A `tf.function` that finetunes a model for `num_finetuning_epochs`,
    evaluates the model *every* epoch, and returns the evaluation metrics.
  """
  # Creates the `optimizer` here instead of inside the `tf.function` below,
  # because a `tf.function` generally does not allow creating new variables.
  optimizer = optimizer_fn()

  @tf.function
  def finetune_eval_fn(model: tff.learning.Model,
                       train_data: tf.data.Dataset,
                       test_data: tf.data.Dataset,
                       context: Optional[Any] = None) -> _MetricsType:
    """Finetunes the model and returns the evaluation metrics."""
    del context  # Unused

    @tf.function
    def train_one_batch(num_examples_sum, batch):
      """Run gradient descent on a batch."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      if finetune_last_layer:
        # Only works for models built via `tff.learning.from_keras_model`.
        last_layer_variables = model._keras_model.layers[-1].trainable_variables  # pylint:disable=protected-access
        grads = tape.gradient(output.loss, last_layer_variables)
        optimizer.apply_gradients(zip(grads, last_layer_variables))
      else:
        grads = tape.gradient(output.loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
      return num_examples_sum + output.num_examples

    # Starts training.
    metrics_dict = collections.OrderedDict()
    train_data = train_data.batch(batch_size)
    for idx in range(1, num_finetuning_epochs + 1):
      num_examples_sum = train_data.reduce(0, train_one_batch)
      # Evaluate the finetuned model every epoch.
      metrics_dict[f'epoch_{idx}'] = evaluate_fn(model, test_data, batch_size)
    metrics_dict[_NUM_FINETUNE_EXAMPLES] = num_examples_sum
    return metrics_dict

  return finetune_eval_fn


@tf.function
def evaluate_fn(model: tff.learning.Model,
                dataset: tf.data.Dataset,
                batch_size: int = 1) -> OrderedDict[str, tf.Tensor]:
  """Evaluates a model on the given dataset."""
  # Resets the model's local variables. This is necessary because
  # `model.report_local_unfinalized_metrics()` aggregates the metrics from *all*
  # previous calls to `forward_pass` (which include the metrics computed in
  # training). Resetting ensures that the returned metrics are computed on test
  # data. Similar to the `reset_states` method of `tf.keras.metrics.Metric`.
  model.reset_metrics()

  @tf.function
  def reduce_fn(num_examples_sum, batch):
    output = model.forward_pass(batch, training=False)
    return num_examples_sum + output.num_examples

  dataset.batch(batch_size).reduce(0, reduce_fn)
  finalized_metrics = collections.OrderedDict(
      # Note that `model.get_metrics` only works for models created via
      # `tff.learning.from_keras_model`. For non-keras based models, you can
      # apply `model.metric_finalizers` to the unfinalized metrics (returned by
      # `model.report_local_unfinalized_metrics`) to get the finalized metrics.
      (metric.name, metric.result()) for metric in model.get_metrics())
  return finalized_metrics


def postprocess_finetuning_metrics(
    valid_metrics_dict: _MetricsType, test_metrics_dict: _MetricsType,
    accuracy_name: str, finetuning_fn_name: str) -> OrderedDict[str, Any]:
  """Postprocesses the finetuning evaluation metrics collected at the server.

  By postprocessing, we will use `valid_metrics_dict` to find the best
  finetuning epoch number in [0, num_finetuning_epochs], and then report the
  test accuracies in `test_metrics_dict` at the best epoch.

  Both `valid_metrics_dict` and `test_metrics_dict` have the same nested
  structure. It has two keys: `_BASELINE_METRICS` and `finetuning_fn_name`.
  The baseline metrics are obtained from evaluating the initial model before
  finetuning happens. The metrics under `personalize_fn_name` are the metrics
  obtained from evaluating the finetuned model *every* epoch (see the docstring
  of `build_finetune_eval_fn`).

  Args:
    valid_metrics_dict: The finetuning evaluation metrics sampled from the
      validation clients.
    test_metrics_dict: The finetuning evaluation metrics sampled from the test
      clients.
    accuracy_name: The key to get the accuracy from `valid_metrics_dict` and
      `test_metrics_dict`. We use this metric to find the best finetuning epoch.
    finetuning_fn_name: The key to get the finetuning metrics from
      `valid_metrics_dict` and `test_metrics_dict`.

  Returns:
    A nested dict of string metric names to the postprocessed metric values.
  """
  # Find the best finetuning epoch using the validation metrics.
  valid_baseline_metrics = valid_metrics_dict[_BASELINE_METRICS]
  valid_finetuning_metrics = valid_metrics_dict[finetuning_fn_name]
  num_finetuning_epochs = len(valid_finetuning_metrics) - 1
  best_epoch = 0
  best_valid_accuracies_mean = np.mean(valid_baseline_metrics[accuracy_name])
  for idx in range(1, num_finetuning_epochs + 1):
    current_valid_accuracies_mean = np.mean(
        valid_finetuning_metrics[f'epoch_{idx}'][accuracy_name])
    if current_valid_accuracies_mean > best_valid_accuracies_mean:
      best_epoch = idx
      best_valid_accuracies_mean = current_valid_accuracies_mean
  # Extract the test accuracies at the best finetuning epoch.
  test_baseline_metrics = test_metrics_dict[_BASELINE_METRICS]
  test_finetuning_metrics = test_metrics_dict[finetuning_fn_name]
  test_accuracies_at_best_epoch_mean = np.mean(
      test_baseline_metrics[accuracy_name])
  if best_epoch > 0:
    test_accuracies_at_best_epoch_mean = np.mean(
        test_finetuning_metrics[f'epoch_{best_epoch}'][accuracy_name])
  # Compute the number of clients whose test accuracy hurts after finetuning.
  num_total_test_clients = len(test_baseline_metrics[accuracy_name])
  num_test_clients_hurt_after_finetuning = 0
  if best_epoch > 0:
    for client_i in range(num_total_test_clients):
      finetuning_accuracy = test_finetuning_metrics[f'epoch_{best_epoch}'][
          accuracy_name][client_i]
      baseline_accuracy = test_baseline_metrics[accuracy_name][client_i]
      if baseline_accuracy > finetuning_accuracy:
        num_test_clients_hurt_after_finetuning += 1
  fraction_clients_hurt = (
      num_test_clients_hurt_after_finetuning / float(num_total_test_clients))
  # Create the postprocessed metrics dictionary.
  postprocessed_metrics = collections.OrderedDict()
  postprocessed_metrics[_BASELINE_METRICS] = collections.OrderedDict()
  postprocessed_metrics[_BASELINE_METRICS][
      f'valid_{accuracy_name}_mean'] = np.mean(
          valid_baseline_metrics[accuracy_name])
  postprocessed_metrics[_BASELINE_METRICS][
      f'test_{accuracy_name}_mean'] = np.mean(
          test_baseline_metrics[accuracy_name])
  postprocessed_metrics[_BASELINE_METRICS][
      'test_num_eval_examples_mean'] = np.mean(
          test_baseline_metrics[_NUM_TEST_EXAMPLES])
  postprocessed_metrics[_BASELINE_METRICS][
      'test_num_finetune_examples_mean'] = np.mean(
          test_finetuning_metrics[_NUM_FINETUNE_EXAMPLES])
  postprocessed_metrics[finetuning_fn_name] = collections.OrderedDict()
  postprocessed_metrics[finetuning_fn_name][
      'best_finetuning_epoch'] = best_epoch
  postprocessed_metrics[finetuning_fn_name][
      f'valid_{accuracy_name}_at_best_epoch_mean'] = best_valid_accuracies_mean
  postprocessed_metrics[finetuning_fn_name][
      f'test_{accuracy_name}_at_best_epoch_mean'] = test_accuracies_at_best_epoch_mean
  postprocessed_metrics[finetuning_fn_name][
      'fraction_clients_hurt_at_best_epoch'] = fraction_clients_hurt
  postprocessed_metrics[_RAW_METRICS_BEFORE_PROCESS] = collections.OrderedDict(
      valid=valid_metrics_dict, test=test_metrics_dict)
  return postprocessed_metrics
