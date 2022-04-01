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
from typing import Any, Callable, Optional, OrderedDict, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


def split_half(ds: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Splits the dataset into two equal-sized datasets."""
  num_elements_total = ds.reduce(0, lambda x, _: x + 1)
  num_elements_half = tf.cast(num_elements_total / 2, dtype=tf.int64)
  first_data = ds.take(num_elements_half)
  second_data = ds.skip(num_elements_half)
  return first_data, second_data


_OptimizerFnType = Callable[[], tf.keras.optimizers.Optimizer]
_PersonalizeFnType = Callable[
    [tff.learning.Model, tf.data.Dataset, tf.data.Dataset, Any],
    OrderedDict[str, Any]]
_NUM_EXAMPLES = 'num_examples'


def build_personalize_fn(optimizer_fn: _OptimizerFnType, batch_size: int,
                         max_num_epochs: int) -> _PersonalizeFnType:
  """Builds a `tf.function` that finetunes the model and evaluates on test data.

  The returned `tf.function` represents the personalization algorithm to run on
  a client. It takes a `tff.learning.Model` (with weights already initialized to
  the desired initial model weights), an unbatched training dataset, an
  unbatched test dataset, and an optional `context` (e.g., extra dataset) as
  input, trains a personalized model on the training dataset, and returns the
  metrics evaluated on the test dataset. The test dataset is split into a
  validation set and a test set, and the finetuned model is evaluated on both
  sets after *every* finetuning epoch.

  Args:
    optimizer_fn: A no-argument function that returns a
      `tf.keras.optimizers.Optimizer`.
    batch_size: An `int` specifying the batch size used in training.
    max_num_epochs: An `int` specifying the number of epochs used in training.

  Returns:
    A `tf.function` that trains a model for `max_num_epochs`, evaluates the
    personalized model *every* epoch, and returns the evaluation metrics.
  """
  # Creates the `optimizer` here instead of inside the `tf.function` below,
  # because a `tf.function` generally does not allow creating new variables.
  optimizer = optimizer_fn()

  @tf.function
  def personalize_fn(
      model: tff.learning.Model,
      train_data: tf.data.Dataset,
      test_data: tf.data.Dataset,
      context: Optional[Any] = None
  ) -> OrderedDict[str, OrderedDict[str, tf.Tensor]]:
    """Finetunes the model and returns the evaluation metrics."""
    del context  # Unused

    @tf.function
    def train_one_batch(num_examples_sum, batch):
      """Run gradient descent on a batch."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      grads = tape.gradient(output.loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      return num_examples_sum + output.num_examples

    # Starts training.
    metrics_dict = collections.OrderedDict()
    train_data = train_data.batch(batch_size)
    # Splits the `test_data` into a validation set and a test set.
    valid_set, test_set = split_half(test_data)
    for idx in range(1, max_num_epochs + 1):
      num_examples_sum = train_data.reduce(0, train_one_batch)
      # Evaluate the trained model every epoch.
      metrics_dict[f'epoch_{idx}_valid'] = evaluate_fn(model, valid_set,
                                                       batch_size)
      metrics_dict[f'epoch_{idx}_test'] = evaluate_fn(model, test_set,
                                                      batch_size)
    # Save the training statistics.
    metrics_dict['num_train_examples'] = num_examples_sum
    return metrics_dict

  return personalize_fn


@tf.function
def baseline_evaluate_fn(
    model: tff.learning.Model,
    dataset: tf.data.Dataset) -> OrderedDict[str, OrderedDict[str, tf.Tensor]]:
  """Splits dataset into validation and test set, and returns eval metrics."""
  valid_set, test_set = split_half(dataset)
  return collections.OrderedDict(
      valid=evaluate_fn(model, valid_set), test=evaluate_fn(model, test_set))


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
  # TODO(b/152633983): Replace it with a `reset_metrics` method.
  for var in model.local_variables:
    if var.initial_value is not None:
      var.assign(var.initial_value)
    else:
      var.assign(tf.zeros_like(var))

  @tf.function
  def reduce_fn(num_examples_sum, batch):
    output = model.forward_pass(batch, training=False)
    return num_examples_sum + output.num_examples

  dataset.batch(batch_size).reduce(0, reduce_fn)
  finalized_metrics = collections.OrderedDict(
      # Only works for models built with `tff.learning.from_keras_model`.
      (metric.name, metric.result()) for metric in model.get_metrics())
  return finalized_metrics


def postprocess_finetuning_metrics(
    metrics_dict: OrderedDict[str, Any], accuracy_name: str,
    personalize_fn_name: str) -> OrderedDict[str, Any]:
  """Postprocesses the finetuning evaluation metrics collected at the server.

  The finetuning evaluation metrics store the accuracies after *every*
  finetuning epoch. By postprocessing, we will find the best epoch number
  in [0, max_num_epochs] using the validation accuracies, and then report the
  test accuracies at the best epoch.

  Args:
    metrics_dict: The finetuning evaluation metrics collected at the server.
      Each leaf in the dictionary stores that metric value from the sampled
      clients (represented as a list of scalar metric values). It is a nested
      dictionary with 2 keys at the top level, which are 'baseline_metrics' and
      `personalize_fn_name`. The 'baseline_metrics' are metrics obtained from
      evaluating the initial model (i.e., no finetuning happens yet, see the
      docstring of `baseline_evaluate_fn`); while the `personalize_fn_name` are
      metrics obtained from evaluating the finetuned model *every* epoch (see
      the docstring of `build_personalize_fn`).
    accuracy_name: The key to use to get the accuracy from `metrics_dict`.
    personalize_fn_name: The key to use to get the personalization metrics from
      `metrics_dict`.

  Returns:
    A nested dict of string metric names to the processed metric values.
  """
  baseline_metrics = metrics_dict['baseline_metrics']
  processed_metrics_dict = collections.OrderedDict()
  baseline_valid_accuracies_mean = np.mean(
      baseline_metrics['valid'][accuracy_name])
  baseline_test_accuracies_mean = np.mean(
      baseline_metrics['test'][accuracy_name])
  processed_metrics_dict['baseline_metrics'] = collections.OrderedDict(
      valid_accuracies_mean=baseline_valid_accuracies_mean,
      valid_num_examples_mean=np.mean(baseline_metrics['valid'][_NUM_EXAMPLES]),
      test_accuracies_mean=baseline_test_accuracies_mean,
      test_num_examples_mean=np.mean(baseline_metrics['test'][_NUM_EXAMPLES]))
  p13n_metrics = metrics_dict[personalize_fn_name]
  num_epochs = int(len(p13n_metrics) / 2)
  num_clients = len(baseline_metrics['valid'][accuracy_name])
  # Find the best epoch number using validation metrics.
  best_epoch = 0
  best_valid_accuracies_mean = baseline_valid_accuracies_mean
  for idx in range(1, num_epochs + 1):
    current_valid_accuracies = p13n_metrics[f'epoch_{idx}_valid'][accuracy_name]
    current_valid_accuracies_mean = np.mean(current_valid_accuracies)
    if current_valid_accuracies_mean > best_valid_accuracies_mean:
      best_epoch = idx
      best_valid_accuracies_mean = current_valid_accuracies_mean
  # Obtain the test accuracies at the best epoch number.
  test_accuracies_at_best_epoch_mean = baseline_test_accuracies_mean
  if best_epoch > 0:
    test_accuracies_at_best_epoch_mean = np.mean(
        p13n_metrics[f'epoch_{best_epoch}_test'][accuracy_name])
  # Get the number of clients whose test accuracy hurts after p13n.
  num_clients_hurt_after_p13n = 0
  if best_epoch > 0:
    for client_i in range(num_clients):
      p13n_test_accuracy = p13n_metrics[f'epoch_{best_epoch}_test'][
          accuracy_name][client_i]
      baseline_test_accuracy = baseline_metrics['test'][accuracy_name][client_i]
      if baseline_test_accuracy > p13n_test_accuracy:
        num_clients_hurt_after_p13n += 1
  fraction_clients_hurt = num_clients_hurt_after_p13n / float(num_clients)
  processed_metrics_dict[personalize_fn_name] = collections.OrderedDict(
      best_epoch=best_epoch,
      valid_accuracies_at_best_epoch_mean=best_valid_accuracies_mean,
      test_accuracies_at_best_epoch_mean=test_accuracies_at_best_epoch_mean,
      fraction_clients_hurt_at_best_epoch=fraction_clients_hurt)
  return processed_metrics_dict
