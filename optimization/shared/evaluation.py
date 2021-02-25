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
"""Builds evaluation functions with user-specified Keras metrics."""

import collections
import enum
from typing import Callable, Sequence

import tensorflow as tf
import tensorflow_federated as tff


class AggregationMethods(enum.Enum):
  EXAMPLE_WEIGHTED = 'example_weighted'
  UNIFORM_WEIGHTED = 'uniform_weighted'


@tf.function
def compute_metrics(model, model_weights, metrics, dataset):
  """Computes metrics for a given model, model weights, and dataset.

  The model must be a `tff.learning.Model` with a single output model
  prediction. In particular, the output of `model.forward_pass(...)` must have
  an attribute `predictions` with shape matching that of the true labels
  in `dataset`.

  Args:
    model: A `tff.learning.Model` used for evaluation.
    model_weights: A `tff.learning.ModelWeights` that can be assigned to the
      model weights of `model`.
    metrics: A sequence of `tf.keras.metrics.Metric` objects.
    dataset: A `tf.data.Dataset` whose batches match the expected structure of
      `model.forward_pass`.

  Returns:
    A `collections.OrderedDict` of metrics values computed for the given model
    at the given model weights over the input dataset.
  """
  initial_weights = tff.learning.ModelWeights.from_model(model)
  tff.utils.assign(initial_weights, model_weights)

  num_examples = tf.constant(0, dtype=tf.int32)
  for batch in dataset:
    if hasattr(batch, '_asdict'):
      batch = batch._asdict()

    output = model.forward_pass(batch, training=False)
    y_pred = output.predictions

    if isinstance(batch, collections.abc.Mapping):
      y_true = batch.get('y')
    else:
      y_true = batch[1]

    for metric in metrics:
      metric.update_state(y_true, y_pred)
    num_examples += tf.shape(y_true)[0]

  metric_results = collections.OrderedDict()
  for metric in metrics:
    metric_results[metric.name] = tf.cast(metric.result(), dtype=tf.float32)
  metric_results['num_examples'] = tf.cast(num_examples, dtype=tf.float32)

  return metric_results


def build_federated_evaluation(
    model_fn: Callable[[], tff.learning.Model],
    metrics_builder: Callable[[], Sequence[tf.keras.metrics.Metric]]
) -> tff.federated_computation:
  """Builds a federated evaluation `tff.federated_computation`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    metrics_builder: A no-arg function that returns a sequence of
      `tf.keras.metrics.Metric` objects. These metrics must have a callable
      `update_state` accepting `y_true` and `y_pred` arguments, corresponding to
      the true and predicted label, respectively.

  Returns:
    A `tff.federated_computation` that accepts model weights and federated data,
    and returns the evaluation metrics, aggregated in both uniform- and
    example-weighted manners.
  """
  # Wrap model construction in a graph to avoid polluting the global context
  # with variables created for this model.
  with tf.Graph().as_default():
    placeholder_model = model_fn()
    model_weights_type = tff.learning.framework.weights_type_from_model(
        placeholder_model)
    model_input_type = tff.SequenceType(placeholder_model.input_spec)

  @tff.tf_computation(model_weights_type, model_input_type)
  def compute_client_metrics(model_weights, federated_dataset):
    model = model_fn()
    metrics = metrics_builder()
    return compute_metrics(model, model_weights, metrics, federated_dataset)

  @tff.federated_computation(
      tff.type_at_server(model_weights_type),
      tff.type_at_clients(model_input_type))
  def federated_evaluate(model_weights, federated_dataset):
    client_model = tff.federated_broadcast(model_weights)
    client_metrics = tff.federated_map(compute_client_metrics,
                                       (client_model, federated_dataset))
    # Extract the number of examples in order to compute client weights
    num_examples = client_metrics.num_examples
    uniform_weighted_metrics = tff.federated_mean(client_metrics, weight=None)
    example_weighted_metrics = tff.federated_mean(
        client_metrics, weight=num_examples)
    # Aggregate the metrics in a single nested dictionary
    aggregate_metrics = collections.OrderedDict()
    aggregate_metrics[
        AggregationMethods.EXAMPLE_WEIGHTED.value] = example_weighted_metrics
    aggregate_metrics[
        AggregationMethods.UNIFORM_WEIGHTED.value] = uniform_weighted_metrics

    return aggregate_metrics

  return federated_evaluate


def build_centralized_evaluation(
    model_fn: Callable[[], tff.learning.Model],
    metrics_builder: Callable[[], Sequence[tf.keras.metrics.Metric]]
) -> tff.federated_computation:
  """Builds a centralized evaluation `tff.federated_computation`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    metrics_builder: A no-arg function that returns a sequence of
      `tf.keras.metrics.Metric` objects. These metrics must have a callable
      `update_state` accepting `y_true` and `y_pred` arguments, corresponding to
      the true and predicted label, respectively.

  Returns:
    A `tff.federated_computation` that accepts model weights and centralized
    data, and returns the evaluation metrics.
  """
  # Wrap model construction in a graph to avoid polluting the global context
  # with variables created for this model.
  with tf.Graph().as_default():
    placeholder_model = model_fn()
    model_weights_type = tff.learning.framework.weights_type_from_model(
        placeholder_model)
    model_input_type = tff.SequenceType(placeholder_model.input_spec)

  @tff.tf_computation(model_weights_type, model_input_type)
  def compute_server_metrics(model_weights, centralized_dataset):
    model = model_fn()
    metrics = metrics_builder()
    return compute_metrics(model, model_weights, metrics, centralized_dataset)

  @tff.federated_computation(
      tff.type_at_server(model_weights_type),
      tff.type_at_server(model_input_type))
  def centralized_evaluate(model_weights, centralized_dataset):
    return tff.federated_map(compute_server_metrics,
                             (model_weights, centralized_dataset))

  return centralized_evaluate
