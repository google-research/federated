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
"""Libraries for computing distributions of metrics across clients."""

import collections
from typing import Callable, Sequence

import tensorflow as tf
import tensorflow_federated as tff

from large_cohort import simulation_specs


def create_centralized_eval_fn(model_spec: simulation_specs.ModelSpec):
  """Creates a function that computes metrics for model weights and datasets."""

  def centralized_eval_fn(model_weights: tff.learning.ModelWeights,
                          dataset: tf.data.Dataset):
    keras_model = model_spec.keras_model_builder()
    keras_model.compile(
        loss=model_spec.loss_builder(), metrics=model_spec.metrics_builder())
    model_weights.assign_weights_to(keras_model)
    metrics = keras_model.evaluate(dataset, return_dict=True)
    return collections.OrderedDict(metrics)

  return centralized_eval_fn


@tf.function
def compute_metrics(model, eval_weights, metrics, dataset):
  """Computes metrics for a given model, model weights, and dataset.

  The model must be a `tff.learning.Model` with a single output model
  prediction. In particular, the output of `model.forward_pass(...)` must have
  an attribute `predictions` with shape matching that of the true labels
  in `dataset`.

  Args:
    model: A `tff.learning.Model` used for evaluation.
    eval_weights: A `tff.learning.ModelWeights` that can be assigned to the
      model weights of `model`. These weights are used for evaluation.
    metrics: A sequence of `tf.keras.metrics.Metric` objects.
    dataset: A `tf.data.Dataset` whose batches match the expected structure of
      `model.forward_pass`.

  Returns:
    A `collections.OrderedDict` of metrics values computed for the given model
    at the given model weights over the input dataset.
  """
  model_weights = tff.learning.ModelWeights.from_model(model)
  tf.nest.map_structure(lambda x, y: x.assign(y), model_weights, eval_weights)

  num_examples = tf.constant(0, dtype=tf.int32)
  for batch in dataset:
    if hasattr(batch, '_asdict'):
      batch = batch._asdict()

    output = model.forward_pass(batch, training=False)
    y_pred = output.predictions

    # TODO(b/187941327): Avoid having to branch here once we are confident that
    # we are only passing in datasets that yield tuples.
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


def create_federated_eval_fn(
    model_fn: Callable[[], tff.learning.Model],
    metrics_builder: Callable[[], Sequence[tf.keras.metrics.Metric]]):
  """Builds a TFF computation for computing distributions of client metrics.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must not capture TensorFlow tensors or variables and use them.
    metrics_builder: A no-arg function that returns a sequence of
      `tf.keras.metrics.Metric` objects. These metrics must have a callable
      `update_state` accepting `y_true` and `y_pred` arguments, corresponding to
      the true and predicted label, respectively.

  Returns:
    A `tff.federated_computation` that accepts a `tff.learning.ModelWeights`
    structure placed at `SERVER` matching the model structure of
    `keras_model_builder()`, and a federated dataset. This computation returns
    a sequence of evaluation metrics computed over all clients.
  """
  # Wrap model construction in a graph to avoid polluting the global context
  # with variables created for this model.
  with tf.Graph().as_default():
    placeholder_model = model_fn()
    model_weights_type = tff.learning.framework.weights_type_from_model(
        placeholder_model)
    model_input_type = tff.SequenceType(placeholder_model.input_spec)

  @tff.tf_computation(model_weights_type, model_input_type)
  def compute_client_metrics(model_weights, dataset):
    model = model_fn()
    metrics = metrics_builder()
    return compute_metrics(model, model_weights, metrics, dataset)

  @tff.federated_computation(
      tff.type_at_server(model_weights_type),
      tff.type_at_clients(model_input_type))
  def federated_evaluate(model_weights, federated_dataset):
    """Computes client metrics across all clients and collects them."""
    client_model = tff.federated_broadcast(model_weights)
    return tff.federated_map(compute_client_metrics,
                             (client_model, federated_dataset))

  return federated_evaluate
