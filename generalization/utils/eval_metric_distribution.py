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
import functools
import itertools
from typing import Callable, Mapping, Sequence

import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_probability as tfp

INPUT_TENSOR_SPEC = tf.TensorSpec(shape=([None]), dtype=tf.float32)
WEIGHTS_TENSOR_SPEC = tf.TensorSpec(shape=([None]), dtype=tf.float32)
StatFnType = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

CLIENTS_PER_CHUNK = 50


@tf.function
def compute_metrics(model: tff.learning.Model,
                    eval_weights: tff.learning.ModelWeights,
                    metrics: Sequence[tf.keras.metrics.Metric],
                    dataset: tf.data.Dataset):
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


def create_federated_eval_distribution_computation(
    model_fn: Callable[[], tff.learning.Model],
    metrics_builder: Callable[[], Sequence[tf.keras.metrics.Metric]]
) -> tff.Computation:
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
    `model_fn()`, and a federated dataset. This computation returns
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
  def fed_eval(model_weights, federated_dataset):
    """Computes client metrics across all clients and collects them."""
    client_model = tff.federated_broadcast(model_weights)
    return tff.federated_map(compute_client_metrics,
                             (client_model, federated_dataset))

  return fed_eval


def create_federated_eval_distribution_fn(
    model_fn: Callable[[], tff.learning.Model],
    metrics_builder: Callable[[], Sequence[tf.keras.metrics.Metric]],
    stat_fns: Mapping[str, StatFnType]):
  """Compute custom statistics across client metrics.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must not capture TensorFlow tensors or variables and use them.
    metrics_builder: A no-arg function that returns a sequence of
      `tf.keras.metrics.Metric` objects. These metrics must have a callable
      `update_state` accepting `y_true` and `y_pred` arguments, corresponding to
      the true and predicted label, respectively.
    stat_fns: A mapping in which each key-value pair represents a custom
      statistic to be evaluated on the client metrics. Each pair consists of a
      string-typed key describing this statistic, and a callable-typed value
      that computes the statistic of metrics. The callable value should accept
      two sequence-typed arguments `all_clients_this_metric` and
      `all_clients_num_examples` and returns the corresponding statistics.

  Returns:
    A callable that accepts a `tff.learning.ModelWeights`
    structure placed at `SERVER` matching the model structure of
    `model_fn()`, and a federated dataset. This computation returns
    an OrderedDict of statistics of metrics computed based on stat_fns.
  """
  fed_eval = create_federated_eval_distribution_computation(
      model_fn, metrics_builder)

  def eval_metric_distribution(model_weights: tff.learning.ModelWeights,
                               federated_dataset):
    take = lambda n, iterable: list(itertools.islice(iterable, n))
    chunked_federated_dataset = iter(
        functools.partial(take, CLIENTS_PER_CHUNK, iter(federated_dataset)), [])

    map_fn = lambda fds: fed_eval(model_weights, fds)
    chunked_all_clients_all_metrics = map(map_fn, chunked_federated_dataset)
    all_clients_all_metrics = list(
        itertools.chain.from_iterable(chunked_all_clients_all_metrics))

    all_clients_num_examples = [
        one_client_all_metrics['num_examples']
        for one_client_all_metrics in all_clients_all_metrics
    ]
    all_clients_num_examples = tf.convert_to_tensor(
        all_clients_num_examples, dtype=tf.float32)

    metric_names = all_clients_all_metrics[0].keys()
    distribution_metrics = collections.OrderedDict()

    for metric_name in metric_names:
      if metric_name == 'num_examples':
        continue
      all_clients_this_metric = [
          one_client_all_metrics[metric_name]
          for one_client_all_metrics in all_clients_all_metrics
      ]

      for stat_name, stat_fn in stat_fns.items():
        distribution_metrics[metric_name + f'/{stat_name}'] = stat_fn(
            tf.convert_to_tensor(all_clients_this_metric, dtype=tf.float32),
            all_clients_num_examples).numpy()

    return distribution_metrics

  return eval_metric_distribution


@tf.function(input_signature=[INPUT_TENSOR_SPEC, WEIGHTS_TENSOR_SPEC])
def unweighted_avg(input_tensor, weights_tensor):
  """Compute the unweighted averaging of a given tensor."""
  del weights_tensor
  return tf.reduce_mean(input_tensor)


@tf.function(input_signature=[INPUT_TENSOR_SPEC, WEIGHTS_TENSOR_SPEC])
def weighted_avg(input_tensor, weights_tensor):
  """Compute the weighted averaging of a given tensor."""
  result_tensor = tf.reduce_sum(
      input_tensor * weights_tensor) / tf.reduce_sum(weights_tensor)
  return result_tensor


@tf.function(input_signature=[INPUT_TENSOR_SPEC, WEIGHTS_TENSOR_SPEC])
def unweighted_std(input_tensor, weights_tensor):
  """Compute the (unweighted) population variance of a given tensor."""
  del weights_tensor
  return tf.math.reduce_std(input_tensor)


@tf.function(input_signature=[INPUT_TENSOR_SPEC, WEIGHTS_TENSOR_SPEC])
def unweighted_var(input_tensor, weights_tensor):
  """Compute the unweighted population standard deviation of a given tensor."""
  del weights_tensor
  return tf.math.reduce_variance(input_tensor)


@tf.function(input_signature=[INPUT_TENSOR_SPEC, WEIGHTS_TENSOR_SPEC])
def pct95(input_tensor, weights_tensor):
  """Compute the 95th percentile of a given tensor."""
  del weights_tensor
  return tfp.stats.percentile(input_tensor, 95)


@tf.function(input_signature=[INPUT_TENSOR_SPEC, WEIGHTS_TENSOR_SPEC])
def pct75(input_tensor, weights_tensor):
  """Compute the 75th percentile of a given tensor."""
  del weights_tensor
  return tfp.stats.percentile(input_tensor, 75)


@tf.function(input_signature=[INPUT_TENSOR_SPEC, WEIGHTS_TENSOR_SPEC])
def median(input_tensor, weights_tensor):
  """Compute the 50th percentile of a given tensor."""
  del weights_tensor
  return tfp.stats.percentile(input_tensor, 50)


@tf.function(input_signature=[INPUT_TENSOR_SPEC, WEIGHTS_TENSOR_SPEC])
def pct25(input_tensor, weights_tensor):
  """Compute the 25th percentile of a given tensor."""
  del weights_tensor
  return tfp.stats.percentile(input_tensor, 25)


@tf.function(input_signature=[INPUT_TENSOR_SPEC, WEIGHTS_TENSOR_SPEC])
def pct5(input_tensor, weights_tensor):
  """Compute the 5th percentile of a given tensor."""
  del weights_tensor
  return tfp.stats.percentile(input_tensor, 5)


ALL_STAT_FNS = {
    'avg': unweighted_avg,
    'wavg': weighted_avg,
    'var': unweighted_var,
    'med': median,
    'pct95': pct95,
    'pct75': pct75,
    'pct25': pct25,
    'pct5': pct5
}
