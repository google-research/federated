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
"""Shared library for setting up federated training experiments."""

import collections
import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


#  Settings for a multiplicative linear congruential generator (aka Lehmer
#  generator) suggested in 'Random Number Generators: Good
#  Ones are Hard to Find' by Park and Miller.
MLCG_MODULUS = 2**(31) - 1
MLCG_MULTIPLIER = 16807

# Default quantiles for federated evaluations.
DEFAULT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


# TODO(b/143440780): Create more comprehensive tuple conversion by adding an
# explicit namedtuple checking utility.
def convert_to_tuple_dataset(dataset):
  """Converts a dataset to one where elements have a tuple structure.

  Args:
    dataset: A `tf.data.Dataset` where elements either have a mapping
      structure of format {"x": <features>, "y": <labels>}, or a tuple-like
        structure of format (<features>, <labels>).

  Returns:
    A `tf.data.Dataset` object where elements are tuples of the form
    (<features>, <labels>).

  """
  example_structure = dataset.element_spec
  if isinstance(example_structure, collections.Mapping):
    # We assume the mapping has `x` and `y` keys.
    convert_map_to_tuple = lambda example: (example['x'], example['y'])
    try:
      return dataset.map(convert_map_to_tuple)
    except:
      raise ValueError('For datasets with a mapping structure, elements must '
                       'have format {"x": <features>, "y": <labels>}.')
  elif isinstance(example_structure, tuple):

    if hasattr(example_structure, '_fields') and isinstance(
        example_structure._fields, collections.Sequence) and all(
            isinstance(f, str) for f in example_structure._fields):
      # Dataset has namedtuple structure
      convert_tuplelike_to_tuple = lambda x: (x[0], x[1])
    else:
      # Dataset does not have namedtuple structure
      convert_tuplelike_to_tuple = lambda x, y: (x, y)

    try:
      return dataset.map(convert_tuplelike_to_tuple)
    except:
      raise ValueError('For datasets with tuple-like structure, elements must '
                       'have format (<features>, <labels>).')
  else:
    raise ValueError(
        'Expected evaluation dataset to have elements with a mapping or '
        'tuple-like structure, found {} instead.'.format(example_structure))


def build_centralized_evaluate_fn(
    eval_dataset: tf.data.Dataset, model_builder: Callable[[], tf.keras.Model],
    loss_builder: Callable[[], tf.keras.losses.Loss],
    metrics_builder: Callable[[], List[tf.keras.metrics.Metric]]
) -> Callable[[tff.learning.ModelWeights], Dict[str, Any]]:
  """Builds a centralized evaluation function for a model and test dataset.

  The evaluation function takes as input a `tff.learning.ModelWeights`, and
  computes metrics on a keras model with the same weights.

  Args:
    eval_dataset: A `tf.data.Dataset` object. Dataset elements should either
      have a mapping structure of format {"x": <features>, "y": <labels>}, or a
        tuple structure of format (<features>, <labels>).
    model_builder: A no-arg function that returns a `tf.keras.Model` object.
    loss_builder: A no-arg function returning a `tf.keras.losses.Loss` object.
    metrics_builder: A no-arg function that returns a list of
      `tf.keras.metrics.Metric` objects.

  Returns:
    A function that take as input a `tff.learning.ModelWeights` and returns
    a dict of (name, value) pairs for each associated evaluation metric.
  """

  def compiled_eval_keras_model():
    model = model_builder()
    model.compile(
        loss=loss_builder(),
        optimizer=tf.keras.optimizers.SGD(),  # Dummy optimizer for evaluation
        metrics=metrics_builder())
    return model

  eval_tuple_dataset = convert_to_tuple_dataset(eval_dataset)

  def evaluate_fn(reference_model: tff.learning.ModelWeights) -> Dict[str, Any]:
    """Evaluation function to be used during training."""

    if not isinstance(reference_model, tff.learning.ModelWeights):
      raise TypeError('The reference model used for evaluation must be a'
                      '`tff.learning.ModelWeights` instance.')

    model_weights_as_list = tff.learning.ModelWeights(
        trainable=list(reference_model.trainable),
        non_trainable=list(reference_model.non_trainable))

    keras_model = compiled_eval_keras_model()
    model_weights_as_list.assign_weights_to(keras_model)
    logging.info('Evaluating the current model')
    eval_metrics = keras_model.evaluate(eval_tuple_dataset, verbose=0)
    return dict(zip(keras_model.metrics_names, eval_metrics))

  return evaluate_fn


def _build_client_evaluate_fn(model, mean_metrics, sum_metrics):
  """Creates a client evaluation function for a given model and metrics.

  This evaluation function is intended to be re-used for multiple clients during
  an evaluation pass over clients.

  Args:
    model: A `tf.keras.Model` used to compute metrics.
    mean_metrics: A list of `tf.keras.metrics.Mean` metrics.
    sum_metrics: A list of `tf.keras.metrics.Sum` metrics.

  Returns:
    A function that takes as input a `tf.data.Dataset`, and returns an ordered
      dictionary of metrics for the input model on that dataset.
  """

  @tf.function
  def get_client_eval_metrics(dataset):

    # Reset metrics
    for metric in mean_metrics:
      metric.reset_states()
    for metric in sum_metrics:
      metric.reset_states()

    # Compute metrics
    num_examples = tf.constant(0, dtype=tf.int32)
    for x_batch, y_batch in dataset:
      output = model(x_batch, training=False)
      for metric in mean_metrics:
        metric.update_state(y_batch, output)
      for metric in sum_metrics:
        metric.update_state(output)
      num_examples += tf.shape(output)[0]

    # Record metrics
    mean_metric_results = collections.OrderedDict()
    for metric in mean_metrics:
      mean_metric_results[metric.name] = metric.result()

    sum_metric_results = collections.OrderedDict()
    for metric in sum_metrics:
      sum_metric_results[metric.name] = metric.result()
    sum_metric_results['num_examples'] = num_examples

    return mean_metric_results, sum_metric_results

  return get_client_eval_metrics


def build_federated_evaluate_fn(
    eval_dataset: tff.simulation.ClientData,
    model_builder: Callable[[], tf.keras.Model],
    metrics_builder: Callable[[], List[tf.keras.metrics.Metric]],
    clients_per_round: int,
    random_seed: Optional[int] = None,
    quantiles: Optional[Iterable[float]] = DEFAULT_QUANTILES,
) -> Callable[[tff.learning.ModelWeights, int], Dict[str, Any]]:
  """Builds a federated evaluation method for a given model and test dataset.

  The evaluation function takes as input a `tff.learning.ModelWeights`, and
  computes metrics on a `tff.learning.Model` with the same weights. This method
  returns a nested structure of (metric_name, metric_value) pairs. For mean-
  based metrics (such as mean squared error), we compute both example-weighted
  and uniform-weighted versions of the metric, while for sum-based metrics
  (such as the number of examples in a client dataset) we compute the sum.

  The resulting nested structure is an ordered dictionary with keys given by
  metric names, and values given by nested structure of the metric value,
  potentially aggregated in different ways.

  For example, if `metrics_builder = lambda:
  [tf.keras.metrics.MeanSquaredError()]`,
  the resulting nested structure will be of the form:
  [
  ('mean_squared_error', [
    ('example_weighted', ...),
    ('uniform_weighted', ...),
    ('quantiles', ...)
  ]),
  ('num_examples', [
    ('summed', ...),
    ('uniform_weighted', ...),
    ('quantiles', ...)
  ]),
  ]

  Args:
    eval_dataset: A `tf.data.Dataset` object. Dataset elements should either
      have a mapping structure of format {"x": <features>, "y": <labels>}, or a
        tuple structure of format (<features>, <labels>).
    model_builder: A no-arg function returning an uncompiled `tf.keras.Model`.
    metrics_builder: A no-arg function that returns a list of
      `tf.keras.metrics.Metric` objects. These metrics must either be instances
      of `tf.keras.metrics.Mean` or `tf.keras.metrics.Sum`.
    clients_per_round: An integer specifying the number of clients to sample
      when performing evaluation.
    random_seed: An integer used to seed the evaluation client selection.
    quantiles: Which quantiles to compute of mean-based metrics. Must be an
      iterable of float values between 0 and 1.

  Returns:
    A function that take as input a `tff.learning.ModelWeights` and a round
    number, and returns a nested structure of (metric_name, metric_value) pairs.
  """

  client_sample_function = build_client_datasets_fn(
      eval_dataset, clients_per_round, random_seed=random_seed)

  keras_model = model_builder()
  metrics = metrics_builder()

  mean_metrics = []
  sum_metrics = []

  for keras_metric in metrics:
    if isinstance(keras_metric, tf.keras.metrics.Mean):
      mean_metrics.append(keras_metric)
    elif isinstance(keras_metric, tf.keras.metrics.Sum):
      sum_metrics.append(keras_metric)
    else:
      raise ValueError('Unsupported metric {}, metrics must be an instance of '
                       'tf.keras.metrics.Mean or tf.keras.metrics.Sum.')

  def evaluate_fn(model_weights: tff.learning.ModelWeights,
                  round_num: int) -> Dict[str, Any]:

    model_weights_as_list = tff.learning.ModelWeights(
        trainable=list(model_weights.trainable),
        non_trainable=list(model_weights.non_trainable))
    model_weights_as_list.assign_weights_to(keras_model)

    client_eval_fn = _build_client_evaluate_fn(keras_model, mean_metrics,
                                               sum_metrics)

    mean_metrics_at_clients = collections.defaultdict(list)
    sum_metrics_at_clients = collections.defaultdict(list)

    # Record all client metrics
    for client_dataset in client_sample_function(round_num):
      tuple_ds = convert_to_tuple_dataset(client_dataset)
      client_mean_metrics, client_sum_metrics = client_eval_fn(tuple_ds)

      for (metric_name, metric_value) in client_mean_metrics.items():
        mean_metrics_at_clients[metric_name].append(metric_value)
      for (metric_name, metric_value) in client_sum_metrics.items():
        sum_metrics_at_clients[metric_name].append(metric_value)

    # Aggregate metrics across clients
    aggregate_metrics = collections.OrderedDict()

    num_examples_at_clients = tf.cast(
        sum_metrics_at_clients['num_examples'], dtype=tf.float32)
    total_num_examples = tf.reduce_sum(num_examples_at_clients)

    for (metric_name, metric_at_clients) in mean_metrics_at_clients.items():
      metric_as_float = tf.cast(metric_at_clients, dtype=tf.float32)

      uniform_weighted_value = tf.reduce_mean(metric_as_float).numpy()
      example_weighted_value = (tf.reduce_sum(
          tf.math.multiply(metric_as_float, num_examples_at_clients)) /
                                total_num_examples).numpy()

      quantile_values = np.quantile(metric_as_float, quantiles)
      quantile_values = collections.OrderedDict(zip(quantiles, quantile_values))

      aggregate_metrics[metric_name] = collections.OrderedDict(
          example_weighted=example_weighted_value,
          uniform_weighted=uniform_weighted_value,
          quantiles=quantile_values)

    for (metric_name, metric_at_clients) in sum_metrics_at_clients.items():
      summed_value = tf.reduce_sum(metric_at_clients).numpy()
      metric_as_float = tf.cast(metric_at_clients, dtype=tf.float32)
      uniform_weighted_value = tf.reduce_mean(metric_as_float).numpy()
      quantile_values = np.quantile(metric_as_float, quantiles)
      quantile_values = collections.OrderedDict(zip(quantiles, quantile_values))

      aggregate_metrics[metric_name] = collections.OrderedDict(
          summed=summed_value,
          uniform_weighted=uniform_weighted_value,
          quantiles=quantile_values)
    return aggregate_metrics

  return evaluate_fn


def build_sample_fn(
    a: Union[Sequence[Any], int],
    size: int,
    replace: bool = False,
    random_seed: Optional[int] = None) -> Callable[[int], np.ndarray]:
  """Builds the function for sampling from the input iterator at each round.

  Args:
    a: A 1-D array-like sequence or int that satisfies np.random.choice.
    size: The number of samples to return each round.
    replace: A boolean indicating whether the sampling is done with replacement
      (True) or without replacement (False).
    random_seed: If random_seed is set as an integer, then we use it as a random
      seed for which clients are sampled at each round. In this case, we set a
      random seed before sampling clients according to a multiplicative linear
      congruential generator (aka Lehmer generator, see 'The Art of Computer
      Programming, Vol. 3' by Donald Knuth for reference). This does not affect
      model initialization, shuffling, or other such aspects of the federated
      training process.

  Returns:
    A function which returns a list of elements from the input iterator at a
    given round round_num.
  """

  if isinstance(random_seed, int):
    mlcg_start = np.random.RandomState(random_seed).randint(1, MLCG_MODULUS - 1)

    def get_pseudo_random_int(round_num):
      return pow(MLCG_MULTIPLIER, round_num,
                 MLCG_MODULUS) * mlcg_start % MLCG_MODULUS

  def sample(round_num, random_seed):
    if isinstance(random_seed, int):
      random_state = np.random.RandomState(get_pseudo_random_int(round_num))
    else:
      random_state = np.random.RandomState()
    return random_state.choice(a, size=size, replace=replace)

  return functools.partial(sample, random_seed=random_seed)


def build_client_datasets_fn(
    dataset: tff.simulation.ClientData,
    clients_per_round: int,
    random_seed: Optional[int] = None
) -> Callable[[int], List[tf.data.Dataset]]:
  """Builds the function for generating client datasets at each round.

  The function samples a number of clients (without replacement within a given
  round, but with replacement across rounds) and returns their datasets.

  Args:
    dataset: A `tff.simulation.ClientData` object.
    clients_per_round: The number of client participants in each round.
    random_seed: If random_seed is set as an integer, then we use it as a random
      seed for which clients are sampled at each round. In this case, we set a
      random seed before sampling clients according to a multiplicative linear
      congruential generator (aka Lehmer generator, see 'The Art of Computer
      Programming, Vol. 3' by Donald Knuth for reference). This does not affect
      model initialization, shuffling, or other such aspects of the federated
      training process. Note that this will alter the global numpy random seed.

  Returns:
    A function which returns a list of `tf.data.Dataset` objects at a
    given round round_num.
  """
  sample_clients_fn = build_sample_fn(
      dataset.client_ids,
      size=clients_per_round,
      replace=False,
      random_seed=random_seed)

  def client_datasets(round_num):
    sampled_clients = sample_clients_fn(round_num)
    return [
        dataset.create_tf_dataset_for_client(client)
        for client in sampled_clients
    ]

  return client_datasets
