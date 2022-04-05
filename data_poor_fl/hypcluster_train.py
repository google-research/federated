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
"""An implementation of the HypCluster training algorithm.

The HypCluster algorithm was proposed in the paper:
Three Approaches for Personalization with Applications to Federated Learning
  Yishay Mansour, Mehryar Mohri, Jae Ro, Ananda Theertha Suresh
  https://arxiv.org/abs/2002.10619
"""

import collections
from typing import Callable, List, Optional

import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import hypcluster_utils


@tf.function
def client_update_tf(model, optimizer, initial_weights, data):
  """Client training loop."""
  model_weights = tff.learning.ModelWeights.from_model(model)
  tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                        initial_weights)

  def reduce_fn(state, batch):
    """Trains a `tff.learning.Model` on a batch of data."""
    num_examples_sum, optimizer_state = state
    with tf.GradientTape() as tape:
      output = model.forward_pass(batch, training=True)

    gradients = tape.gradient(output.loss, model_weights.trainable)
    optimizer_state, updated_weights = optimizer.next(
        optimizer_state, tuple(tf.nest.flatten(model_weights.trainable)),
        tuple(tf.nest.flatten(gradients)))
    updated_weights = tf.nest.pack_sequence_as(model_weights.trainable,
                                               updated_weights)
    tf.nest.map_structure(lambda a, b: a.assign(b), model_weights.trainable,
                          updated_weights)
    num_examples_sum += tf.cast(output.num_examples, tf.int64)
    return num_examples_sum, optimizer_state

  trainable_tensor_specs = tf.nest.map_structure(
      lambda v: tf.TensorSpec(v.shape, v.dtype),
      tuple(tf.nest.flatten(model_weights.trainable)))
  reduce_initial_state = (tf.zeros(shape=[], dtype=tf.int64),
                          optimizer.initialize(trainable_tensor_specs))
  num_examples, _ = data.reduce(reduce_initial_state, reduce_fn)
  num_examples = tf.cast(num_examples, tf.float32)
  client_update = tf.nest.map_structure(tf.subtract, initial_weights.trainable,
                                        model_weights.trainable)
  model_output = model.report_local_unfinalized_metrics()
  return client_update, num_examples, model_output


def build_hypcluster_client_work(model_fn: Callable[[], tff.learning.Model],
                                 optimizer: tff.learning.optimizers.Optimizer,
                                 model_weights_type: tff.types.Type,
                                 num_clusters: int):
  """Builds a client work process for HypCluster."""

  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
    unfinalized_metrics_type = tff.types.type_from_tensors(
        model.report_local_unfinalized_metrics())
    metrics_aggregation_fn = tff.learning.metrics.sum_then_finalize(
        model.metric_finalizers(), unfinalized_metrics_type)

  data_type = tff.types.SequenceType(model.input_spec)
  weights_gather_fn = hypcluster_utils.build_gather_fn(model_weights_type,
                                                       num_clusters)
  list_weights_type = tff.StructWithPythonType(
      [model_weights_type for _ in range(num_clusters)], container_type=list)

  cluster_selection_fn = hypcluster_utils.build_find_best_model_fn(
      model_fn, num_clusters, model_weights_type, data_type)
  pack_update_and_weight_fn = hypcluster_utils.build_scatter_fn(
      model_weights_type.trainable, num_clusters)

  @tff.tf_computation(list_weights_type, data_type)
  def select_model_and_train(weights, dataset):
    # Determine which cluster to select from
    model_index = cluster_selection_fn(weights, dataset)
    # Gather the relevant weights and train
    train_model = model_fn()
    train_model_weights = weights_gather_fn(weights, model_index)
    client_update, num_examples, train_model_output = client_update_tf(
        train_model, optimizer, train_model_weights, dataset)

    one_hot_update, one_hot_weight = pack_update_and_weight_fn(
        client_update, model_index, num_examples)
    client_result = tff.learning.templates.ClientResult(
        update=one_hot_update, update_weight=one_hot_weight)
    return client_result, train_model_output

  @tff.federated_computation
  def init_fn():
    return tff.federated_value((), tff.SERVER)

  @tff.federated_computation(init_fn.type_signature.result,
                             tff.type_at_clients(list_weights_type),
                             tff.type_at_clients(data_type))
  def next_fn(state, list_of_weights, client_data):
    client_result, train_model_outputs = tff.federated_map(
        select_model_and_train, (list_of_weights, client_data))
    train_metrics = metrics_aggregation_fn(train_model_outputs)
    measurements = tff.federated_zip(
        collections.OrderedDict(train=train_metrics))
    return tff.templates.MeasuredProcessOutput(state, client_result,
                                               measurements)

  return tff.learning.templates.ClientWorkProcess(init_fn, next_fn)


DEFAULT_SERVER_OPTIMIZER = tff.learning.optimizers.build_sgdm(learning_rate=1.0)


def build_hypcluster_train(
    model_fn: Callable[[], tff.learning.Model],
    num_clusters: int,
    client_optimizer: tff.learning.optimizers.Optimizer,
    server_optimizer: tff.learning.optimizers
    .Optimizer = DEFAULT_SERVER_OPTIMIZER,
    model_aggregator: Optional[
        tff.aggregators.WeightedAggregationFactory] = None,
    initial_model_weights_list: Optional[List[tff.learning.ModelWeights]] = None
) -> tff.learning.templates.LearningProcess:
  """Builds a learning process that performs HypCluster training.

  This function creates a `tff.learning.templates.LearningProcess` with the
  following methods:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` representing the initial
      state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `{B*}@CLIENTS` represents the client datasets.
      The output `L` contains the updated server state, as well as aggregated
      metrics at the server, including client training metrics and any other
      metrics from distribution and aggregation processes.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matchs the output of `initialize` and `next` and `M` represents the type
      of the model weights. Note that `M` will be a list of length
      `num_clusters` of `tff.learning.ModelWeights` objects, corresponding to
      each cluster's model.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    num_clusters: An integer specifying the number of clusters to use.
    client_optimizer: A `tff.learning.optimizers.Optimizer`.
    server_optimizer: A `tff.learning.optimizers.Optimizer`. Defaults to
      `tff.learning.optimizers.Optimizer.build_sgdm(learning_rate=1.0)`.
    model_aggregator: An optional `tff.aggregators.WeightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    initial_model_weights_list: If provided, used as the initialization.

  Returns:
    A `tff.learning.templates.LearningProcess`.

  Raises:
    ValueError: If the number of models provided by `initial_model_weights_list`
      does not equal `num_clusters`.
  """

  @tff.tf_computation()
  def initial_weights_fn():
    if initial_model_weights_list:
      if len(initial_model_weights_list) != num_clusters:
        raise ValueError(
            'The `initial_model_weights_list` does not equal `num_clusters`: '
            f'the number of provided models: {len(initial_model_weights_list)} '
            f'while the number of desired clusters: {num_clusters}.')
      return initial_model_weights_list
    return [
        tff.learning.ModelWeights.from_model(model_fn())
        for _ in range(num_clusters)
    ]

  weights_list_type = initial_weights_fn.type_signature.result
  model_weights_type = weights_list_type[0]
  trainable_type = model_weights_type.trainable

  weights_distributor = tff.learning.templates.build_broadcast_process(
      weights_list_type)

  if model_aggregator is None:
    model_aggregator = tff.aggregators.MeanFactory()
  base_aggregator = model_aggregator.create(trainable_type,
                                            tff.types.TensorType(tf.float32))
  aggregator = hypcluster_utils.build_coordinate_aggregator(
      base_aggregator, num_coordinates=num_clusters)

  client_work = build_hypcluster_client_work(
      model_fn=model_fn,
      optimizer=client_optimizer,
      model_weights_type=model_weights_type,
      num_clusters=num_clusters)

  base_finalizer = tff.learning.templates.build_apply_optimizer_finalizer(
      optimizer_fn=server_optimizer, model_weights_type=model_weights_type)
  finalizer = hypcluster_utils.build_coordinate_finalizer(
      base_finalizer, num_coordinates=num_clusters)

  return tff.learning.templates.compose_learning_process(
      initial_weights_fn, weights_distributor, client_work, aggregator,
      finalizer)
