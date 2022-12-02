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
"""DP-FedEmb: Federated Averaging with partially local variables."""

import collections
from collections.abc import Callable
import functools
from typing import Any, Optional, Union

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.models import embedding_model
from dp_visual_embeddings.models import keras_utils


DEFAULT_SERVER_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)


def _choose_client_weight(has_non_finite_delta):
  if has_non_finite_delta > 0:
    return tf.constant(0.0, tf.float32)
  else:
    return tf.constant(1.0, tf.float32)  # Uniform weighting for DP.


@tf.function
def _zero_all_if_any_non_finite(structure: Any) -> tuple[Any, tf.Tensor]:
  """Zeroes out all entries in input if any are not finite.

  Args:
    structure: A structure supported by tf.nest.

  Returns:
     A tuple (input, 0) if all entries are finite or the structure is empty, or
     a tuple (zeros, 1) if any non-finite entries were found.
  """
  flat = tf.nest.flatten(structure)
  if not flat:
    return (structure, tf.constant(0))
  flat_bools = [tf.reduce_all(tf.math.is_finite(t)) for t in flat]
  all_finite = functools.reduce(tf.logical_and, flat_bools)
  if all_finite:
    return (structure, tf.constant(0))
  else:
    return (tf.nest.map_structure(tf.zeros_like, structure), tf.constant(1))


def _build_model_delta_update_with_keras_optimizer(model_fn):
  """Creates client update logic in FedAvg using a `tf.keras` optimizer.

  In contrast to using a `tff.learning.optimizers.Optimizer`, we have to
  maintain `tf.Variable`s associated with the optimizer state within the scope
  of the client work. Additionally, the client model weights are modified in
  place by using `optimizer.apply_gradients`).

  Args:
    model_fn: A no-arg callable returning a `embedding_model.Model`.

  Returns:
    A `tf.function`.
  """
  model = model_fn()

  @tf.function
  def client_update(global_optimizer, local_optimizer, initial_weights, data):
    global_weights = tff.learning.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda a, b: a.assign(b), global_weights,
                          initial_weights)

    trainable_variables = (model.trainable_variables,
                           model.client_trainable_variables)
    flat_trainable_variables = tf.nest.flatten(trainable_variables)
    for batch in iter(data):
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)

      gradients = tape.gradient(output.loss, flat_trainable_variables)
      struct_grad = tf.nest.pack_sequence_as(trainable_variables, gradients)
      global_optimizer.apply_gradients(
          zip(struct_grad[0], model.trainable_variables))
      local_optimizer.apply_gradients(
          zip(struct_grad[1], model.client_trainable_variables))

    model_delta = tf.nest.map_structure(tf.subtract, initial_weights.trainable,
                                        model.trainable_variables)
    model_output = model.report_local_unfinalized_metrics()

    # TODO(b/122071074): Consider moving this functionality into
    # tff.federated_mean?
    model_delta, has_non_finite_delta = (
        _zero_all_if_any_non_finite(model_delta))
    client_weight = _choose_client_weight(has_non_finite_delta)
    return tff.learning.templates.ClientResult(
        update=model_delta, update_weight=client_weight), model_output

  return client_update


def _build_model_delta_update_reconstruction(model_fn, reconst_iters):
  """Creates client update logic in FedAvg using a `tf.keras` optimizer.

  In contrast to using a `tff.learning.optimizers.Optimizer`, we have to
  maintain `tf.Variable`s associated with the optimizer state within the scope
  of the client work. Additionally, the client model weights are modified in
  place by using `optimizer.apply_gradients`).

  Args:
    model_fn: A no-arg callable returning a `embedding_model.Model`.
    reconst_iters: First optimize the Head of the embedding model for
      `reconst_iters` iterations before training the Backbone.

  Returns:
    A `tf.function`.
  """
  model = model_fn()

  @tf.function
  def conditional_update(global_optimizer, local_optimizer, cnt, batch):
    trainable_variables = (model.trainable_variables,
                           model.client_trainable_variables)
    flat_trainable_variables = tf.nest.flatten(trainable_variables)
    with tf.GradientTape() as tape:
      output = model.forward_pass(batch, training=True)

    gradients = tape.gradient(output.loss, flat_trainable_variables)
    struct_grad = tf.nest.pack_sequence_as(trainable_variables, gradients)
    if cnt < reconst_iters:
      local_optimizer.apply_gradients(
          zip(struct_grad[1], model.client_trainable_variables))
    else:
      global_optimizer.apply_gradients(
          zip(struct_grad[0], model.trainable_variables))

  @tf.function
  def client_update(global_optimizer, local_optimizer, initial_weights, data):
    global_weights = tff.learning.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda a, b: a.assign(b), global_weights,
                          initial_weights)

    cnt = tf.constant(0, dtype=tf.int32)
    for batch in iter(data):
      conditional_update(global_optimizer, local_optimizer, cnt, batch)
      cnt = cnt + tf.constant(1, dtype=tf.int32)

    model_delta = tf.nest.map_structure(tf.subtract, initial_weights.trainable,
                                        model.trainable_variables)
    model_output = model.report_local_unfinalized_metrics()

    # TODO(b/122071074): Consider moving this functionality into
    # tff.federated_mean?
    model_delta, has_non_finite_delta = (
        _zero_all_if_any_non_finite(model_delta))
    client_weight = _choose_client_weight(has_non_finite_delta)
    return tff.learning.templates.ClientResult(
        update=model_delta, update_weight=client_weight), model_output

  return client_update


def _build_scheduled_client_work(
    model_fn: Callable[[], embedding_model.Model],
    learning_rate_fn: Callable[[int], float],
    optimizer_fn: Callable[[float], tf.keras.optimizers.Optimizer],
    metrics_aggregator: Callable[
        [keras_utils.KerasMetricFinalizerType, tff.types.StructWithPythonType],
        tff.Computation],
    head_lr_scale: float,
    reconst_iters: Optional[int],
) -> tff.learning.templates.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for federated averaging.

  This `ClientWorkProcess` creates a state containing the current round number,
  which is incremented at each call to `ClientWorkProcess.next`. This integer
  round number is used to call `optimizer_fn(round_num)`, in order to construct
  the proper optimizer.

  Args:
    model_fn: A no-arg function that returns a `embedding_model.Model`.
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    learning_rate_fn: A callable accepting an integer round number and returning
      a float to be used as a learning rate for the optimizer. That is, the
      client work will call `optimizer_fn(learning_rate_fn(round_num))` where
      `round_num` is the integer round number.
    optimizer_fn: A callable accepting a float learning rate, and returning a
      `tff.learning.optimizers.Optimizer` or a `tf.keras.Optimizer`.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `embedding_model.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `embedding_model.Model.report_local_unfinalized_metrics()`),
      and returns a `tff.Computation` for aggregating the unfinalized metrics.
    head_lr_scale: Use head_lr_scale to scale the learning rate for updating
      the local variables (head of the embedding model).
    reconst_iters: If not `None`, first optimize the Head of the embedding model
      for `reconst_iters` iterations before training the Backbone.

  Returns:
    A `ClientWorkProcess`.
  """
  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    whimsy_model = model_fn()
    unfinalized_metrics_type = tff.framework.type_from_tensors(
        whimsy_model.report_local_unfinalized_metrics())
    metrics_aggregation_fn = metrics_aggregator(
        whimsy_model.metric_finalizers(), unfinalized_metrics_type)
  data_type = tff.SequenceType(whimsy_model.input_spec)
  weights_type = tff.learning.models.weights_type_from_model(whimsy_model)

  if reconst_iters is None:
    build_client_update_fn = _build_model_delta_update_with_keras_optimizer
  else:
    build_client_update_fn = functools.partial(
        _build_model_delta_update_reconstruction, reconst_iters=reconst_iters)

  @tff.tf_computation(weights_type, data_type, tf.int32)
  def client_update_computation(initial_model_weights, dataset, round_num):
    learning_rate = learning_rate_fn(round_num)
    global_optimizer = optimizer_fn(learning_rate)
    local_optimizer = optimizer_fn(learning_rate * head_lr_scale)
    client_update = build_client_update_fn(model_fn=model_fn)
    return client_update(global_optimizer, local_optimizer,
                         initial_model_weights, dataset)

  @tff.federated_computation
  def init_fn():
    return tff.federated_value(0, tff.SERVER)

  @tff.tf_computation(tf.int32)
  @tf.function
  def add_one(x):
    return x + 1

  @tff.federated_computation(init_fn.type_signature.result,
                             tff.types.at_clients(weights_type),
                             tff.types.at_clients(data_type))
  def next_fn(state, weights, client_data):
    round_num_at_clients = tff.federated_broadcast(state)
    client_result, model_outputs = tff.federated_map(
        client_update_computation, (weights, client_data, round_num_at_clients))
    updated_state = tff.federated_map(add_one, state)
    train_metrics = metrics_aggregation_fn(model_outputs)
    measurements = tff.federated_zip(
        collections.OrderedDict(train=train_metrics))
    return tff.templates.MeasuredProcessOutput(updated_state, client_result,
                                               measurements)

  return tff.learning.templates.ClientWorkProcess(init_fn, next_fn)


def build_unweighted_averaging_with_optimizer_schedule(
    model_fn: Callable[[], embedding_model.Model],
    client_learning_rate_fn: Callable[[int], float],
    client_optimizer_fn: Callable[[float], tf.keras.optimizers.Optimizer],
    server_optimizer_fn: Union[tff.learning.optimizers.Optimizer, Callable[
        [], tf.keras.optimizers.Optimizer]] = DEFAULT_SERVER_OPTIMIZER_FN,
    model_distributor: Optional[
        tff.learning.templates.DistributionProcess] = None,
    model_aggregator: Optional[
        tff.aggregators.UnweightedAggregationFactory] = None,
    metrics_aggregator: Optional[Callable[
        [keras_utils.KerasMetricFinalizerType, tff.types.StructWithPythonType],
        tff.Computation]] = None,
    head_lr_scale: float = 1.,
    reconst_iters: Optional[int] = None,
) -> tff.learning.templates.LearningProcess:
  """Builds a FedAvg process with partially local variables and client optimizer scheduling.

  This function creates a `LearningProcess` that performs federated averaging on
  the global variables of client models. The iterative process has the following
  methods inherited from `LearningProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a `LearningAlgorithmState` representing the
      initial state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `{B*}@CLIENTS` represents the client datasets.
      The output `L` contains the updated server state, as well as aggregated
      metrics at the server, including client training metrics and any other
      metrics from distribution and aggregation processes.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matches the output of `initialize` and `M` represents the type of the
      model weights used during training.

  Each time the `next` method is called, the server model is broadcast to each
  client using a broadcast function. For each client, local training is
  performed using `client_optimizer_fn`. Each client computes the difference
  of global variables between the client model after training and the initial
  broadcast model. These model deltas are then aggregated at the server, and the
  aggregate model delta is applied at the server using a server optimizer.

  This variant of FedAvg has two important features
  (1) Scheduling client optimizer across rounds. The process keeps track of how
  many iterations of `FedAvg.next` have occurred (starting at `0`), and for each
  such `round_num`, the clients use `client_optimizer_fn(round_num)` to perform
  local optimization. This allows learning rate scheduling (eg. starting with
  a large learning rate and decaying it over time) as well as a small learning
  rate (eg. switching optimizers as learning progresses).
  (2) The client local variables of the model is only used on clients, and not
  communicated/aggregated.

  Args:
    model_fn: A no-arg function that returns a `embedding_model.Model`.
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    client_learning_rate_fn: A callable accepting an integer round number and
      returning a float to be used as a learning rate for the optimizer. The
      client work will call `optimizer_fn(learning_rate_fn(round_num))` where
      `round_num` is the integer round number. Note that the round numbers
      supplied will start at `0` and increment by one each time `.next` is
      called on the resulting process. Also note that this function must be
      serializable by TFF.
    client_optimizer_fn: A callable accepting a float learning rate, and
      returning a `tff.learning.optimizers.Optimizer` or a `tf.keras.Optimizer`.
    server_optimizer_fn: A `tff.learning.optimizers.Optimizer`, or a no-arg
      callable that returns a `tf.keras.Optimizer`. By default, this uses
      `tf.keras.optimizers.SGD` with a learning rate of 1.0.
    model_distributor: An optional `DistributionProcess` that distributes the
      model weights on the server to the clients. If set to `None`, the
      distributor is constructed via `distributors.build_broadcast_process`.
    model_aggregator: An optional `tff.aggregators.WeightedAggregationFactory`
      used to aggregate client updates on the server. If `None`, this is set to
      `tff.aggregators.MeanFactory`.
    metrics_aggregator: A function that takes in the metric finalizers (i.e.,
      `embedding_model.Model.metric_finalizers()`) and a
      `tff.types.StructWithPythonType` of the unfinalized metrics (i.e., the TFF
      type of `embedding_model.Model.report_local_unfinalized_metrics()`),
      and returns a `tff.Computation` for aggregating the unfinalized metrics.
      If `None`, this is set to `tff.learning.metrics.sum_then_finalize`.
    head_lr_scale: Use head_lr_scale to scale the learning rate for updating
      the local variables (head of the embedding model).
    reconst_iters: If not `None`, first optimize the Head of the embedding model
      for `reconst_iters` iterations before training the Backbone.

  Returns:
    A `tff.learning.templates.LearningProcess`.
  """

  @tff.tf_computation()
  def initial_model_weights_fn():
    return tff.learning.ModelWeights.from_model(model_fn())

  model_weights_type = initial_model_weights_fn.type_signature.result

  if model_distributor is None:
    model_distributor = tff.learning.templates.build_broadcast_process(
        model_weights_type)

  if model_aggregator is None:
    model_aggregator = tff.aggregators.UnweightedMeanFactory()

  model_aggregator = tff.aggregators.as_weighted_aggregator(model_aggregator)
  aggregator = model_aggregator.create(model_weights_type.trainable,
                                       tff.types.TensorType(tf.float32))
  process_signature = aggregator.next.type_signature
  input_client_value_type = process_signature.parameter[1]
  result_server_value_type = process_signature.result[1]
  if input_client_value_type.member != result_server_value_type.member:
    raise TypeError('`model_update_aggregation_factory` does not produce a '
                    'compatible `AggregationProcess`. The processes must '
                    'retain the type structure of the inputs on the '
                    f'server, but got {input_client_value_type.member} != '
                    f'{result_server_value_type.member}.')

  if metrics_aggregator is None:
    metrics_aggregator = tff.learning.metrics.sum_then_finalize
  client_work = _build_scheduled_client_work(model_fn, client_learning_rate_fn,
                                             client_optimizer_fn,
                                             metrics_aggregator, head_lr_scale,
                                             reconst_iters)
  finalizer = tff.learning.templates.build_apply_optimizer_finalizer(
      server_optimizer_fn, model_weights_type)
  return tff.learning.templates.compose_learning_process(
      initial_model_weights_fn, model_distributor, client_work, aggregator,
      finalizer)
