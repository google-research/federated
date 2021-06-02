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
"""An implementation of FedAvg where client optimizer states are aggregated."""

import collections
import enum
from typing import Callable, Optional, Union

import attr
import tensorflow as tf
import tensorflow_federated as tff

from utils import tensor_utils

# Convenience type aliases.
ModelBuilder = Callable[[], tff.learning.Model]
OptimizerBuilder = Callable[[float], tf.keras.optimizers.Optimizer]
ClientWeightFn = Callable[..., float]
LRScheduleFn = Callable[[Union[int, tf.Tensor]], Union[tf.Tensor, float]]


@attr.s(eq=False)
class OptimizerState(object):
  iterations = attr.ib()
  weights = attr.ib()


class AggregationType(enum.Enum):
  mean = 'mean'
  sum = 'sum'
  min = 'min'
  max = 'max'


def build_aggregator(aggregation_method):
  """Builds a federated aggregation method.

  Args:
    aggregation_method: A string describing the desired aggregation type. Should
      be one of 'mean', 'sum', 'min', or 'max'.

  Returns:
    A function that accepts a federated value with placement `tff.CLIENTS` and
      an optional 'weights' argument, and returns a federated value with
      placement `tff.SERVER`.
  """
  try:
    aggregation_type = AggregationType(aggregation_method)
  except ValueError:
    raise ValueError(
        'Aggregation method {} is not supported. Must be one of {}'.format(
            aggregation_method, list(AggregationType.__members__.keys())))

  if aggregation_type is AggregationType.mean:
    aggregator = tff.federated_mean
  elif aggregation_type is AggregationType.sum:

    def aggregator(federated_values, weight=None):
      del weight
      return tff.federated_sum(federated_values)
  elif aggregation_type is AggregationType.max:

    def aggregator(federated_values, weight=None):
      del weight
      return tff.aggregators.federated_max(federated_values)
  elif aggregation_type is AggregationType.min:

    def aggregator(federated_values, weight=None):
      del weight
      return tff.aggregators.federated_min(federated_values)
  else:
    raise ValueError(
        'Aggregation method {} has no associated TFF computation implemented.')
  return aggregator


def _initialize_optimizer_vars(model: tff.learning.Model,
                               optimizer: tf.keras.optimizers.Optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  model_weights = _get_weights(model)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model_weights.trainable)
  optimizer.apply_gradients(grads_and_vars, name='server_update')
  assert optimizer.variables()


def _get_weights(model: tff.learning.Model) -> tff.learning.ModelWeights:
  return tff.learning.ModelWeights.from_model(model)


def _get_optimizer_state(optimizer):
  return OptimizerState(
      iterations=optimizer.iterations,
      # The first weight of an optimizer is reserved for the iterations count,
      # see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#get_weights pylint: disable=line-too-long]
      weights=tuple(optimizer.weights[1:]))


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of the model's trainable and non-trainable
        weights.
  -   `client_optimizer_state`: A namedtuple of the client optimizer variables.
  -   `server_optimizer_state`: A namedtuple of the server optimizer variables.
  -   `round_num`: The current training round, as a float.
  """
  model = attr.ib()
  client_optimizer_state = attr.ib()
  server_optimizer_state = attr.ib()
  round_num = attr.ib()
  # This is a float to avoid type incompatibility when calculating learning rate
  # schedules.


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta,
                  client_optimizer_state_delta):
  """Updates `server_state` based on `weights_delta`, increase the round number.

  Args:
    model: A `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: An update to the trainable variables of the model.
    client_optimizer_state_delta: An update to the client optimizer variables.

  Returns:
    An updated `ServerState`.
  """
  model_weights = _get_weights(model)
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        server_state.model)

  # Server optimizer variables must be initialized prior to invoking this
  updated_client_optimizer_state = tf.nest.map_structure(
      lambda a, b: a + b, server_state.client_optimizer_state,
      client_optimizer_state_delta)
  server_optimizer_state = _get_optimizer_state(server_optimizer)
  tf.nest.map_structure(lambda v, t: v.assign(t), server_optimizer_state,
                        server_state.server_optimizer_state)

  # Apply the update to the model. We must multiply weights_delta by -1.0 to
  # view it as a gradient that should be applied to the server_optimizer.
  grads_and_vars = [
      (-1.0 * x, v) for x, v in zip(weights_delta, model_weights.trainable)
  ]

  server_optimizer.apply_gradients(grads_and_vars)

  # Create a new state based on the updated model.
  return tff.structure.update_struct(
      server_state,
      model=model_weights,
      client_optimizer_state=updated_client_optimizer_state,
      server_optimizer_state=server_optimizer_state,
      round_num=server_state.round_num + 1.0)


@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `client_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta` and `optimizer_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `optimizer_output`: Additional metrics or other outputs defined by the
      optimizer.
  """
  weights_delta = attr.ib()
  optimizer_state_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


def create_client_update_fn():
  """Returns a tf.function for the client_update.

  This "create" fn is necesessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is only needed because we test the client_update function directly.
  """

  @tf.function
  def client_update(model,
                    dataset,
                    initial_weights,
                    initial_client_optimizer_state,
                    client_optimizer,
                    client_model_weight_fn=None,
                    client_opt_weight_fn=None):
    """Updates client model.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      initial_weights: A `tff.learning.Model.weights` from server.
      initial_client_optimizer_state: The variables to assign to the client
        optimizer.
      client_optimizer: A `tf.keras.optimizer.Optimizer` object.
      client_model_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor that provides the
        weight in the federated average of model deltas. If not provided, the
        default is the total number of examples processed on device.
      client_opt_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor that provides the
        weight in the federated average of the optimizer states. If not
        provided, the default is a uniform weighting.

    Returns:
      A 'ClientOutput`.
    """

    model_weights = _get_weights(model)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          initial_weights)

    # Client optimizer variables must be initialized prior to invoking this
    client_optimizer_state = _get_optimizer_state(client_optimizer)
    tf.nest.map_structure(lambda v, t: v.assign(t), client_optimizer_state,
                          initial_client_optimizer_state)

    num_examples = tf.constant(0, dtype=tf.int32)
    for batch in dataset:
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      grads = tape.gradient(output.loss, model_weights.trainable)
      grads_and_vars = zip(grads, model_weights.trainable)
      client_optimizer.apply_gradients(grads_and_vars)
      num_examples += tf.shape(output.predictions)[0]

    aggregated_outputs = model.report_local_outputs()

    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)
    weights_delta, non_finite_weights_delta = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    if non_finite_weights_delta > 0:
      client_model_weight = tf.constant(0, dtype=tf.float32)
    elif client_model_weight_fn is None:
      client_model_weight = tf.cast(num_examples, dtype=tf.float32)
    else:
      client_model_weight = client_model_weight_fn(aggregated_outputs)

    optimizer_state_delta = tf.nest.map_structure(
        lambda a, b: a - b, client_optimizer_state,
        initial_client_optimizer_state)
    if client_opt_weight_fn is None:
      client_opt_weight = tf.cast(num_examples, dtype=tf.float32)
    else:
      client_opt_weight = client_opt_weight_fn(aggregated_outputs)

    optimizer_output = collections.OrderedDict([('num_examples', num_examples)])
    client_weight = collections.OrderedDict([
        ('model_weight', client_model_weight),
        ('optimizer_weight', client_opt_weight)
    ])

    return ClientOutput(
        weights_delta=weights_delta,
        optimizer_state_delta=optimizer_state_delta,
        client_weight=client_weight,
        model_output=aggregated_outputs,
        optimizer_output=optimizer_output)

  return client_update


def build_server_init_fn(model_fn, client_optimizer_fn, server_optimizer_fn):
  """Builds a `tff.tf_computation` that returns the initial `ServerState`.

  The attributes `ServerState.model`, `ServerState.client_optimizer_state`, and
  `ServerState.server_optimizer_state` are initialized via their constructor
  functions. The attribute `ServerState.round_num` is set to 0.0.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
  """

  @tff.tf_computation
  def server_init_tf():
    client_optimizer = client_optimizer_fn()
    server_optimizer = server_optimizer_fn()
    model = model_fn()
    _initialize_optimizer_vars(model, client_optimizer)
    _initialize_optimizer_vars(model, server_optimizer)
    return ServerState(
        model=_get_weights(model),
        client_optimizer_state=_get_optimizer_state(client_optimizer),
        server_optimizer_state=_get_optimizer_state(server_optimizer),
        round_num=0.0)

  return server_init_tf


def build_iterative_process(
    model_fn: ModelBuilder,
    client_optimizer_fn: OptimizerBuilder,
    client_lr: Union[float, LRScheduleFn] = 0.1,
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_lr: Union[float, LRScheduleFn] = 1.0,
    optimizer_aggregation: AggregationType = 'mean',
    client_model_weight_fn: Optional[ClientWeightFn] = None,
    client_opt_weight_fn: Optional[ClientWeightFn] = None,
) -> tff.templates.IterativeProcess:  # pytype: disable=annotation-type-mismatch
  """Builds an iterative process for FedAvg with client optimizer aggregation.

  This version of FedAvg allows user-selected `tf.keras.Optimizers` on both
  the client and server level. Moreover, the iterative process will aggregate
  both the changes in the client model weights, and the changes in the client
  optimizer states. The aggregated model weights and client optimizer states
  will be broadcast to all clients in the subsequent round.

  For example, if clients use SGD with momentum, then this iterative process
  will aggregate both the client model weights and the momentum parameter in
  the clients' optimizers. This allows clients in the next round of computation
  to start with an estimated momentum parameter, rather than initializing it
  at zero.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    client_optimizer_fn: A function that accepts a `learning_rate` keyword
      argument and returns a `tf.keras.optimizers.Optimizer` instance.
    client_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    server_optimizer_fn: A function that accepts a `learning_rate` argument and
      returns a `tf.keras.optimizers.Optimizer` instance.
    server_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    optimizer_aggregation: What type of aggregation to use for the client
      optimizer states. Must be a member of ['mean', 'sum', 'min', 'max'].
    client_model_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of the client models. If not provided, the
      default is the total number of examples processed on device.
    client_opt_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of the client optimizer states. If not provided,
      the default is the total number of examples processed on device.

  Returns:
    A `tff.templates.IterativeProcess`.
  """
  client_lr_schedule = client_lr
  if not callable(client_lr_schedule):
    client_lr_schedule = lambda round_num: client_lr

  server_lr_schedule = server_lr
  if not callable(server_lr_schedule):
    server_lr_schedule = lambda round_num: server_lr

  optimizer_aggregator = build_aggregator(optimizer_aggregation)

  placeholder_model = model_fn()

  server_init_tf = build_server_init_fn(
      model_fn,
      # Initialize with the learning rate for round zero.
      lambda: client_optimizer_fn(client_lr_schedule(0)),
      lambda: server_optimizer_fn(server_lr_schedule(0)))
  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model
  client_optimizer_state_type = server_state_type.client_optimizer_state
  round_num_type = server_state_type.round_num

  tf_dataset_type = tff.SequenceType(placeholder_model.input_spec)

  @tff.tf_computation(tf_dataset_type, model_weights_type,
                      client_optimizer_state_type, round_num_type)
  def client_update_fn(tf_dataset, initial_model_weights,
                       initial_optimizer_state, round_num):
    """Performs a client update."""
    model = model_fn()
    client_lr = client_lr_schedule(round_num)
    client_optimizer = client_optimizer_fn(client_lr)
    # We initialize the client optimizer variables to avoid creating them
    # within the scope of the tf.function client_update.
    _initialize_optimizer_vars(model, client_optimizer)

    client_update = create_client_update_fn()
    return client_update(model, tf_dataset, initial_model_weights,
                         initial_optimizer_state, client_optimizer,
                         client_model_weight_fn, client_opt_weight_fn)

  @tff.tf_computation(server_state_type, model_weights_type.trainable,
                      client_optimizer_state_type)
  def server_update_fn(server_state, model_delta, optimizer_delta):
    model = model_fn()
    server_lr = server_lr_schedule(server_state.round_num)
    server_optimizer = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta,
                         optimizer_delta)

  @tff.tf_computation(client_optimizer_state_type)
  def _convert_opt_state_to_float(optimizer_state):
    return tf.nest.map_structure(lambda x: tf.cast(x, tf.float32),
                                 optimizer_state)

  @tff.tf_computation(_convert_opt_state_to_float.type_signature.result)
  def _convert_opt_state_to_int(optimizer_state):
    iterations_as_int = tf.cast(optimizer_state.iterations, tf.int64)
    return OptimizerState(
        iterations=iterations_as_int, weights=optimizer_state.weights)

  @tff.federated_computation(
      tff.type_at_server(server_state_type),
      tff.type_at_clients(tf_dataset_type))
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `tff.learning.Model.federated_output_computation`.
    """
    client_model = tff.federated_broadcast(server_state.model)
    client_optimizer_state = tff.federated_broadcast(
        server_state.client_optimizer_state)
    client_round_num = tff.federated_broadcast(server_state.round_num)
    client_outputs = tff.federated_map(
        client_update_fn, (federated_dataset, client_model,
                           client_optimizer_state, client_round_num))

    client_model_weight = client_outputs.client_weight.model_weight
    client_opt_weight = client_outputs.client_weight.optimizer_weight

    model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=client_model_weight)

    # We convert the optimizer state to a float type so that it can be used
    # with thing such as `tff.federated_mean`. This is only necessary because
    # `tf.keras.Optimizer` objects have a state with an integer indicating
    # the number of times it has been applied.
    client_optimizer_state_delta = tff.federated_map(
        _convert_opt_state_to_float, client_outputs.optimizer_state_delta)
    client_optimizer_state_delta = optimizer_aggregator(
        client_optimizer_state_delta, weight=client_opt_weight)
    # We conver the optimizer state back into one with an integer round number
    client_optimizer_state_delta = tff.federated_map(
        _convert_opt_state_to_int, client_optimizer_state_delta)

    server_state = tff.federated_map(
        server_update_fn,
        (server_state, model_delta, client_optimizer_state_delta))

    aggregated_outputs = placeholder_model.federated_output_computation(
        client_outputs.model_output)
    if aggregated_outputs.type_signature.is_struct():
      aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return server_state, aggregated_outputs

  @tff.federated_computation
  def initialize_fn():
    return tff.federated_value(server_init_tf(), tff.SERVER)

  iterative_process = tff.templates.IterativeProcess(
      initialize_fn=initialize_fn, next_fn=run_one_round)

  @tff.tf_computation(server_state_type)
  def get_model_weights(server_state):
    return server_state.model

  iterative_process.get_model_weights = get_model_weights
  return iterative_process
