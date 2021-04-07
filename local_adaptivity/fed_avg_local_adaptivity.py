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
"""An implementation of the FedAvg algorithm with local adaptivity.

This is intended to be a somewhat minimal implementation of Federated
Averaging with local adaptivity.

The original FedAvg is based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

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

# Define supported client optimizers here.
SUPPORTED_CLIENT_OPTIMIZERS = frozenset(['SGD', 'Adagrad', 'Adam', 'Yogi'])


def _initialize_optimizer_vars(model: tff.learning.Model,
                               optimizer: tf.keras.optimizers.Optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  model_weights = _get_weights(model)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model_weights.trainable)
  optimizer.apply_gradients(grads_and_vars)
  assert optimizer.variables()


def _get_weights(model: tff.learning.Model) -> tff.learning.ModelWeights:
  return tff.learning.ModelWeights.from_model(model)


def _check_client_optimizer(optimizer: tf.keras.optimizers.Optimizer):
  config = optimizer.get_config()
  return config['name'] in SUPPORTED_CLIENT_OPTIMIZERS


def _get_optimizer_momentum_beta(optimizer: tf.keras.optimizers.Optimizer):
  """Get the momentum beta value from the optimizer."""
  config = optimizer.get_config()
  config_name = config['name']
  if config_name == 'SGD':
    return config['momentum']
  elif config_name == 'Adam':
    return config['beta_1']
  elif config_name == 'Yogi':
    return config['beta1']
  elif config_name == 'Adagrad':
    return 0.0
  else:
    raise TypeError(
        'client optimizer should be one of these: SGD, Adagrad, Adam, Yogi.')


def _get_optimizer_preconditioner(optimizer: tf.keras.optimizers.Optimizer,
                                  model_weights: tff.learning.ModelWeights):
  """Get the preconditioner states from the optimizer."""
  config = optimizer.get_config()
  config_name = config['name']
  if config_name == 'Adagrad':
    eps = optimizer.epsilon
    v = tf.nest.map_structure(
        lambda var: optimizer.get_slot(var, 'accumulator'),
        model_weights.trainable)
    return tf.nest.map_structure(
        lambda a: tf.math.divide_no_nan(  # pylint: disable=g-long-lambda
            1.0,
            tf.math.sqrt(a) + eps),
        v)
  elif config_name in {'Adam', 'Yogi'}:
    eps = optimizer.epsilon
    v = tf.nest.map_structure(lambda var: optimizer.get_slot(var, 'v'),
                              model_weights.trainable)
    return tf.nest.map_structure(
        lambda a: tf.math.divide_no_nan(  # pylint: disable=g-long-lambda
            1.0,
            tf.math.sqrt(a) + eps),
        v)
  elif config_name == 'SGD':
    return tf.nest.map_structure(tf.ones_like, model_weights.trainable)
  else:
    raise TypeError(
        'client optimizer should be one of these: SGD, Adagrad, Adam, Yogi.')


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of the model's trainable and non-trainable
        weights.
  -   `optimizer_state`: The server optimizer variables.
  -   `round_num`: The current training round, as a float.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()
  # This is a float to avoid type incompatibility when calculating learning rate
  # schedules.


@tf.function
def server_update(model,
                  server_optimizer,
                  server_state,
                  weights_delta,
                  global_cor=None):
  """Updates `server_state` based on `weights_delta`, increase the round number.

  Args:
    model: A `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: An update to the trainable variables of the model.
    global_cor: Optional. A correction to the update of `weights_delta`.

  Returns:
    An updated `ServerState`.
  """
  model_weights = _get_weights(model)
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        server_state.model)
  # Server optimizer variables must be initialized prior to invoking this
  tf.nest.map_structure(lambda v, t: v.assign(t), server_optimizer.variables(),
                        server_state.optimizer_state)

  weights_delta, has_non_finite_weight = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  if has_non_finite_weight > 0:
    return server_state

  if global_cor is not None:
    weights_delta = tf.nest.map_structure(tf.math.divide_no_nan, weights_delta,
                                          global_cor)

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
      optimizer_state=server_optimizer.variables(),
      round_num=server_state.round_num + 1.0)


@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `client_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `local_cor_states`: Local correction to be aggregated for global
      correction.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  local_cor_states = attr.ib()


def create_client_update_fn():
  """Returns a tf.function for the client_update.

  This "create" fn is necesessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is really only needed because we test the client_update function directly.
  """

  @tf.function
  def client_update(model,
                    dataset,
                    initial_weights,
                    client_optimizer,
                    client_weight_fn=None):
    """Updates client model.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      initial_weights: A `tff.learning.ModelWeights` from server.
      client_optimizer: A `tf.keras.optimizer.Optimizer` object.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor that provides the
        weight in the federated average of model deltas. If not provided, the
        default is the total number of examples processed on device.

    Returns:
      A 'ClientOutput`.
    """

    model_weights = _get_weights(model)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          initial_weights)
    num_examples = tf.constant(0, dtype=tf.int32)

    # Initialize local states for local and global corrections
    avg_local_states = tf.nest.map_structure(tf.zeros_like,
                                             model_weights.trainable)
    cum_local_states = tf.nest.map_structure(tf.zeros_like,
                                             model_weights.trainable)
    for batch in iter(dataset):
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      grads = tape.gradient(output.loss, model_weights.trainable)
      grads_and_vars = zip(grads, model_weights.trainable)
      client_optimizer.apply_gradients(grads_and_vars)
      num_examples += tf.shape(output.predictions)[0]

      # Get momentum factor and preconditioner to update local states
      client_opt_beta = _get_optimizer_momentum_beta(client_optimizer)
      client_opt_preconditioner = _get_optimizer_preconditioner(
          client_optimizer, model_weights)
      avg_local_states = tf.nest.map_structure(
          lambda m, p, b=client_opt_beta: b * m + (1 - b) * p,
          avg_local_states,
          client_opt_preconditioner)
      cum_local_states = tf.nest.map_structure(lambda m, n: m + n,
                                               avg_local_states,
                                               cum_local_states)

    aggregated_outputs = model.report_local_outputs()
    weights_delta = tf.nest.map_structure(
        lambda a, b, c: tf.math.divide_no_nan(a - b, c),
        model_weights.trainable, initial_weights.trainable, cum_local_states)
    local_cor_states = tf.nest.map_structure(
        lambda a: tf.math.divide_no_nan(1.0, a), cum_local_states)

    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))

    if has_non_finite_weight > 0:
      client_weight = tf.constant(0, dtype=tf.float32)
    elif client_weight_fn is None:
      client_weight = tf.cast(num_examples, dtype=tf.float32)
    else:
      client_weight = client_weight_fn(aggregated_outputs)

    return ClientOutput(weights_delta, client_weight, aggregated_outputs,
                        local_cor_states)

  return client_update


def build_server_init_fn(
    model_fn: ModelBuilder,
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer]):
  """Builds a `tff.tf_computation` that returns the initial `ServerState`.

  The attributes `ServerState.model` and `ServerState.optimizer_state` are
  initialized via their constructor functions. The attribute
  `ServerState.round_num` is set to 0.0.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
  """

  @tff.tf_computation
  def server_init_tf():
    server_optimizer = server_optimizer_fn()
    model = model_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return ServerState(
        model=_get_weights(model),
        optimizer_state=server_optimizer.variables(),
        round_num=0.0)

  return server_init_tf


def build_fed_avg_process(
    model_fn: ModelBuilder,
    client_optimizer_fn: OptimizerBuilder,
    client_lr: float = 0.1,
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_lr: float = 1.0,
    client_weight_fn: Optional[ClientWeightFn] = None,
    correction: str = 'joint') -> tff.templates.IterativeProcess:
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    client_optimizer_fn: A function that accepts a `learning_rate` keyword
      argument and returns a `tf.keras.optimizers.Optimizer` instance. Only
      supporting SGD, Adagrad, Adam and Yogi optimizers at this moment.
    client_lr: A scalar learning rate.
    server_optimizer_fn: A function that accepts a `learning_rate` argument and
      returns a `tf.keras.optimizers.Optimizer` instance.
    server_lr: A scalar learning rate.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.
    correction: A string that specifies the type of correction method when
       applying local adaptive optimizers. It must be either `local` or `joint`.
       The default is`joint`.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  dummy_model = model_fn()

  client_optimizer = client_optimizer_fn(client_lr)

  if not _check_client_optimizer(client_optimizer):
    raise TypeError(
        'client optimizer should be one of these: SGD, Adagrad, Adam, Yogi.')

  server_init_tf = build_server_init_fn(model_fn,
                                        lambda: server_optimizer_fn(server_lr))
  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model

  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
  model_input_type = tff.SequenceType(dummy_model.input_spec)

  @tff.tf_computation(model_input_type, model_weights_type)
  def client_update_fn(tf_dataset, initial_model_weights):
    client_update = create_client_update_fn()
    return client_update(model_fn(), tf_dataset, initial_model_weights,
                         client_optimizer, client_weight_fn)

  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    server_optimizer = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta)

  @tff.tf_computation(server_state_type, model_weights_type.trainable,
                      model_weights_type.trainable)
  def server_update_joint_cor_fn(server_state, model_delta, global_cor):
    model = model_fn()
    server_optimizer = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta,
                         global_cor)

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

    client_outputs = tff.federated_map(client_update_fn,
                                       (federated_dataset, client_model))

    client_weight = client_outputs.client_weight
    model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=client_weight)
    global_cor_states = tff.federated_mean(
        client_outputs.local_cor_states, weight=client_weight)
    if correction == 'joint':
      server_state = tff.federated_map(
          server_update_joint_cor_fn,
          (server_state, model_delta, global_cor_states))
    elif correction == 'local':
      server_state = tff.federated_map(server_update_fn,
                                       (server_state, model_delta))
    else:
      raise TypeError('Correction method must be local or joint.')

    aggregated_outputs = dummy_model.federated_output_computation(
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
