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
"""An implementation of the Federated Averaging algorithm.

This is forked from TFF/simple_fedavg with the following changes for DP:
(1) clip the norm of the model delta from clients;
(2) aggregate the model delta from clients with uniform weighting.
"""

import collections
import attr

import tensorflow as tf
import tensorflow_federated as tff

from dp_ftrl import optimizer_utils
from utils import tensor_utils


def _dataset_reduce_fn(reduce_fn, dataset, initial_state_fn):
  return dataset.reduce(initial_state=initial_state_fn(), reduce_func=reduce_fn)


def _for_iter_dataset_fn(reduce_fn, dataset, initial_state_fn):
  """Performs dataset reduce for simulation performance."""
  # TODO(b/155208489): this is a workaround for GPU simulation because
  # `tf.device` does not cross the boundary of dataset ops. TF use a different
  # set of ops when we explicitly use `iter` for dataset.
  update_state = initial_state_fn()
  for batch in iter(dataset):
    update_state = reduce_fn(update_state, batch)
  return update_state


def _build_dataset_reduce_fn(simulation_flag=True):
  """Returns a reduce loop function on input dataset."""
  if simulation_flag:
    return _for_iter_dataset_fn
  else:
    return _dataset_reduce_fn


def _unpack_data_label(batch):
  if isinstance(batch, collections.abc.Mapping):
    return batch['x'], batch['y']
  elif isinstance(batch, (tuple, list)):
    if len(batch) < 2:
      raise ValueError('Expecting both data and label from a batch.')
    return batch[0], batch[1]
  else:
    raise ValueError('Unrecognized batch data.')


def _get_model_weights(model):
  if hasattr(model, 'weights'):
    return model.weights
  else:
    return tff.learning.ModelWeights.from_model(model)


@attr.s(eq=False, frozen=True, slots=True)
class ModelWeights(object):
  """A container for the trainable and non-trainable variables of a `Model`.

  Note this does not include the model's local variables.

  It may also be used to hold other values that are parallel to these variables,
  e.g., tensors corresponding to variable values, or updates to model variables.
  """
  trainable = attr.ib()
  non_trainable = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ModelOutputs(object):
  loss = attr.ib()


# TODO(b/172867399): remove `KerasModelWrapper` and use `tff.learning.Model`.
class KerasModelWrapper(object):
  """A standalone keras wrapper to be used in TFF."""

  def __init__(self, keras_model, input_spec, loss):
    """A wrapper class that provides necessary API handles for TFF.

    Args:
      keras_model: A `tf.keras.Model` to be trained.
      input_spec: Metadata of dataset that desribes the input tensors, which
        will be converted to `tff.Type` specifying the expected type of input
        and output of the model.
      loss: A `tf.keras.losses.Loss` instance to be used for training.
    """
    self.keras_model = keras_model
    self.input_spec = input_spec
    self.loss = loss

  def forward_pass(self, batch_input, training=True):
    """Forward pass of the model to get loss for a batch of data.

    Args:
      batch_input: A `collections.abc.Mapping` with two keys, `x` for inputs and
        `y` for labels.
      training: Boolean scalar indicating training or inference mode.

    Returns:
      A scalar tf.float32 `tf.Tensor` loss for current batch input.
    """
    batch_x, batch_y = _unpack_data_label(batch_input)
    predictions = self.keras_model(batch_x, training=training)
    loss = self.loss(batch_y, predictions)
    return ModelOutputs(loss=loss)

  @property
  def weights(self):
    return ModelWeights(
        trainable=self.keras_model.trainable_variables,
        non_trainable=self.keras_model.non_trainable_variables)

  def from_weights(self, model_weights):
    tf.nest.map_structure(lambda v, t: v.assign(t),
                          self.keras_model.trainable_variables,
                          list(model_weights.trainable))
    tf.nest.map_structure(lambda v, t: v.assign(t),
                          self.keras_model.non_trainable_variables,
                          list(model_weights.non_trainable))


def keras_evaluate(model, test_data, metrics):
  for metric in metrics:
    metric.reset_states()
  for batch in test_data:
    batch_x, batch_y = _unpack_data_label(batch)
    preds = model(batch_x, training=False)
    for metric in metrics:
      metric.update_state(y_true=batch_y, y_pred=preds)
  return metrics


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
  """Structure for tensors broadcasted by server during federated optimization.

  Fields:
  -   `model_weights`: A dictionary of model's trainable tensors.
  -   `dp_clip_norm`: Clip norm for client model delta.
  """
  model_weights = attr.ib()
  dp_clip_norm = attr.ib()


@tf.function
def build_server_broadcast_message(server_state):
  """Builds `BroadcastMessage` for broadcasting.

  Args:
    server_state: A `ServerState`.

  Returns:
    A `BroadcastMessage`.
  """
  return BroadcastMessage(
      model_weights=server_state.model, dp_clip_norm=server_state.dp_clip_norm)


@attr.s(eq=False, frozen=True, slots=True)
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
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()


@tf.function
def client_update(model, dataset, server_message, client_optimizer,
                  use_simulation_loop):
  """Performans client local training of `model` on `dataset`.

  Args:
    model: A `tff.learning.Model`.
    dataset: A 'tf.data.Dataset'.
    server_message: A `BroadcastMessage` from server.
    client_optimizer: A `tf.keras.optimizers.Optimizer`.
    use_simulation_loop: Controls the reduce loop function for client dataset.
      Set this flag to True for performant GPU simulations.

  Returns:
    A 'ClientOutput`.
  """
  model_weights = _get_model_weights(model)
  initial_weights = server_message.model_weights
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        initial_weights)

  def reduce_fn(state, batch):
    """Train model on local client batch."""
    num_examples, loss_sum = state
    with tf.GradientTape() as tape:
      outputs = model.forward_pass(batch)

    grads = tape.gradient(outputs.loss, model_weights.trainable)
    client_optimizer.apply_gradients(zip(grads, model_weights.trainable))
    if hasattr(outputs, 'num_examples'):
      batch_size = tf.cast(outputs.num_examples, dtype=tf.int32)
    else:
      batch_x, _ = _unpack_data_label(batch)
      batch_size = tf.shape(batch_x)[0]
    num_examples += batch_size
    loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)
    return num_examples, loss_sum

  num_examples = tf.constant(0, dtype=tf.int32)
  loss_sum = tf.constant(0, dtype=tf.float32)
  dataset_reduce_fn = _build_dataset_reduce_fn(use_simulation_loop)
  num_examples, loss_sum = dataset_reduce_fn(
      reduce_fn, dataset, initial_state_fn=lambda: (num_examples, loss_sum))
  weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                        model_weights.trainable,
                                        initial_weights.trainable)
  client_weight = tf.cast(num_examples, tf.float32)
  # Clip the norm of model delta before sending back.
  clip_norm = tf.cast(server_message.dp_clip_norm, tf.float32)
  if tf.less(tf.constant(0, tf.float32), clip_norm):
    flatten_weights_delta = tf.nest.flatten(weights_delta)
    clipped_flatten_weights_delta, _ = tf.clip_by_global_norm(
        flatten_weights_delta, clip_norm)
    weights_delta = tf.nest.pack_sequence_as(weights_delta,
                                             clipped_flatten_weights_delta)
  return ClientOutput(weights_delta, client_weight, loss_sum / client_weight)


@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of model's trainable variables.
  -   `optimizer_state`: Server optimizer states.
  -   'round_num': Current round index
  """
  # Some attributes are named to be consistent with the private `ServerState` in
  # `tff.learning` to possibly use `tff.learning.build_federated_evaluation`.
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()
  dp_clip_norm = attr.ib()


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `KerasModelWrapper` or `tff.learning.Model`.
    server_optimizer: A `ServerOptimizerBase`.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: A nested structure of tensors holding the updates to the
      trainable variables of the model.

  Returns:
    An updated `ServerState`.
  """
  weights_delta, has_non_finite_weight = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  if has_non_finite_weight > 0:
    return server_state

  # Initialize the model with the current state.
  model_weights = _get_model_weights(model)
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        server_state.model)

  # Apply the update to the model, and return the updated state.
  grad = tf.nest.map_structure(lambda x: -1.0 * x, weights_delta)
  optimizer_state = server_optimizer.model_update(
      state=server_state.optimizer_state,
      weight=model_weights.trainable,
      grad=grad,
      round_idx=server_state.round_num)

  # Create a new state based on the updated model.
  return tff.structure.update_struct(
      server_state,
      model=model_weights,
      optimizer_state=optimizer_state,
      round_num=server_state.round_num + 1)


def build_federated_averaging_process(
    model_fn,
    dp_clip_norm=1.0,
    server_optimizer_fn=lambda w: optimizer_utils.SGDServerOptimizer(  # pylint: disable=g-long-lambda
        learning_rate=1.0),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
    use_simulation_loop=True):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `dp_fedavg_tf.KerasModelWrapper`.
    dp_clip_norm: if < 0, no clipping
    server_optimizer_fn: .
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for client update.
    use_simulation_loop: Controls the reduce loop function for client dataset.
      Set this flag to True for performant GPU simulations.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  example_model = model_fn()

  @tff.tf_computation
  def server_init_tf():
    model = model_fn()
    model_weights = _get_model_weights(model)
    optimizer = server_optimizer_fn(model_weights.trainable)
    return ServerState(
        model=model_weights,
        optimizer_state=optimizer.init_state(),
        round_num=0,
        dp_clip_norm=dp_clip_norm)

  server_state_type = server_init_tf.type_signature.result

  model_weights_type = server_state_type.model

  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    model_weights = _get_model_weights(model)
    optimizer = server_optimizer_fn(model_weights.trainable)
    return server_update(model, optimizer, server_state, model_delta)

  @tff.tf_computation(server_state_type)
  def server_message_fn(server_state):
    return build_server_broadcast_message(server_state)

  server_message_type = server_message_fn.type_signature.result
  tf_dataset_type = tff.SequenceType(example_model.input_spec)

  @tff.tf_computation(tf_dataset_type, server_message_type)
  def client_update_fn(tf_dataset, server_message):
    model = model_fn()
    client_optimizer = client_optimizer_fn()
    return client_update(model, tf_dataset, server_message, client_optimizer,
                         use_simulation_loop)

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(tf_dataset_type)

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.data.Dataset` with placement
        `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and `tf.Tensor` of average loss.
    """
    server_message = tff.federated_map(server_message_fn, server_state)
    server_message_at_client = tff.federated_broadcast(server_message)

    client_outputs = tff.federated_map(
        client_update_fn, (federated_dataset, server_message_at_client))

    # Model deltas are equally weighted in DP.
    round_model_delta = tff.federated_mean(client_outputs.weights_delta)

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, round_model_delta))
    round_loss_metric = tff.federated_mean(client_outputs.model_output)

    return server_state, round_loss_metric

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    return tff.federated_value(server_init_tf(), tff.SERVER)

  return tff.templates.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round)
