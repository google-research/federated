# Copyright 2020, The TensorFlow Federated Authors.
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

This is intended to be a minimal stand-alone implementation of Federated
Averaging, suitable for branching as a starting point for algorithm
modifications; see `tff.learning.build_federated_averaging_process` for a
more full-featured implementation.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import collections
from typing import Union, Sequence, Tuple, List

import attr
import tensorflow as tf
import tensorflow_federated as tff

ModelOutputs = collections.namedtuple('ModelOutputs', 'loss')
WEIGHT_DENOM_TYPE = tf.float32


def get_model_weights(
    model: Union[tff.learning.Model, 'KerasModelWrapper']
) -> tff.learning.ModelWeights:
  """Gets the appropriate ModelWeights object based on the model type."""
  if isinstance(model, tff.learning.Model):
    return tff.learning.ModelWeights.from_model(model)
  else:
    # Using simple_fedavg custom Keras wrapper.
    return model.weights


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
    preds = self.keras_model(batch_input['x'], training=training)
    loss = self.loss(batch_input['y'], preds)
    return ModelOutputs(loss=loss)

  @property
  def weights(self):
    return tff.learning.ModelWeights(
        trainable=self.keras_model.trainable_variables,
        non_trainable=self.keras_model.non_trainable_variables)

  def from_weights(self, model_weights):
    tf.nest.map_structure(lambda v, t: v.assign(t),
                          self.keras_model.trainable_variables,
                          list(model_weights.trainable))
    tf.nest.map_structure(lambda v, t: v.assign(t),
                          self.keras_model.non_trainable_variables,
                          list(model_weights.non_trainable))


def keras_evaluate(model, test_data, metric):
  metric.reset_states()
  for batch in test_data:
    preds = model(batch['x'], training=False)
    metric.update_state(y_true=batch['y'], y_pred=preds)
  return metric.result()


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


@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model_weights`: A dictionary of model's trainable variables.
  -   `optimizer_state`: Variables of optimizer.
  -   'round_num': Current round index
  """
  model_weights = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
  """Structure for tensors broadcasted by server during federated optimization.

  Fields:
  -   `model_weights`: A dictionary of model's trainable tensors.
  -   `round_num`: Round index to broadcast. We use `round_num` as an example to
          show how to broadcast auxiliary information that can be helpful on
          clients. It is not explicitly used, but can be applied to enable
          learning rate scheduling.
  """
  model_weights = attr.ib()
  round_num = attr.ib()


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `KerasModelWrapper` or `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`. If the optimizer
      creates variables, they must have already been created.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: A nested structure of tensors holding the updates to the
      trainable variables of the model.

  Returns:
    An updated `ServerState`.
  """
  # Initialize the model with the current state.
  model_weights = get_model_weights(model)
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        server_state.model_weights)
  tf.nest.map_structure(lambda v, t: v.assign(t), server_optimizer.variables(),
                        server_state.optimizer_state)

  # Apply the update to the model.
  neg_weights_delta = [-1.0 * x for x in weights_delta]
  server_optimizer.apply_gradients(
      zip(neg_weights_delta, model_weights.trainable), name='server_update')

  # Create a new state based on the updated model.
  return tff.structure.update_struct(
      server_state,
      model_weights=model_weights,
      optimizer_state=server_optimizer.variables(),
      round_num=server_state.round_num + 1)


@tf.function
def build_server_broadcast_message(server_state):
  """Builds `BroadcastMessage` for broadcasting.

  This method can be used to post-process `ServerState` before broadcasting.
  For example, perform model compression on `ServerState` to obtain a compressed
  state that is sent in a `BroadcastMessage`.

  Args:
    server_state: A `ServerState`.

  Returns:
    A `BroadcastMessage`.
  """
  return BroadcastMessage(
      model_weights=server_state.model_weights,
      round_num=server_state.round_num)


def flatten_list_of_tensors(
    list_of_tensors: Sequence[tf.Tensor]
) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
  """Flattens and concatenates the tensors in `list_of_tensors` into one vector.

  Args:
    list_of_tensors: A Sequence of Tensors.

  Returns:
    The flattened and concatenated vector,
    a list of sizes of original tensors from `list_of_tensors`,
    a list of shapes of original tensors from `list_of_tensors`.
  """
  list_of_shapes = [tf.shape(x) for x in list_of_tensors]
  list_of_sizes = [tf.size(x) for x in list_of_tensors]
  list_of_flattened = tf.nest.map_structure(lambda x: tf.reshape(x, [-1]),
                                            list_of_tensors)
  concatenated = tf.concat(list_of_flattened, axis=0)
  return concatenated, list_of_sizes, list_of_shapes


def reshape_flattened_tensor(
    concatenated: tf.Tensor, list_of_sizes: Sequence[tf.Tensor],
    list_of_shapes: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
  """Reshapes `concatenated` into the form specified by `list_of_shapes`.

  Args:
    concatenated: A flat Tensor to be reshaped.
    list_of_sizes: a list of desired sizes of returned tensors
    list_of_shapes: a list of desired shapes of returned tensors

  Returns:
    A Sequence of Tensors.
  """
  if len(tf.shape(concatenated)) != 1:
    raise ValueError(
        f'rank of input tensor is {tf.shape(concatenated)}, expected 1.')
  return [
      tf.reshape(flat_tensor, shape=shape) for flat_tensor, shape in zip(
          tf.split(concatenated, list_of_sizes), list_of_shapes)
  ]


def projection(projection_matrix: tf.Tensor,
               flattened_vector: tf.Tensor) -> tf.Tensor:
  """Projects `flattened_vector` using `projection_matrix`.

  Args:
    projection_matrix: A rank-2 Tensor that specifies the projection.
    flattened_vector: A flat Tensor to be projected

  Returns:
    A flat Tensor returned from projection.
  """
  return tf.reshape(
      projection_matrix @ (tf.transpose(projection_matrix) @ tf.reshape(
          flattened_vector, [-1, 1])), [-1])


@tf.function
def client_update(model,
                  dataset,
                  server_message,
                  client_optimizer,
                  projection_matrix=None):
  """Performans client local training of `model` on `dataset`.

  Args:
    model: A `tff.learning.Model`.
    dataset: A 'tf.data.Dataset'.
    server_message: A `BroadcastMessage` from server.
    client_optimizer: A `tf.keras.optimizers.Optimizer`.
    projection_matrix: A projection matrix used to project updates;
      if unspecified, no projection is done.

  Returns:
    A 'ClientOutput`.
  """
  model_weights = get_model_weights(model)
  initial_weights = server_message.model_weights
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        initial_weights)

  num_examples = tf.constant(0, dtype=tf.int32)
  loss_sum = tf.constant(0, dtype=WEIGHT_DENOM_TYPE)
  # Explicit use `iter` for dataset is a trick that makes TFF more robust in
  # GPU simulation and slightly more performant in the unconventional usage
  # of large number of small datasets.
  for batch in iter(dataset):
    with tf.GradientTape() as tape:
      outputs = model.forward_pass(batch)
    grads = tape.gradient(outputs.loss, model_weights.trainable)
    client_optimizer.apply_gradients(zip(grads, model_weights.trainable))
    batch_size = tf.shape(batch['x'])[0]
    num_examples += batch_size
    loss_sum += outputs.loss * tf.cast(batch_size, WEIGHT_DENOM_TYPE)

  weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                        model_weights.trainable,
                                        initial_weights.trainable)

  flattened_weights_delta, list_of_sizes, list_of_shapes = flatten_list_of_tensors(
      weights_delta)

  if projection_matrix is not None:
    projected_flattened_weights_delta = projection(projection_matrix,
                                                   flattened_weights_delta)
  else:
    projected_flattened_weights_delta = flattened_weights_delta

  projected_weights_delta = reshape_flattened_tensor(
      projected_flattened_weights_delta, list_of_sizes, list_of_shapes)
  weights_delta = projected_weights_delta
  client_weight = tf.cast(num_examples, WEIGHT_DENOM_TYPE)
  return ClientOutput(weights_delta, client_weight, loss_sum / client_weight)
  # Note that loss_sum corresponds to the loss of the weights before projection
