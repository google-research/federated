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
from typing import List, Sequence, Tuple, Union

from absl import logging
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
    if isinstance(batch_input, dict):
      x = batch_input['x']
      y = batch_input['y']
    else:
      x, y = batch_input
    preds = self.keras_model(x, training=training)
    loss = self.loss(y, preds)
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
  """A function evaluates the model on test data.

  Args:
    model: a `tf.keras.Model` object
    test_data: a `tf.data.Dataset` object
    metric: a `tf.keras.metric` object

  Returns:
    statisitic computed by metric object.
  """
  metric.reset_states()
  for batch in test_data:
    if isinstance(batch, dict):
      x = batch['x']
      y = batch['y']
    else:
      x, y = batch
    preds = model(x, training=False)
    metric.update_state(y_true=y, y_pred=preds)
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
  round_num = attr.ib()
  shrink_unshrink_dynamic_info = attr.ib(factory=list)


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
  shrink_unshrink_dynamic_info = attr.ib(factory=list)


@attr.s(eq=False, frozen=True, slots=True)
class LayerwiseProjectionShrinkUnshrinkInfoV2(object):
  """Structure for state on the server.

  Fields:
  -   `model_weights`: A dictionary of model's trainable variables.
  -   `optimizer_state`: Variables of optimizer.
  -   'round_num': Current round index
  """
  left_mask = attr.ib()
  right_mask = attr.ib()
  build_projection_matrix = attr.ib()


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
    projection_matrix: A projection matrix used to project updates; if
      unspecified, no projection is done.

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
    if isinstance(batch, dict):
      x = batch['x']
      y = batch['y']
    else:
      x, y = batch
    del y
    batch_size = tf.shape(x)[0]
    num_examples += batch_size
    loss_sum += outputs.loss * tf.cast(batch_size, WEIGHT_DENOM_TYPE)

  weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                        model_weights.trainable,
                                        initial_weights.trainable)
  if projection_matrix is not None:
    flattened_weights_delta, list_of_sizes, list_of_shapes = flatten_list_of_tensors(
        weights_delta)
    projected_flattened_weights_delta = projection(projection_matrix,
                                                   flattened_weights_delta)
    projected_weights_delta = reshape_flattened_tensor(
        projected_flattened_weights_delta, list_of_sizes, list_of_shapes)
    weights_delta = projected_weights_delta

  client_weight = tf.cast(num_examples, WEIGHT_DENOM_TYPE)
  return ClientOutput(
      weights_delta=weights_delta,
      client_weight=client_weight,
      model_output=loss_sum / client_weight,
      round_num=server_message.round_num,
      shrink_unshrink_dynamic_info=server_message.shrink_unshrink_dynamic_info)
  # Note that loss_sum corresponds to the loss of the weights before projection


def left_right_multiply(left_matrix, mid_matrix, right_matrix, is_left_scalar,
                        is_right_scalar):
  """Mutiplies three quantities together.

  Args:
    left_matrix: A tensor of rank <= 2
    mid_matrix: A tensor of rank <= 2
    right_matrix: A tensor of rank <= 2
    is_left_scalar: has value -1 if left_matrix has rank < 2
    is_right_scalar: has value -1 if right_matrix has rank < 2

  Returns:
    A tensor of rank <= 2
  """
  if is_left_scalar == -1:
    return_val = left_matrix * mid_matrix
  else:
    return_val = left_matrix @ mid_matrix

  if is_right_scalar == -1:
    return_val = return_val * right_matrix
  else:
    return_val = return_val @ right_matrix
  return return_val


@tf.function
def project_server_weights(server_state, left_maskval_to_projmat_dict,
                           left_mask, right_mask):
  """Projects the weights stored in `server_state` using information from other arguments.

  Args:
    server_state: A `ServerState`.
    left_maskval_to_projmat_dict: A dictionary mapping the values contained in
      `left_mask` and `right_mask` to matrices.
    left_mask: A list of equal length as the weight matrices in `server_state`.
      The value `k` in the list at index `i` indicates that the weight at
      index`i` should be left multiplied by `left_maskval_to_projmat_dict[k]`
    right_mask: A list of equal length as the weight matrices in `server_state`.
      The value `k` in the list at index `i` indicates that the weight at
      index`i` should be right multiplied by `left_maskval_to_projmat_dict[k]`

  Returns:
    A `ServerState`.
  """

  left_projection_mat_lst = [
      left_maskval_to_projmat_dict[val] for val in left_mask
  ]
  right_projection_mat_lst = [
      tf.transpose(left_maskval_to_projmat_dict[val]) for val in right_mask
  ]
  flat_mask = get_flat_mask(server_state.model_weights.trainable)
  reshaped_model_weights = reshape_flattened_weights(
      server_state.model_weights.trainable, flat_mask)

  new_model_weights = tf.nest.map_structure(left_right_multiply,
                                            left_projection_mat_lst,
                                            reshaped_model_weights,
                                            right_projection_mat_lst, left_mask,
                                            right_mask)
  logging.info('starting reflatten')
  flattened_new_model_weights = flatten_reshaped_weights(
      new_model_weights, flat_mask)
  new_model_weights = tff.structure.update_struct(
      server_state.model_weights,
      trainable=flattened_new_model_weights,
      non_trainable=server_state.model_weights.non_trainable)
  return tff.structure.update_struct(
      server_state,
      model_weights=new_model_weights,
      optimizer_state=server_state.optimizer_state,
      round_num=server_state.round_num)


@tf.function
def unproject_client_weights(client_output, left_maskval_to_projmat_dict,
                             left_mask, right_mask):
  """Unprojects the weights stored in `client_output` using information from other arguments.

  Args:
    client_output: A `ClientOutput`.
    left_maskval_to_projmat_dict: A dictionary mapping the values contained in
      `left_mask` and `right_mask` to matrices.
    left_mask: A list of equal length as the weight matrices in `client_output`.
      The value `k` in the list at index `i` indicates that the weight at
      index`i` should be left multiplied by `left_maskval_to_projmat_dict[k]`
    right_mask: A list of equal length as the weight matrices in
      `client_output`. The value `k` in the list at index `i` indicates that the
      weight at index`i` should be right multiplied by
      `left_maskval_to_projmat_dict[k]`

  Returns:
    A `ClientOutput`.
  """

  left_projection_mat_lst = [
      tf.transpose(left_maskval_to_projmat_dict[val]) for val in left_mask
  ]
  right_projection_mat_lst = [
      left_maskval_to_projmat_dict[val] for val in right_mask
  ]
  flat_mask = get_flat_mask(client_output.weights_delta)
  reshaped_weights_delta = reshape_flattened_weights(
      client_output.weights_delta, flat_mask)
  new_weights_delta = tf.nest.map_structure(left_right_multiply,
                                            left_projection_mat_lst,
                                            reshaped_weights_delta,
                                            right_projection_mat_lst, left_mask,
                                            right_mask)
  flattened_new_weights_delta = flatten_reshaped_weights(
      new_weights_delta, flat_mask)

  # gTODO should model_output be modified?
  return ClientOutput(
      weights_delta=flattened_new_weights_delta,
      client_weight=client_output
      .client_weight,  # note this is not the model weights
      model_output=client_output.model_output,
      round_num=client_output.round_num)


@tf.function
def get_flat_mask(weights_lst):
  """Returns which elements in list are flat tensors.

  Args:
    weights_lst: a list of tensors.

  Returns:
    a list of 1's and 0's of equal length to weights_lst. A 1 means the
    corresponding tesnor in weights_lst is flat; 0 means it is not flat.
  """

  def is_flat(x):
    return 1 if tf.less_equal(tf.rank(x), 1) else 0

  return tf.nest.map_structure(is_flat, weights_lst)


@tf.function
def reshape_flattened_weights(weights_lst, flat_mask):
  """Unflattens weights using information from flat_mask.

  Args:
    weights_lst: a list of tensors.
    flat_mask: a list of 1's and 0's

  Returns:
    A list of tensors consisting of tensors from weights_lst which have been
    reshaped into a rank 2 tensor if the corresponding flat_mask value is equal
    to 1.
  """

  def unflatten(x, m):
    return tf.reshape(x, (1, -1)) if tf.equal(m, 1) else tf.identity(x)

  return tf.nest.map_structure(unflatten, weights_lst, flat_mask)


@tf.function
def flatten_reshaped_weights(weights_lst, flat_mask):
  """Flattens weights using information from flat_mask.

  Args:
    weights_lst: a list of tensors.
    flat_mask: a list of 1's and 0's

  Returns:
    A list of tensors consisting of tensors from weights_lst which have been
    flattened if the corresponding flat_mask value is equal to 1.
  """

  def flatten(x, m):
    return tf.reshape(x, [-1]) if tf.equal(m, 1) else tf.identity(x)

  return tf.nest.map_structure(flatten, weights_lst, flat_mask)


@tf.function
def build_dropout_projection_matrix(seed, desired_shape, is_left_multiply=True):
  """Builds a random projection matrix that corresponds to random dropout.

  Args:
    seed: a tuple of two integers which is the seed
    desired_shape: desired shape of the random projection matrix
    is_left_multiply: a boolean specifying whether the projection matrix is
      being left or right multiplied

  Returns:
    A matrix with values of 0 or 1 in the shape of desired_shape
  """
  if is_left_multiply:
    num_to_select = desired_shape[0]
    num_cand_indices = desired_shape[1]
    one_indices = tf.argsort(
        tf.random.stateless_uniform(shape=[num_cand_indices],
                                    seed=seed))[:num_to_select]
    return tf.one_hot(one_indices, depth=num_cand_indices)
  else:
    num_to_select = desired_shape[1]
    num_cand_indices = desired_shape[0]
    one_indices = tf.argsort(
        tf.random.stateless_uniform(shape=[num_cand_indices],
                                    seed=seed))[:num_to_select]
    return tf.transpose(tf.one_hot(one_indices, depth=num_cand_indices))


@tf.function
def build_normal_projection_matrix(seed, desired_shape, is_left_multiply=True):
  """Builds an random (approximate) projection matrix using a Gaussian sketch.

  Args:
    seed: a tuple of two integers which is the seed
    desired_shape: desired shape of the random projection matrix
    is_left_multiply: a boolean specifying whether the projection matrix is
      being left or right multiplied

  Returns:
    A real valued projection matrix in the shape of desired_shape
  """
  if is_left_multiply:
    latent_dim = tf.cast(desired_shape[0], tf.float32)
  else:
    latent_dim = tf.cast(desired_shape[1], tf.float32)
  return tf.random.stateless_normal(
      shape=desired_shape, seed=seed, stddev=1 / tf.sqrt(latent_dim))


@tf.function
def build_qr_projection_matrix(seed, desired_shape, is_left_multiply=True):
  """Builds a random projection matrix using a QR decomposition.

  Does the same operation as tf.keras.initializers.Orthogonal. Note that the
  random projection matrix generated does not necessarilly follow the Haar
  distribution.

  Look at these references for more information:
  https://stackoverflow.com/questions/48364447/generating-a-tall-and-thin-random-orthonormal-matrix-in-scipy
  Section 4 of https://arxiv.org/pdf/math-ph/0609050.pdf

  Args:
    seed: a tuple of two integers which is the seed
    desired_shape: desired shape of the random projection matrix
    is_left_multiply: a boolean specifying whether the projection matrix is
      being left or right multiplied

  Returns:
    A real valued projection matrix in the shape of desired_shape
  """
  if is_left_multiply:
    tf.debugging.assert_less_equal(desired_shape[0], desired_shape[1])

    # need to transpose to get columns of random_mat to be orthogonal
    transposed_desired_shape = (desired_shape[1], desired_shape[0])
    random_mat = tf.random.stateless_normal(
        shape=transposed_desired_shape, seed=seed, stddev=1)

    q, _ = tf.linalg.qr(random_mat)
    return tf.transpose(q)
  else:
    tf.debugging.assert_less_equal(desired_shape[1], desired_shape[0])
    random_mat = tf.random.stateless_normal(
        shape=desired_shape, seed=seed, stddev=1)
    q, _ = tf.linalg.qr(random_mat)
    return q


@tf.function
def create_left_maskval_to_projmat_dict(seed, whimsy_server_weights,
                                        whimsy_client_weights, left_mask,
                                        right_mask, build_projection_matrix):
  """Creates a dictionary mapping the values of `left_mask` and `right_mask` to projection matrices.

  Args:
    seed: A integer used for seeding
    whimsy_server_weights: A list of weight matrices
    whimsy_client_weights: A list of weight matrices
    left_mask: A list of equal length as the weight matrices in `client_output`.
      The value `k` in the list at index `i` indicates that the weight at
      index`i` should be left multiplied by `left_maskval_to_projmat_dict[k]`
    right_mask: A list of equal length as the weight matrices in
      `client_output`. The value `k` in the list at index `i` indicates that the
      weight at index`i` should be right multiplied by
      `left_maskval_to_projmat_dict[k]`
    build_projection_matrix: A function which builds the projection_matrix
      matrix used in this function

  Returns:
    A dictionary mapping the values of `left_mask` and `right_mask` to
    projection matrices.
  """
  logging.info('starting create_left_maskval_projmat_dict')
  server_flat_mask = get_flat_mask(whimsy_server_weights)
  whimsy_server_trainable_variables = reshape_flattened_weights(
      whimsy_server_weights, server_flat_mask)
  # gTODO change whimsy_server_weights
  # to whimsy_server_model.trainable_variables?
  client_flat_mask = get_flat_mask(whimsy_client_weights)
  whimsy_client_trainable_variables = reshape_flattened_weights(
      whimsy_client_weights, client_flat_mask)
  tf.debugging.assert_equal(
      server_flat_mask, client_flat_mask,
      'there should be a one-to-one correspondence between the flat tensors in server and client models.'
  )
  tf.random.set_seed(seed)
  left_maskval_to_projmat_dict = dict()

  for idx, val in enumerate(left_mask):
    server_weight_mat_shape = tf.shape(whimsy_server_trainable_variables[idx])
    client_weight_mat_shape = tf.shape(whimsy_client_trainable_variables[idx])
    desired_shape = (client_weight_mat_shape[0], server_weight_mat_shape[0]
                    )  # left multiply server_mat to generate client_mat

    if val < 0:
      left_maskval_to_projmat_dict[val] = tf.ones([1], dtype=tf.float32)
      tf.debugging.assert_equal(server_weight_mat_shape[0],
                                client_weight_mat_shape[0])
    elif val not in left_maskval_to_projmat_dict:
      projection_matrix = build_projection_matrix(
          seed=(seed, idx), desired_shape=desired_shape, is_left_multiply=True)

      left_maskval_to_projmat_dict[val] = projection_matrix
      actual_shape = tf.shape(left_maskval_to_projmat_dict[val])
      tf.debugging.assert_equal(actual_shape, desired_shape)
    else:
      actual_mat = left_maskval_to_projmat_dict[val]
      actual_shape = tf.shape(actual_mat)
      tf.debugging.assert_equal(actual_shape, desired_shape)

  for idx, val in enumerate(right_mask):
    server_weight_mat_shape = tf.shape(whimsy_server_trainable_variables[idx])
    client_weight_mat_shape = tf.shape(whimsy_client_trainable_variables[idx])
    desired_shape = (server_weight_mat_shape[1], client_weight_mat_shape[1]
                    )  # right multiply server_mat to generate client_mat
    if val < 0:
      left_maskval_to_projmat_dict[val] = tf.ones([1], dtype=tf.float32)
      tf.debugging.assert_equal(server_weight_mat_shape[1],
                                client_weight_mat_shape[1])
    elif val not in left_maskval_to_projmat_dict:
      projection_matrix = build_projection_matrix(
          seed=(seed, idx), desired_shape=desired_shape, is_left_multiply=False)

      left_maskval_to_projmat_dict[val] = tf.transpose(projection_matrix)
      actual_shape = tf.shape(tf.transpose(left_maskval_to_projmat_dict[val]))
      tf.debugging.assert_equal(actual_shape, desired_shape)
    else:
      actual_mat = tf.transpose(left_maskval_to_projmat_dict[val]
                               )  # transposed because of right multiply
      actual_shape = tf.shape(actual_mat)
      tf.debugging.assert_equal(actual_shape, desired_shape)
  logging.info('finished create_left_maskval_projmat_dict')
  return left_maskval_to_projmat_dict
