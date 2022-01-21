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
"""An implementation of FedTKM for periodic distribution shift.

The core TFF iterative process implments the federated averaging algorithm
together with a federated version of k-means clustering to train the
mutli-branch networks for periodic distribution shift. Note that part of the
server updates are lifted out of the TFF iterative process and defined in
python.
"""

import collections
import math
from typing import Callable, Optional, Union

import attr
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from utils import tensor_utils

# Convenience type aliases.
ModelBuilder = Callable[[], tff.learning.Model]
OptimizerBuilder = Callable[[float], tf.keras.optimizers.Optimizer]
ClientWeightFn = Callable[..., float]
LRScheduleFn = Callable[[Union[int, tf.Tensor]], Union[tf.Tensor, float]]


def initialize_optimizer_vars(model: tff.learning.Model,
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


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of the model's trainable and non-trainable
        weights.
  -   `optimizer_state`: The server optimizer variables.
  -   `round_num`: The current training round, as a float.
  -   `kmeans_centers`: Number of cluster centers. Currently only supports two.
  -   `dist_scalar`: The adaptive scalar for rescaling the distances to the
          cluster 0.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()
  kmeans_centers = attr.ib()
  dist_scalar = attr.ib()


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
  -   `optimizer_output`: Additional metrics or other outputs defined by the
      optimizer.
  -   `kmeans_deltas`: Updates to the cluster centers.
  -   `kmeans_n_samples`: Number of samples used to update each center.
  -   `cluster1_ratio`: A 0-1 variable indicating whether this client was
      assigned to update cluster 1.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()
  kmeans_deltas = attr.ib()
  kmeans_n_samples = attr.ib()
  cluster1_ratio = attr.ib()


@attr.s(eq=False, order=False, frozen=True)
class ClientOutputNoKmeans(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `client_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `optimizer_output`: Additional metrics or other outputs defined by the
      optimizer.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


def create_client_update_no_kmeans_fn():
  """Returns a tf.function for the client_update.

  This "create" fn is necesessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is really only needed because we test the client_update function directly.
  """
  @tf.function
  def client_update(
      model,
      dataset,
      initial_weights,
      client_optimizer,
      client_weight_fn=None,
  ):
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

    # The training loop
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
    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))

    if has_non_finite_weight > 0:
      client_weight = tf.constant(0, dtype=tf.float32)
    elif client_weight_fn is None:
      client_weight = tf.cast(num_examples, dtype=tf.float32)
    else:
      client_weight = client_weight_fn(aggregated_outputs)

    return ClientOutputNoKmeans(
        weights_delta, client_weight, aggregated_outputs,
        collections.OrderedDict([('num_examples', num_examples)]))

  return client_update


def create_client_aggregated_update_fn():
  """Returns a tf.function for the client_update.

  This "create" fn is necesessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is really only needed because we test the client_update function directly.
  """

  def branch_selection(dataset, model, kmeans_centers, kmeans_k, dist_scalar):
    # This includes the case of vanilla kmeans
    total_examples = tf.constant(0, dtype=tf.float32)
    total_g1_examples = tf.constant(0, dtype=tf.float32)
    for batch in iter(dataset):
      output = model.forward_pass(batch, get_head_scores=True)
      total_examples += tf.cast(tf.shape(output.features)[0], tf.float32)
      dists = tf.norm(
          tf.expand_dims(output.features, axis=1) - kmeans_centers, axis=-1)
      c1_dists = tf.reduce_min(dists[:, :kmeans_k // 2], axis=1) * dist_scalar
      c2_dists = tf.reduce_min(dists[:, kmeans_k // 2:], axis=1)
      c1_inds = tf.cast(tf.math.less(c1_dists, c2_dists), tf.float32)
      total_g1_examples += tf.reduce_sum(c1_inds)
    total_g2_examples = total_examples - total_g1_examples
    main_branch = tf.cond(
        tf.math.greater(total_g1_examples, total_g2_examples),
        lambda: tf.constant(0), lambda: tf.constant(1))

    return main_branch, total_examples

  def update_kmeans_centers(feature_dim, dataset, model, kmeans_centers,
                            dist_scalar, total_examples):
    feature_mean = tf.zeros([1, feature_dim], dtype=tf.float32)
    g0_votes = tf.constant(0.)
    for batch in iter(dataset):
      output = model.forward_pass(batch, get_head_scores=True)

      feature_mean += tf.reduce_sum(
          output.features, axis=0, keepdims=True) / total_examples

      dists = tf.norm(
          tf.expand_dims(output.features, axis=1) - kmeans_centers, axis=-1)

      dshape = tf.shape(dists)
      grouped_dists = tf.reshape(dists, [dshape[0], 2, dshape[1] // 2])
      min_dist_ = tf.math.reduce_min(grouped_dists, axis=2)

      # We use the scaled distance to decide the assignment
      dist0 = min_dist_[:, :1] * dist_scalar
      g0_inds = tf.cast(tf.math.less(dist0, min_dist_[:, 1:]), tf.float32)
      g0_votes += tf.reduce_sum(g0_inds)

    new_center = feature_mean
    zeros_delta = tf.zeros_like(feature_mean)
    center0_update = tf.concat([new_center - kmeans_centers[:, 0], zeros_delta],
                               axis=0)
    center1_update = tf.concat([zeros_delta, new_center - kmeans_centers[:, 1]],
                               axis=0)
    cluster0_cond = tf.math.greater(g0_votes, total_examples - g0_votes)
    kmeans_deltas = tf.cond(cluster0_cond,
                            lambda: center0_update * total_examples,
                            lambda: center1_update * total_examples)
    center_samples = tf.cond(cluster0_cond,
                             lambda: tf.constant([[1.], [0.]]) * total_examples,
                             lambda: tf.constant([[0.], [1.]]) * total_examples)
    # Here we convert the index of cluster centers to 1-based. Previously it
    # was 0-based.
    is_cluster1 = tf.cast(cluster0_cond, tf.float32)
    return kmeans_deltas, center_samples, is_cluster1

  @tf.function
  def client_update(model,
                    dataset,
                    initial_weights,
                    client_optimizer,
                    client_weight_fn=None,
                    kmeans_centers=None,
                    kmeans_k=2,
                    feature_dim=128,
                    dist_scalar=1.,
                    clip_norm=-1.):
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
      kmeans_centers: The cluster centers.
      kmeans_k: Number of clusters for k-means.
      feature_dim: Dimension of the feature space in which k-means is defined.
      dist_scalar: Used to rescale the distance on cluster 0.
      clip_norm: Maximum norm for gradient clipping.

    Returns:
      A 'ClientOutput` with the updates to the network weights and cluster
        centers.
    """
    model_weights = _get_weights(model)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          initial_weights)

    # Assign the client to the branches according to results from K-means
    kmeans_centers = tf.expand_dims(kmeans_centers, axis=0)
    main_branch, total_examples = branch_selection(dataset, model,
                                                   kmeans_centers, kmeans_k,
                                                   dist_scalar)
    train_kmeans_centers = kmeans_centers[0]

    # The training loop
    num_examples = tf.constant(0, dtype=tf.int32)
    for batch in iter(dataset):
      with tf.GradientTape() as tape:
        output = model.forward_pass(
            batch, main_branch=main_branch, kmeans_centers=train_kmeans_centers)
      grads = tape.gradient(output.loss, model_weights.trainable)
      grads_and_vars = zip(grads, model_weights.trainable)
      client_optimizer.apply_gradients(grads_and_vars)
      num_examples += tf.shape(output.predictions)[0]

    aggregated_outputs = model.report_local_outputs()
    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)
    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))

    # Clip the weight deltas.
    if clip_norm > 0 and not has_non_finite_weight > 0:
      flatten_weights_delta = tf.nest.flatten(weights_delta)
      clipped_flatten_weights_delta, _ = tf.clip_by_global_norm(
          flatten_weights_delta, clip_norm)
      weights_delta = tf.nest.pack_sequence_as(weights_delta,
                                               clipped_flatten_weights_delta)

    if has_non_finite_weight > 0:
      client_weight = tf.constant(0, dtype=tf.float32)
    elif client_weight_fn is None:
      client_weight = tf.cast(num_examples, dtype=tf.float32)
    else:
      client_weight = client_weight_fn(aggregated_outputs)

    kmeans_deltas, center_samples, is_cluster1 = update_kmeans_centers(
        feature_dim, dataset, model, kmeans_centers, dist_scalar,
        total_examples)

    return ClientOutput(
        weights_delta,
        client_weight,
        aggregated_outputs,
        collections.OrderedDict([('num_examples', num_examples)]),
        kmeans_deltas=kmeans_deltas,
        kmeans_n_samples=center_samples,
        cluster1_ratio=is_cluster1,
    )

  return client_update


def build_server_init_fn(
    model_fn: ModelBuilder, kmeans_k: int,
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    feature_dim: int):
  """Builds a `tff.tf_computation` that returns the initial `ServerState`.

  The attributes `ServerState.model` and `ServerState.optimizer_state` are
  initialized via their constructor functions. The attribute
  `ServerState.round_num` is set to 0.0.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    kmeans_k: Number of cluster centers. Currently only supports two.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    feature_dim: Dimenstion of the feature space.

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
  """

  @tff.tf_computation
  def server_init_tf():
    server_optimizer = server_optimizer_fn()
    model = model_fn()
    initialize_optimizer_vars(model, server_optimizer)
    model_weights = _get_weights(model)

    kmeans_centers = tf.constant(
        np.random.normal(scale=0.1,
                         size=(kmeans_k, feature_dim)).astype(np.float32))
    dist_scalar = tf.constant(1., tf.float32)
    return ServerState(
        model=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=0.0,
        kmeans_centers=kmeans_centers,
        dist_scalar=dist_scalar)

  return server_init_tf


def build_fed_avg_process(
    model_fn: ModelBuilder,
    client_optimizer_fn: OptimizerBuilder,
    client_lr: Union[float, LRScheduleFn] = 0.1,
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_lr: Union[float, LRScheduleFn] = 1.0,
    client_weight_fn: Optional[ClientWeightFn] = None,
    kmeans_k: int = 2,
    feature_dim: int = 128,
    aggregated_kmeans: bool = False,
    clip_norm: float = -1.) -> tff.templates.IterativeProcess:
  """Builds the TFF computations for optimization using federated averaging.

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
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.
    kmeans_k: Number of centers for k-means. Currently only support 2 centers.
    feature_dim: Dimension of the feature space where k-means is based on.
    aggregated_kmeans: Whether to use aggregated k-means.
    clip_norm: the maximum norm for gradient clipping.

  Returns:
    A `tff.templates.IterativeProcess`.
  """
  if kmeans_k != 2:
    raise ValueError(f'Currently only supports two centers. '
                     f'Got kmeans_k={kmeans_k}.')

  # In most cases, we use constant learning rates.
  client_lr_schedule = client_lr
  if not callable(client_lr_schedule):
    client_lr_schedule = lambda round_num: client_lr

  server_lr_schedule = server_lr
  if not callable(server_lr_schedule):
    server_lr_schedule = lambda round_num: server_lr

  placeholder_model = model_fn()

  server_init_tf = build_server_init_fn(
      model_fn,
      kmeans_k,
      # Initialize with the learning rate for round zero.
      lambda: server_optimizer_fn(server_lr_schedule(0)),
      feature_dim=feature_dim)
  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model
  round_num_type = server_state_type.round_num
  kmeans_centers_type = server_state_type.kmeans_centers
  dist_scalar_type = server_state_type.dist_scalar

  tf_dataset_type = tff.SequenceType(placeholder_model.input_spec)
  model_input_type = tff.SequenceType(placeholder_model.input_spec)

  @tff.tf_computation(model_input_type, model_weights_type, round_num_type,
                      kmeans_centers_type, dist_scalar_type)
  def client_update_fn(tf_dataset, initial_model_weights, round_num,
                       kmeans_centers, dist_scalar):
    client_lr = client_lr_schedule(round_num)
    client_optimizer = client_optimizer_fn(client_lr)
    if aggregated_kmeans:
      client_update = create_client_aggregated_update_fn()
      return client_update(
          model_fn(),
          tf_dataset,
          initial_model_weights,
          client_optimizer,
          client_weight_fn,
          kmeans_centers=kmeans_centers,
          kmeans_k=kmeans_k,
          feature_dim=feature_dim,
          dist_scalar=dist_scalar,
          clip_norm=clip_norm)
    else:
      client_update = create_client_update_no_kmeans_fn()
      return client_update(model_fn(), tf_dataset, initial_model_weights,
                           client_optimizer, client_weight_fn)

  @tff.federated_computation(
      tff.type_at_server(server_state_type),
      tff.type_at_clients(tf_dataset_type),
  )
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
    client_round_num = tff.federated_broadcast(server_state.round_num)
    kmeans_centers = tff.federated_broadcast(server_state.kmeans_centers)
    dist_scalar = tff.federated_broadcast(server_state.dist_scalar)

    client_outputs = tff.federated_map(
        client_update_fn, (federated_dataset, client_model, client_round_num,
                           kmeans_centers, dist_scalar))

    client_weight = client_outputs.client_weight
    model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=client_weight)
    aggregated_outputs = placeholder_model.federated_output_computation(
        client_outputs.model_output)
    if aggregated_outputs.type_signature.is_struct():
      aggregated_outputs = tff.federated_zip(aggregated_outputs)
    if not aggregated_kmeans:
      return server_state, aggregated_outputs, model_delta

    # Federated mean will return NaN when all weights are zeros.
    # This happens when all clients are assigned to one of the clusters, so the
    # other cluster will have zero sample for the k-means update.
    # To handle this, we get the federated sums first and then handle
    # dividing by zero on the server.
    kmeans_delta_sum = tff.federated_sum(client_outputs.kmeans_deltas)
    kmeans_n_samples = tff.federated_sum(client_outputs.kmeans_n_samples)
    cluster1_ratio = tff.federated_mean(client_outputs.cluster1_ratio)

    return (server_state, aggregated_outputs, model_delta, kmeans_delta_sum,
            kmeans_n_samples, cluster1_ratio)

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


@tf.function
def server_update(model,
                  server_optimizer,
                  server_state,
                  weights_delta,
                  updated_centers,
                  dist_scalar=1.):
  """Updates `server_state` based on `weights_delta`, increase the round number.

  Args:
    model: A `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: An update to the trainable variables of the model.
    updated_centers: The updated kmeans clustering centers.
    dist_scalar: The updated scalar to adopt temporail prior for clustering.

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
      round_num=server_state.round_num + 1.0,
      kmeans_centers=updated_centers,
      dist_scalar=tf.cast(dist_scalar, tf.float32))


def geometric_scalar_from_g1_ratio_fn(cluster1_ratio,
                                      round_num,
                                      period,
                                      dist_scalar,
                                      geo_lr,
                                      prior_fn='linear',
                                      zero_mid=False):
  """Update the distance scalars for clustering based on temporal prior.

  The geometric update rule is inspired by
  "Differentially Private Learning with Adaptive Clipping"
  https://arxiv.org/pdf/1905.03871.pdf.

  Args:
    cluster1_ratio: The actual ratio of clients coming from the first cluster.
    round_num: Current training round.
    period: The period of the distribution shift.
    dist_scalar: The distance scalar applied on the first cluster.
    geo_lr: The step size of the geometric updates.
    prior_fn: The prior function of the distribution shift.
    zero_mid: Whether to set the geometric step size to 0 in the middle of each
      period.

  Returns:
    The updated distance scalar.
  """
  proc = (round_num - 1) % period / period
  if prior_fn == 'linear':
    lin_ratio = 2 * abs(proc - 0.5)
  else:
    lin_ratio = (math.cos(2 * math.pi * proc) + 1.) / 2

  if zero_mid:
    glr = abs(lin_ratio - 0.5) * 2 * geo_lr
  else:
    glr = geo_lr

  # Force converting all the data types to avoid random runtime errors due to
  # mismatched data types (float32 vs. float64).
  cluster1_ratio = tf.cast(cluster1_ratio, tf.float32)
  dist_scalar = tf.cast(dist_scalar, tf.float32)
  glr = tf.cast(glr, tf.float32)
  lin_ratio = tf.cast(lin_ratio, tf.float32)

  dist_scalar = tf.math.exp(glr * (cluster1_ratio - lin_ratio)) * dist_scalar
  return dist_scalar
