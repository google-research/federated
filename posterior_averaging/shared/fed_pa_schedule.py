# Copyright 2020, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An implementation of the FedPA algorithm.

The key difference between FedPA and FedAvg:
  1. The server logic is essentially the same as in FedAvg and admits different
     server optimizers. This functionality is imported from
     `tensorflow_federated/python/research/optimization/shared/`.
  2. The client logic is somewhat similar, since each client still runs SGD
     for the specified number of steps. However, SGD iterates are used to
     construct approximate samples from the local posteriors and estimate
     corrected deltas.

References:
  - Federated Learning via Posterior Averaging:
    A New Perspective and Practical Algorithms.
        Maruan Al-Shedivat, Jennifer Gillenwater, Eric Xing, Afshin Rostamizadeh
        ICLR 2021 (https://arxiv.org/abs/2010.05273)
"""

import collections
import functools
from typing import Callable, Collection, Optional, Union

import attr
import tensorflow as tf
import tensorflow_federated as tff

from utils import tensor_utils

# Convenience type aliases.
ModelBuilder = Callable[[], tff.learning.Model]
OptimizerBuilder = Callable[[float], tf.keras.optimizers.Optimizer]
ClientMixedinFn = Callable[..., bool]
ClientMixedinScheduleFn = Callable[[int], ClientMixedinFn]
ClientUpdateDeltaFn = Callable[..., Collection[tf.Tensor]]
ClientWeightFn = Callable[..., float]
LRScheduleFn = Callable[[Union[int, tf.Tensor]], Union[tf.Tensor, float]]


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
  return tff.learning.ModelWeights(
      trainable=tuple(model.trainable_variables),
      non_trainable=tuple(model.non_trainable_variables))


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
def server_update(model, server_optimizer, server_state, weights_delta):
  """Updates `server_state` based on `weights_delta`, increase the round number.

  Args:
    model: A `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: An update to the trainable variables of the model.

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
      round_num=server_state.round_num + 1.0)


@attr.s(eq=False, order=False, frozen=True)
class DataPassOutput(object):
  """Structure for outputs returned by a single data pass.

  Fields:
  - `loss`: A scalar tensor that represents the average loss during the pass.
  - `num_examples`: A scalar tensor that represents the number of seen examples.
  - `model_weights_trainable`: A dictionary of model's trainable variables that
      contain the weights obtained at the end of the data pass.
  - `model_weights_trainable_sample`: A dictionary of model's trainable
      variables that contain a candidate posterior sample.
  """
  loss = attr.ib()
  num_examples = attr.ib()
  model_weights_trainable = attr.ib()
  model_weights_trainable_sample = attr.ib()


@attr.s(eq=False, order=False, frozen=True)
class DeltaUpdateOutput(object):
  """Structure for outputs returned by delta update functions.

  This structure contains the sufficient state of the dynamic program used by
  FedPA to update weight deltas given a new approximate posterior sample.
  Each field described below is a dictionary of the same structure as model's
  trainable variables.

  Fields:
  - `num_samples`: The number of approximate posterior samples so far.
  - `weights_delta`: Updates to model's trainable variables.
  - `weights_sample_mean`: The mean of the all posterior samples so far.
      `n`-th sample from the mean of the previous `(n - 1)` samples.
  - `recursion_state`: The state of the dynamic programming recursion
      represented by a structure of `tf.TensorArray`s.
  """
  num_samples = attr.ib()
  weights_delta = attr.ib()
  weights_sample_mean = attr.ib(None)
  recursion_state = attr.ib(default=None)

  @classmethod
  def from_weights(cls, initial_weights, updated_weights, num_samples=0):
    """Creates a delta update output from intial and updated weights."""
    # Initialize updates.
    weights_delta = tf.nest.map_structure(lambda a, b: a - b, updated_weights,
                                          initial_weights)
    # Initial state for dynamic programming updates.
    recursion_state_init = (tf.nest.map_structure(
        lambda d: tf.TensorArray(d.dtype, size=0, dynamic_size=True),
        weights_delta), tf.TensorArray(tf.float32, size=0, dynamic_size=True))
    return cls(
        num_samples=tf.cast(num_samples, dtype=tf.float32),
        weights_delta=weights_delta,
        weights_sample_mean=updated_weights,
        recursion_state=recursion_state_init)


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
  -   `additional_output`: A dictionary of additional outputs that may contain
      various statistics from the local execution.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()
  additional_output = attr.ib()


def create_mixin_check_fn(name, *, num_mixin_epochs=0, start_round=0):
  """Creates a function that checks whether the client has mixed in."""

  if name == 'fixed_epochs':
    mixin_check_fn = functools.partial(
        _mixin_check_fixed_epochs_fn, num_mixin_epochs=num_mixin_epochs)
  else:
    raise ValueError(f'Unknown mixin check: {name}.')

  def _mixedin_schedule_fn(round_num):
    """Returns mixin check function for the given round number."""

    @tf.function
    def _mixin_check_fn(*args, **kwargs):
      return tf.cond(
          tf.math.greater_equal(round_num, start_round),
          true_fn=lambda: mixin_check_fn(*args, **kwargs),
          false_fn=lambda: False)

    return _mixin_check_fn

  return _mixedin_schedule_fn


@tf.function
def _mixin_check_fixed_epochs_fn(epoch, *_, num_mixin_epochs):
  return tf.math.greater_equal(epoch, num_mixin_epochs)


def create_update_delta_fn(name, *, rho=1.):
  """Creates a function for updating weights delta based on weight samples."""

  if name == 'simple_avg':
    update_delta_fn = _update_delta_simple_mean
  elif name == 'posterior_avg':
    update_delta_fn = functools.partial(_update_delta_posterior_mean, rho=rho)
  else:
    raise ValueError(f'Unknown delta update: {name}.')

  @tf.function
  def _update_delta(mixedin, initial_weights, data_pass_outputs,
                    previous_updates):
    """Wraps `update_delta_fn` and computes updates for the common case.

    Args:
      mixedin: A scalar boolean tensor indicating whether SG-MCMC has mixed-in.
      initial_weights: A `tff.learning.Model.weights` from server.
      data_pass_outputs: A `DataPassOutputs` structure returned by a single data
        pass and contains the epoch loss, the final iterate and an approximate
        posterior sample of the model's trainable weights.
      previous_updates: A `DeltaUpdateOutputs` structure that contains the
        information about previous updates to the model's trainable variables.

    Returns:
      A `DeltaUpdateOutputs` structure.
    """
    if tf.logical_not(mixedin):
      # If not mixed-in yet, compute delta based on the last weights iterate.
      # This computation is common across different delta update methods.
      weights_delta = tf.nest.map_structure(
          lambda a, b: a - b, data_pass_outputs.model_weights_trainable,
          initial_weights.trainable)
      updates = DeltaUpdateOutput(
          weights_delta=weights_delta,
          num_samples=previous_updates.num_samples,
          weights_sample_mean=data_pass_outputs.model_weights_trainable_sample,
          recursion_state=previous_updates.recursion_state)
    else:
      if tf.equal(previous_updates.num_samples, 0.):
        # Special case of the first sample.
        # This computation is common across different delta update methods.
        weights_delta = tf.nest.map_structure(
            lambda a, b: a - b,
            data_pass_outputs.model_weights_trainable_sample,
            initial_weights.trainable)
        updates = DeltaUpdateOutput(
            weights_delta=weights_delta,
            num_samples=tf.add(previous_updates.num_samples, 1),
            weights_sample_mean=(
                data_pass_outputs.model_weights_trainable_sample),
            recursion_state=previous_updates.recursion_state)
      else:
        updates = update_delta_fn(
            data_pass_outputs=data_pass_outputs,
            previous_updates=previous_updates)
    return updates

  return _update_delta


@tf.function
def _update_delta_simple_mean(data_pass_outputs, previous_updates):
  """Updates weights delta using a simple running mean over posterior samples.

  Args:
    data_pass_outputs: A `DataPassOutputs` structure returned by a single data
      pass that contains the epoch loss, the final iterate of model's trainable
      weights, and an approximate posterior sample of model's trainable weights.
    previous_updates: A `DeltaUpdateOutputs` structure that contains information
      about the previous updates to the model's trainable variables.

  Returns:
    A `DeltaUpdateOutputs` structure.
  """
  n = tf.add(previous_updates.num_samples, 1)

  # Update delta based on the new weights sample.
  weights_delta = tf.nest.map_structure(
      lambda d, m, s: d + (s - m) / n, previous_updates.weights_delta,
      previous_updates.weights_sample_mean,
      data_pass_outputs.model_weights_trainable_sample)

  # Update the running mean of the weights samples.
  weights_sample_mean = tf.nest.map_structure(
      lambda a, b: ((n - 1) * a + b) / n, previous_updates.weights_sample_mean,
      data_pass_outputs.model_weights_trainable_sample)

  return DeltaUpdateOutput(
      num_samples=n,
      weights_delta=weights_delta,
      weights_sample_mean=weights_sample_mean,
      recursion_state=previous_updates.recursion_state)


@tf.function
def _struct_dot(struct1, struct2):
  """Computes a dot product between two similar structures of tensors."""
  dot_struct = tf.nest.map_structure(lambda x, y: tf.reduce_sum(x * y), struct1,
                                     struct2)
  return sum(tf.nest.flatten(dot_struct))


@tf.function
def _update_delta_posterior_mean(data_pass_outputs, previous_updates, *, rho):
  r"""Updates weights delta by incrementally re-computing posterior mean.

  The variable naming convention in this function ties to closely follow the
  variable naming used in the original paper (Appendix C):
  https://arxiv.org/pdf/2010.05273.pdf#page=18

  The notation is explained below. For more details, please refer to the paper.

  Notation:
    `n`: The number of posterior samples produced so far (including the sample
      that `data_pass_outputs` argument contains). Denoted `\ell` in the paper.
    `un`: A vector that represents the difference between the new posterior
      sample and the mean of the previous samples.
    `vn`: A vector proportional to `uk` multiplied by the inverse covariance
      estimate based on the previous samples.
    `vk_tas`: A `tf.TensorArray` of size `n_prev` that contains all previous
      `vk` values for k = 1, ..., n - 1.
    `dot_vk_uk_ta`: A `tf.TensorArray` of size `n_prev` that contains dot
      products between all previous `uk` and `vk` vectors for k = 1, ..., n - 1.
    `weights_delta_tilde`: The previous weights delta scaled by a constant,
      `1 / (1 + (n_prev - 1) * rho)`. The scaling is necessary to simplify
      the intermediate algebraic computations.
    `weights_delta`: The updated client delta (computed at the end).

  Args:
    data_pass_outputs: A `DataPassOutputs` structure returned by a single data
      pass that contains the epoch loss, the final iterate of model's trainable
      weights, and an approximate posterior sample of model's trainable weights.
    previous_updates: A `DeltaUpdateOutputs` structure that contains information
      about the previous updates to the model's trainable variables.
    rho: A float that specifies the shrinkage coefficient.

  Returns:
    A `DeltaUpdateOutputs` structure.
  """
  n = tf.add(previous_updates.num_samples, 1)
  vk_tas, dot_vk_uk_ta = previous_updates.recursion_state

  # Rescale previous weights delta to use it in the recursion.
  weights_delta_tilde = tf.nest.map_structure(
      lambda wd: wd / (1 + ((n - 1) - 1) * rho), previous_updates.weights_delta)

  # Update the running mean of the weights samples.
  weights_sample_mean = tf.nest.map_structure(
      lambda a, b: ((n - 1) * a + b) / n, previous_updates.weights_sample_mean,
      data_pass_outputs.model_weights_trainable_sample)

  # Compute u_{n} (deviation of the new sample from the previous mean).
  un_flat = tf.nest.map_structure(
      lambda a, b: tf.reshape(a - b, [-1]),  # Each tensor is flattened.
      data_pass_outputs.model_weights_trainable_sample,
      previous_updates.weights_sample_mean)

  # Compute v_{n-1, n} (solution of `sigma_{n-1} x = u_n`).
  if tf.math.greater(n, 2):
    # Step 1: compute `vk_coeff = gamma * dot(v_k, u_n) / (1 + gamma * uv_k)`.
    gammas_range = 2 + tf.range(n - 2, dtype=tf.float32)
    gammas = rho * (gammas_range - 1) / gammas_range
    dot_vk_un = tf.nest.map_structure(
        lambda vk_ta, u: tf.einsum('ij,j->i', vk_ta.stack(), u), vk_tas,
        un_flat)
    dot_vk_un = sum(tf.nest.flatten(dot_vk_un))
    dot_vk_uk = dot_vk_uk_ta.stack()
    vk_coeffs = gammas * dot_vk_un / (1 + gammas * dot_vk_uk)
    # Step 2: compute `vn = u - sum_k vk_coeff * vk` and `dot(v_n, u_n)`.
    vn_flat = tf.nest.map_structure(
        lambda vk_ta, u: u - tf.einsum('i,ij->j', vk_coeffs, vk_ta.stack()),
        vk_tas, un_flat)
  else:
    # Special case of the second sample.
    vn_flat = un_flat

  # Compute `dot(vn, un)`.
  dot_vn_un = _struct_dot(vn_flat, un_flat)

  # Update the state of the recursion represented by `tf.TensorArrays`.
  i = tf.cast(n - 2, dtype=tf.int32)
  vk_tas = tf.nest.map_structure(lambda vk_ta, vn: vk_ta.write(i, vn), vk_tas,
                                 vn_flat)
  dot_vk_uk_ta = dot_vk_uk_ta.write(i, dot_vn_un)
  recursion_state = (vk_tas, dot_vk_uk_ta)

  # Compute weights delta tilde: `weights_delta_tilde += coeff * vn / n`.
  weights_delta_tilde_flat = tf.nest.map_structure(
      lambda x: tf.reshape(x, [-1]), weights_delta_tilde)
  dot_wd_un = _struct_dot(weights_delta_tilde_flat, un_flat)
  gamma = rho * (n - 1) / n
  vn_coeff = 1. - gamma * (n * dot_wd_un + dot_vn_un) / (1. + gamma * dot_vn_un)
  weights_delta_tilde = tf.nest.map_structure(
      lambda wdt, vn: wdt + vn_coeff * tf.reshape(vn, wdt.shape) / n,
      weights_delta_tilde, vn_flat)

  # Obtain new weights delta by rescaling weights delta tilde.
  weights_delta = tf.nest.map_structure(lambda wdt: wdt * (1 + (n - 1) * rho),
                                        weights_delta_tilde)

  return DeltaUpdateOutput(
      num_samples=n,
      weights_delta=weights_delta,
      weights_sample_mean=weights_sample_mean,
      recursion_state=recursion_state)


def create_client_single_data_pass_fn():
  """Returns a tf.function for taking a single pass over the client data.

  This "create" fn is necessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is needed only because we test the client_single_data_pass function directly.
  """

  @tf.function
  def _single_data_pass(model, dataset, client_optimizer):
    """Makes a single pass over the dataset and updates the model.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      client_optimizer: A `tf.keras.optimizer.Optimizer` object.

    Returns:
      A `DataPassOutput` structure.
    """
    loss_avg = tf.constant(0., dtype=tf.float32)
    num_batches = tf.constant(0., dtype=tf.float32)
    num_examples = tf.constant(0., dtype=tf.float32)
    model_weights_trainable = tuple(model.trainable_variables)
    model_weights_trainable_sum = tf.nest.map_structure(
        tf.zeros_like, model_weights_trainable)

    # Make a pass over the dataset.
    for batch in dataset:
      # Do forward pass and update the model.
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      grads = tape.gradient(output.loss, model_weights_trainable)
      grads_and_vars = zip(grads, model_weights_trainable)
      client_optimizer.apply_gradients(grads_and_vars)
      # Accumulate weights.
      model_weights_trainable_sum = tf.nest.map_structure(
          lambda a, b: tf.add(a, tf.identity(b)),
          model_weights_trainable_sum,
          model_weights_trainable)
      # Accumulate losses.
      batch_size = tf.cast(tf.shape(output.predictions)[0], dtype=tf.float32)
      loss_avg += output.loss * batch_size
      num_examples += batch_size
      num_batches += 1.

    # Compute average loss and weights sample.
    loss_avg = tf.math.divide_no_nan(loss_avg, num_examples)
    model_weights_trainable_sample = tf.nest.map_structure(
        lambda x: tf.math.divide_no_nan(x, num_batches),
        model_weights_trainable_sum)

    outputs = DataPassOutput(
        loss=loss_avg,
        num_examples=num_examples,
        model_weights_trainable=model_weights_trainable,
        model_weights_trainable_sample=model_weights_trainable_sample)

    return outputs

  return _single_data_pass


@tf.function
def _compute_l2_difference(weights_struct_a, weights_struct_b):
  """Computes the L2 norm of the difference between two weight vectors.

  Args:
    weights_struct_a: An arbitrary nested structure of tensors.
    weights_struct_b: An arbitrary nested structure of tensors; the structure
      must match `weights_structure_a` and the tensors must have same shapes.

  Returns:
    A scalar tensor that contains the L2 norm of the difference.
  """
  weights_struct_a_flat = tf.nest.map_structure(lambda x: tf.reshape(x, [-1]),
                                                weights_struct_a)
  weights_struct_b_flat = tf.nest.map_structure(lambda x: tf.reshape(x, [-1]),
                                                weights_struct_b)
  weights_vector_a = tf.concat(tf.nest.flatten(weights_struct_a_flat), axis=0)
  weights_vector_b = tf.concat(tf.nest.flatten(weights_struct_b_flat), axis=0)
  return tf.norm(weights_vector_a - weights_vector_b)


@tf.function
def _compute_zeros_percentage(weights_struct):
  """Computes the percentage of zeros on a structure of tensors.

  Args:
    weights_struct: An arbitrary nested structure of tensors.

  Returns:
    A scalar tensor that contains the percentage of zeros in `weights_struct`.
  """

  def sum_zeros_fn(x):
    return tf.reduce_sum(tf.cast(tf.equal(x, 0.0), dtype=tf.float32))

  compute_size_fn = lambda x: tf.cast(tf.size(x), dtype=tf.float32)
  num_zeros = sum(
      tf.nest.flatten(tf.nest.map_structure(sum_zeros_fn, weights_struct)))
  num_elements = sum(
      tf.nest.flatten(tf.nest.map_structure(compute_size_fn, weights_struct)))
  return tf.math.divide_no_nan(num_zeros, num_elements) * 100.


@tf.function
def client_update(model,
                  dataset,
                  num_epochs,
                  initial_weights,
                  client_optimizer,
                  client_mixedin_fn,
                  client_update_delta_fn,
                  client_single_data_pass_fn,
                  client_weight_fn=None):
  """Updates client model.

  Args:
    model: A `tff.learning.Model`.
    dataset: A 'tf.data.Dataset'.
    num_epochs: The number of epochs or dataset passes.
    initial_weights: A `tff.learning.Model.weights` from server.
    client_optimizer: A `tf.keras.optimizer.Optimizer` object.
    client_mixedin_fn: A function that takes the outputs of the previous and
      current epoch and returns a boolean indicating whether the SG-MCMC has
      mixed in, in which case the following epochs can be used to produce
      approximate posterior samples.
    client_update_delta_fn: A function for updating the weights delta as new
      posterior samples become available.
    client_single_data_pass_fn: A function for taking a single pass over the
      client data to update the model and compute necessary outputs.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A 'ClientOutput`.
  """
  model_weights = _get_weights(model)
  initial_weights = tff.learning.ModelWeights(
      trainable=tuple(initial_weights.trainable),
      non_trainable=tuple(initial_weights.non_trainable))
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        initial_weights)

  # Initialize updates.
  mixedin = tf.constant(False, dtype=tf.bool)
  updates = DeltaUpdateOutput.from_weights(
      initial_weights=initial_weights.trainable,
      updated_weights=initial_weights.trainable)

  # Keep iterating over the data and refining weight deltas.
  num_examples = 0.0
  for epoch in tf.range(num_epochs):
    outputs = client_single_data_pass_fn(
        model=model, dataset=dataset, client_optimizer=client_optimizer)
    mixedin = client_mixedin_fn(epoch, mixedin, outputs)
    updates = client_update_delta_fn(
        mixedin=mixedin,
        initial_weights=initial_weights,
        data_pass_outputs=outputs,
        previous_updates=updates)
    num_examples = outputs.num_examples

  # Check for non-finite weights.
  weights_delta, has_non_finite_weight = (
      tensor_utils.zero_all_if_any_non_finite(updates.weights_delta))
  model_output = model.report_local_outputs()
  optimizer_output = collections.OrderedDict(num_examples=num_examples)
  weights_delta_zeros_percent = _compute_zeros_percentage(weights_delta)

  if has_non_finite_weight > 0:
    client_weight = tf.constant(0, dtype=tf.float32)
  elif client_weight_fn is None:
    client_weight = tf.cast(num_examples, dtype=tf.float32)
  else:
    client_weight = client_weight_fn(model_output)

  # Compute the L2 norm of the difference between corrected/uncorrected deltas.
  weights_delta_uncorrected = tf.nest.map_structure(lambda a, b: a - b,
                                                    model_weights.trainable,
                                                    initial_weights.trainable)
  weights_delta_correction = _compute_l2_difference(weights_delta_uncorrected,
                                                    updates.weights_delta)
  additional_output = collections.OrderedDict(
      model_delta_zeros_percent=weights_delta_zeros_percent,
      model_delta_correction_l2_norm=weights_delta_correction,
  )

  return ClientOutput(
      weights_delta=weights_delta,
      client_weight=client_weight,
      model_output=model_output,
      optimizer_output=optimizer_output,
      additional_output=additional_output)


def build_federated_mean_masked(value_type, weight_type):
  """Builds a federated computation of weighted mean with zero masking.

  Args:
    value_type: The type of federated values which weighted mean is computed.
    weight_type: The type of the weights used in the weighted mean.

  Returns:
    A federated computation.
  """

  @tff.tf_computation(value_type, weight_type)
  def _multiply_by_weight(value, weight):
    if hasattr(value, '_asdict'):
      value = value._asdict()
    return tf.nest.map_structure(lambda x: x * weight, value)

  @tff.tf_computation(value_type, weight_type)
  def _create_weighted_mask(value, weight):
    if hasattr(value, '_asdict'):
      value = value._asdict()
    return tf.nest.map_structure(
        lambda x: weight * tf.cast(tf.not_equal(x, 0.0), dtype=tf.float32),
        value)

  mask_type = _create_weighted_mask.type_signature.result

  @tff.tf_computation(value_type, mask_type)
  def _divide_no_nan(value, weight):
    if hasattr(value, '_asdict'):
      value = value._asdict()
    return tf.nest.map_structure(tf.math.divide_no_nan, value, weight)

  @tff.federated_computation(
      tff.FederatedType(value_type, tff.CLIENTS),
      tff.FederatedType(weight_type, tff.CLIENTS))
  def federated_mean_masked(value, weight):
    """Computes federated weighted mean masking out zeros elementwize.

    Masking out zero elements essentially changes the denominator of the
    mean by not counting zero values.

    Args:
      value: The federated value which mean is to be computed.
      weight: The federated weight to be used in a weighted mean.

    Returns:
      A federated weighted mean of the value with omitted zeros elementwise.
    """
    weighted_value = tff.federated_map(_multiply_by_weight, (value, weight))
    weighted_numerator = tff.federated_sum(weighted_value)
    weighted_mask_denominator = tff.federated_sum(
        tff.federated_map(_create_weighted_mask, (value, weight)))
    return tff.federated_map(_divide_no_nan,
                             (weighted_numerator, weighted_mask_denominator))

  return federated_mean_masked


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


def build_fed_pa_process(
    model_fn: ModelBuilder,
    client_update_epochs: int,
    client_mixedin_schedule_fn: ClientMixedinScheduleFn,
    client_update_delta_fn: ClientUpdateDeltaFn,
    client_optimizer_fn: OptimizerBuilder,
    client_lr: Union[float, LRScheduleFn] = 0.1,
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_lr: Union[float, LRScheduleFn] = 1.0,
    client_weight_fn: Optional[ClientWeightFn] = None,
    mask_zeros_in_client_updates: bool = False,
) -> tff.templates.IterativeProcess:
  """Builds the TFF computations for optimization using posterior averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    client_update_epochs: An inteter that represents the number of local
      epochs to run on the clients.
    client_mixedin_schedule_fn: A function that returns a client mixed in check
      function for given round; the latter determines whether the client has
      mixed-in based on the outputs of the previous two epochs; if mixed-in,the
      following epochs can be used to produce samples from the local posterior.
    client_update_delta_fn: A function that computes an updated weights delta
      based on the previous delta and a new posterior sample.
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
    mask_zeros_in_client_updates: A boolean indicating whether to average deltas
      with zero masking that affects the denominator in the average elementwise.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  client_lr_schedule = client_lr
  if not callable(client_lr_schedule):
    client_lr_schedule = lambda round_num: client_lr

  server_lr_schedule = server_lr
  if not callable(server_lr_schedule):
    server_lr_schedule = lambda round_num: server_lr

  placeholder_model = model_fn()

  server_init_tf = build_server_init_fn(
      model_fn,
      # Initialize with the learning rate for round zero.
      lambda: server_optimizer_fn(server_lr_schedule(0)))
  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model
  round_num_type = server_state_type.round_num

  tf_dataset_type = tff.SequenceType(placeholder_model.input_spec)
  model_input_type = tff.SequenceType(placeholder_model.input_spec)

  @tff.tf_computation(model_input_type, model_weights_type, round_num_type)
  def client_update_fn(tf_dataset, initial_model_weights, round_num):
    client_lr = client_lr_schedule(round_num)
    client_optimizer = client_optimizer_fn(client_lr)
    client_mixedin_fn = client_mixedin_schedule_fn(round_num)
    client_single_data_pass_fn = create_client_single_data_pass_fn()
    return client_update(
        model=model_fn(),
        dataset=tf_dataset,
        num_epochs=client_update_epochs,
        initial_weights=initial_model_weights,
        client_optimizer=client_optimizer,
        client_mixedin_fn=client_mixedin_fn,
        client_update_delta_fn=client_update_delta_fn,
        client_single_data_pass_fn=client_single_data_pass_fn,
        client_weight_fn=client_weight_fn)

  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    server_lr = server_lr_schedule(server_state.round_num)
    server_optimizer = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta)

  client_output_type = client_update_fn.type_signature.result
  client_model_delta_type = client_output_type.weights_delta
  client_weight_type = client_output_type.client_weight

  federated_mean_masked = build_federated_mean_masked(client_model_delta_type,
                                                      client_weight_type)

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
    # Run computation on the clients.
    client_model = tff.federated_broadcast(server_state.model)
    client_round_num = tff.federated_broadcast(server_state.round_num)

    client_outputs = tff.federated_map(
        client_update_fn,
        (federated_dataset, client_model, client_round_num))

    # Aggregate model deltas.
    client_weight = client_outputs.client_weight
    if mask_zeros_in_client_updates:
      model_delta = federated_mean_masked(client_outputs.weights_delta,
                                          client_weight)
    else:
      model_delta = tff.federated_mean(client_outputs.weights_delta,
                                       client_weight)

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, model_delta))

    # Aggregate model outputs that contain local metrics and various statistics.
    aggregated_outputs = placeholder_model.federated_output_computation(
        client_outputs.model_output)
    additional_outputs = tff.federated_mean(
        client_outputs.additional_output, weight=client_weight)

    @tff.tf_computation(aggregated_outputs.type_signature.member,
                        additional_outputs.type_signature.member)
    def _update_aggregated_outputs(aggregated_outputs, additional_outputs):
      aggregated_outputs.update(additional_outputs)
      return aggregated_outputs

    aggregated_outputs = tff.federated_map(
        _update_aggregated_outputs, (aggregated_outputs, additional_outputs))
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
