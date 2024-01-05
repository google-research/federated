# Copyright 2023, Google LLC.
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
"""Utilities for auditing + training."""

import math

from absl import logging
import dp_accounting
import scipy.optimize
import tensorflow as tf


RDP_ORDERS = (
    [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
    + list(range(5, 64))
    + [128, 256, 512]
)


def get_optimal_noise_multiplier_dpsgd(
    n: int,
    batch_size: int,
    epochs: int,
    target_epsilon: float,
    target_delta: float,
) -> float:
  """Find the best noise multiplier for given DP-SGD parameters."""
  q = batch_size / n  # the sampling ratio
  assert q <= 1
  steps = int(math.ceil(epochs * n / batch_size))

  def objective(noise_multiplier):
    accountant = dp_accounting.rdp.RdpAccountant(RDP_ORDERS)
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            q, dp_accounting.GaussianDpEvent(noise_multiplier)
        ),
        steps,
    )
    accountant.compose(event)
    eps, _ = accountant.get_epsilon_and_optimal_order(target_delta)
    return eps - target_epsilon

  return scipy.optimize.brentq(objective, 1e-6, 1000)


#################################################
# Gradient clipping utilities
#################################################
def clip_gradients_vmap(g, l2_norm_clip, norm_threshold=tf.constant(1.0)):
  """Clips gradients in a way that is compatible with vectorized_map."""
  # Here, `g`` is the gradient per-sample. It is the nested list of tensors,
  # compatible with model.trainable_variables.
  grads_flat = tf.nest.flatten(g)  # flat list of tensors
  squared_l2_norms = [
      tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat
  ]  # list of per-parameter squared L2 norms
  global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
  div = tf.maximum(global_norm / l2_norm_clip, norm_threshold)
  clipped_flat = [g / div for g in grads_flat]
  clipped_grads = tf.nest.pack_sequence_as(g, clipped_flat)
  l2_norms = [tf.sqrt(norm) for norm in squared_l2_norms]  # flat list only
  return clipped_grads, l2_norms


#################################################
# Stateless random noise generation utilities
#################################################
def get_random_normal_like(
    weights: list[tf.Tensor], seed: int | tf.Tensor, flat_l2_norm: float
) -> list[tf.Tensor]:
  """Return random Gaussian with structure like weights and a given L2 norm."""
  if isinstance(seed, tf.Tensor):
    seed_tuple = tf.stack([seed, seed + 1])
  else:
    seed_tuple = tf.constant([seed, seed + 1], tf.int64)

  # get a seed for each element in the structure
  nest_seeds = seeds_for_structure(seed_tuple, weights)
  # Generate noise
  nest_noise = tf.nest.map_structure(
      lambda x, s: tf.random.stateless_normal(shape=tf.shape(x), seed=s),
      weights,
      nest_seeds,
  )
  # Calculate the norm
  norm = tf.sqrt(
      tf.add_n(
          tf.nest.map_structure(
              lambda x: tf.linalg.norm(x) ** 2, tf.nest.flatten(nest_noise)
          )
      )
  )
  div = tf.maximum(norm / flat_l2_norm, 1e-6)  # to avoid divide by 0
  # normalize to the right norm
  nest_noise = tf.nest.map_structure(lambda x: x / div, nest_noise)
  return nest_noise


def seeds_for_structure(seed_tuple, nest_structure):
  """Returns seed in nested structure and the next state seed."""
  flat_structure = tf.nest.flatten(nest_structure)
  flat_seeds = [
      seed_tuple + tf.constant([0, i], tf.int64)
      for i in range(len(flat_structure))
  ]
  nest_seeds = tf.nest.pack_sequence_as(nest_structure, flat_seeds)
  return nest_seeds


@tf.function
def get_batched_random_normal_like(
    weights: list[tf.Tensor], seeds: tf.Tensor, flat_l2_norm: tf.Tensor
) -> list[tf.Tensor]:
  """Return batched random Gaussians of a given L2 norm and structure.

    It is a batched version of `get_random_normal_like`.

  Args:
    weights: list of tensors, such as the trainable variables.
    seeds: tensor of seeds of shape (num_seeds,).
    flat_l2_norm: L2 norm of each (flattened) element in the batch.

  Returns:
    noise_batched: it is a list of tensors [(num_seeds, *weights[i].shape)].
      The `i`th entries of each tensor should match the corresponding entry
      of `get_random_normal_like(weights, seeds[i], l2_norm)`.
  """
  logging.warning('***Tracing batched noise generation!')
  # NOTE: the next line has to be consistent with `seeds_for_structure`.
  num_seeds = seeds.shape[0]
  seeds_pair = tf.transpose(tf.stack([seeds, seeds + 1]))  # (num_seeds, 2)
  norm_threshold = tf.constant(1e-6)  # to avoid divide by zero
  noise_batched = []
  for i, weight in enumerate(weights):
    seeds_pair_local = seeds_pair + tf.constant(
        [0, i], tf.int64
    )  # (num_seeds, 2)
    local_noise = tf.TensorArray(tf.float32, size=num_seeds)
    for j in tf.range(num_seeds):
      this_noise = tf.random.stateless_normal(
          shape=tf.shape(weight), seed=seeds_pair_local[j]
      )  # shape = weights[i].shape
      local_noise = local_noise.write(j, this_noise)
    local_noise = local_noise.stack()  # (num_seeds, *weights[i].shape)
    noise_batched.append(local_noise)

  # Clip
  clipped_noise, _ = tf.vectorized_map(
      lambda g: clip_gradients_vmap(g, flat_l2_norm, norm_threshold),
      noise_batched,
  )
  return clipped_noise
