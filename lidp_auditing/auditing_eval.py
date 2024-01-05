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
"""Evaluation utilities required for auditing."""

# from absl import logging
import numpy as np
import tensorflow as tf

from lidp_auditing import constants
from lidp_auditing import utils


def get_evaluate_fn():
  """Return a `tf.function` to evaluate a keras model."""

  @tf.function
  def evaluate_model(dataset, model, metric):
    # Note: pass in batched dataset
    metric.reset_state()
    for x, y, _ in dataset:
      predictions = model(x, training=False)
      metric.update_state(y, predictions)
    return metric.result()

  return evaluate_model


def evaluate_canary_dataset(
    canary_type: str,
    canary_dataset: tf.data.Dataset,
    model: tf.keras.Model,
    vector_loss_fn: tf.keras.losses.Loss,
    batch_size: int,
) -> np.ndarray:
  """Run the test to see if the canary can be found."""
  if canary_dataset is None:
    return np.array([])
  if canary_type == constants.RANDOM_GRADIENT_CANARY:
    # return evaluate_random_gradient_canary(canary_dataset, model)
    return evaluate_random_gradient_canary_batched(canary_dataset, model)

  # Static or adaptive data canary
  return evaluate_data_canary(canary_dataset, model, vector_loss_fn, batch_size)


def evaluate_data_canary(canary_dataset, model, vector_loss_fn, batch_size):
  """Compute the loss on the canaries."""
  if canary_dataset is None:
    return np.array([])
  all_losses = []
  for x, y, _ in canary_dataset.batch(batch_size, drop_remainder=False):
    predictions = model(x, training=False)
    loss_vector = vector_loss_fn(y, predictions)
    all_losses.append(loss_vector.numpy())
  return np.concatenate(all_losses)


def evaluate_random_gradient_canary(canary_dataset, model):
  """Compute the cosines of the parameters with the canaries."""
  if canary_dataset is None:
    return np.array([])
  all_cosines = []
  weights = tf.nest.flatten(model.trainable_variables)
  weight_norm = tf.sqrt(
      tf.add_n(
          tf.nest.map_structure(
              lambda x: tf.linalg.norm(x) ** 2, tf.nest.flatten(weights)
          )
      )
  )
  for _, _, z in canary_dataset:  # all examples are canaries
    # Note: We use canaries of norm = 1 because we normalize by the norm of
    # canaries anyway in our final statistic. So the clip norm does not matter.
    noise = utils.get_random_normal_like(weights, z, flat_l2_norm=1)
    dot_product = tf.add_n(
        tf.nest.map_structure(lambda a, b: tf.reduce_sum(a * b), noise, weights)
    )
    cosine = dot_product / weight_norm
    all_cosines.append(cosine.numpy())
  return np.array(all_cosines)


def evaluate_random_gradient_canary_batched(
    canary_dataset, model, max_batch_size=1024
):
  """Batched computation of the cosines of the parameters with the canaries."""
  # Batching gives a 20x speedup on the evaluation.
  if canary_dataset is None:
    return np.array([])
  all_cosines = []
  weights = tf.nest.flatten(model.trainable_variables)
  weight_norm = tf.sqrt(
      tf.add_n(
          tf.nest.map_structure(
              lambda x: tf.linalg.norm(x) ** 2, tf.nest.flatten(weights)
          )
      )
  )
  # All examples are canaries, so no special filtering necessary.
  for _, _, z in canary_dataset.batch(max_batch_size):
    # Note: We use canaries of norm = 1 because we normalize by the norm of
    # canaries anyway in our final statistic. So the clip norm does not matter.
    noise = utils.get_batched_random_normal_like(
        weights, z, flat_l2_norm=tf.constant(1.0)
    )  # list of (batch_size, *weights[i])
    dot_product = tf.add_n(
        tf.nest.map_structure(batched_dot, noise, weights)
    )  # (batch_size,)
    cosine = dot_product / weight_norm
    all_cosines.append(cosine.numpy())
  return np.concatenate(all_cosines)


def batched_dot(a, b):
  """Return [dot(c, b) for c in a] but in TF."""
  # a: (bsz, s1, s2, ...)
  # b: (s1, s2, ...)
  return tf.tensordot(
      tf.reshape(a, (tf.shape(a)[0], -1)),  # (bsz, s1, s2, ...) -> (bsz, s)
      tf.reshape(b, -1),  # (s1, s2, ...) -> (s,)
      axes=1,
  )  # (bsz,)
