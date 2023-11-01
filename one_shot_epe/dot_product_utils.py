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
"""Functions for computing cosines between model and canaries."""

import tensorflow as tf


@tf.function
def flatten(weights: list[tf.Tensor]) -> tf.Tensor:
  """Reshapes structure of tensors into flat vector.

  Args:
    weights: A structure of tensors.

  Returns:
    A flat tensor concatenating all of the weights, in `tf.nest` traversal
    order.
  """
  return tf.concat([tf.reshape(c, [-1]) for c in tf.nest.flatten(weights)], 0)


def _flat_canary(
    model_weights: list[tf.Tensor], canary_id: int, global_seed: int
) -> tf.Tensor:
  """Returns the canary update associated with this canary ID.

  The canary is seeded by the canary client index and the global seed. It is
  drawn from the uniform distribution over the unit sphere. It is returned as
  a flat vector of the same total dimensionality as the weights.

  Args:
    model_weights: A structure of tensors to use as a template for creating the
      canary.
    canary_id: A positive integer identifier for this canary.
    global_seed: The global seed for this run.

  Returns:
    The canary associated with this canary_id, as a flat vector.
  """
  canary_seed = tf.convert_to_tensor([canary_id, global_seed], tf.int64)
  dim = sum(tf.size(w) for w in tf.nest.flatten(model_weights))
  canary = tf.random.stateless_normal((dim,), canary_seed)
  return canary / tf.norm(canary)


def packed_canary(
    model_weights: list[tf.Tensor], canary_id: int, global_seed: int
) -> list[tf.Tensor]:
  """Packs flat canary into the structure of model_weights.

  Args:
    model_weights: A structure of tensors to use as a template for creating the
      canary.
    canary_id: The integer ID of the canary.
    global_seed: The global seed for this run.

  Returns:
    Canary packed into structure like model_weights.
  """
  canary = _flat_canary(model_weights, canary_id, global_seed)
  pieces = []
  begin = 0
  for w in tf.nest.flatten(model_weights):
    size = tf.size(w)
    pieces.append(tf.reshape(tf.slice(canary, (begin,), (size,)), tf.shape(w)))
    begin += size
  return tf.nest.pack_sequence_as(model_weights, pieces)


def compute_cosine(
    model_weights: list[tf.Tensor], canary_id: int, global_seed: int
) -> tf.Tensor:
  """Computes cosine between model and single canary.

  Args:
    model_weights: A structure of tensors to compute cosine with.
    canary_id: The integer ID of the canary.
    global_seed: The global seed for this run.

  Returns:
    Cosine between model_weights and canary.
  """
  flattened_weights = flatten(model_weights)
  canary = _flat_canary(flattened_weights, canary_id, global_seed)
  return tf.reduce_sum(flattened_weights * canary) / tf.norm(flattened_weights)


def compute_negative_cosines_with_all_canaries(
    model_weights: list[tf.Tensor],
    num_canaries: int,
    global_seed: int,
    offset: int = 0,
) -> list[tf.Tensor]:
  """Computes negative cosine of model parameters with all canaries.

  This implementation is optimized for computing all cosines at once.

  Args:
    model_weights: A structure of tensors representing the weights to compute
      cosine with.
    num_canaries: The number of canaries. It is assumed that the canary IDs are
      indexed (0, ..., num_canaries-1).
    global_seed: The global seed for the run.
    offset: Optional offset for individual canary seeds. Can be used to compute
      a different set of canaries, for example, negative canaries.

  Returns:
    List of negative cosines between model and canaries. We use negative cosines
    because the model moves in the opposite direction of the canary.
  """
  flattened_weights = flatten(model_weights)
  weights_norm = tf.norm(flattened_weights)

  def get_neg_cosine(i):
    canary = _flat_canary(flattened_weights, i + offset, global_seed)
    # In exceptional circumstances, the model deltas can be zeroed out. This
    # should be rare, which can be verified by looking at the model_delta_norm
    # metric. If this happens, return zero for the cosine.
    return -tf.math.divide_no_nan(
        tf.reduce_sum(flattened_weights * canary),
        weights_norm,
    )

  return [get_neg_cosine(i) for i in range(num_canaries)]
