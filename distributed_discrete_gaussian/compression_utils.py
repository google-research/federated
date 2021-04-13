# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compression util functions."""

import numpy as np
import tensorflow as tf

DEFAULT_BETA = np.exp(-0.5)


def stochastic_rounding(x, conditional, l2_norm_bound=None, beta=DEFAULT_BETA):
  """Randomly rounds the elements of a tensor to integer values (keeps dtype).

  Args:
    x: The input tensor.
    conditional: A bool constant specifying whether to do conditional rounding
      (i.e. keep retrying rounding until L2 norm of the flattened tensor as a
      vector doesn't grow too much).
    l2_norm_bound: A float constant denoting the bound of the L2 norm of the
      input records. This is useful when `l2_norm_bound` is larger than the
      input norm, in which case we can allow more leeway during conditional
      stochastic rounding rounding. If `None`, defaults to `l2_norm(x)`.
    beta: A constant in [0, 1) controlling the concentration inequality for the
      probabilistic norm bound after rounding.

  Returns:
    The rounded tensor.
  """

  def post_rounding_l2_norm_bound(x, l2_norm_bound, beta):
    """Computes the L2 norm bound of a vector after rounding (Thm. 1, Eq. 2)."""
    beta = tf.cast(beta, x.dtype)
    dim = tf.cast(tf.size(x), x.dtype)
    if l2_norm_bound is None:
      x_norm = tf.norm(x, ord=2)
    else:
      x_norm = tf.cast(l2_norm_bound, x.dtype)

    # We consider 2 (scaled) norm bounds and take the min (Proposition 22).
    bound1 = x_norm + tf.sqrt(dim)
    squared_bound2 = tf.square(x_norm) + 0.25 * dim
    squared_bound2 += (
        tf.sqrt(2.0 * tf.math.log(1.0 / beta)) * (x_norm + 0.5 * tf.sqrt(dim)))
    bound2 = tf.sqrt(squared_bound2)
    # bound2 is inf if beta = 0, in which case we fall back to bound1.
    return tf.minimum(bound1, bound2)

  conditional = tf.cast(conditional, tf.bool)
  l2_norm_threshold = post_rounding_l2_norm_bound(x, l2_norm_bound, beta)
  floored_x = tf.floor(x)
  decimal_x = x - floored_x

  def round_fn(repeat, _):
    # 1. Try stochastic rounding on input (ignore previous iterations' outputs).
    uniform = tf.random.uniform(tf.shape(x), dtype=x.dtype, minval=0, maxval=1)
    bernoulli = uniform < decimal_x
    rounded_x = floored_x + tf.cast(bernoulli, x.dtype)
    # 2. Try again if the rounded vector has excessive L2 norm.
    rounded_l2_norm = tf.norm(rounded_x, ord=2)
    repeat = tf.logical_and(conditional,
                            tf.greater(rounded_l2_norm, l2_norm_threshold))
    return [repeat, rounded_x]

  repeat = tf.constant(True)
  _, result_x = tf.while_loop(
      cond=lambda r, _: r, body=round_fn, loop_vars=[repeat, x])

  return result_x


def scaled_quantization(x,
                        scale,
                        stochastic,
                        conditional,
                        l2_norm_bound,
                        beta=DEFAULT_BETA):
  """Scales the tensors and rounds to integers."""
  scale = tf.cast(scale, x.dtype)
  l2_norm_bound = tf.cast(l2_norm_bound, x.dtype)
  scaled_x = x * scale
  scaled_bound = l2_norm_bound * scale

  quantized_x = tf.cond(
      tf.cast(stochastic, tf.bool),
      lambda: stochastic_rounding(scaled_x, conditional, scaled_bound, beta),
      lambda: tf.round(scaled_x))
  return quantized_x


def inverse_scaled_quantization(x, scale):
  """Restores the value range of `x` from `scaled_quantization`."""
  return x / tf.cast(scale, x.dtype)


def flatten_concat(structure):
  """Flattens each tensor in the structure and concats them as a vector.

  Each tensor within the structure should have rank >= 1 (i.e. no scalars).

  Args:
    structure: The input structure of tensors.

  Returns:
    The flattened and concatenated component tensors as a tf.Tensor with
    shape (d,) where `d` is the total number of elements in the structure.
  """
  flattened_as_list = []
  for x in tf.nest.flatten(structure):
    with tf.control_dependencies([tf.debugging.assert_rank_at_least(x, 1)]):
      flattened_as_list.append(tf.reshape(x, [-1]))
  return tf.concat(flattened_as_list, axis=0)


def inverse_flatten_concat(flat_vector, original_structure):
  """Applies the inverse of `flatten_concat` given the original structure."""
  location, split_tensors = 0, []
  for orig_t in tf.nest.flatten(original_structure):
    length = tf.size(orig_t)
    split_vector = tf.slice(flat_vector, [location], [length])
    split_tensors.append(tf.reshape(split_vector, orig_t.shape))
    location += length
  return tf.nest.pack_sequence_as(original_structure, split_tensors)


def sample_rademacher(shape, dtype, seed_pair):
  """Sample uniform random +1/-1 values with specified shape/dtype/seed_pair."""
  rand_uniform = tf.random.stateless_uniform(shape=shape, seed=seed_pair)
  return tf.cast(tf.sign(rand_uniform - 0.5), dtype)


def pad_zeros(x):
  """Pads a vector with shape (d,) with zeros to the next power of two."""
  dim = tf.shape(x)[0]
  log2_dim = tf.math.log(tf.cast(dim, tf.float32)) / tf.math.log(2.0)
  pad_dim = tf.pow(2, tf.cast(tf.math.ceil(log2_dim), tf.int32))
  with tf.control_dependencies([tf.debugging.assert_rank(x, 1)]):
    return tf.pad(x, [[0, tf.maximum(0, pad_dim - dim)]])


def randomized_hadamard_transform(x, seed_pair, repeat=1):
  """Applies randomized Hadamard transform to a vector with the given seed.

  Args:
    x: The input vector.
    seed_pair: The seed pair for generating randomness.
    repeat: Number of times to repeat the randomized Hadamard transform.

  Returns:
    The transformed vector.
  """

  def apply_transform(repeat_index, x):
    # All sources of randomness depend on the input seed.
    cur_seed = seed_pair + repeat_index
    # Randomly flip signs.
    signs = sample_rademacher(tf.shape(x), dtype=x.dtype, seed_pair=cur_seed)
    rademacher_x = signs * x
    # Apply Hadamard (+ expand/squeeze dims).
    encoded_x = tf.squeeze(
        fast_walsh_hadamard_transform(tf.expand_dims(rademacher_x, axis=0)),
        axis=0)
    return encoded_x

  tf.debugging.assert_type(x, tf.float32)
  padded_x = pad_zeros(x)  # Hadamard transform requires vectors with 2^n dims.
  i, result_x = tf.constant(0), padded_x
  cond_fn = lambda i, _: tf.less(i, repeat)
  body_fn = lambda i, x: [tf.add(i, 1), apply_transform(i, x)]
  _, result_x = tf.while_loop(cond_fn, body_fn, [i, result_x])
  return result_x


def inverse_randomized_hadamard_transform(x, original_dim, seed_pair, repeat=1):
  """Applies inverse of `randomized_hadamard_transform` with the given seed.

  Args:
    x: The transformed vector.
    original_dim: The dimension of the original vector.
    seed_pair: The same seed pair used in the forward transform.
    repeat: Number of times the randomized Hadamard transform was applied.

  Returns:
    The original vector.
  """

  def inverse_transform(repeat_index, x):
    # All sources of randomness depend on the input seed.
    cur_seed = seed_pair + repeat_index
    # Apply Hadamard.
    unrotated_x = fast_walsh_hadamard_transform(tf.expand_dims(x, axis=0))
    unrotated_x = tf.squeeze(unrotated_x, axis=0)
    # Unflip signs.
    signs = sample_rademacher(
        tf.shape(unrotated_x), dtype=x.dtype, seed_pair=cur_seed)
    decoded_x = signs * unrotated_x
    return decoded_x

  # Repeat inverse transforms (with reversed indices).
  tf.debugging.assert_type(x, tf.float32)
  i, result_x = tf.constant(repeat - 1), x
  cond_fn = lambda i, _: tf.greater_equal(i, 0)
  body_fn = lambda i, x: [tf.subtract(i, 1), inverse_transform(i, x)]
  _, result_x = tf.while_loop(cond_fn, body_fn, [i, result_x])

  # Unpad zeros from forward transform.
  return result_x[:original_dim]


def fast_walsh_hadamard_transform(x):
  """Applies the fast Walsh-Hadamard transform to a set of vectors.

  This method uses a composition of existing TensorFlow operations to implement
  the transform.

  This function is forked from https://github.com/tensorflow/model-optimization.

  Args:
    x: A `Tensor`. Must be of shape `[a, b]`, where `a` can be anything (not
      necessarily known), and `b` must be a power of two, not required to be
      statically known.

  Returns:
    A `Tensor` of shape `[a, b]`, where `[i, :]` is the product `x[i, :]*H`,
      where `H` is the Hadamard matrix.

  Raises:
    ValueError: If the input is not rank 2 `Tensor`, and if the second dimension
      is statically known and is not a power of two.
    OpError: If the second dimension is not statically known and is not a power
      of two. Note that in graph execution, this error is not raised during the
      execution of the Python function, but during execution of the resulting
      computation.
  """
  with tf.compat.v1.name_scope(None, 'fast_walsh_hadamard_transform'):
    # Validate input.
    x = tf.convert_to_tensor(x)
    if x.shape.ndims != 2:
      raise ValueError('Number of dimensions of x must be 2. Shape of x: %s' %
                       x.shape)

    original_x_shape = x.shape.as_list()
    dim = x.shape.as_list()[-1]

    if dim is None:  # dim is not statically known.
      dim = tf.shape(x)[-1]
      log2 = tf.cast(
          tf.math.round(
              tf.math.log(tf.cast(dim, tf.float32)) / tf.math.log(2.)),
          tf.int32)
      with tf.control_dependencies([
          tf.compat.v1.assert_equal(
              dim,
              tf.math.pow(2, log2),
              message='The dimension of x must be a power of two.'
              'Provided dimension is: %s' % dim)
      ]):
        x = tf.identity(x)
    else:  # dim is statically known.
      if not (dim and ((dim & (dim - 1)) == 0)):
        raise ValueError('The dimension of x must be a power of two. '
                         'Provided dimension is: %s' % dim)
      log2 = int(np.ceil(np.log2(dim)))
      if dim == 1:  # Equivalent to identity.
        return tf.identity(x)

    h_core = tf.constant([[1., 1.], [1., -1.]],
                         dtype=x.dtype,
                         name='hadamard_weights_2x2')
    permutation = tf.constant([0, 2, 1], name='hadamard_permutation')

    # A step of the fast Walsh-Hadamard algorithm.
    def _hadamard_step(x, dim):
      """A single step in the fast Walsh-Hadamard transform."""
      x_shape = x.shape.as_list()
      x = tf.reshape(x, [-1, 2])  # Reshape so that we have a matrix.
      x = tf.matmul(x, h_core)  # Multiply.
      x = tf.reshape(x, [-1, dim // 2, 2])  # Reshape to rank-3.
      x = tf.transpose(x, perm=permutation)  # Swap last two dimensions.
      x.set_shape(x_shape)  # Failed shape inference in tf.while_loop.
      return x

    def _fwht(x, dim, log2):
      x = tf.reshape(x, [-1, 2, dim // 2])
      # The fast Walsh-Hadamard transform.

      i = tf.constant(0)
      c = lambda i, x: tf.less(i, log2)
      b = lambda i, x: [i + 1, _hadamard_step(x, dim)]
      i, x = tf.while_loop(c, b, [i, x])
      return x

    x = tf.cond(
        tf.equal(dim, 1), lambda: tf.identity(x), lambda: _fwht(x, dim, log2))

    x = tf.reshape(x, [-1, dim])
    x /= tf.sqrt(tf.cast(dim, x.dtype))  # Normalize.
    x.set_shape(original_x_shape)  # Failed shape inference after tf.while_loop.
    return x
