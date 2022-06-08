# Copyright 2022, Google LLC.
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
"""Implements (linear) sketching techniques for compression of gradients."""

import enum
import functools
from typing import Optional, Tuple

import tensorflow as tf


class DecodeMethod(enum.Enum):
  MEAN = 'mean'
  MEDIAN = 'median'


def _get_hash_mapping(width: tf.Tensor, gradient_length: tf.Tensor,
                      hash_id: tf.Tensor, gradient_dtype: tf.DType,
                      index_seeds: tf.Tensor,
                      sign_seeds: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Returns the (indices, signs) of a gradient for a hash function.

  Args:
    width: A tf.Tensor `int` representing the width of the sketch representing
      the number of bins for each hash function.
    gradient_length: A tf.Tensor `int` representing the length of the gradient
      to be encoded.
    hash_id: A tf.Tensor `int` index specifying the hash function to be used.
    gradient_dtype: the `tf.DType` of the gradient/sketch.
    index_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the hash bucket that an input maps to.
    sign_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the multiplicative sign for the stored sketch value.

  Returns:
    A tuple of (indices, signs). Both indices and signs are a tf.Tensor with
    shape [`gradient_length`] and type `tf.int32`.
  """
  hash_id = tf.cast(tf.stack([0, hash_id], 0), index_seeds.dtype)
  index_seeds += hash_id
  sign_seeds += hash_id
  scale = tf.cast(width, tf.float32)
  indices = tf.random.stateless_uniform(
      shape=[gradient_length], seed=index_seeds, dtype=tf.float32)
  indices = tf.cast(scale * indices, tf.int32)
  signs = tf.random.stateless_uniform(
      shape=[gradient_length], seed=sign_seeds, dtype=tf.float32)
  signs = tf.cast(2 * tf.math.round(signs) - 1, gradient_dtype)
  return (indices, signs)


def encode(gradient: tf.Tensor,
           length: tf.Tensor,
           width: tf.Tensor,
           index_seeds: tf.Tensor,
           sign_seeds: tf.Tensor,
           parallel_iterations: Optional[int] = None) -> tf.Tensor:
  """Encodes a gradient vector into a count sketch.

  This method iterates over the coordinates of `gradient`, hashing each
  coordinate's value individually `length` times, once for each of the
  `length` hash functions. The returned sketch can be unsketched to return a
  gradient estimate using the decode method.

  Args:
    gradient: A gradient tensor of rank 1, i.e., gradient vector.
    length: The length of the sketch representing the number of
      pairwise-independent hash functions.
    width: The width of the sketch representing the number of bins for each hash
      function.
    index_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the hash bucket that an input maps to.
    sign_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the multiplicative sign for the stored sketch value.
    parallel_iterations: An integer specifying how many hash rows to compute in
      parallel that is passed directly to `tf.map_fn`. See their documentation
      for more details. If None, when graph building, the default value is 10.
      While executing eagerly, the default value is set to 1.

  Returns:
    A sketch matrix encoding the gradient vector as a tf.Tensor of shape
    (`length`, `width`).

  Raises:
    ValueError: If the passed gradient is more than rank 1 or if
      `parallel_iterations` < 0. If `length` is non-scalar or has value < 1 or
      if `width` is non-scalar or has value < 1.
  """
  tf.debugging.assert_scalar(length, message='Length must be a scalar.')
  # `tf.debugging.assert_greater_equal` raises a ValueError in eager mode and
  # a tf.errors.InvalidArgumentError in graph mode. Because the rest only raise
  # ValueErrors, we force ValueErrors for consistency.

  try:
    tf.debugging.assert_greater_equal(
        length, tf.ones_like(length),
        'Sketch length must be a positive integer')
  except (tf.errors.InvalidArgumentError, ValueError) as e:
    raise ValueError(e) from e
  tf.debugging.assert_scalar(width, message='Width must be a scalar.')
  try:
    tf.debugging.assert_greater_equal(
        width, tf.ones_like(width), 'Sketch width must be a positive integer')
  except (tf.errors.InvalidArgumentError, ValueError) as e:
    raise ValueError(e) from e
  tf.debugging.assert_type(index_seeds, sign_seeds.dtype,
                           'Seeds must be of the same data type.')
  tf.debugging.assert_rank_in(gradient, [0, 1],
                              'Gradient expected to be a vector or scalar.')
  # `parallel_iterations` must be a normal integer as the underlying tf.map_fn
  # code does not support tf.tensor inputs.

  if parallel_iterations is not None and parallel_iterations < 0:
    raise ValueError(
        f'Detected `parallel_iterations`={parallel_iterations} which must be '
        f'>= 0.')

  width = tf.cast(width, tf.int32)
  length = tf.cast(length, tf.int32)
  gradient_length = gradient.shape.num_elements()
  normalization = tf.cast(length, gradient.dtype)

  def get_sketch_row(hash_id):
    indices, signs = _get_hash_mapping(width, gradient_length, hash_id,
                                       gradient.dtype, index_seeds, sign_seeds)
    weights = tf.math.divide(tf.math.multiply(signs, gradient), normalization)
    row = tf.math.bincount(
        indices,
        weights=weights,
        minlength=width,
        maxlength=width,
        dtype=gradient.dtype)
    return tf.reshape(row, (width,))

  return tf.map_fn(
      get_sketch_row,
      tf.range(length, dtype=index_seeds.dtype),
      parallel_iterations=parallel_iterations,
      fn_output_signature=gradient.dtype,
      swap_memory=True)


def _apply_threshold(gradient_vector: tf.Tensor,
                     threshold: tf.Tensor) -> tf.Tensor:
  """Sets absolute values below `threshold` in `gradient_vector` to zero."""
  threshold = tf.cast(threshold, gradient_vector.dtype)
  mask = tf.abs(gradient_vector) >= threshold
  return tf.math.multiply(gradient_vector, tf.cast(mask, gradient_vector.dtype))


def _decode_mean(sketch: tf.Tensor, gradient_length: tf.Tensor,
                 index_seeds: tf.Tensor, sign_seeds: tf.Tensor) -> tf.Tensor:
  """Estimates a gradient vector using a count-mean sketch decoding.

  Args:
    sketch: A tf.Tensor of shape [`length`, `width`] representing the sketched
      estimate of a gradient.
    gradient_length: A tf.int32 Tensor representing the length of the decoded
      gradient vector.
    index_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the hash bucket that an input maps to.
    sign_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the multiplicative sign for the stored sketch value.

  Returns:
    Decoded gradient vector estimate.
  """
  length, width = sketch.shape
  gradient_estimate = tf.zeros([gradient_length], dtype=sketch.dtype)

  def add_to_gradient(hash_id, gradient_estimate):
    indices, signs = _get_hash_mapping(width, gradient_length, hash_id,
                                       sketch.dtype, index_seeds, sign_seeds)
    gradient_estimate += tf.math.multiply(signs,
                                          tf.gather(sketch[hash_id], indices))
    return gradient_estimate

  cond_fn = lambda i, _: tf.less(i, length)
  body_fn = lambda i, gradient: (tf.add(i, 1), add_to_gradient(i, gradient))
  i = tf.constant(0)
  _, gradient_estimate = tf.while_loop(
      cond_fn, body_fn, [i, gradient_estimate], swap_memory=True)
  return gradient_estimate


def _get_batch_estimate(batch_id, sketch, chunk_size, padded_gradient_length,
                        index_seeds, sign_seeds):
  """Computes `chunk_size` gradient estimates for coordinates of `batch_id`.

  This function computes the estimate of `chunk_size` coordinates in the
  gradient vector, at the position indicated by `batch_id` * `chunk_size`. The
  `chunk_size` can be chosen so as to limit the overall memory consumption of
  computing the entire gradient vector estimate.

  Args:
    batch_id: The index of the batch, where there are `gradient_length` /
      `chunk_size` batches. This determines the location of the coordinates
      within the gradient vector.
    sketch: A tf.Tensor of shape [`length`, `width`] and of dtype
      `gradient_dtype` representing the sketched estimate of a gradient.
    chunk_size: the number of coordinates in the gradient estimate to calculate.
    padded_gradient_length: The `gradient_length` rounded up to the nearest
      multiple of `chunk_size`. Required so that this function can be called
      within a tf.map_fn.
    index_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the hash bucket that an input maps to.
    sign_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the multiplicative sign for the stored sketch value.

  Returns:
    Gradient estimate for the `chunk_size` coordinates at indices with
    respect to `batch_id`.
  """
  length, width = sketch.shape
  start = batch_id * chunk_size
  end = start + chunk_size
  selector = tf.range(start, end, dtype=tf.int32)
  mid = tf.cast(length / 2, tf.int32) + 1
  normalization = tf.cast(length, sketch.dtype)

  def get_gradient_row_estimate(hash_id, selector=selector):
    """Computes gradient for row `hash_id` of the sketch."""
    indices, signs = _get_hash_mapping(width, padded_gradient_length, hash_id,
                                       sketch.dtype, index_seeds, sign_seeds)
    indices = tf.gather(indices, selector)
    signs = tf.gather(signs, selector)
    normalized_gradient_row_estimate = tf.math.multiply(
        signs, tf.gather(sketch[hash_id], indices))
    gradient_row_estimate = normalized_gradient_row_estimate * normalization
    return tf.reshape(gradient_row_estimate, [-1])

  gradient_estimates = tf.map_fn(
      get_gradient_row_estimate,
      tf.range(0, length),
      fn_output_signature=sketch.dtype,
      swap_memory=True)

  sorted_coordinates = tf.nn.top_k(tf.transpose(gradient_estimates), mid)[0]

  if length % 2 == 0:
    gradient_estimates = (
        (sorted_coordinates[:, -2] + sorted_coordinates[:, -1]) / 2.0)
  else:
    gradient_estimates = sorted_coordinates[:, -1]
  return gradient_estimates


def _decode_median(sketch: tf.Tensor, gradient_length: tf.Tensor,
                   index_seeds: tf.Tensor, sign_seeds: tf.Tensor) -> tf.Tensor:
  """Estimates a gradient vector using a count-median sketch decoding.

  This function parses the gradient length in chunks of size
  gradient_length / length so as to use approximately
  (gradient_length + gradient_length/length) memory.

  Args:
    sketch: A tf.Tensor of shape [`length`, `width`] and of representing the
      sketched estimate of a gradient.
    gradient_length: A tf.int32 Tensor representing the length of the decoded
      gradient vector.
    index_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the hash bucket that an input maps to.
    sign_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the multiplicative sign for the stored sketch value.

  Returns:
    Decoded gradient vector estimate such that decode(sketch) ~ gradient.
  """
  sketch_length = sketch.shape[0]
  # Process gradient in chunks of size `chunk_size` where
  # `chunk_size` = `gradient_length` / `sketch_length` so that when we store all
  # `sketch_length` * `chunk_size` sketch values, we only store
  # O(`gradient_length`) total parameters.

  chunk_size = tf.reduce_max(
      [tf.cast(gradient_length / sketch_length, tf.int32), 1])
  num_batches = tf.cast(tf.math.ceil(gradient_length / chunk_size), tf.int32)
  batches = tf.range(num_batches, dtype=tf.int32)
  padded_gradient_length = chunk_size * num_batches

  batch_estimate = functools.partial(
      _get_batch_estimate,
      sketch=sketch,
      chunk_size=chunk_size,
      padded_gradient_length=padded_gradient_length,
      index_seeds=index_seeds,
      sign_seeds=sign_seeds)

  gradient_estimate = tf.map_fn(
      batch_estimate,
      batches,
      fn_output_signature=sketch.dtype,
      swap_memory=True)
  gradient_estimate = tf.reshape(gradient_estimate, [padded_gradient_length])
  return tf.gather(gradient_estimate, tf.range(gradient_length))


def decode(sketch: tf.Tensor,
           gradient_length: tf.Tensor,
           index_seeds: tf.Tensor,
           sign_seeds: tf.Tensor,
           method: DecodeMethod = DecodeMethod.MEAN,
           threshold: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Decodes a gradient vector estimate from a sketch.

  This method is to be used in conjuction with the encode method. Since the
  sketch is the representation of encode(gradient), the decode method undoes the
  encoding to return decode(sketch) ~= gradient. The returned rank 1 gradient
  vector's dtype will be the same as the input sketch.dtype.

  Args:
    sketch: A tf.Tensor of shape [`length`, `width`] as returned by the encode
      method. Represents the sketched estimate of a gradient.
    gradient_length: A tf.int32 Tensor representing the length of the decoded
      gradient vector.
    index_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the hash bucket that an input maps to.
    sign_seeds: A tf.int32 tensor with shape [2] used as a seed for generating
      the multiplicative sign for the stored sketch value.
    method: a `DecodeMethod` enum representing which decoding method to use.
      `MEAN` is preferred for runtime and memory.
    threshold: a tf.Tensor `float` representing the threshold value such that
      (absolute) values in the decoded gradient vector that are below this
      threshold are set to 0.0.

  Returns:
    The gradient vector estimate from the count-`sketch` using the specified
    decode `method`, optionally with thresholding.

  Raises:
    ValueError: If method is not one of [`mean`, `median`],
      if `threshold` < 0.0, or if sketch is not a rank 2 tensor.
  """
  tf.debugging.assert_rank(sketch, 2,
                           'Sketch expected to be a (rank 2) matrix.')
  gradient_vector_estimate = tf.zeros([gradient_length], sketch.dtype)
  if threshold is not None:
    try:
      tf.debugging.assert_greater(
          threshold, 0.0, message='Threshold must be a positive float.')
    except tf.errors.InvalidArgumentError as e:
      raise ValueError(str(e)) from e

  if method == DecodeMethod.MEAN:
    gradient_vector_estimate = _decode_mean(sketch, gradient_length,
                                            index_seeds, sign_seeds)
  elif method == DecodeMethod.MEDIAN:
    gradient_vector_estimate = _decode_median(sketch, gradient_length,
                                              index_seeds, sign_seeds)

  if threshold is not None:
    gradient_vector_estimate = _apply_threshold(gradient_vector_estimate,
                                                threshold)
  return gradient_vector_estimate
