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
"""tff.aggregators for compressing flattened gradients via SRHT.

SRHT is the subsampled randomized hadamart transform. A detailed description is
found under GradientSRHTSketchFactory.
"""
import collections

from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

from distributed_dp import compression_utils


class GradientSRHTSketchFactory(tff.aggregators.UnweightedAggregationFactory):
  """A aggregation factory for rotating and subsampling flattened gradients.

  The created `tff.templates.AggregationProcess` does Subsampled Randomized
  Hadamard Transform (SRHT) based sketch on inputs and outputs a
  tensor of shape [original_dim*`compression_rate`] to reduce the dimension. In
  particular, SRHT performs the following (randomized) linear transformation:
  y = S*D*H*x, where S is the sampling matrix, D is a diagonal matrix whose
  entries are independent random signs, and H is a Walsh–Hadamard matrix. See
  https://arxiv.org/pdf/1011.1595.pdf (Page 2) for more details.

  After aggregation, the decode function performs the unsketch operation,
  defined as x_hat = H*D*P*y where P is the `pad_zeros` operation, and outputs
  an estimate of the mean of (uncompressed) gradients.

  This aggregator may be useful as a preprocessing step to reduce the dimension
  (i.e., number of parameters) of gradient vectors. For example, if this factory
  receives type <float32[d]>, and the compression_rate is `cr` < 1, then the
  inner_agg_factory will operate on <float32[int(d*cr)]>.

  Note that the input gradient vector must be flattened first, i.e. it must have
  type <float32[d]>. If the model is not flattened, one should apply a
  `concat` factory first, see, for example tff.aggregators.concat_factory.
  """

  def __init__(self,
               inner_agg_factory: Optional[
                   tff.aggregators.UnweightedAggregationFactory] = None,
               compression_rate: int = 1,
               repeat: int = 3):
    """Initializes the GradientSRHTSketchFactory.

    Args:
      inner_agg_factory: The inner `UnweightedAggregationFactory` to aggregate
        the values after the transform. Defaults to
        `tff.aggregators.SumFactory()` if None is provided.
      compression_rate: A fraction between (0, 1] that determines the compressed
        vector size as `original_dimension` * `compresseion_rate`. Setting
        `compression_rate=1` is equivalent to random rotation.
      repeat: A positive integer specifying the number of times the randomized
        Hadamard transform is performed.

    Raises:
      TypeError: If `inner_agg_factory` is not an instance of
        `tff.aggregators.UnweightedAggregationFactory`
      ValueError: If `compression_rate` is not in (0, 1], or of `repeat` is not
      a positive integer.
    """
    if inner_agg_factory is None:
      inner_agg_factory = tff.aggregators.SumFactory()

    if not isinstance(inner_agg_factory,
                      tff.aggregators.UnweightedAggregationFactory):
      raise TypeError('`inner_agg_factory` must have type '
                      'UnweightedAggregationFactory. '
                      f'Found {type(inner_agg_factory)}.')

    if not isinstance(repeat, int) or repeat < 1:
      raise ValueError('`repeat` should be a positive integer. '
                       f'Found {repeat}.')

    if compression_rate <= 0 or compression_rate > 1:
      raise ValueError('`compression_rate` must be a number in (0, 1].'
                       f'Found {compression_rate}.')

    self._inner_agg_factory = inner_agg_factory
    self._compression_rate = tf.cast(compression_rate, tf.float32)
    self._repeat = repeat

  def create(self, value_type):

    if value_type.dtype != tf.float32:
      raise TypeError('`value_type` must have dtype '
                      'tf.float32. '
                      f'Found {value_type.dtype}.')
    original_dim = value_type.shape[0]
    encoded_dim = int(original_dim * self._compression_rate)
    encoded_type = tff.to_type((tf.float32, [encoded_dim]))
    inner_process = self._inner_agg_factory.create(encoded_type)
    seed_pairs_type = _init_seed_pairs.type_signature.result

    @tff.federated_computation()
    def init_fn():
      state = collections.OrderedDict(
          round_seed=tff.federated_eval(_init_seed_pairs, tff.SERVER),
          inner_agg_process=inner_process.initialize())
      return tff.federated_zip(state)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.FederatedType(value_type, tff.CLIENTS))
    def next_fn(state, value):
      server_seed = state['round_seed']
      client_seed = tff.federated_broadcast(server_seed)
      inner_state = state['inner_agg_process']

      @tff.tf_computation(value_type, seed_pairs_type)
      def encode_fn(x, seeds):
        return srht_encode(
            x,
            encoded_dim,
            seed_sampling=seeds[0],
            seed_rotation=seeds[1],
            repeat=self._repeat)

      encoded_value = tff.federated_map(encode_fn, (value, client_seed))

      inner_output = inner_process.next(inner_state, encoded_value)

      @tff.tf_computation(encoded_type, seed_pairs_type)
      def decode_fn(x, seeds):
        return srht_sketch_decode(
            x,
            original_dim,
            seed_sampling=seeds[0],
            seed_rotation=seeds[1],
            repeat=self._repeat)

      decoded_value = tff.federated_map(decode_fn,
                                        (inner_output.result, server_seed))
      measurements = inner_output.measurements
      new_state = collections.OrderedDict(
          round_seed=tff.federated_map(_next_seed_pairs, server_seed),
          inner_agg_process=inner_output.state)

      return tff.templates.MeasuredProcessOutput(
          state=tff.federated_zip(new_state),
          result=decoded_value,
          measurements=measurements)

    return tff.templates.AggregationProcess(init_fn, next_fn)


@tff.tf_computation()
def _init_seed_pairs():
  return tf.constant([[1, 0], [0, 2]], dtype=tf.int32)


@tff.tf_computation(_init_seed_pairs.type_signature.result)
def _next_seed_pairs(seed_pairs: tf.Tensor) -> tf.Tensor:
  return seed_pairs + [[0, 1], [1, 0]]


@tf.function
def _get_subsampled_indices(
    max_index: int, sampling_length: int,
    seed: tf.Tensor = tf.constant([0, 0])) -> tf.Tensor:
  """Subsamples `sampling_length` indices of [0, 1,..., max_index-1]."""
  v = tf.random.stateless_uniform([max_index], seed=seed)
  return tf.reshape(tf.argsort(v)[:sampling_length], [-1, 1])


@tf.function
def _subsampling(
    vector: tf.Tensor,
    subsample_dim: int,
    seed: tf.Tensor = tf.constant([0, 0])) -> tf.Tensor:
  """Subsamples `subsample_dim` coordinates from `vector`."""
  original_dim = vector.shape[0]
  indices = _get_subsampled_indices(original_dim, subsample_dim, seed=seed)
  return tf.gather(vector, tf.reshape(indices, [subsample_dim]))


@tf.function
def _pad_zeros(
    vector: tf.Tensor, original_dim: int,
    seed: tf.Tensor = tf.constant([0, 0])) -> tf.Tensor:
  """Pads zeros to coordinates that are discarded by `subsampling` function."""
  subsample_dim = vector.shape[0]
  indices = _get_subsampled_indices(original_dim, subsample_dim, seed=seed)
  return tf.scatter_nd(indices=indices, updates=vector, shape=[original_dim])


@tf.function
def srht_encode(vector: tf.Tensor,
                subsample_dim: int,
                seed_sampling: tf.Tensor = tf.constant([0, 0]),
                seed_rotation: tf.Tensor = tf.constant([0, 0]),
                repeat: int = 3) -> tf.Tensor:
  """Performs RHT and subsamples `subsample_dim` coordinates from `vector`.

  This function performs the following (randomized) linear transformation:
  y = S*D*H*x, where S is the sampling matrix, D is a diagonal matrix whose
  entries are independent random signs, and H is a Walsh–Hadamard matrix. See
  https://arxiv.org/pdf/1011.1595.pdf (Page 2) for more details.

  Args:
    vector: A tensor with dtype float32 with one dimension.
    subsample_dim: An integer specifying the dimension of the subsampled 1-d
      tensor.
    seed_sampling: A tensor of shape [1, 2] and dtype=tf.int32 used as the seed
      for random sampling.
    seed_rotation: A tensor of shape [1, 2] and dtype=tf.int32 used as the seed
      for random rotation.
    repeat: A positive integer specifying how many times the RHT is performed.

  Returns:
    A tensor of shape [subsample_dim].

  Raises:
    ValueError: if the passed `subsample_dim` > the dimension of `vector` or if
    `repeat` is not a positive integer.
  """
  original_dim = vector.shape[0]
  if subsample_dim > original_dim:
    raise ValueError(f'Detected subsampled dimension {subsample_dim} must be'
                     f'smaller than the original dimension {original_dim}.')

  if not isinstance(repeat, int) or repeat < 1:
    raise ValueError('`repeat` should be a positive integer. '
                     f'Found {repeat}.')

  rotated_vector = compression_utils.randomized_hadamard_transform(
      vector, seed_pair=seed_rotation, repeat=repeat)
  sampled_vector = _subsampling(rotated_vector, subsample_dim, seed_sampling)
  return tf.multiply(tf.math.sqrt(original_dim / subsample_dim), sampled_vector)


@tf.function
def srht_sketch_decode(subsampled_vector: tf.Tensor,
                       original_dim: int,
                       seed_sampling: tf.Tensor = tf.constant([0, 0]),
                       seed_rotation: tf.Tensor = tf.constant([0, 0]),
                       repeat: int = 3) -> tf.Tensor:
  """Pads zeros and performs inverse RHT to `vector`.

  This function peforms the inverse SRHT by computing x_hat = H*D*P*y where P
  is the `pad_zeros` operation, and outputs an estimate of the original vector.
  Note that the decoder must use the same seed as the encoder.

  Args:
    subsampled_vector: A tensor with one dimension.
    original_dim: An integer specifying the dimension of the recovered 1-d
      tensor.
    seed_sampling: A tensor of shape [1, 2] and dtype=tf.int32 used as the seed
      for random sampling.
    seed_rotation: A tensor of shape [1, 2] and dtype=tf.int32 used as the seed
      for random rotation.
    repeat: A positive integer specifying how many times the inverse RHT is
      performed.

  Returns:
    A tensor of shape [original_dim].

  Raises:
    ValueError: if the passed `subsample_dim` > the dimension of `vector` or if
    `repeat` is not a positive integer.
  """
  subsample_dim = subsampled_vector.shape[0]
  if subsample_dim > int(original_dim):
    raise ValueError(f'Detected subsampled dimension {subsample_dim} must be'
                     f'smaller than the original dimension {original_dim}.')

  log2_dim = tf.math.log(tf.cast(original_dim, tf.float32)) / tf.math.log(2.0)
  pad_dim = tf.pow(2, tf.cast(tf.math.ceil(log2_dim), tf.int32))
  padded_vector = _pad_zeros(subsampled_vector, pad_dim, seed_sampling)
  return compression_utils.inverse_randomized_hadamard_transform(
      padded_vector, original_dim, seed_pair=seed_rotation, repeat=repeat)
