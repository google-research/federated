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
"""tff.aggregators for sketching flattened gradients."""
import collections
from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

from private_linear_compression import count_sketching_utils

_SEED_TF_DTYPE = tf.int64
_SEED_TFF_TYPE = tff.types.TensorType(_SEED_TF_DTYPE, [2])


def _get_server_client_values_for_key(state, key):
  """Gets the state's value at `key` for both SERVER and CLIENTS."""
  server_value = state[key]
  return server_value, tff.federated_broadcast(server_value)


class GradientCountSketchFactory(tff.aggregators.UnweightedAggregationFactory):
  """An `UnweightedAggregationFactory` for sketching flattened gradients.

  The created `tff.templates.AggregationProcess` compresses inputs to a tensor
  of shape [`num_repeats`, num_bins] using a linear count-sketch. This reduces
  the gradient dimension. Given a gradient input dimension d, the num_bins is
  calculated to ensure a `min_compression_rate` given the `num_repeats`, via the
  following formula: num_bins = d / (`num_repeats` * `min_compression_rate`).

  The count-sketch is constructed via two independent hash functions that
  generate random signs in [-1, +1] and random hash buckets [1,..,num_bins].
  Each (index, value) pair of the gradient vector will be hashed into a bucket
  with a random sign and with `value` being its frequency. The details can be
  found in https://arxiv.org/pdf/1411.4357.pdf (Page 17).

  After aggregation, the decode function performs the unsketch operation. Two
  unsketch methods 'mean' or 'median' are currently supported, which are defined
  as the mean (or the median) of `num_repeats` (unsketched) rows.

  This aggregator may be useful as a preprocessing step to reduce the dimension
  (i.e. number of parameters) of gradient vectors. For example, if this factory
  receives type <float32[d]> and the `num_repeats` and num_bins are `nr`,
  `nb`, then the inner_agg_factory will operate on <float32[nr], float32[nb]>.
  This is typically much smaller than d, meaning `min_compression_rate` >> 1.0.

  Note that the input gradient vector must be flattened first, i.e. it must take
  the type of <float32[d]>. If the model is not flattened, one should apply a
  `concat` factory first. See, for example tff.aggregators.concat_factory

  The `num_repeats` is equivalent to the sketch length and the num_bins the
  sketch width in sketching literature.
  """

  def __init__(
      self,
      min_compression_rate: float = 5.,
      decode_method: count_sketching_utils.DecodeMethod = count_sketching_utils
      .DecodeMethod.MEAN,
      num_repeats: int = 15,
      inner_agg_factory: Optional[
          tff.aggregators.UnweightedAggregationFactory] = None,
      parallel_iterations: Optional[int] = None,
  ):
    """Initializes the GradientCountSketchFactory.

    Args:
      min_compression_rate: Ratio of original gradient length to the compressed
        dimension. Because the num_bins must use integer increments, the actual
        compression rate may be slightly larger. The `min_compression_rate` and
        `num_repeats` defines the `num_bins`. Must be >= 1.0.
      decode_method: Defines the method for decoding the sketch.
        DecodeMethod.MEAN is preferred for runtime and memory.
      num_repeats: The number of independent hashes to apply, i.e., the sketch
        length. Must be a non-negative integer.
      inner_agg_factory: The inner `UnweightedAggregationFactory` to aggregate
        the values after the transform. Defaults to
        `tff.aggregators.SumFactory()` if None is provided.
      parallel_iterations: An integer specifying how many hash rows to compute
        in parallel that is passed directly to `tf.map_fn`. See their
        documentation for more details. Must be a non-negative integer.

    Raises:
      TypeError: If `inner_agg_factory` is not an instance of
        `tff.aggregators.UnweightedAggregationFactory`.
    """
    if inner_agg_factory is None:
      inner_agg_factory = tff.aggregators.SumFactory()

    if not isinstance(inner_agg_factory,
                      tff.aggregators.UnweightedAggregationFactory):
      raise TypeError('`inner_agg_factory` must have type '
                      'UnweightedAggregationFactory. '
                      f'Found {type(inner_agg_factory)}.')

    if parallel_iterations is not None and parallel_iterations < 0:
      raise ValueError(
          f'Detected `parallel_iterations`={parallel_iterations} which must be '
          f'> = 0.')
    if min_compression_rate < 1.:
      raise ValueError(
          f'Detected `compression_rate`={min_compression_rate} which must be >='
          f' 1.0.')
    if num_repeats < 1:
      raise ValueError(
          f'Detected `num_repeats`={num_repeats} which must be >= 1.')

    self._inner_agg_factory = inner_agg_factory
    self._decode_method = decode_method
    self._parallel_iterations = parallel_iterations
    self._min_compression_rate = min_compression_rate
    self._num_repeats = num_repeats

  def _get_num_bins(self, gradient_length: float,
                    compression_rate: float) -> int:
    """Gets `num_bins` given `gradient_length` and `compression_rate`."""
    return int(gradient_length // (self._num_repeats * compression_rate))

  def create(self, value_type):
    """Constructs an aggregation factory that applies sketching."""
    if not value_type.is_tensor():
      raise TypeError('Expected `value_type` to be `TensorType`.')
    value_type.shape.assert_has_rank(1)

    gradient_length = value_type.shape.num_elements()
    num_bins = self._get_num_bins(gradient_length, self._min_compression_rate)
    if num_bins < 1:
      raise ValueError(
          f'Chosen `min_compression_rate`={self._min_compression_rate} too '
          f'large for ambient_dimension of value_type:`{value_type}` and '
          f'`num_repeats`={self._num_repeats}.')
    flattened_sketch_size = self._num_repeats * num_bins
    flattened_encoded_type = tff.to_type(
        (value_type.dtype, [flattened_sketch_size]))

    inner_process = self._inner_agg_factory.create(flattened_encoded_type)

    @tff.tf_computation((value_type, _SEED_TFF_TYPE, _SEED_TFF_TYPE))
    def encode_fn(gradient, index_seeds, sign_seeds):
      num_repeats_tf = tf.convert_to_tensor(self._num_repeats, dtype=tf.int32)
      num_bins_tf = tf.convert_to_tensor(num_bins, dtype=tf.int32)
      sketch = count_sketching_utils.encode(gradient, num_repeats_tf,
                                            num_bins_tf, index_seeds,
                                            sign_seeds,
                                            self._parallel_iterations)
      return tf.reshape(sketch, [flattened_sketch_size])

    @tff.tf_computation(
        (flattened_encoded_type, _SEED_TFF_TYPE, _SEED_TFF_TYPE))
    def decode_fn(flattened_sketch, index_seeds, sign_seeds):
      sketch = tf.reshape(flattened_sketch, [self._num_repeats, num_bins])
      grad_length = tf.convert_to_tensor(gradient_length, dtype=tf.int32)
      return count_sketching_utils.decode(sketch, grad_length, index_seeds,
                                          sign_seeds, self._decode_method, None)

    num_seed_pairs = 2
    increment_seed_fn = tff.tf_computation(
        lambda seed: _next_round_seed_pair(seed, num_seed_pairs))

    @tff.federated_computation
    def init_fn():
      index_seed_position = tff.federated_value(0, tff.SERVER)
      sign_seed_position = tff.federated_value(1, tff.SERVER)
      num_seeds = tff.federated_value(num_seed_pairs, tff.SERVER)
      seed_pairs = tff.federated_map(_init_seed_pairs, num_seeds)

      state = collections.OrderedDict(
          inner_agg_process=inner_process.initialize(),
          index_seeds=tff.federated_map(_select_seed,
                                        (seed_pairs, index_seed_position)),
          sign_seeds=tff.federated_map(_select_seed,
                                       (seed_pairs, sign_seed_position)),
      )
      return tff.federated_zip(state)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      inner_state = state['inner_agg_process']
      server_index_seeds, client_index_seeds = (
          _get_server_client_values_for_key(state, 'index_seeds'))
      server_sign_seeds, client_sign_seeds = (
          _get_server_client_values_for_key(state, 'sign_seeds'))

      flattened_encoded_sketch = tff.federated_map(
          encode_fn, (value, client_index_seeds, client_sign_seeds))

      inner_output = inner_process.next(inner_state, flattened_encoded_sketch)
      flattened_aggregated_sketch = inner_output.result

      decoded_gradient = tff.federated_map(
          decode_fn,
          (flattened_aggregated_sketch, server_index_seeds, server_sign_seeds))

      measurements = inner_output.measurements
      new_state = collections.OrderedDict(
          inner_agg_process=inner_output.state,
          index_seeds=tff.federated_map(increment_seed_fn, server_index_seeds),
          sign_seeds=tff.federated_map(increment_seed_fn, server_sign_seeds),
      )

      return tff.templates.MeasuredProcessOutput(
          state=tff.federated_zip(new_state),
          result=decoded_gradient,
          measurements=measurements)

    return tff.templates.AggregationProcess(init_fn, next_fn)


@tff.tf_computation
def _init_seed_pairs(num_seeds: int) -> tf.Tensor:
  """Initializes `num_seeds` unique seed_pairs for tf.random.stateless* ops."""
  scale_factor = 10**6  # Timestamp returns fractional seconds.
  round_seed = tf.cast(tf.timestamp() * scale_factor, _SEED_TF_DTYPE)
  round_seed = tf.repeat(round_seed, num_seeds)
  round_seed += tf.range(num_seeds, dtype=_SEED_TF_DTYPE)
  seed_pairs = tf.stack(
      [round_seed, tf.zeros([num_seeds], dtype=_SEED_TF_DTYPE)], 1)
  return tf.reshape(seed_pairs, [num_seeds, 2])


@tff.tf_computation
def _select_seed(seed_pairs: tf.Tensor, index: int) -> tf.Tensor:
  """Selects the seed_pair from a vertical stack of `seed_pairs` at `index`."""
  return seed_pairs[index]


def _next_round_seed_pair(seed_pair: tf.Tensor, stride: int) -> tf.Tensor:
  """Ensures unique seeds for all `seed_pairs` by incrementing by `stride`."""
  return seed_pair + [stride, 0]
