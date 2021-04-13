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
"""Implements DPQuery interface for queries with compression operations.

This query acts as a wrapper/decorator for discrete-valued DPQueries
to provide compression operations (and their inverses) including:
  (1) randomized Hadamard transform;
  (2) scaling; and
  (3) quantization with stochastic rounding.
"""

import abc
import collections

import attr
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

from distributed_discrete_gaussian import compression_utils


def _attr_bool_check(instance, attribute, value):
  if not isinstance(value, tf.Tensor):
    if not isinstance(value, (bool, np.bool, np.bool_)):
      raise ValueError(f'`{attribute.name}` should be a bool constant. Found '
                       f'`{value}` with type `{type(value)}`.')


@attr.s(eq=False)
class QuantizationParams(metaclass=abc.ABCMeta):
  """Common parameters for quantization.

  Attributes:
    stochastic: A bool constant specifying whether to use stochastic rounding
      for quantization. Use deterministic rounding if set to False.
    conditional: A bool constant specifying whether to use conditional rounding.
      If True, we keep retrying stochastic rounding until the L2 norm of the
      rounded vector doesn't grow over the probabilistic bound. If False, use
      unconditional stochastic rounding. Ignored if `stochastic` is False.
    l2_norm_bound: A float constant denoting the bound of the L2 norm of the
      input records. This is useful when `l2_norm_bound` is larger than the
      input norm, in which case we can allow more leeway during conditional
      stochastic rounding rounding.
    beta: A constant in [0, 1) controlling the concentration inequality for the
      probabilistic norm bound after rounding.
  """
  stochastic = attr.ib(validator=_attr_bool_check)
  conditional = attr.ib(validator=_attr_bool_check)
  l2_norm_bound = attr.ib()
  beta = attr.ib()

  @l2_norm_bound.validator
  def check_l2_norm_bound(self, attribute, value):
    if not isinstance(value, tf.Tensor):
      if value <= 0:
        raise ValueError(f'`l2_norm_bound` must be > 0. Found {value}.')

  @beta.validator
  def check_beta(self, attribute, value):
    if not isinstance(value, tf.Tensor):
      if value < 0 or value >= 1:
        raise ValueError(f'`beta` must be in [0, 1). Found {value}.')

  @abc.abstractmethod
  def to_tf_tensors(self):
    raise NotImplementedError(
        'The function `to_tf_tensors` is not implemented in the '
        'base `QuantizationParams` class. Please, use a subclass.')


@attr.s(eq=False)
class ScaledQuantizationParams(QuantizationParams):
  """Parameters for scaling-based quantization.

  Attributes:
    quantize_scale: A scale factor controlling the quantization granularity; it
      is applied to the input records before rounding to the nearest integer.
      This also encompasses the scaling needed for inner DP mechanisms.
  """
  quantize_scale = attr.ib()

  @quantize_scale.validator
  def check_quantize_scale(self, attribute, value):
    if not isinstance(value, tf.Tensor):
      if value <= 0:
        raise ValueError(f'`quantize_scale` must be positive. Found {value}.')

  def to_tf_tensors(self):
    # Make a copy to ensure TF tensors get created only when needed.
    return attr.evolve(
        self,
        stochastic=tf.cast(self.stochastic, tf.bool),
        conditional=tf.cast(self.conditional, tf.bool),
        l2_norm_bound=tf.cast(self.l2_norm_bound, tf.float32),
        beta=tf.cast(self.beta, tf.float32),
        quantize_scale=tf.cast(self.quantize_scale, tf.float32))


class CompressionSumQuery(tfp.SumAggregationDPQuery):
  """Implements DPQuery interface for wrapping DPQueries with compression ops.

  This query is responsible for (1) encoding the records before applying DP
  mechanisms and (2) decoding the aggregated record. It delegates the actual
  DP operations (noise addition and norm checks) to an `inner_query` which
  must operate on tf.int32 record values.

  The encoding/decoding operations include the following (and their inverses):
    (1) randomized Hadamard transform;
    (2) scaling; and
    (2) quantization with stochastic rounding.

  The input records to this DPQuery first gets encoded/compressed, and
  then sent to the `inner_query` noise addition mechanisms; the aggregated
  output from the `inner_query` are then decoded by applying the above steps
  in reverse. No noises are added within this DPQuery.
  """
  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ('sample_hadamard_seed', 'quantization_params',
                       'inner_query_state', 'record_template'))

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple(
      '_SampleParams',
      ('sample_hadamard_seed', 'quantization_params', 'inner_query_params'))

  def __init__(self, quantization_params: QuantizationParams,
               inner_query: tfp.SumAggregationDPQuery, record_template):
    """Initializes the CompressionSumQuery.

    Args:
      quantization_params: A config object of type `QuantizationParams`.
      inner_query: A `SumAggregationDPQuery` that operates on discrete records
        with dtype `tf.int32`. This query is responsible for the DP mechanisms.
      record_template: A nested structure of tensors, TensorSpecs, or numpy
        arrays used as a template to create the initial sample state. It is
        required here as compression operations (particularly randomized HT) can
        change the record shapes and we need to keep track of the original
        record shapes for decoding the aggregated records.

    Raises:
      ValueError: If `inner_query` is not a `tfp.SumAggregationDPQuery`.
    """
    if not isinstance(inner_query, tfp.SumAggregationDPQuery):
      raise ValueError('`inner_query` should be a `SumAggregationDPQuery`. '
                       f'Found {type(inner_query)}.')
    self._quantization_params = quantization_params
    self._inner_query = inner_query
    self._record_template = record_template

    if isinstance(quantization_params, ScaledQuantizationParams):
      self._quantization_fn = scaled_quantization
      self._inverse_quantization_fn = inverse_scaled_quantization
    else:
      raise ValueError(
          f'Unknown type(quantization_params) of {type(quantization_params)} '
          f'with value {quantization_params}.')

  def set_ledger(self, ledger):
    raise NotImplementedError(
        'Ledger has not yet been implemented for this query!')

  def initial_sample_state(self, template):
    # We operate on the entire tensor structure as a single vector.
    template_as_vector = compression_utils.flatten_concat(template)
    # Hadamard transform does padding, so we also pad the agg template.
    padded_template_as_vector = compression_utils.pad_zeros(template_as_vector)
    # Quantization involves casting to int32.
    inner_template = tf.cast(padded_template_as_vector, tf.int32)
    return self._inner_query.initial_sample_state(inner_template)

  def initial_global_state(self):
    return self._GlobalState(
        new_seed_pair(), self._quantization_params.to_tf_tensors(),
        self._inner_query.initial_global_state(),
        tf.nest.map_structure(tf.zeros_like, self._record_template))

  def derive_sample_params(self, global_state):
    return self._SampleParams(
        global_state.sample_hadamard_seed, global_state.quantization_params,
        self._inner_query.derive_sample_params(global_state.inner_query_state))

  def _encode_record(self, record, sample_hadamard_seed: tf.Tensor,
                     quantization_params: QuantizationParams):
    """Applies compression to the record as a single concatenated vector."""
    record_as_vector = compression_utils.flatten_concat(record)
    casted_record = tf.cast(record_as_vector, tf.float32)
    rotated_record = compression_utils.randomized_hadamard_transform(
        casted_record, sample_hadamard_seed)
    encoded_record = self._quantization_fn(rotated_record, quantization_params)
    return encoded_record

  def _decode_agg_record(self, agg_record, record_template,
                         sample_hadamard_seed: tf.Tensor,
                         quantization_params: QuantizationParams):
    """Reverts the operations by `_encode_record` after aggregation."""

    def cast_to_input_dtype(t, t_input):
      if t_input.dtype.is_integer:
        t = tf.round(t)
      return tf.cast(t, t_input.dtype)

    template_as_vector = compression_utils.flatten_concat(record_template)
    dequantized_record = self._inverse_quantization_fn(agg_record,
                                                       quantization_params)
    unrotated_record = compression_utils.inverse_randomized_hadamard_transform(
        dequantized_record,
        original_dim=tf.size(template_as_vector),
        seed_pair=sample_hadamard_seed)
    uncasted_record = cast_to_input_dtype(unrotated_record, template_as_vector)
    decoded_record = compression_utils.inverse_flatten_concat(
        uncasted_record, record_template)
    return decoded_record

  def preprocess_record(self, params, record):
    """Compress the record and delegate to inner query for DP mechanisms."""
    encoded_record = self._encode_record(record, params.sample_hadamard_seed,
                                         params.quantization_params)
    return self._inner_query.preprocess_record(params.inner_query_params,
                                               encoded_record)

  def get_noised_result(self, sample_state, global_state):
    # Delegate to inner query for final aggregation result.
    agg_record, new_inner_query_state = self._inner_query.get_noised_result(
        sample_state, global_state.inner_query_state)

    # Decode the aggregated result.
    decoded_agg_record = self._decode_agg_record(
        agg_record, global_state.record_template,
        global_state.sample_hadamard_seed, global_state.quantization_params)

    # Generate new seed_pair for the next sample here.
    new_global_state = global_state._replace(
        sample_hadamard_seed=new_seed_pair(),
        inner_query_state=new_inner_query_state)

    return decoded_agg_record, new_global_state


def new_seed_pair() -> tf.Tensor:
  """Create a seed pair with shape=[2] to be used by `tf.random.stateless_*`."""
  return tf.random.uniform(
      shape=[2], minval=tf.int32.min, maxval=tf.int32.max, dtype=tf.int32)


def scaled_quantization(record, quantization_params: ScaledQuantizationParams):
  """Quantization by scaling up the inputs and rounding to integers."""

  def quantization(t):
    quantized_t = compression_utils.scaled_quantization(
        t,
        quantization_params.quantize_scale,
        stochastic=quantization_params.stochastic,
        conditional=quantization_params.conditional,
        l2_norm_bound=quantization_params.l2_norm_bound,
        beta=quantization_params.beta)
    return tf.cast(quantized_t, tf.int32)  # Inner discrete query uses int32.

  return tf.nest.map_structure(quantization, record)


def inverse_scaled_quantization(agg_record,
                                quantization_params: ScaledQuantizationParams):
  """Applies the inverse of `scaled_quantization` after aggregation."""

  def dequantization(t):
    # Revert to float32 for decoding operations.
    t = tf.cast(t, tf.float32)
    return compression_utils.inverse_scaled_quantization(
        t, quantization_params.quantize_scale)

  return tf.nest.map_structure(dequantization, agg_record)
