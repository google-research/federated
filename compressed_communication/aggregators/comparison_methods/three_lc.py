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
"""A tff.aggregator for implementing 3LC."""
import collections

import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication.aggregators.utils import quantize_utils


class ThreeLCFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that implements 3LC.

  Expects `value_type` to be a `TensorType`.

  Paper: https://arxiv.org/abs/1802.07389
  """

  def __init__(self, sparsity_factor=1.):
    """Initializer for ThreeLCFactory.

    Args:
      sparsity_factor: By default 1.
    """
    self._sparsity_factor = sparsity_factor

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be a float tensor, "
                       f"found {value_type}.")

    @tff.tf_computation((value_type, tf.float32))
    def decode(encoded_value):
      quantized_value, scale_factor = encoded_value
      decoded_value = scale_factor * quantized_value
      return decoded_value

    @tff.tf_computation
    def get_zero_run_lengths(value):
      # Append nonzero value at start and end to capture length of any leading
      # or trailing zeros.
      value = tf.cast(value, tf.int32)
      padded_value = tf.concat(
          [tf.constant([1]), value, tf.constant([1])], axis=0)
      nonzero_indices = tf.where(tf.not_equal(padded_value, 0))
      zero_run_lengths = nonzero_indices[1:] - nonzero_indices[:-1]
      # Account for case where there are no trailing zeros.
      zero_run_lengths = tf.cond(
          tf.equal(zero_run_lengths[-1], 1), lambda: zero_run_lengths[:-1],
          lambda: zero_run_lengths)
      zero_run_lengths = tf.subtract(zero_run_lengths, 1)
      zero_run_lengths = tf.reshape(zero_run_lengths,
                                    [tf.size(zero_run_lengths)])
      zero_run_lengths = tf.gather(zero_run_lengths,
                                   tf.where(zero_run_lengths > 0))
      return tf.cast(zero_run_lengths, tf.float32)

    @tff.tf_computation(value_type)
    def encode(value):
      max_magnitude = tf.reduce_max(tf.abs(value))
      scale_factor = max_magnitude * self._sparsity_factor
      seed = tf.cast(tf.stack([tf.timestamp() * 1e6, tf.timestamp() * 1e6]),
                     dtype=tf.int64)
      quantized_value = tf.cast(quantize_utils.stochastic_quantize(
          value, scale_factor, seed), tf.float32)
      encoded_value = (quantized_value, scale_factor)

      decoded_value = decode(encoded_value)
      value_size = tf.size(value, out_type=tf.float32)
      distortion = tf.reduce_sum(
          tf.square(value - decoded_value)) / value_size

      @tf.function
      def get_pad(size):
        pad = 0
        if tf.math.floormod(size, 5) > 0:
          pad = 5 - tf.cast(tf.math.floormod(size, 5), tf.int32)
        return pad

      padded_value = tf.pad(quantized_value, [[0, get_pad(value_size)]])
      quintuples = tf.reshape(padded_value, (-1, 5))
      binarized_value = tf.cast(tf.logical_not(
          tf.reduce_all(tf.equal(quintuples, 0), axis=-1)), tf.float32)
      nonzero_bits = tf.reduce_sum(binarized_value) * 8.
      runlengths = get_zero_run_lengths(binarized_value)
      # base-3^5 encoding represents 2 <= runlengths <= 14 with a single byte
      zero_bits = tf.reduce_sum(tf.math.ceil(runlengths / 14.)) * 8.
      bitrate = (nonzero_bits + zero_bits + 32.) / value_size

      return encoded_value, bitrate, distortion

    @tff.federated_computation()
    def init_fn():
      return tff.federated_value((), tff.SERVER)

    def sum_encoded_value(value):

      @tff.tf_computation
      def get_accumulator():
        return tf.zeros(shape=value_type.shape, dtype=tf.float32)

      @tff.tf_computation
      def decode_accumulate_values(accumulator, encoded_value):
        decoded_value = decode(encoded_value)
        return accumulator + decoded_value

      @tff.tf_computation
      def merge_decoded_values(decoded_value_1, decoded_value_2):
        return decoded_value_1 + decoded_value_2

      @tff.tf_computation
      def report_decoded_summation(summed_decoded_values):
        return summed_decoded_values

      return tff.federated_aggregate(
          value,
          zero=get_accumulator(),
          accumulate=decode_accumulate_values,
          merge=merge_decoded_values,
          report=report_decoded_summation)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      encoded_value, bitrate, distortion = tff.federated_map(encode, value)
      avg_bitrate = tff.federated_mean(bitrate)
      avg_distortion = tff.federated_mean(distortion)

      result = sum_encoded_value(encoded_value)

      return tff.templates.MeasuredProcessOutput(
          state=state,
          result=result,
          measurements=tff.federated_zip(
              collections.OrderedDict(avg_bitrate=avg_bitrate,
                                      avg_distortion=avg_distortion)))

    return tff.templates.AggregationProcess(init_fn, next_fn)
