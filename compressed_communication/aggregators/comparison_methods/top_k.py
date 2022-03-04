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
"""A tff.aggregator for implementing top-k sparsity."""
import collections

import tensorflow as tf
import tensorflow_federated as tff


class TopKFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that implements top-k sparsity.

  Expects `value_type` to be a `TensorType`.

  Paper: https://arxiv.org/abs/1704.05021
  """

  def __init__(self, fraction_to_select):
    """Initializer for TopKFactory.

    Args:
      fraction_to_select: What fraction of elements to select on each client,
        within range (0.0, 1.0].
    """
    if fraction_to_select <= 0.0 or fraction_to_select > 1.0:
      raise ValueError("Expect 0.0 < fraction_to_select <= 1.0, found "
                       f"{fraction_to_select}.")
    self._fraction_to_select = fraction_to_select

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be a float tensor, "
                       f"found {value_type}.")

    @tff.tf_computation
    def decode(encoded_value):
      values, indices = encoded_value
      indices = tf.expand_dims(indices, 1)
      decoded_value = tf.scatter_nd(indices, values, value_type.shape)
      return decoded_value

    @tff.tf_computation(value_type)
    def encode(value):
      value_size = tf.size(value, out_type=tf.float32)
      k_size = tf.cast(tf.math.ceil(self._fraction_to_select * value_size),
                       tf.int32)
      _, topk_indices = tf.math.top_k(tf.abs(value), k=k_size, sorted=False)
      topk_values = tf.gather(value, topk_indices)

      encoded_value = (topk_values, topk_indices)
      index_bits = tf.math.ceil(tf.math.log(value_size) / tf.math.log(2.))
      bitrate = (tf.cast(k_size, tf.float32) * 32. + index_bits) / value_size

      decoded_value = decode(encoded_value)
      distortion = tf.reduce_sum(
          tf.square(value - decoded_value)) / value_size

      return encoded_value, bitrate, distortion

    @tff.federated_computation()
    def init_fn():
      return tff.federated_value((), tff.SERVER)

    def decode_and_sum(value):
      """Encoded client values do not commute with sum: decode then sum."""

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

      result = decode_and_sum(encoded_value)

      return tff.templates.MeasuredProcessOutput(
          state=state,
          result=result,
          measurements=tff.federated_zip(
              collections.OrderedDict(avg_bitrate=avg_bitrate,
                                      avg_distortion=avg_distortion)))

    return tff.templates.AggregationProcess(init_fn, next_fn)
