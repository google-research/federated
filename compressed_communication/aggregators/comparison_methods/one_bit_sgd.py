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
"""A tff.aggregator for implementing 1bit SGD."""
import collections

import tensorflow as tf
import tensorflow_federated as tff


class OneBitSGDFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that quantizes to 1 bit.

  Expects `value_type` to be a `TensorType`.

  Paper:
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/IS140694.pdf
  """

  def __init__(self, threshold=0.):
    """Initializer for OneBitSGDFactory.

    Args:
      threshold: Float that specifies the boundary between the two quantization
        levels, by default 0. `threshold` values are mapped to level above.
    """
    self._threshold = threshold

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be a float tensor, "
                       f"found {value_type}.")

    @tff.tf_computation
    def decode(encoded_value):
      mask_above_threshold, means = encoded_value
      mean_below_threshold, mean_above_threshold = means
      decoded_above_threshold = tf.multiply(mask_above_threshold,
                                            mean_above_threshold)
      decoded_below_threshold = tf.multiply(
          (1.0 - mask_above_threshold), mean_below_threshold)
      decoded_value = decoded_above_threshold + decoded_below_threshold
      return decoded_value

    @tff.tf_computation(value_type)
    def encode(value):
      mask_below_threshold = tf.math.less(value, self._threshold)
      mask_above_threshold = tf.logical_not(mask_below_threshold)
      mask_below_threshold = tf.cast(mask_below_threshold, tf.float32)
      mask_above_threshold = tf.cast(mask_above_threshold, tf.float32)

      mean_below_threshold = tf.divide(
          tf.reduce_sum(tf.multiply(value, mask_below_threshold)),
          tf.maximum(tf.reduce_sum(mask_below_threshold), 1.0))
      mean_above_threshold = tf.divide(
          tf.reduce_sum(tf.multiply(value, mask_above_threshold)),
          tf.maximum(tf.reduce_sum(mask_above_threshold), 1.0))

      means = (mean_below_threshold, mean_above_threshold)

      encoded_value = (mask_above_threshold, means)

      value_size = tf.size(value, out_type=tf.float32)
      bitrate = (value_size + 64.) / value_size

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
