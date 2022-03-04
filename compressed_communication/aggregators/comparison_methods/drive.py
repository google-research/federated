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
"""A tff.aggregator for implementing DRIVE."""
import collections

import tensorflow as tf
import tensorflow_federated as tff


class DRIVEFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that implements DRIVE algorithm.

  Expects `value_type` to be a `TensorType`. Assumes random rotation has been
  applied already.

  Paper: https://arxiv.org/pdf/2105.08339.pdf
  """

  def __init__(self, scaling_factor="unbiased"):
    """Initializer for DRIVEFactory.

    Args:
      scaling_factor: A string indicating what type of scaling to perform, one
        of ["unbiased", "min_distortion"], by default "unbiased."
    """
    if scaling_factor not in ["unbiased", "min_distortion"]:
      raise ValueError("Expect scaling_factor to be one of [\"unbiased\", "
                       f"\"min_distortion\"], found {scaling_factor}.")
    self._scaling_factor = scaling_factor

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be a float tensor, "
                       f"found {value_type}.")

    @tff.tf_computation
    def decode(encoded_value):
      mask_negatives, scale = encoded_value
      decoded_negatives = tf.multiply(
          tf.cast(mask_negatives, tf.float32), scale)
      decoded_non_negatives = tf.multiply(
          tf.cast(tf.logical_not(mask_negatives), tf.float32), scale)
      decoded_value = decoded_non_negatives - decoded_negatives
      return decoded_value

    @tff.tf_computation(value_type)
    def encode(value):
      mask_negatives = tf.math.less(value, 0.)
      value_size = tf.size(value, out_type=tf.float32)
      if self._scaling_factor == "min_distortion":
        scale = tf.norm(value, ord=1) / value_size
      else:
        scale = tf.math.divide_no_nan(tf.norm(value, ord=2)**2,
                                      tf.norm(value, ord=1))

      encoded_value = (mask_negatives, scale)

      bitrate = (value_size + 32.) / value_size

      decoded_value = decode(encoded_value)
      distortion = tf.reduce_sum(
          tf.square(value - decoded_value)) / value_size

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
