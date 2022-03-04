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
"""A tff.aggregator for implementing TernGrad."""
import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


class TernGradFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that implements TernGrad.

  Expects `value_type` to be a `TensorType`.

  Paper: https://arxiv.org/abs/1705.07878
  """

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be a float tensor, "
                       f"found {value_type}.")

    @tff.tf_computation
    def decode(encoded_value):
      inf_norm, sign, bitmask = encoded_value
      decoded_value = inf_norm * tf.multiply(sign, bitmask)
      return decoded_value

    @tff.tf_computation(value_type)
    def encode(value):
      inf_norm = tf.norm(value, ord=np.inf)
      sign = tf.math.sign(value)
      prob = tf.abs(value) / inf_norm
      seed = tf.cast(tf.stack([tf.timestamp() * 1e6, tf.timestamp() * 1e6]),
                     dtype=tf.int64)
      random = tf.random.stateless_uniform(value.shape, seed=seed,
                                           dtype=tf.float32)
      bitmask = tf.where(tf.less_equal(random, prob),
                         tf.ones(value.shape, dtype=tf.float32),
                         tf.zeros(value.shape, dtype=tf.float32))

      encoded_value = (inf_norm, sign, bitmask)

      value_size = tf.size(value, out_type=tf.float32)
      bitrate = (2. * value_size + 32.) / value_size

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
