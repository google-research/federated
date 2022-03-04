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
"""A tff.aggregator for implementing QSGD."""
import collections

import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_federated as tff

from compressed_communication.aggregators.utils import quantize_utils


_SEED_TYPE = tff.TensorType(tf.int64, [2])


@tff.tf_computation
def get_bitstring_length(value):
  """Return size (in bits) of encoded value."""
  bitstring, _ = value
  return 32. + 8. * tf.cast(tf.strings.length(bitstring, unit="BYTE"),
                            dtype=tf.float64)


class QSGDFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that implements QSGD.

  Expects `value_type` to be a `TensorType`.

  Paper: https://arxiv.org/abs/1610.02132
  """

  def __init__(self, num_steps):
    """Initializer for QSGDFactory.

    Defines the initial quantization step size, as well as what type of
    quantization should be applied and what normalization (if any) should be
    used to scale client updates.

    Args:
      num_steps: Float that parametrizes the quantization levels,
        equal to the number of steps.
    """
    self._num_steps = num_steps

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be a float tensor, "
                       f"found {value_type}.")

    @tff.tf_computation(value_type)
    def quantize_encode(value):
      seed = tf.cast(tf.stack([tf.timestamp() * 1e6, tf.timestamp() * 1e6]),
                     dtype=tf.int64)
      norm = tf.norm(value, ord=2)
      q_step_size = norm / tf.cast(self._num_steps, tf.float32)
      quantized_value = quantize_utils.stochastic_quantize(
          value, q_step_size, seed)
      dequantized_value = quantize_utils.uniform_dequantize(
          quantized_value, q_step_size, None)
      value_size = tf.size(quantized_value, out_type=tf.float32)
      distortion = tf.reduce_sum(
          tf.square(value - dequantized_value)) / value_size
      value_nonzero_ct = tf.math.count_nonzero(
          quantized_value, dtype=tf.float32)
      sparsity = (value_size - value_nonzero_ct) / value_size
      encoded_value = (tfc.run_length_gamma_encode(data=quantized_value), norm)
      return encoded_value, distortion, sparsity

    def dequantize(value, norm):
      q_step_size = norm / tf.cast(self._num_steps, tf.float32)
      return quantize_utils.uniform_dequantize(value, q_step_size, None)

    def sum_encoded_value(value):

      @tff.tf_computation
      def get_accumulator():
        return tf.zeros(shape=value_type.shape, dtype=tf.float32)

      @tff.tf_computation
      def decode_accumulate_values(accumulator, encoded_value):
        bitstring, norm = encoded_value
        decoded_value = tfc.run_length_gamma_decode(code=bitstring,
                                                    shape=value_type.shape)
        dequantized_value = dequantize(decoded_value, norm)
        return accumulator + dequantized_value

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

    @tff.federated_computation()
    def init_fn():
      return tff.federated_value((), tff.SERVER)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      encoded_value, distortion, sparsity = tff.federated_map(
          quantize_encode, value)

      avg_distortion = tff.federated_mean(distortion)
      avg_sparsity = tff.federated_mean(sparsity)

      bitstring_lengths = tff.federated_map(get_bitstring_length, encoded_value)
      avg_bitstring_length = tff.federated_mean(bitstring_lengths)
      num_elements = tff.federated_mean(tff.federated_map(
          tff.tf_computation(lambda x: tf.size(x, out_type=tf.float64)), value))
      avg_bitrate = tff.federated_map(
          tff.tf_computation(
              lambda x, y: tf.math.divide_no_nan(x, y, name="tff_divide")),
          (avg_bitstring_length, num_elements))

      decoded_value = sum_encoded_value(encoded_value)

      return tff.templates.MeasuredProcessOutput(
          state=state,
          result=decoded_value,
          measurements=tff.federated_zip(
              collections.OrderedDict(avg_bitrate=avg_bitrate,
                                      avg_distortion=avg_distortion,
                                      avg_sparsity=avg_sparsity)))

    return tff.templates.AggregationProcess(init_fn, next_fn)
