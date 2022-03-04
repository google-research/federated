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
"""A tff.aggregator for encoding client model weight updates."""

import collections
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_federated as tff


def get_bitstring_length(value):
  """Return size (in bits) of encoded value."""
  return 8. * tf.cast(tf.strings.length(value, unit="BYTE"), dtype=tf.float64)


class EliasGammaEncodedSumFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that encodes input integer tensor elements.

  The created tff.templates.AggregationProcess encodes the input tensor as a
  bitstring. The `value_type` is expected to be a Type of integer tensors,
  with the expecataion that it should have relatively large amount of zeros.
  Each value is encoded according to the following protocol.

  First, one more than the number of zeros preceding the first non-zero integer
  in the tensor is encoded using the Elias Gamma code, a universal code which
  maps positive integers to a bitstring representation. Next, the sign of the
  non-zero integer is encoded with a single bit. The magnitude of the integer is
  encoded using the Elias Gamma code. This process is repeated for the remaining
  elements of the integer tensor and the substrings are concatenated into a
  single bitstring.

  Information about the Elias Gamma code can be found here:
  https://ieeexplore.ieee.org/document/1055349. Notably, the Elias Gamma code
  is used to compress positive integers whose values are unbounded but for which
  smaller values are more likely to occur than larger values.

  The bitstrings are aggregated at SERVER and decoded to the same shape as the
  original input integer tensors. This aggregator computes the sum over decoded
  client values at SERVER and outputs the sum placed at SERVER.

  The process returns an empty `state`, the summed client values in `result` and
  records the average number of encoded bits sent from CLIENT --> SERVER in
  `measurements` as a dictionary with key `avg_bitrate`.
  """

  def create(self, value_type):
    if not tff.types.is_structure_of_integers(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be an integer tensor, "
                       f"found {value_type}.")

    def sum_encoded_value(value):

      @tff.tf_computation
      def get_accumulator():
        return tf.zeros(shape=value_type.shape, dtype=tf.int32)

      @tff.tf_computation
      def decode_accumulate_values(accumulator, encoded_value):
        decoded_value = tfc.run_length_gamma_decode(code=encoded_value,
                                                    shape=value_type.shape)
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

    @tff.federated_computation()
    def init_fn():
      return tff.federated_value((), tff.SERVER)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      encoded_value = tff.federated_map(
          tff.tf_computation(lambda x: tfc.run_length_gamma_encode(data=x)),
          value)
      bitstring_lengths = tff.federated_map(
          tff.tf_computation(get_bitstring_length), encoded_value)
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
              collections.OrderedDict(avg_bitrate=avg_bitrate)))

    return tff.templates.AggregationProcess(init_fn, next_fn)
