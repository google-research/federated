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

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_federated as tff

from compressed_communication.aggregators import elias_gamma_encode


_test_value_type_integer_tensor_rank_1 = (tf.int32, (4,))
_test_value_type_integer_tensor_rank_2 = (tf.int32, (2, 4,))
_test_value_type_float_tensor = (tf.float32, (3,))
_test_value_type_list_integer_tensors = [(tf.int32, (2,)),
                                         (tf.int32, (3,))]

_test_client_values_integer_tensor_rank_1 = [[-5, 3, 0, 0], [-3, 1, 0, 0]]
_test_client_values_integer_tensor_rank_2 = [[[-5, 3, 0, 0], [-3, 1, 0, 0]],
                                             [[-5, 3, 0, 0], [-3, 1, 0, 0]]]

_test_avg_bitstring_len_integer_tensor_rank_1 = 16
_test_avg_bitstring_len_integer_tensor_rank_2 = 32


class EncodeComputationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('integer_tensor_rank_1', _test_value_type_integer_tensor_rank_1),
      ('integer_tensor_rank_2', _test_value_type_integer_tensor_rank_2))
  def test_encode_properties(self, value_type):
    factory = elias_gamma_encode.EliasGammaEncodedSumFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, tff.templates.AggregationProcess)

    server_state_type = tff.type_at_server(())
    expected_initialize_type = tff.FunctionType(
        parameter=None, result=server_state_type)
    tff.test.assert_types_equivalent(process.initialize.type_signature,
                                     expected_initialize_type)

    expected_measurements_type = tff.StructType([
        ('avg_bitrate', tf.float64)
    ])
    expected_measurements_type = tff.type_at_server(expected_measurements_type)
    expected_next_type = tff.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type, value=tff.type_at_clients(value_type)),
        result=tff.templates.MeasuredProcessOutput(
            state=server_state_type,
            result=tff.type_at_server(value_type),
            measurements=expected_measurements_type))
    tff.test.assert_types_equivalent(process.next.type_signature,
                                     expected_next_type)

  @parameterized.named_parameters(
      ('float_tensor', _test_value_type_float_tensor),
      ('list_integer_tensors', _test_value_type_list_integer_tensors))
  def test_encode_create_raises(self, value_type):
    factory = elias_gamma_encode.EliasGammaEncodedSumFactory()
    value_type = tff.to_type(value_type)
    self.assertRaises(ValueError, factory.create, value_type)


class EncodeExecutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('integer_tensor_rank_1',
       _test_value_type_integer_tensor_rank_1,
       _test_client_values_integer_tensor_rank_1,
       _test_avg_bitstring_len_integer_tensor_rank_1),
      ('integer_tensor_rank_2',
       _test_value_type_integer_tensor_rank_2,
       _test_client_values_integer_tensor_rank_2,
       _test_avg_bitstring_len_integer_tensor_rank_2))
  def test_encode_impl(self, value_type, client_values, avg_bitstring_length):
    factory = elias_gamma_encode.EliasGammaEncodedSumFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    expected_result = tf.reduce_sum(client_values, axis=0)
    expected_measurements = collections.OrderedDict(
        avg_bitrate=avg_bitstring_length/tf.cast(tf.size(expected_result),
                                                 tf.float64))

    measurements = process.next(state, client_values).measurements
    self.assertAllClose(measurements, expected_measurements)
    result = process.next(state, client_values).result
    self.assertAllClose(result, expected_result)

  @parameterized.named_parameters(
      ('integer_tensor_rank_1', _test_client_values_integer_tensor_rank_1,
       _test_avg_bitstring_len_integer_tensor_rank_1),
      ('integer_tensor_rank_2', _test_client_values_integer_tensor_rank_2,
       _test_avg_bitstring_len_integer_tensor_rank_2))
  def test_bitstring_impl(self, client_values, avg_bitstring_length):
    bitstrings = [tfc.run_length_gamma_encode(x) for x in client_values]
    bitstring_lengths = [elias_gamma_encode.get_bitstring_length(x)
                         for x in bitstrings]
    self.assertEqual(np.mean(bitstring_lengths), avg_bitstring_length)


if __name__ == '__main__':
  tf.test.main()
