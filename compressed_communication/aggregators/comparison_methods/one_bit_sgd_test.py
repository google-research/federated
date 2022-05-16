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
import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication.aggregators.comparison_methods import one_bit_sgd


_test_integer_tensor_type = (tf.int32, (3,))
_test_float_struct_type = [(tf.float32, (2,)), (tf.float32, (3,))]
_test_float_tensor_type = (tf.float32, (3,))


class OneBitSGDComputationTest(tff.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("float_tensor", _test_float_tensor_type))
  def test_one_bit_sgd_properties(self, value_type):
    factory = one_bit_sgd.OneBitSGDFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, tff.templates.AggregationProcess)

    server_state_type = tff.type_at_server(())
    expected_initialize_type = tff.FunctionType(
        parameter=None, result=server_state_type)
    tff.test.assert_types_equivalent(process.initialize.type_signature,
                                     expected_initialize_type)

    expected_measurements_type = tff.type_at_server(
        collections.OrderedDict(
            avg_bitrate=tf.float32,
            avg_distortion=tf.float32))
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
      ("integer_tensor", _test_integer_tensor_type),
      ("float_struct", _test_float_struct_type))
  def test_one_bit_sgd_create_raises(self, value_type):
    factory = one_bit_sgd.OneBitSGDFactory()
    value_type = tff.to_type(value_type)
    self.assertRaises(ValueError, factory.create, value_type)


class OneBitSGDExecutionTest(tff.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("float_tensor", _test_float_tensor_type))
  def test_correctness_positive(self, value_type):
    factory = one_bit_sgd.OneBitSGDFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [tf.ones(value_type.shape, value_type.dtype)
                     for _ in range(2)]
    expected_result = tf.ones(value_type.shape, value_type.dtype) * 2
    bitstring_length = tf.size(expected_result, out_type=tf.float32) + 64.
    expected_avg_bitrate = bitstring_length / tf.size(expected_result,
                                                      out_type=tf.float32)

    expected_measurements = collections.OrderedDict(
        avg_bitrate=expected_avg_bitrate,
        avg_distortion=0.0)

    output = process.next(state, client_values)
    self.assertAllClose(output.result, expected_result)
    self.assertAllClose(output.measurements, expected_measurements)

  @parameterized.named_parameters(
      ("float_tensor", _test_float_tensor_type))
  def test_correctness_negative(self, value_type):
    factory = one_bit_sgd.OneBitSGDFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [-1.0 * tf.ones(value_type.shape, value_type.dtype)
                     for _ in range(2)]
    expected_result = tf.ones(value_type.shape, value_type.dtype) * -2
    bitstring_length = tf.size(expected_result, out_type=tf.float32) + 64.
    expected_avg_bitrate = bitstring_length / tf.size(expected_result,
                                                      out_type=tf.float32)

    expected_measurements = collections.OrderedDict(
        avg_bitrate=expected_avg_bitrate,
        avg_distortion=0.0)

    output = process.next(state, client_values)
    self.assertAllClose(output.result, expected_result)
    self.assertAllClose(output.measurements, expected_measurements)

  @parameterized.named_parameters(
      ("float_tensor", _test_float_tensor_type))
  def test_correctness_positive_negative(self, value_type):
    factory = one_bit_sgd.OneBitSGDFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [[0.0, 2.0, -1.0] for _ in range(2)]
    expected_result = [2.0, 2.0, -2.0]
    bitstring_length = tf.size(expected_result, out_type=tf.float32) + 64.
    expected_avg_bitrate = bitstring_length / tf.size(expected_result,
                                                      out_type=tf.float32)

    expected_measurements = collections.OrderedDict(
        avg_bitrate=expected_avg_bitrate,
        avg_distortion=2./3.)

    output = process.next(state, client_values)
    self.assertAllClose(output.result, expected_result)
    self.assertAllClose(output.measurements, expected_measurements)

  @parameterized.named_parameters(
      ("float_tensor", _test_float_tensor_type))
  def test_correctness_nonzero_threshold(self, value_type):
    factory = one_bit_sgd.OneBitSGDFactory(2.)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [[-1.0, 1.0, 2.0] for _ in range(2)]
    expected_result = [0.0, 0.0, 4.0]
    bitstring_length = tf.size(expected_result, out_type=tf.float32) + 64.
    expected_avg_bitrate = bitstring_length / tf.size(expected_result,
                                                      out_type=tf.float32)

    expected_measurements = collections.OrderedDict(
        avg_bitrate=expected_avg_bitrate,
        avg_distortion=2./3.)

    output = process.next(state, client_values)
    self.assertAllClose(output.result, expected_result)
    self.assertAllClose(output.measurements, expected_measurements)

  @parameterized.named_parameters(
      ("float_tensor", _test_float_tensor_type))
  def test_correctness_one_client(self, value_type):
    factory = one_bit_sgd.OneBitSGDFactory(2.)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [[-1.0, 1.0, 2.0]]
    expected_result = [0.0, 0.0, 2.0]
    bitstring_length = tf.size(expected_result, out_type=tf.float32) + 64.
    expected_avg_bitrate = bitstring_length / tf.size(expected_result,
                                                      out_type=tf.float32)

    expected_measurements = collections.OrderedDict(
        avg_bitrate=expected_avg_bitrate,
        avg_distortion=2./3.)

    output = process.next(state, client_values)
    self.assertAllClose(output.result, expected_result)
    self.assertAllClose(output.measurements, expected_measurements)

  @parameterized.named_parameters(
      ("float_tensor", _test_float_tensor_type))
  def test_correctness_different_clients(self, value_type):
    factory = one_bit_sgd.OneBitSGDFactory(2.)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [[-1.0, 1.0, 2.0], [1.0, 1.0, 1.0]]
    expected_result = [1.0, 1.0, 3.0]
    bitstring_length = tf.size(expected_result, out_type=tf.float32) + 64.
    expected_avg_bitrate = bitstring_length / tf.size(expected_result,
                                                      out_type=tf.float32)

    expected_measurements = collections.OrderedDict(
        avg_bitrate=expected_avg_bitrate,
        avg_distortion=2./6.)

    output = process.next(state, client_values)
    self.assertAllClose(output.result, expected_result)
    self.assertAllClose(output.measurements, expected_measurements)


if __name__ == "__main__":
  tff.test.main()
