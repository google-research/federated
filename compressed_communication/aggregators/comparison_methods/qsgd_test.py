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

from compressed_communication.aggregators.comparison_methods import qsgd


_test_integer_tensor_type = (tf.int32, (3,))
_test_float_struct_type = [(tf.float32, (2,)), (tf.float32, (3,))]
_test_one_float_tensor_type = (tf.float32, (1,))
_test_three_float_tensor_type = (tf.float32, (3,))


class QSGDComputationTest(tff.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("float_tensor", _test_three_float_tensor_type))
  def test_qsgd_properties(self, value_type):
    factory = qsgd.QSGDFactory(1.0)
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
            avg_bitrate=tf.float64,
            avg_distortion=tf.float32,
            avg_sparsity=tf.float32))
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
  def test_qsgd_create_raises(self, value_type):
    factory = qsgd.QSGDFactory(1.0)
    value_type = tff.to_type(value_type)
    self.assertRaises(ValueError, factory.create, value_type)


class QSGDExecutionTest(tff.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("float_tensor", _test_one_float_tensor_type))
  def test_correctness_one_element_one_client(self, value_type):
    factory = qsgd.QSGDFactory(1.0)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [tf.ones(value_type.shape, value_type.dtype)]
    expected_result = tf.ones(value_type.shape, value_type.dtype)
    bitstring_length = tf.math.ceil(
        tf.size(expected_result, out_type=tf.float32) * 3.0 / 8.0) * 8.0 + 32.
    expected_avg_bitrate = bitstring_length / tf.size(expected_result,
                                                      out_type=tf.float32)

    expected_measurements = collections.OrderedDict(
        avg_bitrate=tf.cast(expected_avg_bitrate, tf.float64),
        avg_distortion=0.0,
        avg_sparsity=0.0)

    output = process.next(state, client_values)
    self.assertAllClose(output.result, expected_result)
    self.assertAllClose(output.measurements, expected_measurements)

  @parameterized.named_parameters(
      ("float_tensor", _test_one_float_tensor_type))
  def test_correctness_one_element_different_clients(self, value_type):
    factory = qsgd.QSGDFactory(2.0)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [[1.0], [2.0]]
    expected_result = [3.0]
    bitstring_length = tf.math.ceil(
        tf.size(expected_result, out_type=tf.float32) * 5.0 / 8.0) * 8.0 + 32.
    expected_avg_bitrate = bitstring_length / tf.size(expected_result,
                                                      out_type=tf.float32)

    expected_measurements = collections.OrderedDict(
        avg_bitrate=tf.cast(expected_avg_bitrate, tf.float64),
        avg_distortion=0.0,
        avg_sparsity=0.0)

    output = process.next(state, client_values)
    self.assertAllClose(output.result, expected_result)
    self.assertAllClose(output.measurements, expected_measurements)

  @parameterized.named_parameters(
      ("float_tensor", _test_three_float_tensor_type))
  def test_correctness_multiple_elements_identical_clients(self, value_type):
    factory = qsgd.QSGDFactory(7.0)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [[2.0, 3.0, 6.0] for _ in range(2)]  # norm=7.0
    expected_result = [4.0, 6.0, 12.0]
    bitstring_length = tf.math.ceil((5.0 + 5.0 + 7.0) / 8.0) * 8.0 + 32.
    expected_avg_bitrate = bitstring_length / tf.size(expected_result,
                                                      out_type=tf.float32)

    expected_measurements = collections.OrderedDict(
        avg_bitrate=tf.cast(expected_avg_bitrate, tf.float64),
        avg_distortion=0.0,
        avg_sparsity=0.0)

    output = process.next(state, client_values)
    self.assertAllClose(output.result, expected_result)
    self.assertAllClose(output.measurements, expected_measurements)

  @parameterized.named_parameters(
      ("float_tensor", _test_three_float_tensor_type))
  def test_correctness_multiple_elements_different_clients(self, value_type):
    factory = qsgd.QSGDFactory(7.0)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [[2.0, 3.0, 6.0],  # norm=7.0, codelength=17, distortion=0
                     [1.0, 1.0, 1.0]]  # norm~1.73, codelength=21
    min_value = (tf.sqrt(3.) / 7.) * 4.
    max_value = (tf.sqrt(3.) / 7.) * 5.
    min_diff = 1. - min_value
    max_diff = max_value - 1.
    expected_max_distortion = tf.maximum(min_diff, max_diff)**2
    expected_max_avg_distortion = expected_max_distortion / 2.
    expected_min_result = [2.0 + min_value, 3.0 + min_value, 6.0 + min_value]
    expected_max_result = [2.0 + max_value, 4.0 + max_value, 6.0 + max_value]
    bitstring_length = tf.math.ceil(((17.0 + 21.0) / 2.0) / 8.0) * 8.0 + 32.
    expected_avg_bitrate = bitstring_length / 3.0
    expected_avg_bitrate = tf.cast(expected_avg_bitrate, tf.float64)

    output = process.next(state, client_values)
    self.assertAllGreaterEqual(output.result - expected_min_result, 0.)
    self.assertAllLessEqual(output.result - expected_max_result, 0.)
    self.assertAllClose(output.measurements["avg_bitrate"],
                        expected_avg_bitrate)
    self.assertLessEqual(output.measurements["avg_distortion"],
                         expected_max_avg_distortion)


if __name__ == "__main__":
  tff.test.main()
