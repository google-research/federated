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

from compressed_communication.aggregators import quantize


_test_integer_tensor_type = (tf.int32, (3,))
_test_float_tensor_type = (tf.float32, (3,))
_test_float_tensor_quantized_type = (tf.int32, (3,))
_test_float_struct_type = [(tf.float32, (2,)), (tf.float32, (3,))]
_test_float_struct_quantized_type = [(tf.int32, (2,)), (tf.int32, (3,))]

_measurement_aggregator = tff.aggregators.add_measurements(
    tff.aggregators.SumFactory(), client_measurement_fn=tff.federated_sum)


class QuantizeComputationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("float_tensor_uniform", _test_float_tensor_type,
       _test_float_tensor_quantized_type, "uniform"),
      ("float_tensor_stochastic", _test_float_tensor_type,
       _test_float_tensor_quantized_type, "stochastic"))
  def test_quantize_properties(self, value_type, quantize_type,
                               rounding_type):
    factory = quantize.QuantizeFactory(0.1, _measurement_aggregator,
                                       rounding_type=rounding_type)
    value_type = tff.to_type(value_type)
    quantize_type = tff.to_type(quantize_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, tff.templates.AggregationProcess)

    server_state_type = tff.type_at_server(())
    expected_initialize_type = tff.FunctionType(
        parameter=None, result=server_state_type)
    tff.test.assert_types_equivalent(process.initialize.type_signature,
                                     expected_initialize_type)

    expected_measurements_type = tff.StructType([("avg_distortion", tf.float32),
                                                 ("inner_agg", quantize_type)])
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

  @parameterized.named_parameters(("integer_tensor", _test_integer_tensor_type),
                                  ("float_struct", _test_float_struct_type))
  def test_quantize_create_raises(self, value_type):
    factory = quantize.QuantizeFactory(0.1, _measurement_aggregator)
    value_type = tff.to_type(value_type)
    self.assertRaises(ValueError, factory.create, value_type)


class QuantizeExecutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("float_tensor_uniform", _test_float_tensor_type,
       _test_float_tensor_quantized_type, "uniform"),
      ("float_tensor_stochastic", _test_float_tensor_type,
       _test_float_tensor_quantized_type, "stochastic"))
  def test_quantize_impl(self, value_type, quantize_type, rounding_type):
    factory = quantize.QuantizeFactory(0.1, _measurement_aggregator,
                                       rounding_type=rounding_type)
    value_type = tff.to_type(value_type)
    quantize_type = tff.to_type(quantize_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [tf.ones(value_type.shape, value_type.dtype)
                     for _ in range(2)]
    expected_result = tf.ones(value_type.shape, value_type.dtype) * 2
    expected_quantized_result = tf.ones(quantize_type.shape,
                                        quantize_type.dtype) * 20
    expected_measurements = collections.OrderedDict(
        avg_distortion=0., inner_agg=expected_quantized_result)
    measurements = process.next(state, client_values).measurements
    self.assertAllClose(measurements, expected_measurements)
    result = process.next(state, client_values).result
    self.assertAllClose(result, expected_result)


if __name__ == "__main__":
  tf.test.main()
