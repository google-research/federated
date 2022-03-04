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
import math

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication.aggregators import zero_run_length


_test_value_type_integer_tensor = (tf.int32, (3,))
_test_value_type_float_tensor = (tf.float32, (3,))
_test_value_type_list_integer_tensors = [(tf.int32, (2,)),
                                         (tf.int32, (3,))]

_test_leading_zeros_tensor = [0, 0, 1]  # zero run lengths: [3]
_test_leading_zeros_bincount = [0, 0, 0, 1]
_test_leading_zeros_entropy = 0.
_test_leading_zeros_ce_gamma = 3.
_test_leading_zeros_ce_delta = 4.

_test_trailing_zeros_tensor = [1, 2, 0]   # zero run lengths: [1, 1, 2]
_test_trailing_zeros_bincount = [0, 2, 1]
_test_trailing_zeros_entropy = math.log(1.5, 2) / 1.5 + math.log(3., 2) / 3.
_test_trailing_zeros_ce_gamma = 5./3.
_test_trailing_zeros_ce_delta = 6./3.

_test_middle_zeros_tensor = [1, 0, 2]   # zero run lengths: [1, 2]
_test_middle_zeros_bincount = [0, 1, 1]
_test_middle_zeros_entropy = 1.
_test_middle_zeros_ce_gamma = 4./2.
_test_middle_zeros_ce_delta = 5./2.

_test_mixed_zeros_clients = [_test_leading_zeros_tensor,
                             _test_middle_zeros_tensor,
                             _test_trailing_zeros_tensor]
_test_mixed_zeros_summed_bincount = [0, 3, 2, 1]
_test_mixed_zeros_entropy = math.log(6., 2) / 6. + math.log(
    3., 2) / 3. + math.log(2., 2) / 2.
_test_mixed_zeros_ce_gamma = (3. + (1. + 1. + 3.) / 3. + (1. + 3.) / 2.) / 3.
_test_mixed_zeros_ce_delta = (4. + (1. + 1. + 4.) / 3. + (1. + 4.) / 2.) / 3.


class ZeroRunLengthComputationTest(tff.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('integer_tensor', _test_value_type_integer_tensor))
  def test_zero_run_length_properties(self, value_type):
    factory = zero_run_length.ZeroRunLengthSumFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, tff.templates.AggregationProcess)

    server_state_type = tff.type_at_server(())
    expected_initialize_type = tff.FunctionType(
        parameter=None, result=server_state_type)
    self.assert_types_equivalent(process.initialize.type_signature,
                                 expected_initialize_type)

    expected_measurements_type = tff.StructType([
        ('zero_run_lengths', tff.TensorType(tf.int32, shape=(None,))),
        ('entropy', tf.float64),
        ('cross_entropy_gamma', tf.float32),
        ('cross_entropy_delta', tf.float32)
    ])
    expected_measurements_type = tff.type_at_server(expected_measurements_type)
    expected_next_type = tff.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type, value=tff.type_at_clients(value_type)),
        result=tff.templates.MeasuredProcessOutput(
            state=server_state_type,
            result=tff.type_at_server(value_type),
            measurements=expected_measurements_type))
    self.assert_types_equivalent(process.next.type_signature,
                                 expected_next_type)

  @parameterized.named_parameters(
      ('float_tensor', _test_value_type_float_tensor),
      ('list_integer_tensors', _test_value_type_list_integer_tensors))
  def test_zero_run_length_create_raises(self, value_type):
    factory = zero_run_length.ZeroRunLengthSumFactory()
    value_type = tff.to_type(value_type)
    self.assertRaises(ValueError, factory.create, value_type)


class ZeroRunLengthExecutionTest(tff.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('leading_zeros', _test_value_type_integer_tensor,
       [_test_leading_zeros_tensor for _ in range(3)],
       [x * 3 for x in _test_leading_zeros_bincount],
       _test_leading_zeros_entropy, _test_leading_zeros_ce_gamma,
       _test_leading_zeros_ce_delta),
      ('trailing_zeros', _test_value_type_integer_tensor,
       [_test_trailing_zeros_tensor for _ in range(3)],
       [x * 3 for x in _test_trailing_zeros_bincount],
       _test_trailing_zeros_entropy, _test_trailing_zeros_ce_gamma,
       _test_trailing_zeros_ce_delta),
      ('middle_zeros', _test_value_type_integer_tensor,
       [_test_middle_zeros_tensor for _ in range(3)],
       [x * 3 for x in _test_middle_zeros_bincount],
       _test_middle_zeros_entropy, _test_middle_zeros_ce_gamma,
       _test_middle_zeros_ce_delta),
      ('mixed_zeros', _test_value_type_integer_tensor,
       _test_mixed_zeros_clients,
       _test_mixed_zeros_summed_bincount,
       _test_mixed_zeros_entropy, _test_mixed_zeros_ce_gamma,
       _test_mixed_zeros_ce_delta))
  def test_zero_run_length_impl(self, value_type, client_values,
                                expected_zero_run_length_summed_bincount,
                                expected_entropy, expected_ce_gamma,
                                expected_ce_delta):
    factory = zero_run_length.ZeroRunLengthSumFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    expected_result = tf.reduce_sum(client_values, axis=0)
    expected_measurements = collections.OrderedDict(
        zero_run_lengths=expected_zero_run_length_summed_bincount,
        entropy=expected_entropy,
        cross_entropy_gamma=expected_ce_gamma,
        cross_entropy_delta=expected_ce_delta)

    measurements = process.next(state, client_values).measurements
    self.assertAllClose(measurements, expected_measurements)
    result = process.next(state, client_values).result
    self.assertAllClose(result, expected_result)

if __name__ == '__main__':
  tff.test.main()
