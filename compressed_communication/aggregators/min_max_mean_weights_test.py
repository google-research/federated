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

from compressed_communication.aggregators import min_max_mean_weights


_test_value_type_struct_float_tensor = (tf.float32, (3,))
_test_value_type_struct_list_float_tensors = [(tf.float32, (2,)),
                                              (tf.float32, (3,))]

_test_measurements_type_struct_float_tensor = tf.float32
_test_measurements_type_struct_list_float_tensors = [tf.float32, tf.float32]


def _min_max_mean_weights():
  return min_max_mean_weights.MinMaxMeanWeightsFactory()


class MinMaxMeanWeightsComputationTest(tf.test.TestCase,
                                       parameterized.TestCase):

  @parameterized.named_parameters(
      ('struct_float_tensor', _test_value_type_struct_float_tensor,
       _test_measurements_type_struct_float_tensor),
      ('struct_list_float_tensors', _test_value_type_struct_list_float_tensors,
       _test_measurements_type_struct_list_float_tensors))
  def test_min_max_mean_type_properties(self, value_type,
                                        expected_measurements_type):
    factory = _min_max_mean_weights()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, tff.templates.AggregationProcess)

    server_state_type = tff.type_at_server(())
    expected_initialize_type = tff.FunctionType(
        parameter=None, result=server_state_type)
    tff.test.assert_types_equivalent(process.initialize.type_signature,
                                     expected_initialize_type)

    expected_measurements_type = tff.to_type(expected_measurements_type)
    expected_next_type = tff.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type, value=tff.type_at_clients(value_type)),
        result=tff.templates.MeasuredProcessOutput(
            state=server_state_type,
            result=tff.type_at_server(value_type),
            measurements=tff.type_at_server(collections.OrderedDict(
                min=expected_measurements_type,
                max=expected_measurements_type,
                mean=expected_measurements_type))))
    tff.test.assert_types_equivalent(process.next.type_signature,
                                     expected_next_type)


class MinMaxMeanWeightsExecutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('struct_list_float_scalars', _test_value_type_struct_float_tensor,
       [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], collections.OrderedDict(
           [('min', 1.0), ('max', 3.0), ('mean', 2.0)]), [2.0, 4.0, 6.0]),
      ('struct_list_float_tensors', _test_value_type_struct_list_float_tensors,
       [[[1.0, 1.0], [1.0, 2.0, 3.0]], [[9.0, 9.0], [1.0, 2.0, 3.0]]],
       collections.OrderedDict([('min', [1.0, 1.0]), ('max', [9.0, 3.0]),
                                ('mean', [5.0, 2.0])]),
       [[10.0, 10.0], [2.0, 4.0, 6.0]]))
  def test_min_max_mean_impl(self, value_type, client_values,
                             expected_min_max_mean, expected_result):
    value_type = tff.to_type(value_type)
    min_max_mean_agg = _min_max_mean_weights().create(value_type)
    state = min_max_mean_agg.initialize()
    min_max_mean = min_max_mean_agg.next(state, client_values).measurements
    self.assertAllEqual(min_max_mean, expected_min_max_mean)
    result = min_max_mean_agg.next(state, client_values).result
    self.assertAllClose(result, expected_result)


if __name__ == '__main__':
  tf.test.main()
