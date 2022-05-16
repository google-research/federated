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

from compressed_communication.broadcasters import histogram_model


_mn = 0.0
_mx = 10.0
_nbins = 4

_test_weights_type_float_tensor = (tf.float32, (3,))
_test_weights_type_struct_float_tensor = (
    (tf.float32, (3,)), (tf.float32, (2,)))

_test_measurements_type = (tf.int32, (_nbins,))


def _histogram_model():
  return histogram_model.HistogramModelBroadcastProcess(
      _test_weights_type_struct_float_tensor, _mn, _mx, _nbins)


class HistogramModelComputationTest(tff.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('float_tensor', _test_weights_type_float_tensor,
       _test_measurements_type),
      ('struct_float_tensor', _test_weights_type_struct_float_tensor,
       _test_measurements_type))
  def test_historgram_type_properties(self, weights_type,
                                      expected_measurements_type):
    broadcast_process = histogram_model.HistogramModelBroadcastProcess(
        weights_type, _mn, _mx, _nbins)
    self.assertIsInstance(broadcast_process, tff.templates.MeasuredProcess)

    server_state_type = tff.type_at_server(())
    expected_initialize_type = tff.FunctionType(
        parameter=None, result=server_state_type)
    tff.test.assert_types_equivalent(
        broadcast_process.initialize.type_signature, expected_initialize_type)

    expected_measurements_type = tff.to_type(expected_measurements_type)
    expected_next_type = tff.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type, weights=tff.type_at_server(weights_type)),
        result=tff.templates.MeasuredProcessOutput(
            state=server_state_type,
            result=tff.type_at_clients(weights_type, all_equal=True),
            measurements=tff.type_at_server(expected_measurements_type)))
    tff.test.assert_types_equivalent(broadcast_process.next.type_signature,
                                     expected_next_type)


class HistogramModelExecutionTest(tff.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('float_tensor', _test_weights_type_float_tensor,
       [1.0, 2.0, 3.0], [2, 1, 0, 0]),
      ('struct_float_tensor', _test_weights_type_struct_float_tensor,
       [[1.0, 2.0, 3.0], [1.0, 1.0]], [4, 1, 0, 0]))
  def test_histogram_impl(self, weights_type, weights, expected_histogram):
    weights_type = tff.to_type(weights_type)
    histogram_broadcaster = histogram_model.HistogramModelBroadcastProcess(
        weights_type, _mn, _mx, _nbins)
    state = histogram_broadcaster.initialize()
    histogram = histogram_broadcaster.next(state, weights).measurements
    self.assertAllEqual(histogram, expected_histogram)
    result = histogram_broadcaster.next(state, weights).result
    self.assertAllClose(result, weights)


if __name__ == '__main__':
  tff.backends.native.set_local_python_execution_context(10)
  tff.test.main()
