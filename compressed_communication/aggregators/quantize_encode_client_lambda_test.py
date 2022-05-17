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

from compressed_communication.aggregators import quantize_encode_client_lambda


_step_size_options = [0.5, 1.0, 2.0]
_step_size_options_type = [tf.float32, tf.float32, tf.float32]
_step_size_votes_type = (tf.int32, (3,))
_test_integer_tensor_type = (tf.int32, (3,))
_test_float_struct_type = [(tf.float32, (2,)), (tf.float32, (3,))]
_test_float_tensor_type = (tf.float32, (3,))


class QuantizeEncodeClientLambdaComputationTest(tf.test.TestCase,
                                                parameterized.TestCase):

  @parameterized.named_parameters(
      ("float_tensor", _test_float_tensor_type))
  def test_quantize_encode_client_lambda_impl(self, value_type):
    factory = quantize_encode_client_lambda.QuantizeEncodeClientLambdaFactory(
        1.0, 1.0, _step_size_options)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, tff.templates.AggregationProcess)

    server_state_type = tff.type_at_server(collections.OrderedDict(
        step_size=tf.float32,
        inner_state=()))
    expected_initialize_type = tff.FunctionType(
        parameter=None, result=server_state_type)
    tff.test.assert_types_equivalent(process.initialize.type_signature,
                                     expected_initialize_type)

    expected_measurements_type = tff.type_at_server(
        collections.OrderedDict(
            step_size=tf.float32,
            step_size_options=_step_size_options_type,
            step_size_vote_counts=_step_size_votes_type))
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
  def test_quantize_encode_client_lambda_create_raises(self, value_type):
    factory = quantize_encode_client_lambda.QuantizeEncodeClientLambdaFactory(
        1.0, 1.0, _step_size_options)
    value_type = tff.to_type(value_type)
    self.assertRaises(ValueError, factory.create, value_type)


class QuantizeEncodeClientLambdaExecutionTest(tf.test.TestCase,
                                              parameterized.TestCase):

  @parameterized.named_parameters(
      ("float_tensor", _test_float_tensor_type))
  def test_quantize_encode_client_lambda_impl(self, value_type):
    factory = quantize_encode_client_lambda.QuantizeEncodeClientLambdaFactory(
        1.0, 1.0, _step_size_options)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [tf.ones(value_type.shape, value_type.dtype)
                     for _ in range(2)]
    expected_result = tf.ones(value_type.shape, value_type.dtype) * 2

    # step_size=0.5 --> quantized=[2., 2., 2.]
    expected_distortion_twos = 0.
    bitstring_length_twos = tf.math.ceil(
        tf.size(expected_result, out_type=tf.float32) * 5.0 / 8.0) * 8.0
    expected_avg_bitrate_twos = bitstring_length_twos / tf.size(
        expected_result, out_type=tf.float32)

    # step_size=1. --> quantized=[1., 1., 1.]
    expected_distortion_ones = 0.
    bitstring_length_ones = tf.math.ceil(
        tf.size(expected_result, out_type=tf.float32) * 3.0 / 8.0) * 8.0
    expected_avg_bitrate_ones = bitstring_length_ones / tf.size(
        expected_result, out_type=tf.float32)

    # step_size=2. --> quantized=[0., 0., 0.]
    expected_distortion_zeros = 1.
    bitstring_length_zeros = tf.math.ceil(
        (1 + tf.size(expected_result, out_type=tf.float32)) * 1.0 / 8.0) * 8.0
    expected_avg_bitrate_zeros = bitstring_length_zeros / tf.size(
        expected_result, out_type=tf.float32)

    expected_distortions = [expected_distortion_twos,
                            expected_distortion_ones,
                            expected_distortion_zeros]
    expected_rates = [expected_avg_bitrate_twos,
                      expected_avg_bitrate_ones,
                      expected_avg_bitrate_zeros]
    expected_losses = [d + r for (d, r) in zip(
        expected_distortions, expected_rates)]

    expected_votes = [
        2 * int(x == min(expected_losses)) for x in expected_losses
    ]
    expected_next_step_size = _step_size_options[expected_votes.index(
        max(expected_votes))]

    expected_measurements = collections.OrderedDict(
        step_size=1.0,
        step_size_options=_step_size_options,
        step_size_vote_counts=expected_votes)

    output = process.next(state, client_values)
    self.assertAllClose(output.result, expected_result)
    self.assertAllClose(output.measurements, expected_measurements)
    self.assertAllClose(output.state["step_size"], expected_next_step_size)

  @parameterized.named_parameters(
      ("float_tensor", (tf.float32, (1000,))))
  def test_quantize_encode_client_lambda_votes_across_rounds(self, value_type):
    factory = quantize_encode_client_lambda.QuantizeEncodeClientLambdaFactory(
        0.001, 0.5, [0.01, 0.1, 0.5, 1.0, 2.0])
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    step_sizes = []
    votes = [0, 0, 0, 0, 0]
    for i in range(20):
      magnitude = 1.0 / float(i + 1)
      client_values = [
          tf.random.uniform((1000,), minval=-magnitude, maxval=magnitude,
                            dtype=tf.float32, seed=i*j) for j in range(20)]
      output = process.next(state, client_values)
      state = output.state
      step_sizes.append(output.measurements["step_size"])
      votes = [x + y for (x, y) in zip(
          votes, output.measurements["step_size_vote_counts"])]

    self.assertGreater(len(set(step_sizes)), 1)
    self.assertNotEqual(sum(votes), max(votes))

if __name__ == "__main__":
  tf.test.main()
