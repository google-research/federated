# Copyright 2022, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ddp_compression."""
from typing import List
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from private_linear_compression import ddp_compression

_FloatMatrixType = tff.TensorType(tf.float32, [20, 30])
_TestStructType = [(tf.float32, (15,))]


def _make_test_struct_value(x: float) -> List[tf.Tensor]:
  return [tf.constant(x, dtype=tf.float32, shape=(15,))]


class DDPCompressionTest(tf.test.TestCase, parameterized.TestCase):

  def test_ddp_compression_aggregator_uses_only_secure_aggregation(self):
    aggregator = ddp_compression.compressed_ddp_factory(
        noise_multiplier=1e-2,
        expected_clients_per_round=10,
        bits=12,
        compression_rate=5.0).create(_FloatMatrixType)

    try:
      tff.test.assert_not_contains_unsecure_aggregation(aggregator.next)
    except AssertionError:
      self.fail('Secure aggregator contains non-secure aggregation.')

  @parameterized.named_parameters(('zeroing_float_matrix', True),
                                  ('no_zeroing_float_matrix', False))
  def test_ddp_compression_aggregator_unweighted(self, zeroing):
    aggregator = ddp_compression.compressed_ddp_factory(
        noise_multiplier=1e-2,
        expected_clients_per_round=10,
        bits=12,
        compression_rate=5.0,
        zeroing=zeroing)

    self.assertIsInstance(aggregator,
                          tff.aggregators.UnweightedAggregationFactory)
    process = aggregator.create(_FloatMatrixType)
    self.assertIsInstance(process, tff.templates.AggregationProcess)
    self.assertFalse(process.is_weighted)

  def test_ddp_compression_aggregator_zeroing_added(self):
    aggregator = ddp_compression.compressed_ddp_factory(
        noise_multiplier=0.0,
        expected_clients_per_round=3,
        bits=12,
        compression_rate=1.0,
        zeroing=True)
    value_type = tff.to_type(_TestStructType)
    process = aggregator.create(value_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [0.0, 0.0, 100.0]]
    output = process.next(state, client_data)

    self.assertAllClose(_make_test_struct_value(0.0), output.result)
    self.assertEqual(1, output.measurements['zeroed_count'])

  def test_ddp_compression_aggregator_no_zeroing(self):
    tf.random.set_seed(2)
    aggregator = ddp_compression.compressed_ddp_factory(
        noise_multiplier=0.0,
        expected_clients_per_round=3,
        bits=12,
        compression_rate=1.0,
        num_repeats=1,
        zeroing=False)
    value_type = tff.to_type(_TestStructType)
    process = aggregator.create(value_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [0.0, 0.0, 100.0]]
    output = process.next(state, client_data)

    self.assertNotAllEqual(_make_test_struct_value(0.0), output.result)

  @parameterized.named_parameters([('zeroing_compression', True, 1.0),
                                   ('no_zeroing_compression', False, 1.0),
                                   ('zeroing_no_compression', True, 0.0),
                                   ('no_zeroing_no_compression', False, 0.0)])
  def test_ddp_compression_aggregator_noise_added(self, zeroing,
                                                  compression_rate):
    # Instead of enforcing that DDP is used, we enforce that noise was added.

    # Ensure random seeds are fixed regardless of time of execution and that
    # tensorflow_privacy stateful random operations are seeded.
    tf.random.set_seed(2)
    with mock.patch.object(tf, 'timestamp', return_value=0):
      aggregator = ddp_compression.compressed_ddp_factory(
          noise_multiplier=10.0,
          expected_clients_per_round=int(10.0 / 0.05),
          bits=16,
          compression_rate=compression_rate,
          zeroing=zeroing)
      value_type = tff.to_type(_TestStructType)
      process = aggregator.create(value_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [0.01, 0.5, 1.0]]
    output = process.next(state, client_data)

    uniques, _ = tf.unique(output.result[0])
    self.assertLen(uniques, len(output.result[0]))

    signs, _ = tf.unique(tf.sign(output.result[0]))
    self.assertLen(signs, 2)

  def test_ddp_compression_aggregator_no_compression_added(self):
    aggregator = ddp_compression.compressed_ddp_factory(
        noise_multiplier=0.0,
        expected_clients_per_round=3,
        bits=22,
        compression_rate=0.0,
        zeroing=False)
    value_type = tff.to_type(_TestStructType)
    process = aggregator.create(value_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [0.005, 0.01, 0.02]]
    output = process.next(state, client_data)

    expected_aggregator = tff.aggregators.UnweightedMeanFactory()
    expected_process = expected_aggregator.create(value_type)
    expected_result = expected_process.next(expected_process.initialize(),
                                            client_data)

    self.assertAllClose(output.result[0], expected_result.result[0])

  @parameterized.named_parameters([
      ('low_compression', 1.),
      ('high_compression', 5.),
  ])
  def test_ddp_compression_aggregator_compression_added(self, compression_rate):
    aggregator = ddp_compression.compressed_ddp_factory(
        noise_multiplier=0.0,
        expected_clients_per_round=3,
        bits=18,
        compression_rate=compression_rate,
        zeroing=False,
        num_repeats=3,
    )
    process = aggregator.create(tff.to_type(_TestStructType))
    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [0.005, 0.01, 0.02]]
    output = process.next(state, client_data)

    expected_aggregator = tff.aggregators.UnweightedMeanFactory()
    expected_process = expected_aggregator.create(tff.to_type(_TestStructType))
    expected_result = expected_process.next(expected_process.initialize(),
                                            client_data)

    self.assertNotAllClose(output.result[0], expected_result.result[0])


class CentralDPCompressionTest(tf.test.TestCase, parameterized.TestCase):

  def test_central_dp_compression_aggregator_uses_unsecure_aggregation(self):
    aggregator = ddp_compression.compressed_central_dp_factory(
        noise_multiplier=1e-2,
        expected_clients_per_round=10,
        compression_rate=5.0).create(_FloatMatrixType)

    try:
      tff.test.assert_contains_unsecure_aggregation(aggregator.next)
    except AssertionError:
      self.fail('Unsecure aggregator contains secure aggregation.')

  @parameterized.named_parameters(('zeroing_float_matrix', True),
                                  ('no_zeroing_float_matrix', False))
  def test_central_dp_compression_aggregator_unweighted(self, zeroing):
    aggregator = ddp_compression.compressed_central_dp_factory(
        noise_multiplier=1e-2,
        expected_clients_per_round=10,
        compression_rate=5.0,
        zeroing=zeroing)

    self.assertIsInstance(aggregator,
                          tff.aggregators.UnweightedAggregationFactory)
    process = aggregator.create(_FloatMatrixType)
    self.assertIsInstance(process, tff.templates.AggregationProcess)
    self.assertFalse(process.is_weighted)

  def test_central_dp_compression_aggregator_zeroing_added(self):
    aggregator = ddp_compression.compressed_ddp_factory(
        noise_multiplier=0.0,
        expected_clients_per_round=3,
        bits=12,
        compression_rate=1.0,
        zeroing=True)
    value_type = tff.to_type(_TestStructType)
    process = aggregator.create(value_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [0.0, 0.0, 100.0]]
    output = process.next(state, client_data)

    self.assertAllClose(_make_test_struct_value(0.0), output.result)
    self.assertEqual(1, output.measurements['zeroed_count'])

  def test_central_dp_compression_aggregator_no_zeroing(self):
    tf.random.set_seed(2)
    aggregator = ddp_compression.compressed_central_dp_factory(
        noise_multiplier=0.0,
        expected_clients_per_round=3,
        compression_rate=1.0,
        num_repeats=1,
        zeroing=False)
    value_type = tff.to_type(_TestStructType)
    process = aggregator.create(value_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [0.0, 0.0, 100.0]]
    output = process.next(state, client_data)

    self.assertNotAllEqual(_make_test_struct_value(0.0), output.result)

  @parameterized.named_parameters([('zeroing_compression', True, 1.0),
                                   ('no_zeroing_compression', False, 1.0),
                                   ('zeroing_no_compression', True, 0.0),
                                   ('no_zeroing_no_compression', False, 0.0)])
  def test_central_dp_compression_aggregator_noise_added(
      self, zeroing, compression_rate):
    tf.random.set_seed(2)
    with mock.patch.object(tf, 'timestamp', return_value=0):
      aggregator = ddp_compression.compressed_central_dp_factory(
          noise_multiplier=10.0,
          expected_clients_per_round=int(10.0 / 0.05),
          compression_rate=compression_rate,
          zeroing=zeroing)
      value_type = tff.to_type(_TestStructType)
      process = aggregator.create(value_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [0.01, 0.5, 1.0]]
    output = process.next(state, client_data)

    uniques, _ = tf.unique(output.result[0])
    self.assertLen(uniques, len(output.result[0]))

    signs, _ = tf.unique(tf.sign(output.result[0]))
    self.assertLen(signs, 2)

  def test_central_dp_compression_aggregator_no_compression_added(self):
    aggregator = ddp_compression.compressed_ddp_factory(
        noise_multiplier=0.0,
        expected_clients_per_round=3,
        compression_rate=0.0,
        zeroing=False)
    value_type = tff.to_type(_TestStructType)
    process = aggregator.create(value_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [0.005, 0.01, 0.02]]
    output = process.next(state, client_data)

    expected_aggregator = tff.aggregators.UnweightedMeanFactory()
    expected_process = expected_aggregator.create(value_type)
    expected_result = expected_process.next(expected_process.initialize(),
                                            client_data)

    self.assertAllClose(output.result[0], expected_result.result[0])

  @parameterized.named_parameters([
      ('low_compression', 1.),
      ('high_compression', 5.),
  ])
  def test_ddp_compression_aggregator_compression_added(self, compression_rate):
    aggregator = ddp_compression.compressed_central_dp_factory(
        noise_multiplier=0.0,
        expected_clients_per_round=3,
        compression_rate=compression_rate,
        zeroing=False,
        num_repeats=3,
    )
    process = aggregator.create(tff.to_type(_TestStructType))
    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [0.005, 0.01, 0.02]]
    output = process.next(state, client_data)

    expected_aggregator = tff.aggregators.UnweightedMeanFactory()
    expected_process = expected_aggregator.create(tff.to_type(_TestStructType))
    expected_result = expected_process.next(expected_process.initialize(),
                                            client_data)

    self.assertNotAllClose(output.result[0], expected_result.result[0])


if __name__ == '__main__':
  # Required because the Secure Aggregation intrinsics are not implemented
  # in the default resolving executor.
  tff.backends.test.set_test_python_execution_context()
  tf.test.main()
