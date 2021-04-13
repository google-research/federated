# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for compression_query."""

from unittest import mock

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import test_utils

from distributed_discrete_gaussian import compression_query
from distributed_discrete_gaussian import distributed_discrete_gaussian_query

ddg_sum_query = distributed_discrete_gaussian_query.DistributedDiscreteGaussianSumQuery


class CompressionQueryTest(tf.test.TestCase, parameterized.TestCase):

  DEFAULT_L2_NORM_BOUND = 1000
  DEFAULT_QUANTIZATION_PARAMS = compression_query.ScaledQuantizationParams(
      stochastic=True,
      conditional=False,
      l2_norm_bound=DEFAULT_L2_NORM_BOUND,
      beta=0.88,
      quantize_scale=123)

  # Use a noiseless discrete sum query.
  DEFAULT_INNER_QUERY = ddg_sum_query(
      l2_norm_bound=DEFAULT_L2_NORM_BOUND, local_scale=0.0)

  @parameterized.product(dtype=[tf.int32, tf.float32])
  def test_inner_query_no_noise_dtypes(self, dtype):
    t1 = tf.constant([-2, 4], dtype=dtype)
    t2 = tf.constant([-4, 2], dtype=dtype)
    record = [t1, t2]
    sample = [record]

    # Use a noiseless discrete sum query.
    comp_query = compression_query.CompressionSumQuery(
        quantization_params=self.DEFAULT_QUANTIZATION_PARAMS,
        inner_query=self.DEFAULT_INNER_QUERY,
        record_template=record)

    query_result, _ = test_utils.run_query(comp_query, sample)
    result = self.evaluate(query_result)
    expected = self.evaluate([t1, t2])

    self.assertAllClose(result, expected, atol=0)

  @parameterized.product(sample_size=[1, 3])
  def test_inner_query_no_noise_sum(self, sample_size):
    t1 = tf.constant([1, 2], dtype=tf.float32)
    t2 = tf.constant([3], dtype=tf.float32)
    record = [t1, t2]
    sample = [record] * sample_size

    comp_query = compression_query.CompressionSumQuery(
        quantization_params=self.DEFAULT_QUANTIZATION_PARAMS,
        inner_query=self.DEFAULT_INNER_QUERY,
        record_template=record)

    query_result, _ = test_utils.run_query(comp_query, sample)
    result = self.evaluate(query_result)
    expected = self.evaluate([t1 * sample_size, t2 * sample_size])
    self.assertAllClose(result, expected, atol=0)

  @parameterized.product(sample_size=[1, 3])
  def test_inner_query_no_noise_sum_nested(self, sample_size):
    t1 = tf.constant([1, 0], dtype=tf.int32)
    t2 = tf.constant([1, 1, 1], dtype=tf.int32)
    t3 = tf.constant([1], dtype=tf.int32)
    t4 = tf.constant([[1, 1], [1, 1]], dtype=tf.int32)
    record = [t1, dict(a=t2, b=[t3, (t4, t1)])]
    sample = [record] * sample_size

    compress_query = compression_query.CompressionSumQuery(
        quantization_params=self.DEFAULT_QUANTIZATION_PARAMS,
        inner_query=self.DEFAULT_INNER_QUERY,
        record_template=record)

    query_result, _ = test_utils.run_query(compress_query, sample)
    result = self.evaluate(query_result)

    s = sample_size
    expected = [t1 * s, dict(a=t2 * s, b=[t3 * s, (t4 * s, t1 * s)])]
    # Use `assertAllClose` to compare structures
    self.assertAllClose(result, expected, atol=0)

  def test_calls_inner_query(self):
    t = tf.constant([1, 2], dtype=tf.float32)
    record = [t, t]
    sample_size = 2
    sample = [record] * sample_size

    # Mock inner query to check that the methods get called.
    in_q = ddg_sum_query(
        l2_norm_bound=self.DEFAULT_L2_NORM_BOUND, local_scale=0.0)
    in_q.initial_sample_state = mock.MagicMock(wraps=in_q.initial_sample_state)
    in_q.initial_global_state = mock.MagicMock(wraps=in_q.initial_global_state)
    in_q.derive_sample_params = mock.MagicMock(wraps=in_q.derive_sample_params)
    in_q.preprocess_record = mock.MagicMock(wraps=in_q.preprocess_record)
    in_q.get_noised_result = mock.MagicMock(wraps=in_q.get_noised_result)

    comp_query = compression_query.CompressionSumQuery(
        quantization_params=self.DEFAULT_QUANTIZATION_PARAMS,
        inner_query=in_q,
        record_template=record)

    query_result, _ = test_utils.run_query(comp_query, sample)
    result = self.evaluate(query_result)
    expected = self.evaluate([t * sample_size, t * sample_size])
    self.assertAllClose(result, expected, atol=0)

    # Check calls
    self.assertEqual(in_q.initial_sample_state.call_count, 1)
    self.assertEqual(in_q.initial_global_state.call_count, 1)
    self.assertEqual(in_q.derive_sample_params.call_count, 1)
    self.assertEqual(in_q.preprocess_record.call_count, sample_size)
    self.assertEqual(in_q.get_noised_result.call_count, 1)


if __name__ == '__main__':
  tf.test.main()
