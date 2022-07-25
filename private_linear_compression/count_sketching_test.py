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
"""Tests for count_sketching."""
from unittest import mock
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from private_linear_compression import count_sketching
from private_linear_compression import count_sketching_utils


class GradientCountSketchFactoryTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.default_sketching_kwargs = {
        'min_compression_rate': 2.,
        'num_repeats': 5,
        'decode_method': count_sketching_utils.DecodeMethod.MEAN,
    }

  def test_raises_weighted_inner_factory_type_error(self):
    kwargs = self.default_sketching_kwargs
    with self.assertRaises(TypeError):
      count_sketching.GradientCountSketchFactory(
          **kwargs, inner_agg_factory=tff.aggregators.MeanFactory())

  @parameterized.named_parameters([('negative', -1)])
  def test_raises_parallel_iterations_value_error(self, parallel_iterations):
    kwargs = self.default_sketching_kwargs
    with self.assertRaises(ValueError):
      count_sketching.GradientCountSketchFactory(
          **kwargs, parallel_iterations=parallel_iterations)

  def test_raises_compression_rate_value_error(self):
    kwargs = self.default_sketching_kwargs
    kwargs = dict(kwargs, min_compression_rate=0.98)
    with self.assertRaises(ValueError):
      count_sketching.GradientCountSketchFactory(**kwargs)

  def test_raises_num_repeats_value_error(self):
    kwargs = self.default_sketching_kwargs
    kwargs = dict(kwargs, num_repeats=0)
    with self.assertRaises(ValueError):
      count_sketching.GradientCountSketchFactory(**kwargs)

  @parameterized.named_parameters([('rank_2', (tf.float32, [1, 1]))])
  def test_raises_value_type_shape_error(self, value_type):
    kwargs = self.default_sketching_kwargs
    factory = count_sketching.GradientCountSketchFactory(**kwargs)
    with self.assertRaises(ValueError):
      factory.create(tff.to_type(value_type))

  @parameterized.named_parameters([
      ('federated_type',
       tff.FederatedType(tff.TensorType(tf.float32, [1]), tff.SERVER)),
      ('struct_type', tff.types.to_type((tf.float32, tf.float32)))
  ])
  def test_raises_not_tensortype_error(self, value_type):
    kwargs = self.default_sketching_kwargs
    factory = count_sketching.GradientCountSketchFactory(**kwargs)
    with self.assertRaises(TypeError):
      factory.create(value_type)

  def test_aggregation_compression_too_high_value_error(self):
    """Integrated test for sum and mean with `compression_rate`=1."""
    kwargs = self.default_sketching_kwargs
    factory = count_sketching.GradientCountSketchFactory(**kwargs)

    with self.assertRaises(ValueError):
      factory.create(tff.to_type((tf.float32, [kwargs['num_repeats']])))

  # Due to the linearity of count-sketch and the sum aggregator, if the sum of
  # clients' data is zero, then the expected results should be all zeros.
  @parameterized.named_parameters([
      ('all_zeros', (tf.float32, [25]),
       [tf.constant([0.0] * 25),
        tf.constant([0.0] * 25)], tf.constant([0.0] * 25)),
      ('sum_zeros_1', (tf.float32, [25]),
       [tf.constant([1.0] + [0.0] * 24),
        tf.constant([-1.0] + [0.0] * 24)], tf.constant([0.0] * 25)),
      ('sum_zeros_2', (tf.float32, [25]), [
          tf.constant([1.0, -0.8] + [0.0] * 23),
          tf.constant([-1.0, 0.8] + [0.0] * 23)
      ], tf.constant([0.0] * 25)),
  ])
  def test_aggregation_sum_across_clients_is_zero(self, value_type, client_data,
                                                  expected_result):
    """Integrated test for sum and mean with `compression_rate`=1."""
    kwargs = self.default_sketching_kwargs
    factory = count_sketching.GradientCountSketchFactory(**kwargs)

    aggregation_process = factory.create(tff.to_type(value_type))
    state = aggregation_process.initialize()
    output = aggregation_process.next(state, client_data)
    self.assertAllClose(output.result, expected_result)

  @parameterized.named_parameters([
      ('length_10_mean_0_method_mean_seed_2', 10, 0.0,
       count_sketching_utils.DecodeMethod.MEAN, (2, 2)),
      ('length_100_mean_0_method_mean_seed_3', 100, 0.0,
       count_sketching_utils.DecodeMethod.MEAN, (3, 3)),
      ('length_10_mean_0_method_median_seed_2', 10, 0.0,
       count_sketching_utils.DecodeMethod.MEDIAN, (2, 2)),
      ('length_100_mean_0_method_median_seed_3', 100, 0.0,
       count_sketching_utils.DecodeMethod.MEDIAN, (3, 3)),
      ('length_10_mean_5_method_mean_seed_2', 10, 5.0,
       count_sketching_utils.DecodeMethod.MEAN, (2, 2)),
      ('length_100_mean_5_method_mean_seed_3', 100, 5.0,
       count_sketching_utils.DecodeMethod.MEAN, (3, 3)),
      ('length_10_mean_5_method_median_seed_2', 10, 5.0,
       count_sketching_utils.DecodeMethod.MEDIAN, (2, 2)),
      ('length_100_mean_5_method_median_seed_3', 100, 5.0,
       count_sketching_utils.DecodeMethod.MEDIAN, (3, 3)),
  ])
  def test_aggregator_sum_matches_sketching(self, gradient_length, mean_value,
                                            method, seed):
    """Integrated test for sum and mean with `compression_rate`=1."""
    kwargs = self.default_sketching_kwargs
    kwargs.pop('decode_method')
    value_type = tff.to_type((tf.float32, [gradient_length]))
    client_data = [
        tf.random.stateless_normal([gradient_length], seed, mean_value),
        tf.random.stateless_uniform([gradient_length], seed, mean_value - 1,
                                    mean_value + 1),
    ]

    # Compute the sum with `GradientCountSketchFactory`.
    # Fix flakiness by fixing the seed value.
    with mock.patch.object(tf, 'timestamp', return_value=0):
      factory = count_sketching.GradientCountSketchFactory(
          **kwargs, decode_method=method)
      aggregation_process = factory.create(tff.to_type(value_type))

    state = aggregation_process.initialize()
    output = aggregation_process.next(state, client_data)

    # Compute the sum directly via sketching
    width = factory._get_num_bins(gradient_length,
                                  kwargs['min_compression_rate'])
    index_seeds = state['index_seeds']
    sign_seeds = state['sign_seeds']
    client_data_sum = tf.add_n(client_data)
    client_data_sketch = count_sketching_utils.encode(
        client_data_sum,
        kwargs['num_repeats'],
        width,
        index_seeds=index_seeds,
        sign_seeds=sign_seeds)
    expected_result = count_sketching_utils.decode(
        client_data_sketch,
        value_type.shape.num_elements(),
        index_seeds=index_seeds,
        sign_seeds=sign_seeds,
        method=method)
    self.assertAllClose(output.result, expected_result, atol=1e-4)

  def test_aggregator_unique_seeds_each_round(self):
    kwargs = self.default_sketching_kwargs

    value_type = (tf.float32, [100])
    client_data = [tf.random.normal([100]), tf.random.uniform([100])]

    # Ensure that unique seeds are achieved regardless of the time of execution.
    with mock.patch.object(tf, 'timestamp', return_value=0):
      factory = count_sketching.GradientCountSketchFactory(**kwargs)
      aggregation_process = factory.create(tff.to_type(value_type))
    state = aggregation_process.initialize()

    # We check only the global seeds (index 0) as then random states are unique.
    index_seeds = [state['index_seeds'][0]]
    sign_seeds = [state['sign_seeds'][0]]

    # Perform n_rounds of FL.
    for _ in range(10):
      output = aggregation_process.next(state, client_data)
      state = output.state
      index_seeds.append(state['index_seeds'][0])
      sign_seeds.append(state['sign_seeds'][0])

    # Expect a different sign and index seed each round.
    self.assertLen(set(index_seeds + sign_seeds), len(index_seeds + sign_seeds))

  def test_aggregator_correct_dtype(self):
    kwargs = self.default_sketching_kwargs

    dtype = tf.float32
    value_type = (dtype, [45])
    client_data = [tf.random.normal([45]), tf.random.uniform([45])]

    factory = count_sketching.GradientCountSketchFactory(**kwargs)
    aggregation_process = factory.create(tff.to_type(value_type))
    state = aggregation_process.initialize()

    output = aggregation_process.next(state, client_data)

    self.assertDTypeEqual(output.result, dtype)


if __name__ == '__main__':
  tf.test.main()
