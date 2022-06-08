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
"""Tests for subsampled_random_hadamard."""
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from private_linear_compression import subsampled_random_hadamard


class SRHTSketchTest(tf.test.TestCase, parameterized.TestCase):
  """Test SRHT encoder, decoder, and other helper functions."""

  @parameterized.named_parameters([('identity', tf.constant([1.0] * 10), 10),
                                   ('subsampling', tf.constant([1.0] * 10), 6)])
  def test_subsampling_shape(self, vector, subsample_dim):
    vector_sampled = subsampled_random_hadamard._subsampling(
        vector, subsample_dim)
    self.assertEqual(vector_sampled.shape, [subsample_dim])
    self.assertAllEqual(vector_sampled, [1.0] * subsample_dim)

  @parameterized.named_parameters([('identity', tf.constant([0.0] * 10), 10),
                                   ('all_zeros', tf.constant([0.0] * 10), 20)])
  def test_pad_zeros_shape(self, vector, original_dim):
    vector_padded = subsampled_random_hadamard._pad_zeros(vector, original_dim)
    self.assertEqual(vector_padded.shape, [original_dim])
    self.assertAllEqual(vector_padded, [0.0] * original_dim)

  @parameterized.named_parameters([('range_99', tf.range(99) + 1, 50, (1, 1))])
  def test_subsampling_pad_zeros_jointly(self, vector, subsample_dim,
                                         seed_sampling):
    original_dim = vector.shape[0]
    vector_subsampled = subsampled_random_hadamard._subsampling(
        vector, subsample_dim, seed_sampling)
    vector_padded = subsampled_random_hadamard._pad_zeros(
        vector_subsampled, original_dim, seed_sampling)
    num_equal_coordinates = tf.where(vector == vector_padded).shape[0]
    self.assertEqual(num_equal_coordinates, subsample_dim)

  @parameterized.named_parameters([('identity', tf.constant([1.0] * 10), 10),
                                   ('all_zeros', tf.constant([1.0] * 10), 20)])
  def test_pad_zeros_same_sum(self, vector, original_dim):
    vector_padded = subsampled_random_hadamard._pad_zeros(vector, original_dim)
    subsample_dim = vector.shape[0]
    self.assertEqual(vector_padded.shape, [original_dim])
    self.assertEqual(tf.reduce_sum(vector_padded), subsample_dim)

  @parameterized.named_parameters([('identity', tf.constant([0.0] * 10), 10),
                                   ('subsampling', tf.constant([0.0] * 10), 5)])
  def test_srht_encode_shape(self, vector, subsample_dim):
    vector_encoded = subsampled_random_hadamard.srht_encode(
        vector, subsample_dim)
    self.assertEqual(vector_encoded.shape, [subsample_dim])
    self.assertAllEqual(vector_encoded, [0.0] * subsample_dim)

  @parameterized.named_parameters([('identity', tf.constant([0.0] * 10), 10),
                                   ('all_zeros', tf.constant([0.0] * 10), 20)])
  def test_srht_sketch_decode_shape(self, vector, original_dim):
    vector_decoded = subsampled_random_hadamard.srht_sketch_decode(
        vector, original_dim)
    self.assertEqual(vector_decoded.shape, [original_dim])
    self.assertAllEqual(vector_decoded, [0.0] * original_dim)

  @parameterized.named_parameters([('identity',
                                    tf.constant([1.0, 2.0, 3.0, 4.0]), 4)])
  def test_srht_encode_decode(self, vector, subsample_dim):
    original_dim = vector.shape[0]
    vector_enc = subsampled_random_hadamard.srht_encode(vector, subsample_dim)
    vector_dec = subsampled_random_hadamard.srht_sketch_decode(
        vector_enc, original_dim)
    self.assertEqual(vector_dec.shape, [original_dim])
    self.assertAllEqual(vector_dec, vector)

  @parameterized.named_parameters([('sampling_outside_range',
                                    tf.constant([1.0, 2.0, 3.0, 4.0]), 5)])
  def test_srht_encode_raises_value_error(self, vector, subsample_dim):
    with self.assertRaises(ValueError):
      subsampled_random_hadamard.srht_encode(vector, subsample_dim)

  @parameterized.named_parameters([('padding_outside_range',
                                    tf.constant([1.0, 2.0, 3.0, 4.0]), 3)])
  def test_srht_decode_raises_value_error(self, vector, original_dim):
    with self.assertRaises(ValueError):
      subsampled_random_hadamard.srht_sketch_decode(vector, original_dim)


def _identity_srht_mean():
  return subsampled_random_hadamard.GradientSRHTSketchFactory(
      compression_rate=1,
      inner_agg_factory=tff.aggregators.UnweightedMeanFactory())


def _identity_srht_sum():
  return subsampled_random_hadamard.GradientSRHTSketchFactory(
      compression_rate=1, inner_agg_factory=tff.aggregators.SumFactory())


def _srht_mean():
  return subsampled_random_hadamard.GradientSRHTSketchFactory(
      compression_rate=0.5,
      inner_agg_factory=tff.aggregators.UnweightedMeanFactory())


def _srht_sum():
  return subsampled_random_hadamard.GradientSRHTSketchFactory(
      compression_rate=0.5, inner_agg_factory=tff.aggregators.SumFactory())


class GradientSRHTSketchFactoryExecutionTest(tf.test.TestCase,
                                             parameterized.TestCase):

  @parameterized.product(
      (dict(
          name='shape_4_tensor_sum',
          value_type=(tf.float32, [4]),
          client_data=[
              tf.constant([1.0, 2.0, 3.0, 4.0]),
              tf.constant([4.0, 3.0, 2.0, 1.0])
          ],
          expected_result=tf.constant([5.0, 5.0, 5.0, 5.0]),
          factory_fn=_identity_srht_sum,
          comment='`compression_rate`=1 is equivalent to random rotation, sum'
          ' of clients vectors is expected'),
       dict(
           name='shape_4_tensor_mean',
           value_type=(tf.float32, [4]),
           client_data=[
               tf.constant([1.0, 2.0, 3.0, 4.0]),
               tf.constant([4.0, 3.0, 2.0, 1.0])
           ],
           expected_result=tf.constant([2.5, 2.5, 2.5, 2.5]),
           factory_fn=_identity_srht_mean,
           comment='`compression_rate`=1 is equivalent to random rotation, mean'
           ' of clients vectors is expected'),
       dict(
           name='shape_1_tensor_mean',
           value_type=(tf.float32, [1]),
           client_data=[tf.constant([1.0]),
                        tf.constant([4.0])],
           expected_result=tf.constant([5.0]),
           factory_fn=_identity_srht_sum,
           comment='`compression_rate`=1 is equivalent to random rotation, sum'
           ' of clients vectors is expected')))
  def test_as_identitical_mapping(self, name, value_type, client_data,
                                  expected_result, factory_fn, comment):
    """Integrated test for sum and mean with `compression_rate`=1."""
    del name, comment  # Unused.
    factory = factory_fn()
    aggregation_process = factory.create(tff.to_type(value_type))
    state = aggregation_process.initialize()
    output = aggregation_process.next(state, client_data)
    self.assertAllClose(output.result, expected_result)

  @parameterized.named_parameters([
      ('rank_1_tensor_sum', (tf.float32, [10]),
       [tf.constant([0.0] * 10),
        tf.constant([0.0] * 10)], tf.constant([0.0] * 10))
  ])
  def test_aggregation_process_state_type(self, value_type, client_data,
                                          expected_result):
    """Integrated test for sum and mean with `compression_rate=0.5`."""
    del expected_result  # Unused.
    factory = _srht_sum()
    aggregation_process = factory.create(tff.to_type(value_type))
    state = aggregation_process.initialize()
    for _ in range(3):
      output = aggregation_process.next(state, client_data)
      self.assertEqual(output.state['round_seed'].dtype,
                       tf.int32.as_numpy_dtype)
      self.assertEqual(output.state['round_seed'].shape, (2, 2))
      self.assertEqual(output.state['inner_agg_process'], ())
      state = output.state

  @parameterized.named_parameters([
      ('rank_1_tensor_all_zeros_sum', (tf.float32, [10]),
       [tf.constant([0.0] * 10),
        tf.constant([0.0] * 10)], tf.constant([0.0] * 10))
  ])
  def test_encode_decode_values(self, value_type, client_data, expected_result):
    """Integrated test for sum and mean with `compression_rate=0.5`."""
    factory = _srht_sum()
    aggregation_process = factory.create(tff.to_type(value_type))
    state = aggregation_process.initialize()
    for _ in range(3):
      output = aggregation_process.next(state, client_data)
      self.assertAllClose(output.result, expected_result)
      state = output.state

  def test_raises_inner_factory_type_error(self):
    with self.assertRaises(TypeError):
      subsampled_random_hadamard.GradientSRHTSketchFactory(
          tff.aggregators.MeanFactory())

  @parameterized.product(repeat=[0, -1, 0.5], compression_rate=[0, -1, 1.5])
  def test_raises_srht_factory_value_error(self, repeat, compression_rate):
    with self.assertRaises(ValueError):
      subsampled_random_hadamard.GradientSRHTSketchFactory(
          repeat=repeat, compression_rate=compression_rate)


if __name__ == '__main__':
  tf.test.main()
