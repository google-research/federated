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
"""Tests for compression_utils."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from distributed_discrete_gaussian import compression_utils

SEED_PAIR = (12345678, 87654321)


class StochasticRoundingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([('unconditional', False),
                                   ('conditional', True)])
  def test_noop(self, conditional):
    x = tf.range(100, dtype=tf.float32)
    rounded_x = compression_utils.stochastic_rounding(x, conditional)
    x, rounded_x = self.evaluate([x, rounded_x])
    self.assertAllEqual(x, rounded_x)
    self.assertEqual(rounded_x.dtype, np.float32)

  @parameterized.named_parameters([('unconditional', False),
                                   ('conditional', True)])
  def test_expected_value(self, conditional):
    num_trials = 200
    x = tf.random.uniform([100], minval=-10, maxval=10, dtype=tf.float32)

    avg_rounded_x = tf.zeros_like(x)
    for _ in range(num_trials):
      # Defaults to input norm as the norm bound for conditional rounding.
      avg_rounded_x += compression_utils.stochastic_rounding(x, conditional)
    avg_rounded_x /= num_trials

    # Expected value over trials of rounding should be close to original input;
    # use slightly larger tolerance for conditional rounding.
    x, avg_rounded_x = self.evaluate([x, avg_rounded_x])
    self.assertAllClose(x, avg_rounded_x, atol=0.2 if conditional else 0.1)

  @parameterized.product(
      conditional=[False, True], dtype=[tf.float16, tf.float32, tf.float64])
  def test_keeps_dtype(self, conditional, dtype):
    x = tf.random.uniform([100], minval=-10, maxval=10, dtype=dtype)
    rounded_x = compression_utils.stochastic_rounding(x, conditional)
    self.assertAllEqual(rounded_x.dtype, dtype)


class ScaledQuantizationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      quantize_scale=[0.1, 0.25, 0.33, 0.5, 1, 2, 3.14, 4, 8, 16, 32])
  def test_quantization_stochastic_rounding(self, quantize_scale):
    min_max = [-10, 10]
    x = tf.random.uniform([10], minval=min_max[0], maxval=min_max[1])

    encoded_x = compression_utils.scaled_quantization(
        x,
        scale=quantize_scale,
        stochastic=True,
        conditional=False,
        l2_norm_bound=tf.norm(x))
    decoded_x = compression_utils.inverse_scaled_quantization(
        encoded_x, quantize_scale)
    x, decoded_x = self.evaluate([x, decoded_x])

    # For stochastic rounding, the error incurred by scaled quantization is
    # bounded by the effective bin width.
    self.assertAllClose(x, decoded_x, rtol=0.0, atol=1 / quantize_scale)

  @parameterized.product(
      quantize_scale=[0.25, 0.33, 0.5, 1, 2, 3.14, 4, 8, 16, 32])
  def test_quantization_deterministic_rounding(self, quantize_scale):
    min_max = [-10, 10]
    x = tf.random.uniform([10], minval=min_max[0], maxval=min_max[1])

    encoded_x = compression_utils.scaled_quantization(
        x,
        scale=quantize_scale,
        stochastic=False,
        conditional=False,
        l2_norm_bound=tf.norm(x))
    decoded_x = compression_utils.inverse_scaled_quantization(
        encoded_x, quantize_scale)
    x, decoded_x = self.evaluate([x, decoded_x])

    # For deterministic rounding, the error incurred by scaled quantization is
    # bounded by the half of the effective bin width.
    self.assertAllClose(x, decoded_x, rtol=0.0, atol=0.5 / quantize_scale)

  @parameterized.product(shape=[(10,), (10, 10), (10, 5, 2)])
  def test_quantization_different_shapes(self, shape):
    quantize_scale = 20.0
    min_x, max_x = 0, 1
    x = tf.random.uniform(shape=shape, minval=min_x, maxval=max_x)
    encoded_x = compression_utils.scaled_quantization(
        x,
        quantize_scale,
        stochastic=True,
        conditional=False,
        l2_norm_bound=tf.norm(x))
    decoded_x = compression_utils.inverse_scaled_quantization(
        encoded_x, quantize_scale)
    x, decoded_x = self.evaluate([x, decoded_x])
    # Check value and shape.
    self.assertAllEqual(x.shape, decoded_x.shape)
    self.assertAllClose(x, decoded_x, rtol=0.0, atol=1 / quantize_scale)


class FlattenConcatTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([('atomic_record', 10),
                                   ('list_record', [1, 2, np.int32(3)]),
                                   ('nested_record', (1, dict(a=2, b=[3, 4])))])
  def test_raise_on_scalar_tensors(self, record):
    # Need to convert np arrays to tensors first.
    record = tf.nest.map_structure(tf.constant, record)
    with self.assertRaises(ValueError):
      compression_utils.flatten_concat(record)

  @parameterized.named_parameters([
      ('vector_record', np.arange(10), np.arange(10)),
      ('tensor_record', np.arange(24).reshape(3, 2, 4), np.arange(24)),
      ('vector_list_record', [np.arange(2),
                              np.arange(3)], np.array([0, 1, 0, 1, 2])),
      ('tensor_list_record', [np.array([[0, 1], [2, 3]]),
                              np.array([4, 5])], np.arange(6)),
      ('tensor_nested_record',
       (np.array([0]),
        [np.array([[1], [2]]),
         dict(a=np.array([3]), b=np.array([4, 5]))]), np.arange(6)),
      ('large_vector_list_record', [np.arange(1e4),
                                    np.arange(1e4, 5e4)], np.arange(5e4))
  ])
  def test_flatten_concat(self, record, record_as_vector_expected):
    """Checks the structure gets flattened/concatenated and packed correctly."""
    # Need to convert np arrays to tensors first.
    record = tf.nest.map_structure(tf.constant, record)
    record_as_vector = self.evaluate(compression_utils.flatten_concat(record))
    self.assertAllEqual(record_as_vector, record_as_vector_expected)

  @parameterized.named_parameters([
      ('vector_record', np.arange(10), np.arange(10)),
      ('tensor_record', np.arange(24).reshape(3, 2, 4), np.arange(24)),
      ('vector_list_record', [np.arange(2),
                              np.arange(3)], np.array([0, 1, 0, 1, 2])),
      ('tensor_list_record', [np.array([[0, 1], [2, 3]]),
                              np.array([4, 5])], np.arange(6)),
      ('tensor_nested_record',
       (np.array([0]),
        [np.array([[1], [2]]),
         dict(a=np.array([3]), b=np.array([4, 5]))]), np.arange(6)),
      ('large_vector_list_record', [np.arange(1e4),
                                    np.arange(1e4, 5e4)], np.arange(5e4))
  ])
  def test_inverse_flatten_concat(self, original_record, flat_vector):
    # Need to convert np arrays to tensors first.
    original_record = tf.nest.map_structure(tf.constant, original_record)
    flat_vector = tf.constant(flat_vector)

    packed_record = self.evaluate(
        compression_utils.inverse_flatten_concat(flat_vector, original_record))
    tf.nest.assert_same_structure(packed_record, original_record)
    tf.nest.map_structure(self.assertAllEqual, packed_record, original_record)


class PaddingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(dim=[1, 2, 4, 8, 16, 32, 64])
  def test_no_padding(self, dim):
    """Checks the same input is returned if no padding is requried."""
    x = tf.range(dim, dtype=tf.float32)
    padded_x = compression_utils.pad_zeros(x)
    x, padded_x = self.evaluate([x, padded_x])
    self.assertEqual(padded_x.dtype, np.float32)
    self.assertAllEqual(x, padded_x)

  @parameterized.named_parameters(('dim3', 3, 4), ('dim9', 9, 16),
                                  ('dim31', 31, 32), ('dim65', 65, 128))
  def test_padding(self, input_dim, padded_dim):
    """Checks size, dtype, and content of the padded vector."""
    x = tf.range(input_dim, dtype=tf.float32)
    padded_x = compression_utils.pad_zeros(x)
    x, padded_x = self.evaluate([x, padded_x])
    num_zeros = padded_dim - input_dim
    self.assertEqual(padded_x.shape, (padded_dim,))
    self.assertEqual(padded_x.dtype, np.float32)
    self.assertAllEqual(padded_x[:input_dim], x)
    self.assertAllEqual(padded_x[input_dim:], np.zeros((num_zeros,)))

  @parameterized.named_parameters(('rank2', (2, 2)), ('rank4', (2, 2, 2, 2)))
  def test_raise_on_extra_tensor_ranks(self, shape):
    x = tf.random.uniform(shape=shape)
    with self.assertRaises(ValueError):
      _ = compression_utils.pad_zeros(x)


class RandomizedHadamardTransformTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(dim=[2, 4, 8, 16, 32])
  def test_fixed_l2_norm(self, dim):
    """Checks that the L2 norm doesn't change much."""
    x = tf.random.uniform((dim,))
    transformed_x = compression_utils.randomized_hadamard_transform(
        x, SEED_PAIR, repeat=1)
    x, transformed_x = self.evaluate([x, transformed_x])
    self.assertAllClose(np.linalg.norm(x), np.linalg.norm(transformed_x))

  @parameterized.product(dim=[2, 4, 8, 16, 32])
  def test_has_rotation(self, dim):
    """Checks that the input/output differences is not zero."""
    x = tf.random.uniform((dim,))
    transformed_x = compression_utils.randomized_hadamard_transform(
        x, SEED_PAIR, repeat=1)
    x, transformed_x = self.evaluate([x, transformed_x])
    self.assertGreater(np.linalg.norm(transformed_x - x), 0.5)

  @parameterized.named_parameters(('dim16', 16, 3), ('dim64', 64, 7))
  def test_use_different_seed_on_repeat_forward(self, dim, difference_norm):
    """Checks that forward randomized HT repeats with different seed."""
    x = tf.random.uniform((dim,))
    # Repeat transformations by making a single call.
    repeat_transformed_x = compression_utils.randomized_hadamard_transform(
        x, seed_pair=SEED_PAIR, repeat=2)
    # Repeat by making separate calls.
    separate_transformed_x = compression_utils.randomized_hadamard_transform(
        x, seed_pair=SEED_PAIR, repeat=1)
    separate_transformed_x = compression_utils.randomized_hadamard_transform(
        separate_transformed_x, seed_pair=SEED_PAIR, repeat=1)

    repeat_transformed_x = self.evaluate(repeat_transformed_x)
    separate_transformed_x = self.evaluate(separate_transformed_x)
    difference = separate_transformed_x - repeat_transformed_x
    self.assertGreater(np.linalg.norm(difference), difference_norm)

  @parameterized.product(
      seed_pair=[(11, 22), (33, 44), (55, 66)], repeat=[1, 3, 10])
  def test_inverse_transformation(self, seed_pair, repeat):
    """Verifies that the inverse transform reverses the ops with same seeds."""
    dim = 128
    x = tf.random.uniform((dim,))
    transformed_x = compression_utils.randomized_hadamard_transform(
        x, seed_pair=seed_pair, repeat=repeat)
    reverted_x = compression_utils.inverse_randomized_hadamard_transform(
        transformed_x, original_dim=dim, seed_pair=seed_pair, repeat=repeat)
    x, reverted_x = self.evaluate([x, reverted_x])
    self.assertAllClose(x, reverted_x)

  @parameterized.product(dtype=[tf.float64, tf.int32, tf.int64])
  def test_raise_on_invalid_dtype(self, dtype):
    x = tf.range(32, dtype=dtype)
    with self.assertRaises(TypeError):
      _ = compression_utils.randomized_hadamard_transform(x, SEED_PAIR)

    with self.assertRaises(TypeError):
      _ = compression_utils.inverse_randomized_hadamard_transform(
          x, 32, SEED_PAIR)

  @parameterized.named_parameters(('rank2', (2, 2)), ('rank4', (2, 2, 2, 2)))
  def test_raise_on_extra_tensor_ranks(self, shape):
    x = tf.random.uniform(shape=shape)
    with self.assertRaises(ValueError):
      _ = compression_utils.randomized_hadamard_transform(x, SEED_PAIR)

    with self.assertRaises(ValueError):
      _ = compression_utils.inverse_randomized_hadamard_transform(
          x, np.prod(shape), SEED_PAIR)

  @parameterized.named_parameters(('dim1', 1, 1), ('dim3', 3, 4),
                                  ('dim15', 15, 16), ('dim31', 31, 32),
                                  ('dim33', 33, 64))
  def test_forward_transform_pads_to_power_of_2(self, input_dim, padded_dim):
    # Tests the integration with the padding function.
    x = tf.range(input_dim, dtype=tf.float32)
    transformed_x = self.evaluate(
        compression_utils.randomized_hadamard_transform(x, SEED_PAIR))
    self.assertEqual(transformed_x.shape, (padded_dim,))

  @parameterized.named_parameters(('dim1', 1, 1), ('dim3', 3, 4),
                                  ('dim15', 15, 16), ('dim31', 31, 32),
                                  ('dim33', 33, 64))
  def test_inverse_transform_unpads(self, input_dim, padded_dim):
    x = tf.range(padded_dim, dtype=tf.float32)
    reverted_x = compression_utils.inverse_randomized_hadamard_transform(
        x, original_dim=input_dim, seed_pair=SEED_PAIR)
    reverted_x = self.evaluate(reverted_x)
    self.assertEqual(reverted_x.shape, (input_dim,))


class SampleRademacherTests(tf.test.TestCase, parameterized.TestCase):
  """Tests for `sample_rademacher` method."""

  def _assert_signs(self, x):
    """Check every element is +1/-1."""
    assert isinstance(x, np.ndarray)
    size = x.size
    flat_x = x.reshape([-1])
    self.assertAllEqual([True] * size,
                        np.logical_or(
                            np.isclose(1.0, flat_x), np.isclose(-1.0, flat_x)))

  @parameterized.product(num_elements=[1, 10, 101])
  def test_expected_output_values(self, num_elements):
    shape = (1, num_elements)
    signs = compression_utils.sample_rademacher(
        shape=shape, dtype=tf.int32, seed_pair=SEED_PAIR)
    signs = self.evaluate(signs)
    self._assert_signs(signs)

  def test_both_values_present(self):
    signs = compression_utils.sample_rademacher(
        shape=(100, 10), dtype=tf.int32, seed_pair=SEED_PAIR)
    signs = self.evaluate(signs)
    self._assert_signs(signs)
    self.assertGreater(np.sum(np.isclose(1.0, signs)), 450)
    self.assertGreater(np.sum(np.isclose(-1.0, signs)), 450)

  @parameterized.product(dtype=[tf.float32, tf.float64, tf.int32, tf.int64])
  def test_expected_dtype(self, dtype):
    shape = (1000,)
    signs = compression_utils.sample_rademacher(shape, dtype, SEED_PAIR)
    self.assertEqual(dtype, signs.dtype)
    signs = self.evaluate(signs)
    self._assert_signs(signs)

  @parameterized.product(shape=[[10], [10, 10], [10, 10, 10], [10, 1, 1, 1]])
  def test_expected_shape(self, shape):
    shape = tuple(shape)
    signs = compression_utils.sample_rademacher(shape, tf.int32, SEED_PAIR)
    signs = self.evaluate(signs)
    self._assert_signs(signs)
    self.assertEqual(signs.shape, shape)

  def test_differs_given_different_seed(self):
    shape = (100,)
    seed1, seed2 = (42, 42), (4200, 4200)
    signs1 = compression_utils.sample_rademacher(shape, tf.int32, seed1)
    signs2 = compression_utils.sample_rademacher(shape, tf.int32, seed2)
    signs1, signs2 = self.evaluate([signs1, signs2])
    self.assertFalse(np.array_equal(signs1, signs2))


if __name__ == '__main__':
  tf.test.main()
