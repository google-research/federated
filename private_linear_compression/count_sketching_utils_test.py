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
import functools
from typing import Any, Mapping

from absl.testing import parameterized
import numpy as np
import scipy.stats
import tensorflow as tf

from private_linear_compression import count_sketching_utils


def perform_encode_decode(gradient: tf.Tensor, gradient_length: tf.Tensor,
                          kwargs: Mapping[str, Any], method: str) -> tf.Tensor:
  """Returns the reconstructed `gradient_estimate` from the `gradient`."""
  sketch = count_sketching_utils.encode(
      tf.convert_to_tensor(gradient), **kwargs)
  kwargs.pop('length')
  kwargs.pop('width')
  return count_sketching_utils.decode(
      sketch, gradient_length, **kwargs, method=method)


class RandomnessTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.default_kwargs = {
        'width': 25,
        'gradient_length': 1000,
        'gradient_dtype': tf.float16,
        'sign_seeds': tf.constant([0, 0], tf.int32),
        'index_seeds': tf.constant([1, 1], tf.int32),
    }

  @parameterized.named_parameters(('hash_id_0', 0), ('hash_id_1', 1))
  def test_indices_are_uniform(self, hash_id):

    indices, _ = count_sketching_utils._get_hash_mapping(
        **self.default_kwargs, hash_id=hash_id)
    _, pval = scipy.stats.kstest(indices, 'uniform')

    self.assertLess(pval, 1e-5, 'Indices should follow a uniform distribution.')

  @parameterized.named_parameters(('hash_id_0', 0), ('hash_id_1', 1))
  def test_signs_are_uniform(self, hash_id):
    _, signs = count_sketching_utils._get_hash_mapping(
        **self.default_kwargs, hash_id=hash_id)
    _, pval = scipy.stats.kstest(signs, 'uniform')

    self.assertLess(pval, 1e-5, 'Signs should follow a uniform distribution.')

  @parameterized.named_parameters(('hash_id_0', 0), ('hash_id_1', 1))
  def test_signs_are_signs(self, hash_id):
    _, signs = count_sketching_utils._get_hash_mapping(
        **self.default_kwargs, hash_id=hash_id)
    self.assertAllInSet(signs, [-1, 1])

  @parameterized.named_parameters(('hash_id_0', 0, 1), ('hash_id_1', 1, 2))
  def test_hashes_independent_indices(self, hash_id1, hash_id2):
    indices_hash_1, _ = count_sketching_utils._get_hash_mapping(
        **self.default_kwargs, hash_id=hash_id1)
    indices_hash_2, _ = count_sketching_utils._get_hash_mapping(
        **self.default_kwargs, hash_id=hash_id2)
    self.assertNotAllEqual(indices_hash_1, indices_hash_2)

  @parameterized.named_parameters(('hash_id_0', 0, 1), ('hash_id_1', 1, 2))
  def test_hashes_independent_signs(self, hash_id1, hash_id2):
    _, signs_hash_1 = count_sketching_utils._get_hash_mapping(
        **self.default_kwargs, hash_id=hash_id1)
    _, signs_hash_2 = count_sketching_utils._get_hash_mapping(
        **self.default_kwargs, hash_id=hash_id2)
    self.assertNotAllEqual(signs_hash_1, signs_hash_2)


class GradientVectorCountSketchEncodeTest(parameterized.TestCase,
                                          tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.default_sketching_kwargs = {
        'length': tf.constant(2, dtype=tf.int32),
        'width': tf.constant(5, dtype=tf.int32),
        'index_seeds': tf.constant([1, 1], dtype=tf.int32),
        'sign_seeds': tf.constant([2, 2], dtype=tf.int32),
    }

  def test_encode_requires_length(self):
    kwargs = self.default_sketching_kwargs
    kwargs['length'] = tf.constant(0, dtype=tf.int32)
    with self.assertRaisesRegex(ValueError, '.*[lL]ength.*positive integer.*'):
      count_sketching_utils.encode(tf.constant(1.0), **kwargs)

  def test_encode_length_required_scalar(self):
    kwargs = self.default_sketching_kwargs
    kwargs['length'] = tf.constant([1, 1], dtype=tf.int32)
    with self.assertRaisesRegex(ValueError, '.*[lL]ength.*scalar.*'):
      count_sketching_utils.encode(tf.constant(1.0), **kwargs)

  def test_encode_requires_width(self):
    kwargs = self.default_sketching_kwargs
    kwargs['width'] = tf.constant(0, dtype=tf.int32)
    with self.assertRaisesRegex(ValueError, '.*[wW]idth.*positive integer.*'):
      count_sketching_utils.encode(tf.constant(1.0), **kwargs)

  def test_encode_width_required_scalar(self):
    kwargs = self.default_sketching_kwargs
    kwargs['width'] = tf.constant([1, 1], dtype=tf.int32)
    with self.assertRaisesRegex(ValueError, '.*[wW]idth.*scalar.*'):
      count_sketching_utils.encode(tf.constant(1.0), **kwargs)

  def test_encode_parallel_iterations_error(self):
    kwargs = self.default_sketching_kwargs
    gradient = tf.constant([1.0, 2.0], dtype=tf.float16)

    with self.assertRaisesRegex(ValueError,
                                '.*parallel_iterations.*must be >= 0*'):
      count_sketching_utils.encode(gradient, **kwargs, parallel_iterations=-1)

  def test_encode_gradient_dimension_error(self):
    kwargs = self.default_sketching_kwargs
    gradient = tf.constant([[1.0, 2.0]], dtype=tf.float16)

    with self.assertRaisesRegex(ValueError,
                                '.*[gG]radient.*vector or scalar.*'):
      count_sketching_utils.encode(gradient, **kwargs)

  def test_encode_output_type(self):
    kwargs = self.default_sketching_kwargs
    rng = np.random.default_rng(seed=8)
    dtype = tf.float16
    gradient = tf.constant(rng.uniform(-5.0, 5.0, size=25), dtype=dtype)

    sketch = count_sketching_utils.encode(gradient, **kwargs)

    self.assertAllEqual(sketch.shape, [kwargs['length'], kwargs['width']])
    self.assertDTypeEqual(sketch, dtype.as_numpy_dtype())

  # Below, we test that the encode function operates as desired with respect to
  # the underlying randomness. Given this test and the next operate as desired,
  # the decode methods can be tested separately.
  @parameterized.named_parameters([('seed_2', 2), ('seed_3', 3), ('seed_8', 8)])
  def test_encode_rows_different(self, seed):
    kwargs = self.default_sketching_kwargs
    kwargs['length'] = tf.constant(2, dtype=tf.int32)
    kwargs['sign_seeds'] += [0, seed]
    kwargs['index_seeds'] += [0, seed]
    rng = np.random.default_rng(seed=2)
    dtype = tf.float16
    gradient = tf.constant(rng.uniform(-5.0, 5.0, size=2), dtype=dtype)

    sketch = count_sketching_utils.encode(gradient, **kwargs)

    self.assertNotAllEqual(
        sketch[0], sketch[1],
        'Rows (repeats) of the sketch must be pair-wise independent.')

  @parameterized.named_parameters([('seed_2', 2), ('seed_3', 3), ('seed_8', 8)])
  def test_encode_sketch_follows_uniform_distribution(self, seed):
    kwargs = self.default_sketching_kwargs
    kwargs['length'] = tf.constant(100, tf.int32)
    kwargs['width'] = tf.constant(100, tf.int32)
    rng = np.random.default_rng(seed=seed)
    gradient = tf.constant(rng.uniform(-5.0, 5.0, size=[1]), dtype=tf.float16)

    sketch = count_sketching_utils.encode(gradient, **kwargs)

    _, nonzero_col_indices = np.nonzero(sketch)
    _, pval = scipy.stats.kstest(nonzero_col_indices, 'uniform')

    self.assertLess(
        pval, 1e-5,
        'Values across hashes should follow a uniform distribution.')

  @parameterized.named_parameters([('seed_2', 2), ('seed_3', 3), ('seed_8', 8)])
  def test_encode_sketch_uses_signs_correctly(self, seed):
    kwargs = self.default_sketching_kwargs
    kwargs['length'] = tf.constant(100, tf.int32)
    kwargs['width'] = tf.constant(100, tf.int32)
    sketch_length, sketch_width = kwargs.pop('length'), kwargs.pop('width')
    rng = np.random.default_rng(seed=seed)
    dtype = tf.float16
    gradient = tf.constant(rng.uniform(-5.0, 5.0, size=[1]), dtype=dtype)

    sketch = count_sketching_utils.encode(gradient, sketch_length, sketch_width,
                                          **kwargs)

    nonzero_row_indices, nonzero_col_indices = np.nonzero(sketch)
    sketch_nonzero = sketch.numpy()[nonzero_row_indices, nonzero_col_indices]

    sketch_length, sketch_width = sketch_length.numpy(), sketch_width.numpy()
    expected_sketch_value = gradient.numpy() / tf.cast(sketch_length,
                                                       tf.float16).numpy()
    self.assertAllClose(
        tf.abs(sketch_nonzero),
        tf.ones([sketch_width], dtype) * tf.abs(expected_sketch_value),
        msg='Count sketch with a single element contained values other than '
        '+/-gradient.')
    self.assertEqual((sketch_nonzero == expected_sketch_value).any(), 1,
                     'Sketch should contain positive gradient value.')
    self.assertEqual((sketch_nonzero == -expected_sketch_value).any(), 1,
                     'Sketch should contain negative gradient value.')

  @parameterized.named_parameters([('seed_2', 2), ('seed_3', 3), ('seed_8', 8)])
  def test_encode_sketch_encodes_single_element_per_row(self, seed):
    kwargs = self.default_sketching_kwargs
    kwargs['length'] = tf.constant(100, tf.int32)
    kwargs['width'] = tf.constant(100, tf.int32)
    sketch_length = kwargs.pop('length')
    rng = np.random.default_rng(seed=seed)
    gradient = tf.constant(rng.uniform(-5.0, 5.0, size=[1]), dtype=tf.float16)

    sketch = count_sketching_utils.encode(gradient, sketch_length, **kwargs)

    nonzero_row_indices, _ = np.nonzero(sketch)

    self.assertAllEqual(
        nonzero_row_indices, np.arange(sketch_length),
        'count sketch with a single element should constain one nonzero entry '
        'per row.')


class GradientVectorCountSketchDecodeTest(parameterized.TestCase,
                                          tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.default_sketching_kwargs = {
        'index_seeds': tf.constant([1, 1], dtype=tf.int32),
        'sign_seeds': tf.constant([1, 1], dtype=tf.int32),
    }

  @parameterized.named_parameters([('method_mean', 'mean'),
                                   ('method_median', 'median')])
  def test_decode_threshold_error(self, method):
    kwargs = self.default_sketching_kwargs
    sketch = tf.zeros((2, 5))

    with self.assertRaisesRegex(ValueError, '.*[tT]hreshold.*positive float.*'):
      count_sketching_utils.decode(
          sketch, 10, **kwargs, method=method, threshold=-1e-6)

  @parameterized.named_parameters([
      ('rank_0_method_mean', 0, count_sketching_utils.DecodeMethod.MEAN),
      ('rank_1_method_mean', 1, count_sketching_utils.DecodeMethod.MEAN),
      ('rank_3_method_mean', 3, count_sketching_utils.DecodeMethod.MEAN),
      ('rank_0_method_median', 0, count_sketching_utils.DecodeMethod.MEDIAN),
      ('rank_1_method_median', 1, count_sketching_utils.DecodeMethod.MEDIAN),
      ('rank_3_method_median', 3, count_sketching_utils.DecodeMethod.MEDIAN),
  ])
  def test_decode_sketch_shape_error(self, rank, method):
    kwargs = self.default_sketching_kwargs
    sketch_shape = [1] * rank
    sketch = tf.zeros(sketch_shape)

    with self.assertRaises(ValueError):
      count_sketching_utils.decode(sketch, 10, **kwargs, method=method)

  @parameterized.named_parameters([
      ('method_mean', count_sketching_utils.DecodeMethod.MEAN),
      ('method_median', count_sketching_utils.DecodeMethod.MEDIAN),
  ])
  def test_decode_threshold_nearly_matches_no_threshold(self, method):
    kwargs = self.default_sketching_kwargs
    sketch_width = 25
    sketch_length = 10
    rng = np.random.default_rng(seed=8)
    sketch = rng.uniform(
        -5.0, 5.0, size=[sketch_length, sketch_width]).astype(np.float32)
    gradient_length = 100
    threshold = np.median(np.abs(sketch))

    gradient_estimate_thresholded = count_sketching_utils.decode(
        sketch, gradient_length, **kwargs, method=method, threshold=threshold)
    gradient_estimate_not_thresholded = count_sketching_utils.decode(
        sketch, gradient_length, **kwargs, method=method)

    exact_thresholded = np.copy(gradient_estimate_not_thresholded.numpy())
    exact_thresholded[np.abs(exact_thresholded) < threshold] = 0.0
    self.assertAllClose(
        gradient_estimate_thresholded,
        exact_thresholded,
        msg='Incorrect values were thresholded.')


class GradientVectorCountSketchEncodeDecodeIntegrationTest(
    parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.default_sketching_kwargs = {
        'index_seeds': tf.constant([1, 1], dtype=tf.int32),
        'sign_seeds': tf.constant([1, 1], dtype=tf.int32),
    }

  @parameterized.product(
      seed=[2, 8],
      method=[
          count_sketching_utils.DecodeMethod.MEAN,
          count_sketching_utils.DecodeMethod.MEDIAN
      ],
      gradient_length=[10, 20],
  )
  def test_encode_decode_no_collision(self, seed, method, gradient_length):
    kwargs = self.default_sketching_kwargs
    kwargs['length'] = tf.constant(1, tf.int32)
    kwargs['width'] = tf.constant(gradient_length * 100, tf.int32)
    rng = np.random.default_rng(seed=seed)
    n_hot = int(gradient_length * 0.2)
    indices = rng.choice(np.arange(gradient_length), size=[n_hot, 1])
    updates = tf.ones([n_hot])
    gradient = tf.scatter_nd(indices, updates, shape=[gradient_length])

    gradient_estimate = perform_encode_decode(gradient, gradient_length, kwargs,
                                              method)

    self.assertAllClose(
        gradient_estimate,
        gradient,
        msg=f'Exact gradient reconstruction using method: `{method}` failed.')

  @parameterized.named_parameters([
      ('seed_2_length_10', 2, 10),
      ('seed_2_length_20', 2, 20),
      ('seed_3_length_10', 3, 10),
      ('seed_3_length_20', 3, 20),
      ('seed_8_length_10', 8, 10),
      ('seed_8_length_20', 8, 20),
  ])
  def test_encode_decode_approximate_under_collision_mean(
      self, seed, gradient_length):
    kwargs = self.default_sketching_kwargs
    kwargs['length'] = tf.constant(gradient_length * 30, tf.int32)
    kwargs['width'] = tf.constant(gradient_length * 5, tf.int32)
    rng = np.random.default_rng(seed=seed)
    gradient = tf.constant(rng.normal(size=[gradient_length]))

    gradient_estimate = perform_encode_decode(
        gradient, gradient_length, kwargs,
        count_sketching_utils.DecodeMethod.MEAN)

    self.assertAllClose(
        gradient_estimate,
        gradient,
        atol=0.15,
        msg='Approximate gradient reconstruction using the `mean` method did '
        'not approach the original gradient.')

  @parameterized.named_parameters([
      ('seed_2_length_21', 2, 21),
      ('seed_2_length_20', 2, 20),
      ('seed_3_length_21', 3, 21),
      ('seed_3_length_20', 3, 20),
      ('seed_8_length_21', 8, 21),
      ('seed_8_length_20', 8, 20),
  ])
  def test_encode_decode_approximate_under_collision_median(
      self, seed, gradient_length):
    kwargs = self.default_sketching_kwargs
    kwargs['length'] = tf.constant(gradient_length, tf.int32)
    kwargs['width'] = tf.constant(gradient_length, tf.int32)
    rng = np.random.default_rng(seed=seed)
    n_hot = int(0.3 * gradient_length)
    indices = rng.choice(np.arange(gradient_length), size=[n_hot, 1])
    updates = rng.normal(size=[n_hot])
    gradient = tf.scatter_nd(indices, updates, shape=[gradient_length])

    gradient_estimate = perform_encode_decode(
        gradient, gradient_length, kwargs,
        count_sketching_utils.DecodeMethod.MEDIAN)

    self.assertAllClose(
        gradient_estimate,
        gradient,
        atol=0.02,
        msg='Approximate gradient reconstruction using the `median` method did '
        'not approach the original gradient.')

  @parameterized.product(
      chunk_size=[1, 5, 10],
      gradient_length=[10, 20],
      sketch_length=[5, 6],
      seed=[3, 8])
  def test_encode_decode_batch_esimate_exact_correct_no_collision(
      self, chunk_size, gradient_length, sketch_length, seed):
    kwargs = self.default_sketching_kwargs
    sketch_length = tf.constant(sketch_length, tf.int32)
    kwargs['length'] = sketch_length
    kwargs['width'] = tf.constant(int(gradient_length * 5), tf.int32)
    chunk_size = tf.cast(chunk_size, tf.int32)
    rng = np.random.default_rng(seed=seed)
    n_hot = int(gradient_length * 0.3)
    indices = rng.choice(np.arange(gradient_length), size=[n_hot, 1])
    updates = tf.ones([n_hot])
    gradient = tf.scatter_nd(indices, updates, shape=[gradient_length])

    sketch = count_sketching_utils.encode(gradient, **kwargs)

    batch_estimate = functools.partial(
        count_sketching_utils._get_batch_estimate,
        sketch=sketch,
        chunk_size=chunk_size,
        padded_gradient_length=chunk_size,
        index_seeds=kwargs['index_seeds'],
        sign_seeds=kwargs['sign_seeds'])

    batch_id = tf.constant(0, tf.int32)
    gradient_estimate = batch_estimate(batch_id)

    self.assertAllClose(
        gradient_estimate,
        gradient[:chunk_size],
        atol=0.02,
        msg='Approximate parital gradient reconstruction using '
        '`_get_batch_estimate` did not approach the original gradient.')

  @parameterized.product(
      chunk_size=[1, 3, 5], sketch_length=[10, 11, 5], seed=[5, 8])
  def test_encode_decode_batch_esimate_approximately_correct(
      self, chunk_size, sketch_length, seed):
    kwargs = self.default_sketching_kwargs
    gradient_length = 10
    kwargs['length'] = tf.constant(sketch_length, tf.int32)
    kwargs['width'] = tf.constant(gradient_length * 5, tf.int32)
    chunk_size = tf.cast(chunk_size, tf.int32)
    rng = np.random.default_rng(seed=seed)
    n_hot = int(0.3 * gradient_length)
    indices = rng.choice(np.arange(gradient_length), size=[n_hot, 1])
    updates = rng.normal(size=[n_hot])
    gradient = tf.scatter_nd(indices, updates, shape=[gradient_length])

    sketch = count_sketching_utils.encode(gradient, **kwargs)

    batch_estimate = functools.partial(
        count_sketching_utils._get_batch_estimate,
        sketch=sketch,
        chunk_size=chunk_size,
        padded_gradient_length=chunk_size,
        index_seeds=kwargs['index_seeds'],
        sign_seeds=kwargs['sign_seeds'])

    batch_id = tf.constant(0, tf.int32)
    gradient_estimate = batch_estimate(batch_id)

    self.assertAllClose(
        gradient_estimate,
        gradient[:chunk_size],
        atol=0.02,
        msg='Approximate parital gradient reconstruction using '
        '`_get_batch_estimate` did not approach the original gradient.')


if __name__ == '__main__':
  tf.test.main()
