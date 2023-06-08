# Copyright 2023, Google LLC.
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
"""Tests for matrix_io."""

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from multi_epoch_dp_matrix_factorization import matrix_io

_LOADED_TENSORS = (tf.zeros(shape=(4, 4)), tf.ones(shape=(4, 4)), None)


class MatrixIoTest(parameterized.TestCase, tf.test.TestCase):

  @flagsaver.flagsaver(matrix_root_path='/test_root')
  def test_get_matrix_path(self):
    self.assertEqual(
        matrix_io.get_matrix_path(n=1024, mechanism_name='foo'),
        '/test_root/foo/size=1024',
    )

  def test_infer_momentum_from_path(self):
    self.assertEqual(matrix_io.infer_momentum_from_path('momentum_0p95'), 0.95)
    self.assertEqual(matrix_io.infer_momentum_from_path('momentum_0p00'), 0.0)
    self.assertIsNone(matrix_io.infer_momentum_from_path('foo'))

  # Slightly tricky to make all these decorators play well together,
  # note extra parameter in function def, and order here matters.
  # We test both the new/preferred/named-by-constants mechanism names,
  # and the "legacy" names used by aggregator_builder.
  @parameterized.named_parameters(
      ('opt_prefix_sum_matrix_new', matrix_io.PREFIX_OPT, matrix_io.PREFIX_OPT),
      (
          'streaming_honaker_matrix_new',
          matrix_io.PREFIX_ONLINE_HONAKER,
          matrix_io.PREFIX_ONLINE_HONAKER,
      ),
      (
          'full_honaker_matrix_new',
          matrix_io.PREFIX_FULL_HONAKER,
          matrix_io.PREFIX_FULL_HONAKER,
      ),
      ('opt_prefix_sum_matrix', 'opt_prefix_sum_matrix', matrix_io.PREFIX_OPT),
      (
          'streaming_honaker_matrix',
          'streaming_honaker_matrix',
          matrix_io.PREFIX_ONLINE_HONAKER,
      ),
      (
          'full_honaker_matrix',
          'full_honaker_matrix',
          matrix_io.PREFIX_FULL_HONAKER,
      ),
  )
  @mock.patch.object(
      matrix_io, 'load_w_h_and_maybe_lr', return_value=_LOADED_TENSORS
  )
  @flagsaver.flagsaver(matrix_root_path='/test_root')
  def test_get_prefix_sum(self, mechanism_name, path_part, mock_load_w_h):
    w, h = matrix_io.get_prefix_sum_w_h(4, mechanism_name)
    self.assertAllClose(w, _LOADED_TENSORS[0])
    self.assertAllClose(h, _LOADED_TENSORS[1])
    mock_load_w_h.assert_called_once_with(f'/test_root/{path_part}/size=4')

  @parameterized.named_parameters(('_no_lr', False), ('_write_lr', True))
  @mock.patch.object(tf.io, 'write_file')
  def test_verify_and_write(self, write_lr, mock_write):
    output_dir = '/test_root'
    dim = 4
    w = tf.eye(dim, dtype=tf.float64)
    h = tf.ones(shape=(dim, dim), dtype=tf.float64)
    s = tf.ones(shape=(dim, dim), dtype=tf.float64)
    lr = tf.ones(dim) if write_lr else None
    matrix_io.verify_and_write(w, h, s, output_dir=output_dir, lr_sched=lr)
    expected_calls = [
        mock.call(output_dir + '/' + matrix_io.W_MATRIX_STRING, mock.ANY),
        mock.call(output_dir + '/' + matrix_io.H_MATRIX_STRING, mock.ANY),
    ]
    if write_lr:
      expected_calls.append(
          mock.call(output_dir + '/' + matrix_io.LR_VECTOR_STRING, mock.ANY)
      )
    mock_write.assert_has_calls(expected_calls, any_order=True)

  @mock.patch.object(tf.io, 'write_file')
  def test_verify_and_write_non_square(self, mock_write):
    output_dir = '/test_root'
    n = 4
    w = tf.ones(shape=(n, 1), dtype=tf.float64)
    h = tf.ones(shape=(1, n), dtype=tf.float64)
    s = tf.ones(shape=(n, n), dtype=tf.float64)
    lr = tf.ones(n)
    matrix_io.verify_and_write(w, h, s, output_dir=output_dir, lr_sched=lr)
    expected_calls = [
        mock.call(output_dir + '/' + matrix_io.W_MATRIX_STRING, mock.ANY),
        mock.call(output_dir + '/' + matrix_io.H_MATRIX_STRING, mock.ANY),
        mock.call(output_dir + '/' + matrix_io.LR_VECTOR_STRING, mock.ANY),
    ]
    mock_write.assert_has_calls(expected_calls, any_order=True)

  def test_verify_and_write_bad_factorization(self):
    dim = 4
    w = tf.zeros(shape=(dim, dim), dtype=tf.float64)
    h = tf.ones(shape=(dim, dim), dtype=tf.float64)
    s = tf.ones(shape=(dim, dim), dtype=tf.float64)
    with self.assertRaisesRegex(AssertionError, 'Not equal to tolerance'):
      matrix_io.verify_and_write(w, h, s, output_dir='foo/')

  def test_scale_w_h_by_single_participation_sensitivity(self):
    dim = 4
    w = np.ones(shape=(dim, dim), dtype=np.float64)
    h = np.diag(np.linspace(0.0, 3.0, num=dim, dtype=np.float64))
    w, h = matrix_io.scale_w_h_by_single_participation_sensitivity(w, h)
    np.testing.assert_allclose(w, 3 * np.ones(shape=(dim, dim)))
    np.testing.assert_allclose(h, np.diag(np.linspace(0.0, 1.0, num=dim)))


if __name__ == '__main__':
  absltest.main()
