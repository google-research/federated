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
"""Tests for factorize_multi_epoch_prefix_sum."""
import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import numpy as np

from multi_epoch_dp_matrix_factorization import matrix_io
from multi_epoch_dp_matrix_factorization.multiple_participations import factorize_multi_epoch_prefix_sum

FLAGS = flags.FLAGS

NUM_EPOCHS = 2
STEPS_PER_EPOCH = 3


class FactorizeMultiEpochPrefixSumTest(absltest.TestCase):

  @flagsaver.flagsaver(
      init_matrix_dir='',
      constraint_pattern='all_positive',
      matrix_to_factor='prefix_sum',
      max_iterations=6,
      steps_per_eval=3,
      num_epochs=NUM_EPOCHS,
      steps_per_epoch=STEPS_PER_EPOCH,
      run_name='test',
  )
  def test_prefix_sum_all_positive(self):
    root_dir = self.create_tempdir('root').full_path
    FLAGS.root_output_dir = root_dir

    factorize_multi_epoch_prefix_sum.main(['foo'])

    flags.FLAGS.matrix_root_path = os.path.join(root_dir, 'test')
    n = NUM_EPOCHS * STEPS_PER_EPOCH
    w_matrix, h_matrix = matrix_io.get_prefix_sum_w_h(n, matrix_io.PREFIX_OPT)
    self.assertEqual(w_matrix.dtype, np.float64)
    self.assertEqual(h_matrix.dtype, np.float64)
    self.assertEqual(w_matrix.shape, (n, n))
    self.assertEqual(h_matrix.shape, (n, n))

  @flagsaver.flagsaver(
      init_matrix_dir='',
      constraint_pattern='none',
      matrix_to_factor='momentum_with_cooldown',
      max_iterations=6,
      steps_per_eval=3,
      num_epochs=NUM_EPOCHS,
      steps_per_epoch=STEPS_PER_EPOCH,
      run_name='test',
  )
  def test_momentum_full_constraints(self):
    root_dir = self.create_tempdir('root').full_path
    FLAGS.root_output_dir = root_dir

    factorize_multi_epoch_prefix_sum.main(['foo'])

    flags.FLAGS.matrix_root_path = os.path.join(root_dir, 'test')
    n = NUM_EPOCHS * STEPS_PER_EPOCH
    matrix_path = matrix_io.get_momentum_path(n, 0.95)
    w_matrix, h_matrix, lr_vector = matrix_io.load_w_h_and_maybe_lr(matrix_path)
    self.assertLen(lr_vector, n)
    self.assertEqual(w_matrix.dtype, np.float64)
    self.assertEqual(h_matrix.dtype, np.float64)
    self.assertEqual(w_matrix.shape, (n, n))
    self.assertEqual(h_matrix.shape, (n, n))


if __name__ == '__main__':
  absltest.main()
