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
"""Tests for fixed_point_library."""

import numpy as np
import tensorflow as tf

from multi_epoch_dp_matrix_factorization import fixed_point_library


class FixedPointLibraryTest(tf.test.TestCase):

  def test_raises_non_matrix(self):
    bad_target = np.random.uniform(size=[3])
    with self.assertRaises(ValueError):
      fixed_point_library.compute_phi_fixed_point(
          bad_target, np.ones(3), iterations=1
      )

  def test_raises_nonsquare_matrix(self):
    bad_target = np.random.uniform(size=[3, 4])
    with self.assertRaises(ValueError):
      fixed_point_library.compute_phi_fixed_point(
          bad_target, np.ones(3), iterations=1
      )

  def test_raises_matrix_with_nans(self):
    random_square = np.random.uniform(size=[2, 2])
    hermitian_square = random_square.T @ random_square
    hermitian_square[0][0] = float('nan')
    with self.assertRaises(ValueError):
      fixed_point_library.compute_phi_fixed_point(
          hermitian_square, np.ones(3), iterations=1
      )

  def test_computes_fixed_point_for_three_dimensional_s(self):
    rtol = 1e-6
    s_matrix = np.tril(np.ones(shape=(3, 3)))
    target = s_matrix.T @ s_matrix
    multiplier = fixed_point_library.init_fixed_point(target)
    multiplier = fixed_point_library.compute_phi_fixed_point(
        target, multiplier, iterations=50
    )
    expected_multiplier = np.array(
        [2.651861, 1.257355, 0.698646], dtype=np.float32
    )
    self.assertAllClose(expected_multiplier, multiplier, rtol=2 * rtol)


if __name__ == '__main__':
  tf.test.main()
