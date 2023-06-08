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
"""Tests for contrib_matrix_builders."""
from absl.testing import absltest
import numpy as np

from multi_epoch_dp_matrix_factorization.multiple_participations import contrib_matrix_builders


class ContribMatrixBuildersTest(absltest.TestCase):

  def test_epoch_participation_matrix_4_2(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected = np.array(
        [[1,  1, 0,  0],
         [0,  0, 1,  1],
         [1, -1, 0,  0],
         [0,  0, 1, -1]])
    # pyformat: enable
    np.testing.assert_allclose(
        contrib_matrix_builders.epoch_participation_matrix(n=4, num_epochs=2),
        expected,
    )

  def test_epoch_participation_matrix_4_1(self):
    # pyformat: disable
    expected = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])
    # pyformat: enable
    np.testing.assert_allclose(
        contrib_matrix_builders.epoch_participation_matrix(n=4, num_epochs=1),
        expected,
    )

  def test_epoch_participation_matrix_3_3(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected = np.array(
        [[1,  1,  1,  1],
         [1,  1, -1, -1],
         [1, -1,  1, -1]])
    # pyformat: enable
    np.testing.assert_allclose(
        contrib_matrix_builders.epoch_participation_matrix(n=3, num_epochs=3),
        expected,
    )

  def test_epoch_participation_matrix_all_positive_4_2(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected = np.array(
        [[1, 0],
         [0, 1],
         [1, 0],
         [0, 1]])
    # pyformat: enable
    np.testing.assert_allclose(
        contrib_matrix_builders.epoch_participation_matrix_all_positive(
            n=4, num_epochs=2
        ),
        expected,
    )

  def test_epoch_participation_matrix_all_positive_2_2(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected = np.array(
        [[1],
         [1]])
    # pyformat: enable
    np.testing.assert_allclose(
        contrib_matrix_builders.epoch_participation_matrix_all_positive(
            n=2, num_epochs=2
        ),
        expected,
    )


if __name__ == '__main__':
  absltest.main()
