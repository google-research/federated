# Copyright 2021, Google LLC.
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
"""Tests for SQKR."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scipy import stats
from mean_estimation import sqkr


class SqkrTest(parameterized.TestCase):

  def test_kashin_representation_is_reconstructible(self):
    """Test whether vector x can be recovered from its Kashin's representation.
    """
    for n in [50, 100]:
      for d in [50, 100]:
        x = np.random.normal(1, 1, (d, n))
        kashin_dim = 2 * d
        frame = stats.ortho_group.rvs(dim=kashin_dim).T[:, 0:d]
        [k_rpst, _] = sqkr.kashin_representation(x, frame)
        x_hat = frame.T @ k_rpst
        self.assertLessEqual(np.mean(np.abs(x - x_hat)), 0.0001)

  def test_rand_quantize_has_correct_range(self):
    """Test whether each coordinate of x is +1/-1."""
    for n in [50, 100]:
      for d in [50, 100]:
        x = np.random.uniform(-1, 1, (d, n))
        x_quantized = sqkr.rand_quantize(x, 1)
        self.assertLessEqual(
            np.max(np.abs(x_quantized) - np.ones((d, n))), 0.00001)

  @parameterized.named_parameters(("1", 1), ("2", 2), ("5", 5), ("10", 10))
  def test_rand_sampling_is_k_sparse(self, k):
    """Test whether the sampling matrix is k-sparse."""
    n = 100
    d = 100
    x = np.random.uniform(-1, 1, (d, n))
    [_, sampling_mtrx, _] = sqkr.rand_sampling(x, k)
    mean_abs_error = np.mean(
        np.abs(np.sum(sampling_mtrx, axis=0) - k * np.ones(n)))
    self.assertLessEqual(mean_abs_error, 0.00001)

  @parameterized.named_parameters(("1", 1), ("2", 2), ("5", 5), ("10", 10))
  def test_krr_remains_k_sparse(self, k):
    """Test whether the privatized vector remains k-sparse."""
    n = 100
    d = 100
    x = np.random.uniform(-1, 1, (d, n))
    x_quantized = sqkr.rand_quantize(x, 1)
    [spl_mtrx_list, _, x_sampled] = sqkr.rand_sampling(x_quantized, k)
    x_krred = sqkr.krr(k, 1, x_sampled, spl_mtrx_list, 1)
    nonzero_cnt = np.count_nonzero(x_krred, axis=0)
    self.assertLessEqual(np.max(nonzero_cnt), k)


if __name__ == "__main__":
  absltest.main()
