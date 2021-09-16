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
"""Tests for privunit."""

from absl.testing import absltest
import numpy as np
from rcc_dp import privunit


class PrivunitTest(absltest.TestCase):

  def test_gamma_is_in_range(self):
    """Test whether gamma adheres to (16a) or (16b) in the original paper."""
    eps_space = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    d_space = [100, 1000, 10000]
    for eps in eps_space:
      for d in d_space:
        gamma, _ = privunit.find_best_gamma(d, eps)
        self.assertLessEqual(0, gamma)
        self.assertLessEqual(gamma, 1)
        # Test if (16a) is true.
        if gamma <= np.sqrt(np.pi /
                            (2 *
                             (d - 1))) * (np.exp(eps) - 1) / (np.exp(eps) + 1):
          flag_16a = True
        # Test if (16b) is true.
        if (eps >= 0.5 * np.log(d) + np.log(6) + np.log(gamma) -
            (d - 1) * np.log(1 - gamma**2) / 2) and (gamma >= np.sqrt(2 / d)):
          flag_16b = True
        # Test if either (16a) or (16b) is true.
        self.assertTrue(flag_16a or flag_16b)

  def test_c2_is_less_equal_c1(self):
    """Tests if c2 is less than or equal to c1."""
    eps_space = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    d_space = [100, 1000, 10000]
    p_space = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for eps in eps_space:
      for d in d_space:
        gamma, _ = privunit.find_best_gamma(d, eps)
        for p in p_space:
          c1, c2 = privunit.get_privunit_densities(d, gamma, p)
          self.assertLessEqual(c2, c1)

  def test_m_is_less_equal_one(self):
    """Tests whether the inverse norm m is less than or equal to 1."""
    eps_space = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    d_space = [100, 1000, 10000]
    p_space = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for eps in eps_space:
      for d in d_space:
        gamma, _ = privunit.find_best_gamma(d, eps)
        for p in p_space:
          m = privunit.getm(d, gamma, p)
          self.assertLessEqual(m, 1)

  def test_bias_and_norm_privunit(self):
    """Checks whether the privatized x is unbiased and has the right norm."""
    eps_space = [4, 5]
    d_space = [100]
    n = 50000
    for eps in eps_space:
      for d in d_space:
        x = np.random.normal(0, 1, (d, 1))
        x = np.divide(x, np.linalg.norm(x, axis=0).reshape(1, -1))
        x = np.repeat(x, n, axis=1)
        x_privunit, m = privunit.apply_privunit(x, eps)
        x_avg_privunit = np.mean(x_privunit, axis=1).reshape(-1, 1)
        x_mse = np.linalg.norm(
            np.mean(x, axis=1).reshape(-1, 1) - x_avg_privunit)**2
        self.assertLessEqual(x_mse, 0.01)
        x_norm = np.linalg.norm(x_privunit, axis=0) - np.ones(n) / m
        self.assertLessEqual(np.max(x_norm), 0.000001)


if __name__ == "__main__":
  absltest.main()
