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
"""Tests for get_parameters."""

from absl.testing import absltest
import numpy as np
from rcc_dp import get_parameters
from rcc_dp import miracle
from rcc_dp import modify_pi


class GetParametersTest(absltest.TestCase):

  def test_unbiased_miracle_is_unbiased(self):
    """Test if unbiased miracle is unbiased."""
    number_of_budget_intervals = 99
    epsilon = 1
    d = 50
    coding_cost = 6
    n = 40000

    x = np.random.normal(0, 1, (d, 1))
    x /= np.linalg.norm(x, axis=0)
    x = np.repeat(x, n, axis=1)

    x_unbiased_miracle = np.zeros((d, n))
    c1, c2, m, gamma = get_parameters.get_parameters_unbiased_miracle(
        epsilon / 2, d, 2**coding_cost, number_of_budget_intervals)
    for i in range(n):
      k, _, _ = miracle.encoder(i, x[:, i], 2**coding_cost, c1, c2, gamma)
      z_k = miracle.decoder(i, k, d, 2**coding_cost)
      x_unbiased_miracle[:, i] = z_k / m

    x_unbiased_miracle = np.mean(x_unbiased_miracle, axis=1, keepdims=True)
    x_mse = np.linalg.norm(
        np.mean(x, axis=1, keepdims=True) - x_unbiased_miracle)**2
    self.assertLessEqual(x_mse, 0.05)

  def test_unbiased_approx_miracle_is_unbiased(self):
    """Test if unbiased approx miracle is unbiased."""
    budget = 0.5
    delta = 10**(-6)
    epsilon = 1
    d = 50
    coding_cost = 6
    n = 40000

    x = np.random.normal(0, 1, (d, 1))
    x /= np.linalg.norm(x, axis=0)
    x = np.repeat(x, n, axis=1)

    x_unbiased_approx_miracle = np.zeros((d, n))
    c1, c2, m, gamma, _ = get_parameters.get_parameters_unbiased_approx_miracle(
        epsilon, d, 2**coding_cost, budget, delta)
    for i in range(n):
      k, _, _ = miracle.encoder(i, x[:, i], 2**coding_cost, c1, c2, gamma)
      z_k = miracle.decoder(i, k, d, 2**coding_cost)
      x_unbiased_approx_miracle[:, i] = z_k / m
    x_unbiased_approx_miracle = np.mean(
        x_unbiased_approx_miracle, axis=1, keepdims=True)
    x_mse = np.linalg.norm(
        np.mean(x, axis=1, keepdims=True) - x_unbiased_approx_miracle)**2
    self.assertLessEqual(x_mse, 0.05)

  def test_unbiased_modified_miracle_is_unbiased(self):
    """Test if unbiased modified miracle is unbiased."""
    number_of_budget_intervals = 99
    epsilon = 1
    d = 50
    coding_cost = 6
    n = 40000

    x = np.random.normal(0, 1, (d, 1))
    x /= np.linalg.norm(x, axis=0)
    x = np.repeat(x, n, axis=1)

    x_unbiased_modified_miracle = np.zeros((d, n))
    c1, c2, m, gamma = get_parameters.get_parameters_unbiased_modified_miracle(
        epsilon, d, 2**coding_cost, epsilon / 2, number_of_budget_intervals)
    for i in range(n):
      _, _, pi = miracle.encoder(i, x[:, i], 2**coding_cost, c1, c2, gamma)
      pi_all = modify_pi.modify_pi(pi, epsilon / 2)
      k = np.random.choice(2**coding_cost, 1, p=pi_all[-1])[0]
      z_k = miracle.decoder(i, k, d, 2**coding_cost)
      x_unbiased_modified_miracle[:, i] = z_k / m
    x_unbiased_modified_miracle = np.mean(
        x_unbiased_modified_miracle, axis=1, keepdims=True)
    x_mse = np.linalg.norm(
        np.mean(x, axis=1, keepdims=True) - x_unbiased_modified_miracle)**2
    self.assertLessEqual(x_mse, 0.05)


if __name__ == "__main__":
  absltest.main()
