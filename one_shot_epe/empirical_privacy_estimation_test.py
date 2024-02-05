# Copyright 2024, Google LLC.
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
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_federated as tff

from one_shot_epe import empirical_privacy_estimation_lib as epm


_TEST_SEED = 0xBAD5EED
_norm_logpdf = scipy.stats.norm.logpdf


class EmpiricalPrivacyEstimationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      eps=(0.5, 1, 5),
      delta=(1e-5, 1e-2),
      l2_sensitivity=(1e-2, 1, 10),
      diff=(-1e-12, 0, 1e-12),
  )
  def test_epsilon_two_gaussians_stds_close(
      self, eps, delta, l2_sensitivity, diff
  ):
    # When sigmas are numerically equal, a different code path is used. The use
    # of 'diff' here ensures that both code paths are tested. Because 'diff' is
    # small, the result should be within tolerance.
    sigma = tff.analytics.differential_privacy.analytic_gauss_stddev(
        eps, delta, l2_sensitivity
    )
    test_eps = epm.epsilon_two_gaussians(
        0, sigma, l2_sensitivity, sigma + diff, delta
    )
    self.assertAllClose(eps, test_eps)

  @parameterized.product(
      delta=(2e-2, 1e-1, 2e-1),
      trial=tuple(range(20)),
  )
  def test_epsilon_two_gaussians_stds_not_equal(self, delta, trial):
    # Uses privacy loss distribution to verify computation of epsilon from two
    # arbitrary Gaussians.
    rng = np.random.default_rng(seed=_TEST_SEED + trial)
    mu1, std1, mu2, std2 = rng.uniform(0, 5, 4)

    eps = epm.epsilon_two_gaussians(mu1, std1, mu2, std2, delta)

    samples = 1000000
    ys1 = rng.normal(mu1, std1, size=samples)
    zs1 = _norm_logpdf(ys1, mu1, std1) - _norm_logpdf(ys1, mu2, std2)
    deltas1 = np.maximum(0, 1 - np.exp(eps - zs1))
    delta1 = np.mean(deltas1)

    ys2 = rng.normal(mu2, std2, size=samples)
    zs2 = _norm_logpdf(ys2, mu2, std2) - _norm_logpdf(ys2, mu1, std1)
    deltas2 = np.maximum(0, 1 - np.exp(eps - zs2))
    delta2 = np.mean(deltas2)

    max_delta = max(delta1, delta2)
    if eps == 0.0:
      # If eps is zero, delta bound can be loose.
      self.assertLess(max_delta, delta)
    else:
      self.assertNear(max_delta, delta, 1e-2)

  def test_optimal_epsilon_lower_bound_regression(self):
    sigma = 1.0
    num_samples = 100
    alpha = 1e-2
    delta = 1e-2
    rng = np.random.default_rng(seed=_TEST_SEED)
    cosines = rng.normal(loc=sigma, scale=sigma, size=num_samples)
    unseen = rng.normal(loc=0.0, scale=sigma, size=num_samples)
    eps = epm.optimal_epsilon_lower_bound(cosines, unseen, alpha, delta)
    expected_eps = 0.6839449
    self.assertAllClose(eps, expected_eps)

  @parameterized.product(
      sigma=(0.1, 0.5, 2.0),
      delta=(1e-5, 1e-2),
  )
  def test_lower_bound_increases_with_alpha(self, sigma, delta):
    num_samples = 100
    rng = np.random.default_rng(seed=_TEST_SEED)
    cosines = rng.normal(1.0, sigma, num_samples)
    unseen = rng.normal(0, sigma, num_samples)
    eps_bounds = [
        epm.optimal_epsilon_lower_bound(cosines, unseen, alpha, delta)
        for alpha in np.logspace(-3, -1, 10)
    ]
    self.assertAllGreaterEqual(np.diff(eps_bounds), 0)

  @parameterized.product(
      sigma=(0.1, 0.5, 2.0),
      alpha=(1e-3, 1e-2),
  )
  def test_lower_bound_decreases_with_delta(self, sigma, alpha):
    num_samples = 100
    rng = np.random.default_rng(seed=_TEST_SEED)
    cosines = rng.normal(1.0, sigma, num_samples)
    unseen = rng.normal(0, sigma, num_samples)
    eps_bounds = [
        epm.optimal_epsilon_lower_bound(cosines, unseen, alpha, delta)
        for delta in np.logspace(-6, -1, 10)
    ]
    self.assertAllLessEqual(np.diff(eps_bounds), 0)

  @parameterized.named_parameters(
      ('low_dim', 200, 1.394838),
      ('high_dim', 2000, 1.3880321),
  )
  def test_lower_bound_closed_form_null_regression(self, dim, expected_eps):
    sigma = dim**-0.5
    num_samples = 100
    alpha = 1e-2
    delta = 1e-2
    rng = np.random.default_rng(seed=_TEST_SEED)
    cosines = rng.normal(loc=sigma, scale=sigma, size=num_samples)
    eps = epm.optimal_epsilon_lower_bound_closed_form_null(
        cosines, dim, alpha, delta
    )
    self.assertAllClose(eps, expected_eps)

  @parameterized.product(
      delta=(1e-5, 1e-2),
      dim=(200, 2000),
  )
  def test_lower_bound_closed_form_null_increases_with_alpha(self, delta, dim):
    num_samples = 100
    rng = np.random.default_rng(seed=_TEST_SEED)
    cosines = rng.uniform(1e-1, 2e-1, num_samples)
    eps_bounds = [
        epm.optimal_epsilon_lower_bound_closed_form_null(
            cosines, dim, alpha, delta
        )
        for alpha in np.logspace(-3, -1, 10)
    ]
    self.assertAllGreaterEqual(np.diff(eps_bounds), 0)

  @parameterized.product(
      sigma=(0.1, 0.5, 2.0),
      alpha=(1e-3, 1e-2),
      delta=(1e-5, 1e-2),
  )
  def test_lower_bound_invariant_to_affine_transform(self, sigma, alpha, delta):
    num_samples = 100
    rng = np.random.default_rng(seed=_TEST_SEED)
    cosines = rng.normal(1.0, sigma, num_samples)
    unseen = rng.normal(0, sigma, num_samples)
    eps1 = epm.optimal_epsilon_lower_bound(cosines, unseen, alpha, delta)
    shift = 3.14159
    scale = 2.71828
    cosines = cosines * scale + shift
    unseen = unseen * scale + shift
    eps2 = epm.optimal_epsilon_lower_bound(cosines, unseen, alpha, delta)
    self.assertAllClose(eps1, eps2)

  @parameterized.product(
      alpha=(1e-3, 1e-2),
      dim=(200, 2000),
  )
  def test_lower_bound_closed_form_null_decreases_with_delta(self, alpha, dim):
    num_samples = 100
    rng = np.random.default_rng(seed=_TEST_SEED)
    cosines = rng.uniform(1e-1, 2e-1, num_samples)
    eps_bounds = [
        epm.optimal_epsilon_lower_bound_closed_form_null(
            cosines, dim, alpha, delta
        )
        for delta in np.logspace(-6, -1, 10)
    ]
    self.assertAllLessEqual(np.diff(eps_bounds), 0)

  @parameterized.product(
      delta=(1e-5, 1e-2),
  )
  def test_two_gaussians_invariant_to_affine_transform(self, delta):
    rng = np.random.default_rng(seed=_TEST_SEED)
    mu0, std0, mu1, std1 = (0.0, 1.0, 1.0, 1.0) + rng.uniform(-0.2, 0.2, 4)
    eps1 = epm.epsilon_two_gaussians(mu0, std0, mu1, std1, delta)
    shift = 3.14159
    scale = 2.71828
    mu0 = mu0 * scale + shift
    mu1 = mu1 * scale + shift
    std0 *= scale
    std1 *= scale
    eps2 = epm.epsilon_two_gaussians(mu0, std0, mu1, std1, delta)
    self.assertAllClose(eps1, eps2)

  def test_closed_form_null_calculator(self):
    # For dim == 999, the exact null distribution should be very close to
    # Gaussian. This tests the implementation of the exact pdf as well as our
    # assumption that we can substitute the Gaussian for higher dims.
    dim = 999
    calculator = epm.ClosedFormNullCalculator(dim)
    x_vals = np.arange(-0.3, 0.3, 100)

    exact_pdf = calculator.pdf(x_vals)
    gauss_pdf = scipy.stats.norm.pdf(x_vals, scale=dim**-0.5)
    self.assertAllClose(exact_pdf, gauss_pdf)

    exact_cdf = [calculator.cdf(x) for x in x_vals]
    gauss_cdf = scipy.stats.norm.cdf(x_vals, scale=dim**-0.5)
    self.assertAllClose(exact_cdf, gauss_cdf)

    # The ppf is where the approximation is loosest. But it is only used to
    # find optimization bounds, so it doesn't have to be perfect. So we don't
    # get too close to the bounds of {0, 1}, and use a loose rtol of 1e-3.
    p_vals = np.arange(0.01, 0.99, 100)
    exact_ppf = [calculator.ppf(p) for p in p_vals]
    gauss_ppf = scipy.stats.norm.ppf(p_vals, scale=dim**-0.5)
    self.assertAllClose(exact_ppf, gauss_ppf, rtol=1e-3)

  @parameterized.named_parameters(
      ('low_dim', 200, 2.375326),
      ('high_dim', 2000, 2.317789),
  )
  def test_closed_form_null_gaussian_alt_regression(self, dim, expected_eps):
    sigma = dim**-0.5
    delta = 1e-2
    eps = epm.epsilon_closed_form_null_gaussian_alt(
        mu=sigma, std=sigma, dim=dim, delta=delta
    )
    self.assertAllClose(eps, expected_eps)

  @parameterized.product(
      dim=(200, 2000),
  )
  def test_closed_form_null_gaussian_alt_decreases_with_delta(self, dim):
    sigma = dim**-0.5
    eps_bounds = [
        epm.epsilon_closed_form_null_gaussian_alt(
            mu=sigma, std=sigma, dim=dim, delta=delta
        )
        for delta in np.logspace(-6, -1, 10)
    ]
    self.assertAllLessEqual(np.diff(eps_bounds), 0)

  @parameterized.product(
      dim=(200, 2000),
      delta=(1e-5, 1e-2),
  )
  def test_closed_form_null_gaussian_alt_increases_with_mu(self, dim, delta):
    sigma = dim**-0.5
    eps_bounds = [
        epm.epsilon_closed_form_null_gaussian_alt(mu, sigma, dim, delta)
        for mu in np.linspace(sigma / 4, 2 * sigma, 8)
    ]
    self.assertAllGreaterEqual(np.diff(eps_bounds), 0)


if __name__ == '__main__':
  absltest.main()
