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
"""DP Accounting with composition for Gaussian and DDGaussian."""

import math

import numpy as np
from scipy import optimize
from scipy import special
import tensorflow_privacy as tfp

RDP_ORDERS = tuple(map(float, range(2, 21)))
RDP_ORDERS += (22., 24., 28., 32., 64., 128., 256.)
DIV_EPSILON = 1e-22


def log_comb(n, k):
  gammaln = special.gammaln
  return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _compute_rdp_subsampled(alpha, gamma, eps, upper_bound=True):
  """Computes RDP with subsampling.

  Reference: http://proceedings.mlr.press/v97/zhu19c/zhu19c.pdf.

  Args:
    alpha: The RDP order.
    gamma: The subsampling probability.
    eps: The RDP function taking alpha as input.
    upper_bound: A bool indicating whether to use Theorem 5 of the referenced
      paper above (if set to True) or Theorem 6 (if set to False).

  Returns:
    The RDP with subsampling.
  """
  if isinstance(alpha, float):
    assert alpha.is_integer()
    alpha = int(alpha)
  assert alpha > 1
  assert 0 < gamma <= 1

  if upper_bound:
    a = [0, eps(2)]
    b = [((1 - gamma)**(alpha - 1)) * (alpha * gamma - gamma + 1),
         special.comb(alpha, 2) * (gamma**2) * (1 - gamma)**(alpha - 2)]

    for l in range(3, alpha + 1):
      a.append((l - 1) * eps(l) + log_comb(alpha, l) +
               (alpha - l) * np.log(1 - gamma) + l * np.log(gamma))
      b.append(3)

  else:
    a = [0]
    b = [((1 - gamma)**(alpha - 1)) * (alpha * gamma - gamma + 1)]

    for l in range(2, alpha + 1):
      a.append((l - 1) * eps(l) + log_comb(alpha, l) +
               (alpha - l) * np.log(1 - gamma) + l * np.log(gamma))
      b.append(1)

  return special.logsumexp(a=a, b=b) / (alpha - 1)


#####################################
######## Gaussian Accounting ########
#####################################


def guass_noise_stddev_direct(epsilon, delta, norm_bound, tol=1.e-12):
  """Compute the stddev for the Gaussian mechanism with the given DP params.

  Calibrate a Gaussian perturbation for differential privacy using the
  analytic Gaussian mechanism of [Balle and Wang, ICML'18].

  Reference: http://proceedings.mlr.press/v80/balle18a/balle18a.pdf.

  Arguments:
    epsilon: Target epsilon (epsilon > 0).
    delta: Target delta (0 < delta < 1).
    norm_bound: Upper bound on L2 global sensitivity (norm_bound >= 0).
    tol: Error tolerance for binary search (tol > 0).

  Returns:
    sigma: Standard deviation of Gaussian noise needed to achieve
      (epsilon,delta)-DP under the given norm_bound.
  """

  exp = math.exp
  sqrt = math.sqrt

  def phi(t):
    return 0.5 * (1.0 + special.erf(float(t) / sqrt(2.0)))

  def case_one(eps, s):
    return phi(sqrt(eps * s)) - exp(eps) * phi(-sqrt(eps * (s + 2.0)))

  def case_two(eps, s):
    return phi(-sqrt(eps * s)) - exp(eps) * phi(-sqrt(eps * (s + 2.0)))

  def doubling_trick(predicate_stop, s_inf, s_sup):
    while not predicate_stop(s_sup):
      s_inf = s_sup
      s_sup = 2.0 * s_inf
    return s_inf, s_sup

  def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
    s_mid = s_inf + (s_sup - s_inf) / 2.0
    while not predicate_stop(s_mid):
      if predicate_left(s_mid):
        s_sup = s_mid
      else:
        s_inf = s_mid
      s_mid = s_inf + (s_sup - s_inf) / 2.0
    return s_mid

  delta_thr = case_one(epsilon, 0.0)

  if delta == delta_thr:
    alpha = 1.0

  else:
    if delta > delta_thr:
      predicate_stop_dt = lambda s: case_one(epsilon, s) >= delta
      function_s_to_delta = lambda s: case_one(epsilon, s)
      predicate_left_bs = lambda s: function_s_to_delta(s) > delta
      function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)

    else:
      predicate_stop_dt = lambda s: case_two(epsilon, s) <= delta
      function_s_to_delta = lambda s: case_two(epsilon, s)
      predicate_left_bs = lambda s: function_s_to_delta(s) < delta
      function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

    predicate_stop_bs = lambda s: abs(function_s_to_delta(s) - delta) <= tol

    s_inf, s_sup = doubling_trick(predicate_stop_dt, 0.0, 1.0)
    s_final = binary_search(predicate_stop_bs, predicate_left_bs, s_inf, s_sup)
    alpha = function_s_to_alpha(s_final)

  sigma = alpha * norm_bound / sqrt(2.0 * epsilon)
  return sigma


def get_eps_gaussian(q, noise_multiplier, steps, target_delta, orders):
  """Compute eps for the Gaussian mechanism given the DP params."""
  rdp = tfp.compute_rdp(
      q=q, noise_multiplier=noise_multiplier, steps=steps, orders=orders)
  eps, _, _ = tfp.get_privacy_spent(orders, rdp, target_delta=target_delta)
  return eps


def get_gauss_noise_multiplier(target_eps,
                               target_delta,
                               target_sampling_rate,
                               steps,
                               orders=RDP_ORDERS):
  """Compute the Gaussian noise multiplier given the DP params."""

  def get_eps_for_noise_multiplier(z):
    eps = get_eps_gaussian(
        q=target_sampling_rate,
        noise_multiplier=z,
        steps=steps,
        target_delta=target_delta,
        orders=orders)
    return eps

  def opt_fn(z):
    return get_eps_for_noise_multiplier(z) - target_eps

  min_nm, max_nm = 0.001, 1000
  result, r = optimize.brentq(opt_fn, min_nm, max_nm, full_output=True)
  if r.converged:
    return result
  else:
    return -1


############################################################
######## (Distributed) Discrete Gaussian Accounting ########
############################################################


def compute_rdp_discrete_gaussian_simplified(q, l2_scale, tau, dimension, steps,
                                             orders):
  """Compute RDP of the Sampled (Distributed) Discrete Gaussian Mechanism.

  See Proposition 14 / Eq. 17 (Page 16) of the main paper. This function omits
  the RDP term with L1 sensitivity.

  Args:
    q: The sampling rate.
    l2_scale: The l2 scale of the discrete Gaussian mechanism (i.e.,
      l2_sensitivity/stddev). For distributed version, stddev is the noise
      variance after summing all the noise shares.
    tau: The inflation parameter due to adding multiple discrete Gaussians. Set
      to zero when analyzing the the discrete Gaussian mechanism. For the
      distributed discrete Gaussian mechanisn, see Theorem 1.
    dimension: The dimension of the vector query.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders, must all be greater than 1. If
      provided orders are not integers, they are rounded down to the nearest
      integer.

  Returns:
    The RDPs at all orders, can be np.inf.
  """
  orders = [int(order) for order in orders]

  def eps(order):
    dim = dimension
    return _compute_rdp_discrete_gaussian_simplified(l2_scale, tau, dim, order)

  if q == 1:
    rdp = np.array([eps(order) for order in orders])
  else:
    rdp = np.array([
        min(_compute_rdp_subsampled(order, q, eps), eps(order))
        for order in orders
    ])

  return rdp * steps


def _compute_rdp_discrete_gaussian_simplified(l2_scale, tau, dimension, order):
  """See Proposition 14 / Eq. 17 (Page 16) of the main paper."""
  assert order >= 1, "alpha must be greater than or equal to 1."
  term_1 = order * (l2_scale**2) / 2.0 + tau * dimension
  term_2 = (order / 2.0) * (l2_scale + math.sqrt(dimension) * tau)**2
  return min(term_1, term_2)


def compute_l2_sensitivy_squared(l2_clip_norm, gamma, beta, dimension):
  """See Theorem 1 of the paper."""
  term_1 = math.pow(l2_clip_norm + gamma * math.sqrt(dimension), 2)
  if beta is None or beta == 0:
    return term_1

  term_2 = math.pow(l2_clip_norm, 2)
  term_2 += 0.25 * math.pow(gamma, 2) * dimension
  term_2 += math.sqrt(2.0 * math.log(1 / beta)) * gamma * (
      l2_clip_norm + 0.5 * gamma * math.sqrt(dimension))
  return min(term_1, term_2)


def get_ddgauss_epsilon(q,
                        noise_stddev,
                        l2_clip_norm,
                        gamma,
                        beta,
                        steps,
                        num_clients,
                        dimension,
                        delta,
                        orders=RDP_ORDERS):
  """Computes the overall epsilon. See Theorem 1 of the paper."""
  variance = math.pow(noise_stddev, 2)
  l2_sensitivity_squared = compute_l2_sensitivy_squared(l2_clip_norm, gamma,
                                                        beta, dimension)
  sq_l2_scale = l2_sensitivity_squared / (num_clients * variance + DIV_EPSILON)
  l2_scale = math.sqrt(sq_l2_scale)

  tau = 0
  for k in range(1, num_clients):
    tau += math.exp(-2 * (variance / (math.pow(gamma, 2) + DIV_EPSILON)) *
                    math.pi**2 * (k * 1.0 / (k + 1)))
  tau *= 10

  rdp = compute_rdp_discrete_gaussian_simplified(q, l2_scale, tau, dimension,
                                                 steps, orders)
  eps, _, _ = tfp.get_privacy_spent(orders, rdp, target_delta=delta)
  return eps


def get_ddgauss_noise_stddev(q,
                             epsilon,
                             l2_clip_norm,
                             gamma,
                             beta,
                             steps,
                             num_clients,
                             dimension,
                             delta,
                             orders=RDP_ORDERS):
  """Selects the local stddev for the distributed discrete Gaussian."""

  def stddev_opt_fn(z):
    cur_epsilon = get_ddgauss_epsilon(q, z, l2_clip_norm, gamma, beta, steps,
                                      num_clients, dimension, delta, orders)
    return (epsilon - cur_epsilon)**2

  stddev_result = optimize.minimize_scalar(stddev_opt_fn)
  if stddev_result.success:
    return stddev_result.x
  else:
    return -1


def get_ddgauss_gamma(q,
                      epsilon,
                      l2_clip_norm,
                      bits,
                      num_clients,
                      dimension,
                      delta,
                      beta,
                      steps,
                      k=1,
                      rho=1,
                      sqrtn_norm_growth=False,
                      orders=RDP_ORDERS):
  """Selects gamma from the given parameters.

  Args:
    q: The sampling factor.
    epsilon: The target overall epsilon.
    l2_clip_norm: The l2 clipping norm for the client vectors.
    bits: The number of bits per coordinate for the aggregated noised vector.
    num_clients: The number of clients per step.
    dimension: The dimension of the vector query.
    delta: The target delta.
    beta: The constant in [0, 1) controlling conditional randomized rounding.
      See Proposition 22 of the paper.
    steps: The total number of steps.
    k: The number of standard deviations of the signal to bound (see Thm. 34 /
      Eq. 61 of the paper).
    rho: The flatness parameter of the random rotation (see Lemma 29).
    sqrtn_norm_growth: A bool indicating whether the norm of the sum of the
      vectors grow at a rate of `sqrt(n)` (i.e. norm(sum_i x_i) <= sqrt(n) * c).
      If `False`, we use the upper bound `norm(sum_i x_i) <= n * c`. See also
      Eq. 61 of the paper.
    orders: The RDP orders.

  Returns:
    The selected gamma.
  """
  if sqrtn_norm_growth:
    n_factor = num_clients
  else:
    n_factor = num_clients**2

  def stddev(x):
    return get_ddgauss_noise_stddev(q, epsilon, l2_clip_norm, x, beta, steps,
                                    num_clients, dimension, delta, orders)

  def mod_min(x):
    return k * math.sqrt(rho / dimension * l2_clip_norm**2 * n_factor +
                         (x**2 / 4.0 + stddev(x)**2) * num_clients)

  def gamma_opt_fn(x):
    return (math.pow(2, bits) - 2 * mod_min(x) / (x + DIV_EPSILON))**2

  gamma_result = optimize.minimize_scalar(gamma_opt_fn)
  if gamma_result.success:
    return gamma_result.x
  else:
    return -1
