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

RDP_ORDERS = tuple(range(2, 129)) + (256,)
DIV_EPSILON = 1e-22


####################
###### Shared ######
####################


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


def rounded_l2_norm_bound(l2_norm_bound, beta, dim):
  """Computes the L2 norm bound after stochastic rounding to integers.

  Note that this function is *agnostic* to the actual vector whose coordinates
  are to be rounded, and it does *not* consider the effect of scaling (i.e.
  we assume the input norm is already scaled before rounding).

  See Theorem 1 of https://arxiv.org/pdf/2102.06387.pdf.

  Args:
    l2_norm_bound: The L2 norm (bound) of the vector whose coordinates are to be
      stochastically rounded to the integer grid.
    beta: A float constant in [0, 1). See the initializer docstring of the
      aggregator.
    dim: The dimension of the vector to be rounded.

  Returns:
    The inflated L2 norm bound after stochastic rounding (conditionally
    according to beta).
  """
  assert int(dim) == dim and dim > 0, f'Invalid dimension: {dim}'
  assert 0 <= beta < 1, 'beta must be in the range [0, 1)'
  assert l2_norm_bound > 0, 'Input l2_norm_bound should be positive.'

  bound_1 = l2_norm_bound + np.sqrt(dim)
  if beta == 0:
    return bound_1

  squared_bound_2 = np.square(l2_norm_bound) + 0.25 * dim
  squared_bound_2 += (
      np.sqrt(2.0 * np.log(1.0 / beta)) * (l2_norm_bound + 0.5 * np.sqrt(dim)))
  bound_2 = np.sqrt(squared_bound_2)
  return min(bound_1, bound_2)


def rounded_l1_norm_bound(l2_norm_bound, dim):
  # In general we have L1 <= sqrt(d) * L2. In the scaled and rounded domain
  # where coordinates are integers we also have L1 <= L2^2.
  return l2_norm_bound * min(np.sqrt(dim), l2_norm_bound)


def heuristic_scale_factor(local_stddev,
                           l2_clip,
                           bits,
                           num_clients,
                           dim,
                           k_stddevs,
                           rho=1.0):
  """Selects a scaling factor by assuming subgaussian aggregates.

  Selects scale_factor = 1 / gamma such that k stddevs of the noisy, quantized,
  aggregated client values are bounded within the bit-width. The aggregate at
  the server is assumed to follow a subgaussian distribution. See Section 4.2
  and 4.4 of https://arxiv.org/pdf/2102.06387.pdf for more details.

  Specifically, the implementation is solving for gamma using the following
  expression:

    2^b = 2k * sqrt(rho / dim * (cn)^2 + (gamma^2 / 4 + sigma^2) * n) / gamma.

  Args:
    local_stddev: The local noise standard deviation.
    l2_clip: The initial L2 clip norm. See the __init__ docstring.
    bits: The bit-width. See the __init__ docstring.
    num_clients: The expected number of clients. See the __init__ docstring.
    dim: The dimension of the client vector that includes any necessary padding.
    k_stddevs: The number of standard deviations of the noisy and quantized
      aggregate values to bound within the bit-width.
    rho: (Optional) The subgaussian flatness parameter of the random orthogonal
      transform as part of the DDP procedure. See Section 4.2 of the above paper
      for more details.

  Returns:
    The selected scaling factor in tf.float64.
  """
  c = l2_clip
  n = num_clients
  sigma = local_stddev

  if 2.0**(2.0 * bits) < n * k_stddevs**2:
    raise ValueError(f'The selected bit-width ({bits}) is too small for the '
                     f'given parameters (num_clients = {n}, k_stddevs = '
                     f'{k_stddevs}). You may decrease `num_clients`, '
                     f'increase `bits`, or decrease `k_stddevs`.')

  numer = np.sqrt(2.0**(2.0 * bits) - n * k_stddevs**2)
  denom = 2.0 * k_stddevs * np.sqrt(rho / dim * c**2 * n**2 + n * sigma**2)
  scale_factor = numer / denom
  return scale_factor


#####################################
######## Gaussian Accounting ########
#####################################


def analytic_gauss_stddev(epsilon, delta, norm_bound, tol=1.e-12):
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


##################################################
######## (Distributed) Discrete Gaussian  ########
##################################################


def compute_rdp_dgaussian_simplified(q, l2_scale, tau, dim, steps, orders):
  """Compute RDP of the Sampled (Distributed) Discrete Gaussian Mechanism."""
  orders = [int(order) for order in orders]

  def eps(order):
    """See Proposition 14 / Eq. 17 (Page 16) of the main paper."""
    assert order >= 1, 'alpha must be greater than or equal to 1.'
    term_1 = order * (l2_scale**2) / 2.0 + tau * dim
    term_2 = (order / 2.0) * (l2_scale + math.sqrt(dim) * tau)**2
    return min(term_1, term_2)

  if q == 1:
    rdp = np.array([eps(order) for order in orders])
  else:
    rdp = np.array([
        min(_compute_rdp_subsampled(order, q, eps), eps(order))
        for order in orders
    ])

  return rdp * steps


def compute_rdp_dgaussian(q, l1_scale, l2_scale, tau, dim, steps, orders):
  """Compute RDP of the Sampled (Distributed) Discrete Gaussian Mechanism.

  See Proposition 14 / Eq. 17 (Page 16) of the main paper.

  Args:
    q: The sampling rate.
    l1_scale: The l1 scale of the discrete Gaussian mechanism (i.e.,
      l1_sensitivity/stddev). For distributed version, stddev is the noise
      stddev after summing all the noise shares.
    l2_scale: The l2 scale of the discrete Gaussian mechanism (i.e.,
      l2_sensitivity/stddev). For distributed version, stddev is the noise
      stddev after summing all the noise shares.
    tau: The inflation parameter due to adding multiple discrete Gaussians. Set
      to zero when analyzing the the discrete Gaussian mechanism. For the
      distributed discrete Gaussian mechanisn, see Theorem 1.
    dim: The dimension of the vector query.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders, must all be greater than 1. If
      provided orders are not integers, they are rounded down to the nearest
      integer.

  Returns:
    The RDPs at all orders, can be np.inf.
  """
  orders = [int(order) for order in orders]

  def eps(order):
    assert order > 1, 'alpha must be greater than 1.'
    # Proposition 14 of https://arxiv.org/pdf/2102.06387.pdf.
    term_1 = (order / 2.0) * l2_scale**2 + tau * dim
    term_2 = (order / 2.0) * (l2_scale**2 + 2 * l1_scale * tau + tau**2 * dim)
    term_3 = (order / 2.0) * (l2_scale + np.sqrt(dim) * tau)**2
    return min(term_1, term_2, term_3)

  if q == 1:
    rdp = np.array([eps(order) for order in orders])
  else:
    rdp = np.array([
        min(_compute_rdp_subsampled(order, q, eps), eps(order))
        for order in orders
    ])

  return rdp * steps


def ddgauss_epsilon(gamma,
                    local_stddev,
                    num_clients,
                    l2_sens,
                    beta,
                    dim,
                    q,
                    steps,
                    delta,
                    l1_sens=None,
                    rounding=True,
                    orders=RDP_ORDERS):
  """Computes epsilon of (distributed) discrete Gaussian via RDP."""
  scale = 1.0 / (gamma + DIV_EPSILON)
  l1_sens = l1_sens or (l2_sens * np.sqrt(dim))
  if rounding:
    l2_sens = rounded_l2_norm_bound(l2_sens * scale, beta, dim) / scale
    l1_sens = rounded_l1_norm_bound(l2_sens * scale, dim) / scale

  tau = 0
  for k in range(1, num_clients):
    tau += math.exp(-2 * (math.pi * local_stddev * scale)**2 * (k / (k + 1)))
  tau *= 10

  l1_scale = l1_sens / (np.sqrt(num_clients) * local_stddev)
  l2_scale = l2_sens / (np.sqrt(num_clients) * local_stddev)
  rdp = compute_rdp_dgaussian(q, l1_scale, l2_scale, tau, dim, steps, orders)
  eps, _, order = tfp.get_privacy_spent(orders, rdp, target_delta=delta)
  return eps, order


def ddgauss_local_stddev(q,
                         epsilon,
                         l2_clip_norm,
                         gamma,
                         beta,
                         steps,
                         num_clients,
                         dim,
                         delta,
                         orders=RDP_ORDERS):
  """Selects the local stddev for the distributed discrete Gaussian."""

  def stddev_opt_fn(stddev):
    stddev += DIV_EPSILON
    cur_epsilon, _ = ddgauss_epsilon(
        gamma,
        stddev,
        num_clients,
        l2_clip_norm,
        beta,
        dim,
        q,
        steps,
        delta,
        orders=orders)
    return (epsilon - cur_epsilon)**2

  stddev_result = optimize.minimize_scalar(stddev_opt_fn)
  if stddev_result.success:
    return stddev_result.x
  else:
    return -1


def ddgauss_params(q,
                   epsilon,
                   l2_clip_norm,
                   bits,
                   num_clients,
                   dim,
                   delta,
                   beta,
                   steps,
                   k=4,
                   rho=1,
                   sqrtn_norm_growth=False,
                   orders=RDP_ORDERS):
  """Selects gamma and local noise standard deviation from the given parameters.

  Args:
    q: The sampling factor.
    epsilon: The target overall epsilon.
    l2_clip_norm: The l2 clipping norm for the client vectors.
    bits: The number of bits per coordinate for the aggregated noised vector.
    num_clients: The number of clients per step.
    dim: The dimension of the vector query.
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
    The selected gamma and the local noise standard deviation.
  """
  n_factor = num_clients**(1 if sqrtn_norm_growth else 2)

  def stddev(x):
    return ddgauss_local_stddev(q, epsilon, l2_clip_norm, x, beta, steps,
                                num_clients, dim, delta, orders)

  def mod_min(x):
    return k * math.sqrt(rho / dim * l2_clip_norm**2 * n_factor +
                         (x**2 / 4.0 + stddev(x)**2) * num_clients)

  def gamma_opt_fn(x):
    return (math.pow(2, bits) - 2 * mod_min(x) / (x + DIV_EPSILON))**2

  gamma_result = optimize.minimize_scalar(gamma_opt_fn)
  if not gamma_result.success:
    raise ValueError('Cannot compute gamma.')

  gamma = gamma_result.x
  # Select the local_stddev that gave the best gamma.
  local_stddev = ddgauss_local_stddev(q, epsilon, l2_clip_norm, gamma, beta,
                                      steps, num_clients, dim, delta, orders)
  return gamma, local_stddev


####################################
############ Skellam  ##############
####################################


def _skellam_rdp(l1_sens, l2_sens, central_var, scale, order):
  assert order > 1, f'alpha must be greater than 1. Found {order}.'
  a, s, mu = order, scale, central_var
  rdp = a / (2 * mu) * l2_sens**2
  rdp += min(((2 * a - 1) * s * l2_sens**2 + 6 * l1_sens) / (4 * s**3 * mu**2),
             3 * l1_sens / (2 * s * mu))
  return rdp


def skellam_epsilon(scale,
                    central_stddev,
                    l2_sens,
                    beta,
                    dim,
                    q,
                    steps,
                    delta,
                    l1_sens=None,
                    rounding=True,
                    orders=RDP_ORDERS):
  """Computes epsilon of (distributed) Skellam via RDP."""
  l1_sens = l1_sens or (l2_sens * np.sqrt(dim))
  if rounding:
    l2_sens = rounded_l2_norm_bound(l2_sens * scale, beta, dim) / scale
    l1_sens = rounded_l1_norm_bound(l2_sens * scale, dim) / scale

  orders = [int(order) for order in orders]
  central_var = central_stddev**2

  def eps_fn(order):
    return _skellam_rdp(l1_sens, l2_sens, central_var, scale, order)

  if q == 1:
    rdp = np.array([eps_fn(order) for order in orders])
  else:
    # Take min between subsampled RDP and unamplified RDP, for all orders.
    rdp = np.array([
        min(_compute_rdp_subsampled(order, q, eps_fn), eps_fn(order))
        for order in orders
    ])

  eps, _, order = tfp.get_privacy_spent(orders, rdp * steps, target_delta=delta)
  return eps, order


def skellam_local_stddev(epsilon,
                         scale,
                         l2_clip,
                         num_clients,
                         beta,
                         dim,
                         q,
                         steps,
                         delta,
                         orders=RDP_ORDERS):
  """Selects the local stddev for the distributed discrete Gaussian."""

  def stddev_opt_fn(local_stddev):
    local_stddev += DIV_EPSILON
    central_stddev = local_stddev * np.sqrt(num_clients)
    cur_epsilon, _ = skellam_epsilon(
        scale,
        central_stddev,
        l2_clip,
        beta,
        dim,
        q,
        steps,
        delta,
        orders=orders)
    return (epsilon - cur_epsilon)**2

  local_stddev_result = optimize.minimize_scalar(stddev_opt_fn)
  if not local_stddev_result.success:
    raise ValueError('Cannot compute local_stddev for Skellam.')

  return local_stddev_result.x


def skellam_params(epsilon,
                   l2_clip,
                   bits,
                   num_clients,
                   beta,
                   dim,
                   q,
                   steps,
                   delta,
                   k=3,
                   rho=1,
                   sqrtn_norm_growth=False,
                   orders=RDP_ORDERS):
  """Computes the scaling and local noise stddev for Skellam."""
  n_factor = num_clients**(1 if sqrtn_norm_growth else 2)

  # The implementation optimizes for gamma = 1 / scale for stability.
  def local_stddev(gamma):
    scale = 1.0 / (gamma + DIV_EPSILON)
    return skellam_local_stddev(epsilon, scale, l2_clip, num_clients, beta, dim,
                                q, steps, delta, orders)

  def mod_min(gamma):
    var = rho / dim * l2_clip**2 * n_factor
    var += (gamma**2 / 4 + local_stddev(gamma)**2) * num_clients
    return k * math.sqrt(var)

  def gamma_opt_fn(gamma):
    return (math.pow(2, bits) - 2 * mod_min(gamma) / (gamma + DIV_EPSILON))**2

  gamma_result = optimize.minimize_scalar(gamma_opt_fn)
  if not gamma_result.success:
    raise ValueError('Cannot compute scaling factor.')

  scale = 1. / gamma_result.x
  # Select the local_stddev that gave the best scale.
  local_stddev = skellam_local_stddev(epsilon, scale, l2_clip, num_clients,
                                      beta, dim, q, steps, delta, orders)

  return scale, local_stddev
