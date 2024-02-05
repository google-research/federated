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
"""Functions for empirical estimation of privacy loss.

These functions compute the estimates of privacy loss described and used
in Andrew et al. (2023) "One-shot Empirical Privacy Estimation for
Federated Learning". https://arxiv.org/pdf/2302.03098.pdf
"""
import functools
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special
import scipy.stats


def _jeffreys_high(k: int, n: int, alpha: float) -> float:
  """Computes Jeffreys one-sided upper binomial confidence interval.

  Args:
    k: The number of successes.
    n: The number of trials.
    alpha: Allowed probability of failure (one minus confidence).

  Returns:
    A value p such that the true success probability is at most p with 1-alpha
    confidence.
  """
  # Multiply alpha by 2 for one-sided interval.
  return scipy.stats.beta.interval(1 - 2 * alpha, 0.5 + k, 0.5 + n - k)[1]


def _epsilon_lower_bound(log_fpr: float, fnr: float, delta: float) -> float:
  """Computes epsilon lower bound from false positive/negative rates.

  Args:
    log_fpr: Log false positive rate. Pass in log space for more accurate
      computation with closed form null hypothesis distribution.
    fnr: False negative rate (not in log space).
    delta: Delta.

  Returns:
    The epsilon lower bound.
  """
  # We want to ignore invalid values in the log here. If (1 - delta - fnr) is
  # less or equal to zero, the bound is invalid. Let it return np.nan or -np.inf
  # and we will filter it with np.nanmax.
  err_settings = np.seterr(divide='ignore', invalid='ignore')
  bound1 = np.log(1 - delta - fnr) - log_fpr
  bound2 = np.log(1 - delta - np.exp(log_fpr)) - np.log(fnr)
  np.seterr(**err_settings)
  return np.clip(np.nanmax([bound1, bound2], axis=0), 0, None)


def optimal_epsilon_lower_bound(
    seen: list[float], unseen: list[float], alpha: float, delta: float
) -> float:
  """Finds optimal epsilon lower bound from statistics of seen/unseen canaries.

  This is the classic method of Jagielsky et. al (2020) but with Jeffreys
  intervals instead of Clopper-Pearson.

  Args:
    seen: A list of floats representing observed canary statistics.
    unseen: A list of floats representing unobserved canary statistics.
    alpha: Allowed probability of failure (one minus confidence).
    delta: Delta.

  Returns:
    Optimal epsilon lower bound.
  """
  # Sort all stats, using imaginary component to track seen canary stats.
  seen = np.array(seen) + 1.0j
  unseen = np.array(unseen) + 0.0j
  values = np.sort_complex(np.concatenate([seen, unseen]))

  # Now extract positions of positive and negative canaries.
  seen_pos = np.imag(values)
  unseen_pos = 1 - seen_pos

  # Cumulative sum gives false pos/neg counts. Cumsum starts with first entry--
  # there is no value for summing the first 0 entries-- so prepend it.
  fp_counts = len(unseen) - np.concatenate([[0], np.cumsum(unseen_pos)])
  fn_counts = np.concatenate([[0], np.cumsum(seen_pos)])

  log_fprs = np.log(_jeffreys_high(fp_counts, len(unseen), alpha))
  fnrs = _jeffreys_high(fn_counts, len(seen), alpha)
  eps = _epsilon_lower_bound(log_fprs, fnrs, delta)

  return np.max(eps)


class ClosedFormNullCalculator:
  """Calculates properties of the exact cosine distribution.

  The current implementation uses the exact pdf and numerical integration for
  the cdf. If dim is large (>= 1000), it is recommended to use the Gaussian
  approximation N(0,1/d) which is essentially exact. For smaller dim (<200?)
  it might be faster to use the exact cdf in terms of the hypergeometric
  function (https://tinyurl.com/cosine-distribution), but this has not been
  implemented.
  """

  def __init__(self, dim: int):
    """Initializes the ClosedFormNullCalculator.

    Args:
      dim: The dimensionality. Must be < 1000. For dim >= 1000, it is
        recommended to use the Gaussian approximation N(0,1/d).

    Raises:
      ValueError: if dim >= 1000.
    """
    if dim >= 1000:
      raise ValueError('For dim >= 1000, use the normal approximation.')

    self._z = np.exp(
        scipy.special.gammaln(dim / 2) - scipy.special.gammaln((dim - 1) / 2)
    ) / np.sqrt(np.pi)
    self._dim = dim

  def pdf(self, t: float) -> float:
    """The exact pdf of the null hypothesis distribution."""
    return self._z * (1 - t**2) ** ((self._dim - 3) / 2)

  def cdf(self, t: float) -> float:
    """The exact cdf of the null hypothesis distribution."""
    return scipy.integrate.quad(self.pdf, -1, t)[0]

  def ppf(self, p: float) -> float:
    """The exact ppf (inverse cdf) of the null hypothesis distribution."""
    return scipy.optimize.brentq(lambda t: self.cdf(t) - p, -1, 1)


def optimal_epsilon_lower_bound_closed_form_null(
    seen_cosines: list[float], dim: int, alpha: float, delta: float
) -> float:
  """Finds optimal epsilon lower bound using the closed form null distribution.

  This is the classic method of Jagielsky et. al (2020) with two modifications:
    1. Jeffreys interval is used instead of Clopper-Pearson.
    2. It uses the closed-form, Gaussian-approximated null hypothesis
      distribution for the cosine with unobserved canaries described in section
      4 of https://arxiv.org/pdf/2302.03098.pdf.

  For dim >= 1000 the Gaussian approximation to the closed form null is used.

  Args:
    seen_cosines: A list of floats representing observed canary cosines.
    dim: Dimensionality. Must be at least 1000.
    alpha: Allowed probability of failure (one minus confidence).
    delta: Delta.

  Returns:
    Optimal epsilon lower bound.
  """
  # Consider how eps changes as a function of threshold t. As t crosses a cosine
  # value, FNR increases discontinuously, and eps decreases. Between cosines,
  # FPR decreases, so eps increases. Hence the threshold maximizing eps will be
  # infinitessimally less than a cosine value. So we consider thresholds
  # equaling cosine values, but the cosine at t is not counted as a false
  # negative for the purpose of computing epsilon for that t.

  # Note it is not usually possible to get an "upper bound" on epsilon by using
  # lower confidence intervals on FPR/FNR. Typically (for small enough delta)
  # the first cosine value will be at a point where the TPR exceeds delta, so
  # the epsilon estimate is log((TPR - delta) / 0) = inf.

  thresholds = np.sort(list(seen_cosines))

  # FPR(t) = Pr(X0 >= t) = Pr(X0 <= -t).
  if dim < 1000:
    calculator = ClosedFormNullCalculator(dim)
    log_fprs = np.log(np.array([calculator.cdf(-t) for t in thresholds]))
  else:
    log_fprs = scipy.stats.norm.logcdf(-thresholds, scale=dim**-0.5)

  false_neg_counts = np.arange(len(thresholds))
  fnrs = _jeffreys_high(false_neg_counts, len(seen_cosines), alpha)
  eps = _epsilon_lower_bound(log_fprs, fnrs, delta)
  return np.max(eps)


_norm_logcdf = scipy.stats.norm.logcdf


@functools.cache
def _log_delta_two_gaussians(
    mu1: float,
    std1: float,
    mu2: float,
    std2: float,
    eps: float,
) -> float:
  """Computes delta from two Gaussian distributions."""
  a = 0.5 * (std2**-2 - std1**-2)
  b = mu1 / (std1**2) - mu2 / (std2**2)
  c = (
      0.5 * ((mu2 / std2) ** 2 - (mu1 / std1) ** 2)
      + np.log(std2)
      - np.log(std1)
  )

  # In the following, pr_z1 corresponds to log Pr[Z_1 > eps] in the
  # writup, and pr_z2 corresponds to log Pr[Z_2 < -eps].

  if std1 == std2:
    # Here a == 0. We need Pr(bY + c > eps) = Pr(Y > (eps - c) / b), where
    # Y ~ N(mu1, std1). For stable computation in log space, we use the
    # fact that Pr(Y > t; mu, std) = Pr(Y < mu; t, std).
    pr_z1 = _norm_logcdf(mu1, (eps - c) / b, std1)
    pr_z2 = _norm_logcdf(mu2, (eps - c) / b, std2)
  else:
    # Find zeros of (a y^2 + b y + c - eps), then use mass of X1 or X2
    # between zeros or outside depending on sign of a.

    determinant = b**2 - 4 * a * (c - eps)
    if determinant <= 0:
      # Quadratic has no intercept. pr_z1 and pr_z2 are either both 1.0 or
      # both 0.0. Either way, delta is 0.
      return -np.inf

    i_1 = (-b - np.sqrt(determinant)) / (2 * a)
    i_2 = (-b + np.sqrt(determinant)) / (2 * a)
    i_left, i_right = min(i_1, i_2), max(i_1, i_2)

    if a >= 0:
      # Quadratic opens upward. Return mass outside zeros. For stable
      # computation in log space, we use the fact that
      # Pr(Y > t; mu, std) = Pr(Y < mu; t, std).
      pr_z1 = np.logaddexp(
          _norm_logcdf(i_left, mu1, std1), _norm_logcdf(mu1, i_right, std1)
      )
      pr_z2 = np.logaddexp(
          _norm_logcdf(i_left, mu2, std2), _norm_logcdf(mu2, i_right, std2)
      )
    else:
      # Quadratic opens downward. Return mass between zeros.
      pr_z1 = scipy.special.logsumexp(
          [_norm_logcdf(i_right, mu1, std1), _norm_logcdf(i_left, mu1, std1)],
          b=[1, -1],
      )
      pr_z2 = scipy.special.logsumexp(
          [_norm_logcdf(i_right, mu2, std2), _norm_logcdf(i_left, mu2, std2)],
          b=[1, -1],
      )

  factor = eps + pr_z2 - pr_z1
  if factor >= 0:
    # Delta bound is negative, hence vacuous.
    return -np.inf
  return pr_z1 + np.log1p(-np.exp(factor))


def epsilon_two_gaussians(
    mu1: float, std1: float, mu2: float, std2: float, delta: float
) -> float:
  """Computes epsilon from two Gaussian distributions.

  Uses theory from Thomas Steinke "Composition of Differential Privacy & Privacy
  Amplification by subsampling" https://arxiv.org/pdf/2210.00597.pdf to compute
  epsilon exactly.

  Args:
    mu1: The mean of the first Gaussian.
    std1: The std of the first Gaussian.
    mu2: The mean of the second Gaussian.
    std2: The std of the second Gaussian.
    delta: Delta.

  Returns:
    The estimate of epsilon.
  """

  # pylint: disable=arguments-out-of-order
  def _max_log_delta(eps):
    # We could implement _log_delta_two_gaussians to compute this
    # max and save a few operations but this is conceptually simpler.
    return max(
        _log_delta_two_gaussians(mu1, std1, mu2, std2, eps),
        _log_delta_two_gaussians(mu2, std2, mu1, std1, eps),
    )

  # pylint: enable=arguments-out-of-order

  log_delta = np.log(delta)
  if _max_log_delta(0) <= log_delta:
    # We have (0, delta)-DP.
    return 0

  eps_lb = 0
  eps_ub = 1
  while _max_log_delta(eps_ub) > log_delta:
    eps_lb = eps_ub
    eps_ub *= 10
  return scipy.optimize.brentq(
      lambda eps: _max_log_delta(eps) - log_delta, eps_lb, eps_ub
  )


def epsilon_closed_form_null_gaussian_alt(
    mu: float, std: float, dim: int, delta: float
) -> float:
  """Computes eps from closed-form null and Gaussian alternate hypothesis.

  Computes log false positive rate at threshold t using closed form of null
  hypothesis distribution. See sec. 4 of https://arxiv.org/pdf/2302.03098.pdf.
  For dim >= 1000, the Gaussian approximation is used.

  Args:
    mu: The mean of the (observed) canary cosine distribution.
    std: The std of the (observed) canary cosine distribution.
    dim: The dimensionality.
    delta: Delta.

  Returns:
    The estimate of epsilon.
  """
  if dim >= 1000:
    return epsilon_two_gaussians(
        mu1=0,
        std1=dim**-0.5,
        mu2=mu,
        std2=std,
        delta=delta,
    )

  # The method below will fail if the mean statistic of the observed canaries is
  # less than that of the unobserved.
  if mu <= 0:
    raise ValueError(f'mu must be positive. Found mu={mu}.')

  calculator = ClosedFormNullCalculator(dim)

  # Bound search based on where epsilon bound becomes meaningless.
  # t < lb implies FPR > 1-delta.
  lb = calculator.ppf(delta)
  # t > ub implies FNR > 1-delta.
  ub = scipy.stats.norm.ppf(1 - delta, loc=mu, scale=std)

  ts = np.linspace(lb, ub, 10000)
  # FPR(t) = Pr(X0 >= t) = Pr(X0 <= -t).
  log_fprs = np.log(np.array([calculator.cdf(-t) for t in ts]))
  fnrs = scipy.stats.norm.cdf(ts, loc=mu, scale=std)
  eps = _epsilon_lower_bound(log_fprs, fnrs, delta)
  return np.max(eps)
