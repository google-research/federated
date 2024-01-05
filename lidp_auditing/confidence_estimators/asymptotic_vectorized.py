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
"""Asymptotic confidence intervals for exchangeable Bernoulli (xBern) tuples.

Solves the quadratics with a vectorized version.
Gives a 7-12x (avg: 9x) speedup over writing loops.
"""

import numpy as np
import pandas as pd
import scipy.stats

ArrayTriplet = tuple[np.ndarray, np.ndarray, np.ndarray]
ConfidenceIntervalReturnType = (
    tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, ArrayTriplet]
)


def get_asymptotic_confidence_intervals(
    xs: np.ndarray,
    beta: float = 0.05,
    thresholds: np.ndarray | None = None,
    return_statistics: bool = False,
) -> ConfidenceIntervalReturnType:
  """Get asymptotic confidence intervals.

  Args:
    xs: binary matrix of shape (n, k, num_thresholds). Represents the
      k-dimensional output of n trials over T thresholds.
    beta: confidence parameter.
    thresholds: optional np.ndarray to name as columns.
    return_statistics: if true, return difference between moments.

  Returns:
    outputs_left, outputs_right: Each is a pd.DataFrame
    containing various asymptotic CIs (on the index) and
    the threshold (vectorized dimension) along the columns

  NOTE: both left and right do not hold simultaneously!
  In practice, we only look for one of the two bounds.
  """
  n, k, num_thresholds = xs.shape  # (n, k, T)
  outputs_left = {}
  outputs_right = {}

  # Compute the moments
  m1 = np.mean(xs, axis=1)  # (n, T)
  mu1_hat = m1.mean(axis=0)  # (n, T) -> (T,)
  if k > 1:
    m2 = m1 * (k * m1 - 1) / (k - 1)  # (n, T)
    mu2_hat = m2.mean(axis=0)  # (n, T) -> (T,)
  else:
    mu2_hat = None
    m2 = 0.0
  if k > 2:
    m3 = m2 * (k * m1 - 2) / (k - 2)  # (n, T)
    mu3_hat = m3.mean(axis=0)  # (n, T) -> (T,)
  else:
    mu3_hat = None
    m3 = 0.0
  if k > 3:
    m4 = m3 * (k * m1 - 3) / (k - 3)  # (n, T)
    mu4_hat = m4.mean(axis=0)  # (n, T) -> (T,)
  else:
    mu4_hat = None

  #### Clopper Pearson
  key = 'Clopper-Pearson'
  n_pos = xs[:, 0, :].sum(axis=0)  # (T,)
  left = scipy.stats.beta.ppf(beta, n_pos, n - n_pos + 1)
  right = scipy.stats.beta.ppf(1 - beta, n_pos + 1, n - n_pos)
  outputs_left[key] = left
  outputs_right[key] = right

  #### First-order Wilson
  key = '1st-Order Wilson'
  left = solve_first_order_wilson_left_tail(mu1_hat, n, beta)
  right = solve_first_order_wilson_right_tail(mu1_hat, n, beta)
  outputs_left[key] = left
  outputs_right[key] = right

  #### Second-order Wilson
  key = '2nd-Order Wilson'
  if k > 1:
    mu2_upper = solve_first_order_wilson_right_tail(mu2_hat, n, beta / 2)
    left = solve_second_order_wilson_left_tail(
        mu1_hat, mu2_upper, n, k, beta / 2
    )
    right = solve_second_order_wilson_right_tail(
        mu1_hat, mu2_upper, n, k, beta / 2
    )
    outputs_left[key] = left
    outputs_right[key] = right

  ####  Fourth order Wilson
  key = '4th-Order Wilson'
  if k > 3:
    mu3_upper = solve_first_order_wilson_right_tail(mu3_hat, n, beta / 4)
    mu4_upper = solve_first_order_wilson_right_tail(mu4_hat, n, beta / 4)
    mu2_upper = solve_fourth_order_wilson_right_tail_for_mu2(
        mu2_hat, mu3_upper, mu4_upper, n, k, beta / 4
    )
    left = solve_second_order_wilson_left_tail(
        mu1_hat, mu2_upper, n, k, beta / 4
    )
    right = solve_second_order_wilson_right_tail(
        mu1_hat, mu2_upper, n, k, beta / 4
    )
    outputs_left[key] = left
    outputs_right[key] = right

  outputs_left = pd.DataFrame(outputs_left, index=thresholds).T
  outputs_right = pd.DataFrame(outputs_right, index=thresholds).T
  if return_statistics:
    if k > 1:
      mu2_minus_mu1sq = mu2_hat - mu1_hat**2  # (T,)
    else:
      mu2_minus_mu1sq = np.full(num_thresholds, np.nan)  # (T,)
    if k > 2:
      mu3_minus_mu2sq = mu3_hat - mu2_hat**2  # (T,)
    else:
      mu3_minus_mu2sq = np.full(num_thresholds, np.nan)  # (T,)
    if k > 3:
      mu4_minus_mu2sq = mu4_hat - mu2_hat**2  # (T,)
    else:
      mu4_minus_mu2sq = np.full(num_thresholds, np.nan)  # (T,)
    outputs_statistics = (mu2_minus_mu1sq, mu3_minus_mu2sq, mu4_minus_mu2sq)
    return outputs_left, outputs_right, outputs_statistics
  else:
    return outputs_left, outputs_right


def solve_first_order_wilson_right_tail(mu_hat, n, alpha):
  """First order Wilson bound (right tail)."""
  z_alpha = scipy.stats.norm.ppf(
      1 - alpha
  )  # (1-alpha) quantile of the Gaussian
  a = n + z_alpha**2
  b = 2 * n * mu_hat + z_alpha**2
  c = n * mu_hat**2
  sol = (b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
  return np.clip(sol, 0, 1)  # clip to [0, 1]


def solve_first_order_wilson_left_tail(mu_hat, n, alpha):
  """First order Wilson bound (left tail)."""
  z_alpha = scipy.stats.norm.ppf(
      1 - alpha
  )  # (1-alpha) quantile of the Gaussian
  a = n + z_alpha**2
  b = 2 * n * mu_hat + z_alpha**2
  c = n * mu_hat**2
  sol = (b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
  return np.clip(sol, 0, 1)  # clip to [0, 1]


def solve_second_order_wilson_right_tail(mu_hat, mu2, n, k, alpha):
  """Second order Wilson bound (right tail)."""
  z_alpha = scipy.stats.norm.ppf(
      1 - alpha
  )  # (1-alpha) quantile of the Gaussian
  a = n + z_alpha**2
  b = 2 * n * mu_hat + z_alpha**2 / k
  c = n * mu_hat**2 - z_alpha**2 * (k - 1) / k * mu2
  sol = (b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
  return np.clip(sol, 0, 1)  # clip to [0, 1]


def solve_second_order_wilson_left_tail(mu_hat, mu2, n, k, alpha):
  """Second order Wilson bound (left tail)."""
  z_alpha = scipy.stats.norm.ppf(
      1 - alpha
  )  # (1-alpha) quantile of the Gaussian
  a = n + z_alpha**2
  b = 2 * n * mu_hat + z_alpha**2 / k
  c = n * mu_hat**2 - z_alpha**2 * (k - 1) / k * mu2
  sol = (b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
  return np.clip(sol, 0, 1)  # clip to [0, 1]


def solve_fourth_order_wilson_right_tail_for_mu2(
    mu2_hat, mu3, mu4, n, k, alpha
):
  """Fourth order Wilson bound (right tail)."""
  z_alpha = scipy.stats.norm.ppf(
      1 - alpha
  )  # (1-alpha) quantile of the Gaussian
  a = n + 2 * z_alpha**2 * (1 + 2 * (k - 2)) / (k**2 - k)
  b = 2 * n * mu2_hat + 2 * z_alpha**2 / (k**2 - k)
  c1 = (k - 2) * (k - 3) / (k**2 - k) * (mu4 - mu3**2) + 4 * (k - 2) / (
      k**2 - k
  ) * mu3
  c = n * mu2_hat**2 - z_alpha**2 * c1
  sol = (b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
  return np.clip(sol, 0, 1)  # clip to [0, 1]
