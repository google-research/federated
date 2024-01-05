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

Find the intervals directly by using the closed form solutions of quadratics.
This leads to a 10% speed-up over the root-finding version.
"""

import math

import numpy as np
import pandas as pd
import scipy.stats


def get_asymptotic_confidence_intervals(
    xs: np.ndarray, beta: float = 0.05, return_statistics: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, tuple[float, float, float]]:
  """Get asymptotic confidence intervals.

  Args:
    xs: binary matrix of shape (n, k). Represents the k-dimensional output of n
      trials.
    beta: confidence parameter.
    return_statistics: if true, return difference between moments.

  Returns:
    outputs: pd.DataFrame containing various asymptotic CIs

  NOTE: both left and right do not hold simultaneously!
  In practice, we only look for one of the two bounds.
  """
  n, k = xs.shape
  outputs_left = {}
  outputs_right = {}

  # Compute the moments
  m1 = np.mean(xs, axis=1)  # (n,)
  mu1_hat = m1.mean()  # (n,) -> scalar
  if k > 1:
    m2 = m1 * (k * m1 - 1) / (k - 1)  # (n,)
    mu2_hat = m2.mean()  # (n,) -> scalar
  else:
    mu2_hat = None
    m2 = 0.0
  if k > 2:
    m3 = m2 * (k * m1 - 2) / (k - 2)  # (n,)
    mu3_hat = m3.mean()  # (n,) -> scalar
  else:
    mu3_hat = None
    m3 = 0.0
  if k > 3:
    m4 = m3 * (k * m1 - 3) / (k - 3)  # (n,)
    mu4_hat = m4.mean()  # (n,) -> scalar
  else:
    mu4_hat = None

  ##### Clopper Pearson
  key = 'Clopper-Pearson'
  n_pos = xs[:, 0].sum()
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

  outputs = pd.DataFrame(dict(left=outputs_left, right=outputs_right))
  if return_statistics:
    if k > 1:
      mu2_minus_mu1sq = mu2_hat - mu1_hat**2
    else:
      mu2_minus_mu1sq = np.nan
    if k > 2:
      mu3_minus_mu2sq = mu3_hat - mu2_hat**2
    else:
      mu3_minus_mu2sq = np.nan
    if k > 3:
      mu4_minus_mu2sq = mu4_hat - mu2_hat**2
    else:
      mu4_minus_mu2sq = np.nan
    return outputs, (mu2_minus_mu1sq, mu3_minus_mu2sq, mu4_minus_mu2sq)
  else:
    return outputs


def solve_first_order_wilson_right_tail(mu_hat, n, alpha):
  """First order Wilson bound (right tail)."""
  z_alpha = scipy.stats.norm.ppf(
      1 - alpha
  )  # (1-alpha) quantile of the Gaussian
  a = n + z_alpha**2
  b = 2 * n * mu_hat + z_alpha**2
  c = n * mu_hat**2
  sol = (b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
  sol = max(0, min(1, sol))  # clip to [0, 1]
  # Worst-case solution using var <= 1/4
  sol2 = max(0, min(1, mu_hat + z_alpha * math.sqrt(0.25 / n)))
  return min(sol, sol2)


def solve_first_order_wilson_left_tail(mu_hat, n, alpha):
  """First order Wilson bound (left tail)."""
  z_alpha = scipy.stats.norm.ppf(
      1 - alpha
  )  # (1-alpha) quantile of the Gaussian
  a = n + z_alpha**2
  b = 2 * n * mu_hat + z_alpha**2
  c = n * mu_hat**2
  sol = (b - math.sqrt(b**2 - 4 * a * c)) / (2 * a)
  sol = max(0, min(1, sol))  # clip to [0, 1]
  # Worst-case solution using var <= 1/4
  sol2 = max(0, min(1, mu_hat - z_alpha * math.sqrt(0.25 / n)))
  return max(sol, sol2)


def solve_second_order_wilson_right_tail(mu_hat, mu2, n, k, alpha):
  """Second order Wilson bound (right tail)."""
  z_alpha = scipy.stats.norm.ppf(
      1 - alpha
  )  # (1-alpha) quantile of the Gaussian
  a = n + z_alpha**2
  b = 2 * n * mu_hat + z_alpha**2 / k
  c = n * mu_hat**2 - z_alpha**2 * (k - 1) / k * mu2
  sol = (b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
  sol = max(0, min(1, sol))  # clip to [0, 1]
  # Worst-case solution using var <= 1/4
  sol2 = max(0, min(1, mu_hat + z_alpha * math.sqrt(0.25 / n)))
  return min(sol, sol2)


def solve_second_order_wilson_left_tail(mu_hat, mu2, n, k, alpha):
  """Second order Wilson bound (left tail)."""
  z_alpha = scipy.stats.norm.ppf(
      1 - alpha
  )  # (1-alpha) quantile of the Gaussian
  a = n + z_alpha**2
  b = 2 * n * mu_hat + z_alpha**2 / k
  c = n * mu_hat**2 - z_alpha**2 * (k - 1) / k * mu2
  sol = (b - math.sqrt(b**2 - 4 * a * c)) / (2 * a)
  sol = max(0, min(1, sol))  # clip to [0, 1]
  # Worst-case solution using var <= 1/4
  sol2 = max(0, min(1, mu_hat - z_alpha * math.sqrt(0.25 / n)))
  return max(sol, sol2)


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
  sol = (b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
  sol = max(0, min(1, sol))  # clip to [0, 1]
  # Worst-case solution using var <= 1/4
  sol2 = max(0, min(1, mu2_hat + z_alpha * math.sqrt(0.25 / n)))
  return min(sol, sol2)
