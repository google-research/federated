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
"""Code to un-bias miracle & modified miracle for ss.

Please note that this unbiasing techniques are specific to ss + miracle
and are not general strategies for miracle. The un-biasing happens by
modifying the corresponding m and b parameters.
"""

import numpy as np
from scipy import stats
import scipy.special as sc
from rcc_dp.frequency_estimation import ss


def unbias_miracle(k, epsilon, number_candidates, z, n, normalization = 1):
  """Get the unbiased estimate for miracle.

  Args:
    k: The dimension of the one-hot encoded x.
    epsilon: The privacy parameter epsilon.
    number_candidates: The number of candidates to be sampled.
    z: The privatized data.
    n: The number of users.
    normalization: Indicator whether to clip and normalize or
    project on the simplex

  Returns:
    p_estimate: The unbiased estimate for miracle
  """
  d = int(np.ceil(k/(np.exp(epsilon)+1)))
  c1 = np.exp(epsilon)/(sc.comb(k-1,d-1)* np.exp(epsilon) + sc.comb(k-1,d))
  c2 = 1/(sc.comb(k-1,d-1)* np.exp(epsilon) + sc.comb(k-1,d))
  beta = np.array(range(number_candidates+1))/number_candidates
  # The probability with which you choose a candidate inside the cap.
  pi_in = 1/number_candidates*(c1/(beta*c1+(1-beta)*c2))
  expectation = np.sum(stats.binom.pmf(range(number_candidates+1),
    number_candidates, d/k)*range(number_candidates+1)*pi_in)
  b_hat = (d - expectation)/(k-1)
  m_hat = k*expectation/(k-1) - d/(k-1)

  p_estimate = (1.0*np.sum(z, axis=0)/(n*m_hat))-b_hat/m_hat

  if normalization == 0:
    # Clip and normalize.
    p_estimate = ss.probability_normalize(p_estimate)
  elif normalization == 1:
    # Project on the simplex.
    p_estimate = ss.probability_project_simplex(p_estimate)
  return p_estimate


def unbias_modified_miracle(k, epsilon, number_candidates, z, n,
  normalization = 1):
  """Get the unbiased estimate for modified miracle.

  Args:
    k: The dimension of the one-hot encoded x.
    epsilon: The privacy parameter epsilon.
    number_candidates: The number of candidates to be sampled.
    z: The privatized data.
    n: The number of users.
    normalization: Indicator whether to clip and normalize or
    project on the simplex

  Returns:
    p_estimate: The unbiased estimate for modified miracle
  """
  expected_beta = np.ceil(k/(np.exp(epsilon)+1))/k
  d = int(np.ceil(k/(np.exp(epsilon)+1)))
  c1 = np.exp(epsilon)/(sc.comb(k-1,d-1)* np.exp(epsilon) + sc.comb(k-1,d))
  c2 = 1/(sc.comb(k-1,d-1)* np.exp(epsilon) + sc.comb(k-1,d))
  # The fraction of candidates that lie inside the cap.
  beta = np.array(range(number_candidates + 1)) / number_candidates

  pi_in = 1 / number_candidates * (c1 / (beta * c1 + (1 - beta) * c2))
  tilde_pi_in_1 = (
      pi_in * (1 + beta * (np.exp(epsilon) - 1)) / (1 + expected_beta *
                                                    (np.exp(epsilon) - 1)))
  tilde_pi_in_2 = (
      tilde_pi_in_1 * (beta + expected_beta * (np.exp(epsilon) - 1)) /
      (beta * np.exp(epsilon)))
  tilde_pi_in_2[0,] = 0
  indicator = beta <= expected_beta
  # The probability with which you choose a candidate inside the cap.
  tilde_pi_in = indicator * tilde_pi_in_1 + (1 - indicator) * tilde_pi_in_2
  expectation = np.sum(
      stats.binom.pmf(range(number_candidates + 1), number_candidates,
        d / k) * range(number_candidates + 1) * tilde_pi_in)
  b_tilde = (d - expectation)/(k-1)
  m_tilde = k*expectation/(k-1) - d/(k-1)

  p_estimate = (1.0*np.sum(z, axis=0)/(n*m_tilde))-b_tilde/m_tilde

  if normalization == 0:
    # Clip and normalize.
    p_estimate = ss.probability_normalize(p_estimate)
  elif normalization == 1:
    # Project on the simplex.
    p_estimate = ss.probability_project_simplex(p_estimate)
  return p_estimate
