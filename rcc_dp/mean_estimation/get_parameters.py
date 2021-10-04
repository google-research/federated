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
"""Code to obtain parameters of various miracle methods."""

import numpy as np
from mean_estimation import optimize_unbias
from mean_estimation import privunit


def get_parameters_unbiased_approx_miracle(epsilon_target, d, number_candidates,
                                           budget, delta):
  """Get privunit parameters for miracle with approxmiate-DP guarantees.

  The approximate DP guarantee of miracle is as follows: If a mechanism is
  epsilon-DP, simulating it with MIRACLE (with N candidates) gives a mechanism
  which is (epsilon + log(1 + t) - log(1 - t), delta)-DP where
  delta = 2 * exp(-N * t^2 / (c_1 - c_2)^2). An important point to note here is
  that c_1 and c_2 are themselves functions of epsilon. So for given delta,
  number of candidates, and epsilon_target, the idea is to come up with the
  largest epsilon for which epsilon + log(1 + t) - log(1 - t) is smaller than
  epsilon_target.

  Args:
    epsilon_target: The privacy guarantee we desire.
    d: The number of dimensions.
    number_candidates: The number of candidates.
    budget: The default budget splitting between the gamma and p parameters.
    delta: The delta in the differential privacy guarantee.

  Returns:
    c1: The larger constant that the privunit density is proportional to.
    c2: The smaller constant that the privunit density is proportional to.
    m: The inverse of the scalar norm that the decoder should use to get
    an unbiased estimator.
    gamma: The gamma parameter of privunit.
    epsilon_approx: The resulting epsilon that this version of miracle ends
    up using.
  """

  epsilon_search_space = np.linspace(0, epsilon_target, 200)
  epsilon_search_space = epsilon_search_space[:-1]
  epsilon_approx = 0
  # Find the largest epsilon for PrivUnit so that MIRACLE meets epsilon_target
  for epsilon in epsilon_search_space:
    gamma, _ = privunit.find_best_gamma(d, budget * epsilon)
    p = np.exp((1 - budget) * epsilon) / (1 + np.exp((1 - budget) * epsilon))
    c1, c2 = privunit.get_privunit_densities(d, gamma, p)
    t = np.abs(c1 - c2) * np.sqrt((np.log(2 / delta)) / (2 * number_candidates))
    if -1 < t < 1:
      if epsilon + np.log(1 + t) - np.log(1 - t) <= epsilon_target:
        epsilon_approx = epsilon
  gamma, _ = privunit.find_best_gamma(d, budget * epsilon_approx)
  p = np.exp((1 - budget) * epsilon_approx) / (1 + np.exp(
      (1 - budget) * epsilon_approx))
  c1, c2 = privunit.get_privunit_densities(d, gamma, p)
  p_hat = optimize_unbias.get_unbiased_p_hat(number_candidates, c1, c2, p)
  m_hat = privunit.getm(d, gamma, p_hat)

  return c1, c2, m_hat, gamma, epsilon_approx


def get_parameters_unbiased_miracle(epsilon, d, number_candidates, budget):
  """Get privunit parameters for unbiased miracle."""
  # Get the optimized budget.
  gamma, _ = privunit.find_best_gamma(d, budget * epsilon)
  p = np.exp((1 - budget) * epsilon) / (1 + np.exp((1 - budget) * epsilon))
  c1, c2 = privunit.get_privunit_densities(d, gamma, p)
  p_hat = optimize_unbias.get_unbiased_p_hat(number_candidates, c1, c2, p)
  m_hat = privunit.getm(d, gamma, p_hat)

  return c1, c2, m_hat, gamma


def get_parameters_unbiased_modified_miracle(epsilon, d, number_candidates,
                                             eta, budget):
  """Get privunit parameters for unbiased modified miracle."""
  # Get the optimized budget.
  gamma, _ = privunit.find_best_gamma(d, budget * epsilon)
  p = np.exp((1 - budget) * epsilon) / (1 + np.exp((1 - budget) * epsilon))
  c1, c2 = privunit.get_privunit_densities(d, gamma, p)
  p_tilde = optimize_unbias.get_unbiased_p_tilde(number_candidates, c1, c2, p,
                                                 epsilon)
  m_tilde = privunit.getm(d, gamma, p_tilde)

  return c1, c2, m_tilde, gamma
