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
"""Code to optimize & un-bias miracle & modified miracle.

Please note that this optimization and unbiasing techniques are specific to
privunit + miracle and are not general strategies for miracle.

1. The un-biasing happens by modifying the corresponding p and m.

2. The optimization for unbiased miracle happens by choosing the budget that
minimizes the variance or maximizes the corresponding m.

3. The optimization for unbiased modified miracle happens by first choosing the
range budget of budget for which epsilon_kink > epsilon_target and then
minimizing the variance or maximizing the corresponding m in this range.
"""

import numpy as np
from scipy import stats
from rcc_dp.mean_estimation import privunit


def get_unbiased_p_hat(number_candidates, c1, c2, p):
  """Get the p_hat to unbias miracle.

  Args:
    number_candidates: The number of candidates to be sampled.
    c1: The factor that the conditional density of z given x is proportional to
    if the inner product between x and z is more than gamma.
    c2: The factor that the conditional density of z given x is proportional to
    if the inner product between x and z is less than gamma.
    p: The probability with which privunit samples an unit vector from the
    shaded spherical cap associated with input (see original privunit paper).

  Returns:
    p_hat: The probability with which unbiased miracle will sample an unit
    vector from the shaded spherical cap associated with input.
  """
  # Compute the fraction of candidates that lie inside the cap.
  beta = np.array(range(number_candidates + 1)) / number_candidates
  pi_in = 1 / number_candidates * (c1 / (beta * c1 + (1 - beta) * c2))
  p_hat = np.sum(
      stats.binom.pmf(range(number_candidates + 1), number_candidates, p / c1) *
      range(number_candidates + 1) * pi_in)

  return p_hat


def get_unbiased_p_tilde(number_candidates, c1, c2, p, epsilon):
  """Get the p_tilde to unbias modified miracle.

  Args:
    number_candidates: The number of candidates to be sampled.
    c1: The factor that the conditional density of z given x is proportional to
    if the inner product between x and z is more than gamma.
    c2: The factor that the conditional density of z given x is proportional to
    if the inner product between x and z is less than gamma.
    p: The probability with which privunit samples an unit vector from the
    shaded spherical cap associated with input (see original privunit paper).
    epsilon: The privacy parameter.

  Returns:
    p_tilde: The probability with which unbiased modified miracle will sample
    an unit vector from the shaded spherical cap associated with input.
  """
  expected_beta = p / c1
  # The fraction of candidates that lie inside the cap.
  beta = np.array(range(number_candidates + 1)) / number_candidates

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
  p_tilde = np.sum(
      stats.binom.pmf(range(number_candidates + 1), number_candidates, p / c1) *
      range(number_candidates + 1) * tilde_pi_in)

  return p_tilde


def get_optimized_budget_unbiased_miracle(epsilon, d, number_candidates,
                                          number_of_budget_intervals):
  """Get the optimal budget for unbiased miracle."""
  budget_space = np.linspace(0.01, 0.99, number_of_budget_intervals)
  m_hat = np.zeros(len(budget_space))
  for step, budget in enumerate(budget_space):
    gamma, _ = privunit.find_best_gamma(d, budget*epsilon)
    p = np.exp((1-budget)*epsilon)/(1+np.exp((1-budget)*epsilon))
    c1, c2 = privunit.get_privunit_densities(d, gamma, p)
    p_hat = get_unbiased_p_hat(number_candidates, c1, c2, p)
    m_hat[step] = privunit.getm(d, gamma, p_hat)
  return budget_space[np.argmax(m_hat)]


def get_optimized_budget_unbiased_modified_miracle(epsilon, d,
                                                   number_candidates,
                                                   number_of_budget_intervals):
  """Get the optimal budget for unbiased modified miracle."""
  maximum_budget = get_budget_range(epsilon, d, number_of_budget_intervals)
  budget_space = np.linspace(0.01, maximum_budget, number_of_budget_intervals)
  m_tilde = np.zeros(len(budget_space))
  for step, budget in enumerate(budget_space):
    gamma, _ = privunit.find_best_gamma(d, budget*epsilon)
    p = np.exp((1-budget)*epsilon)/(1+np.exp((1-budget)*epsilon))
    c1, c2 = privunit.get_privunit_densities(d, gamma, p)
    p_tilde = get_unbiased_p_tilde(number_candidates, c1, c2, p, epsilon)
    m_tilde[step] = privunit.getm(d, gamma, p_tilde)
  return budget_space[np.argmax(m_tilde)]


def get_epsilon_kink(budget, epsilon_target, d):
  """Find where the epsilon kink lies."""
  eps_space = np.linspace(0.01, epsilon_target, 100)
  flags = np.zeros(len(eps_space))
  for step, epsilon in enumerate(eps_space):
    _, flags[step] = privunit.find_best_gamma(d, budget*epsilon)
  return eps_space[int(np.sum(flags)-1)]


def get_budget_range(epsilon_target, d, number_of_budget_intervals):
  """Get the budget range for which epsilon_kink > epsilon_target."""
  budget_space = np.linspace(0.01, 0.99, number_of_budget_intervals)
  maximum_budget = 0.00
  for budget in budget_space:
    epsilon_kink = get_epsilon_kink(budget, epsilon_target, d)
    if epsilon_kink < epsilon_target:
      break
    else:
      maximum_budget = budget
  return maximum_budget
