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
"""Privunit definitions.

This is the code to be used to simulate the privunit algorithm.

The privunit algorithm was introduced by Bhowmick et al in "Protection Against
Reconstruction and Its Applications in Private Federated Learning" -
https://arxiv.org/pdf/1812.00984.pdf.
"""

import numpy as np
import scipy.special as sc


def find_best_gamma(d, eps):
  """This function finds the best gamma in an iterative fashion.

  Gamma is essentially the parameter in the privunit algorithm that specifies
  the distance from the equator (see figure 2 in the original paper linked
  above). The best gamma here refers to the one that achieves maximum accuracy.
  Gamma always adheres to (16a) or (16b) in the original paper (linked above).

  Args:
    d: Number of dimensions.
    eps: The privacy parameter epsilon.

  Returns:
    gamma: The best gamma.
    flag: Flag indicating how best gamma was calculated - True if 16a was used,
      and False is 16b was used.
  """
  flag = False
  gamma_a = (np.exp(eps) - 1) / (np.exp(eps) + 1) * np.sqrt(np.pi / (2 * d - 2))

  # Calculate an upper bound on gamma as the initialization step.
  gamma_b = min(np.exp(eps) / (6 * np.sqrt(d)), 1)
  while eps < 1 / 2 * np.log(
      d * 36) - (d - 1) / 2 * np.log(1 - gamma_b**2) + np.log(gamma_b):
    gamma_b = gamma_b / 1.01

  if gamma_b > np.sqrt(2 / d):
    gamma = max(gamma_b, gamma_a)
  else:
    gamma = gamma_a

  if gamma == gamma_a:
    flag = True
  return gamma, flag


def get_privunit_densities(d, gamma, p):
  """Compute the constants that the conditional density is proportional to.

  The conditional density of z (i.e., the output of the privunit) given x (i.e.,
  the input to the privunit) is proportional to c1 if the inner product between
  x and z is more than gamma and is proportional to c2 otherwise.

  Args:
    d: The number of dimensions.
    gamma: The best gamma.
    p : The probability with which an unit vector is sampled from the shaded
    spherical cap associated with the input (see the original paper).

  Returns:
    c1: The factor that the conditional density of z given x is proportional to
    if the inner product between x and z is more than gamma.
    c2: The factor that the conditional density of z given x is proportional to
    if the inner product between x and z is less than gamma.
  """
  c1 = 2 * p / (sc.betainc((d - 1) / 2, 1 / 2, (1 - gamma**2)))
  c2 = 2 * (1 - p) / (2 - sc.betainc((d - 1) / 2, 1 / 2, (1 - gamma**2)))
  return c1, c2


def getm(d, gamma, p):
  """Get the parameter m (eq (15) in the paper) in the privunit mechanism."""
  alpha = (d - 1) / 2
  tau = (1 + gamma) / 2
  if d > 1000:
    # For large d, Stirling's formula is used to approximate eq (15).
    m = (d - 2) / (d - 1) * (1 - gamma**2)**alpha / (
        2 * np.sqrt(np.pi * (d - 3) / 2)) * (
            p / (1 - sc.betainc(alpha, alpha, tau)) -
            (1 - p) / sc.betainc(alpha, alpha, tau))
  else:
    # For small d, eq (15) is used directly
    m = ((1 - gamma**2)**alpha) * (
        (p / (sc.betainc(alpha, alpha, 1) - sc.betainc(alpha, alpha, tau))) -
        ((1 - p) / sc.betainc(alpha, alpha, tau))) / (
            (2**(d - 2)) * (d - 1) * sc.beta(alpha, alpha))
  return m


def get_optimized_budget(epsilon, d):
  budget_space = np.linspace(0.01, 0.99, 99)
  m = np.zeros(len(budget_space))
  for step, budget in enumerate(budget_space):
    gamma, _ = find_best_gamma(d, budget * epsilon)
    p = np.exp((1 - budget) * epsilon) / (1 + np.exp((1 - budget) * epsilon))
    m[step] = getm(d, gamma, p)
  return budget_space[np.argmax(m)]


def apply_privunit(x, eps, budget):
  """This function applies the privunit mechanism.

  The privunit mechanism produces an unbiased estimator of x that has
  a small norm and is eps-differentially private. See algortihm 1 in the
  original paper (linked above).

  Args:
    x: The 2-dimensional array to be privatized.
    eps: The privacy factor epsilon.

  Returns:
    x_perturbed: The x privatized using privunit. This has the same dimensions
    as x.
    m: The scalar norm that x_perturbed should be divided with to get
    an unbiased estimator.
  """
  (d, n) = x.shape
  gamma, _ = find_best_gamma(d, budget * eps)
  p = np.exp((1 - budget) * eps) / (1 + np.exp((1 - budget) * eps))
  m = getm(d, gamma, p)
  x_perturbed = np.zeros((d, n))

  for i in range(n):
    u = x[:, i]
    if np.random.uniform(0, 1) < p:
      while True:
        v = np.random.normal(0, 1, (1, d))
        v /= np.linalg.norm(v)
        if np.abs(np.inner(u, v)) >= gamma:
          break
      if np.inner(u, v) < 0:
        v = -v
    else:
      while True:
        v = np.random.normal(0, 1, (1, d))
        v /= np.linalg.norm(v)
        if np.inner(u, v) < gamma:
          break
    x_perturbed[:, i] = v / m

  return x_perturbed, m
