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
"""SQKR definitions.

This is the code to be used to simulate SQKR.

SQKR was introduced by Chen et al in "Breaking the
Communication-Privacy-Accuracy Trilemma" - https://arxiv.org/pdf/2007.11707.pdf
"""

import numpy as np


def kashin_representation(x, u, eta=0.4, delta=0.8):
  """Computes the kashin representation of x."""
  (kashin_n, _) = u.shape
  a = np.zeros((kashin_n, 1))

  k = 1 / ((1 - eta) * np.sqrt(delta))  # Set kashin level to be k
  m = eta / np.sqrt(delta * kashin_n)

  y = x
  itr = int(np.log(kashin_n))
  for _ in range(itr):
    b = u @ y
    b_hat = np.clip(b, -m, m)
    y = y - u.T @ b_hat
    a = a + b_hat
    m = eta * m

  b = u @ y
  tilde_y = u.T @ b
  y = y - tilde_y
  a = a + b
  return [a, k / np.sqrt(kashin_n)]


def rand_quantize(a, a_bdd):
  return (np.random.binomial(1, (np.clip(a, -a_bdd, a_bdd) + a_bdd) /
                             (2 * a_bdd)) - 1 / 2) * 2 * a_bdd


def rand_sampling(q, k):
  """Outputs k sampling matrices and an aggregation of q*sampling_mat."""
  (kashin_n, n) = q.shape
  sampling_mat_sum = np.zeros((n, kashin_n))
  sampling_mat_list = []
  for _ in range(k):
    spl = np.eye(kashin_n)[np.random.choice(kashin_n, n)]
    sampling_mat_sum = sampling_mat_sum + spl
    sampling_mat_list.append(spl.T)

  return [sampling_mat_list, sampling_mat_sum.T, q * sampling_mat_sum.T / k]


def krr(k, eps, q_sampling, sampling_mat_list, a_bdd):
  """Perturb each column of q, as a k-bit string, via k-RR mechanism."""
  q_perturb = q_sampling.copy()
  (kashin_n, n) = q_sampling.shape
  for j in range(n):
    if np.random.uniform(0, 1) > (np.exp(eps) - 1) / (np.exp(eps) + 2**k - 1):
      noise = np.zeros(kashin_n)
      for i in range(k):
        # create a random {-1, +1}^N vector and filter it by sampling matrices
        noise = noise + (2 * np.random.binomial(1, 1 / 2 * np.ones(kashin_n)) -
                         1) * sampling_mat_list[i][:, j].reshape(-1,) / k

      q_perturb[:, j] = noise * a_bdd
  return q_perturb


def estimate(k, eps, q_perturb):
  return (np.exp(eps) + 2**k - 1) / (np.exp(eps) - 1) * q_perturb


def kashin_encode(u, x, k, eps):
  """This function Kashin encodes the data.

  Args:
    u: A tight frame (i.e. (n_kashin, d) matrix) used to compute the
    Kashin's representation.
    x: A (d, n)-matrix consisting of n clients data.
    k: The communication cost (i.e., the number of bits of the compressed data).
    eps: The privacy budget.

  Returns:
    q: The quantized data.
    q_sampling: The quantized-subsampled data.
    q_perturb: The quantized-subsampled-privatized data.
  """
  [a, a_bdd] = kashin_representation(x, u)
  q = rand_quantize(a, a_bdd)
  [sampling_mat_list, _, q_sampling] = rand_sampling(q, k)
  q_perturb = krr(k, eps, q_sampling, sampling_mat_list, a_bdd)
  return [q, q_sampling, q_perturb]


def kashin_decode(u, k, eps, q_perturb):
  """This function Kashin decodes.

  Args:
    u: A tight frame (i.e. (n_kashin, d) matrix) used to compute the
    Kashin's representation.
    k: The communication cost (i.e., the number of bits of the compressed data).
    eps: The privacy budget.
    q_perturb: The quantized-subsampled-privatized data.

  Returns:
    x_estimated: The estimated mean vector.
  """
  (capital_n, _) = u.shape
  q_unbiased = estimate(k, eps, q_perturb)
  x_estimated = u.T @ (np.mean(q_unbiased * capital_n, axis=1)).reshape(-1, 1)
  return x_estimated
