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
"""Subset selection definitions.

This is the code to be used to simulate the subset selection algorithm.

The subset selection algorithm was introduced by Ye et al in
"Optimal Schemes for Discrete Distribution Estimation under
Locally Differential Privacy" - https://arxiv.org/pdf/1702.00610.pdf.
"""

import numpy as np


def probability_project_simplex(p):
  """Project the probabilitiess on the simplex."""
  k = len(p)  # Infer the size of the alphabet.
  p_sorted = np.sort(p)
  p_sorted[:] = p_sorted[::-1]
  p_sorted_cumsum = np.cumsum(p_sorted)
  i = 1
  while i < k:
    if p_sorted[i] + (1.0 / (i + 1)) * (1 - p_sorted_cumsum[i]) < 0:
      break
    i += 1
  lmd = (1.0 / i) * (1 - p_sorted_cumsum[i - 1])
  return np.maximum(p + lmd, 0)


def probability_normalize(p):
  """Normalize the probabilities so that they sum to 1."""
  p = np.maximum(p,0) # Map it to be positive.
  norm = np.sum(p)
  p = np.true_divide(p,norm)
  return p


def encode_string_fast(k, epsilon, x):
  """A fast implementation of the subset selection protocol."""
  d = int(np.ceil(k/(np.exp(epsilon)+1)))
  p = d*np.exp(epsilon)/(d*np.exp(epsilon)+k-d)
  q = (d-p)/(k-1)

  n = len(x)
  z = np.zeros((n, k))
  flip = np.random.random_sample((n, k))

  # Instead of selecting exactly d ones, set each bit to be one 
  # independently with true expectation.
  for i in range(n):
    z[i,x[i]] = np.logical_or(0,flip[i,x[i]] < p)
  return np.logical_or(z, flip < q)

def decode_string(k, epsilon, z, length, normalization = 1):
  """Learn the original distribution from the privatized strings faster."""
  d = int(np.ceil(k/(np.exp(epsilon)+1)))

  temp1 = ((k-1)*np.exp(epsilon)
    +1.0*(k-1)*(k-d)/d)/((k-d)*(np.exp(epsilon)-1))
  temp2 = ((d-1)*np.exp(epsilon)+k-d) / (1.0*(k-d)*(np.exp(epsilon)-1))

  # The input z is a matrix consisting of all the bit vectors.
  p_estimate = (1.0*np.sum(z, axis=0)*temp1/length)-temp2

  if normalization == 0:
    # Clip and normalize.
    p_estimate = probability_normalize(p_estimate)
  if normalization == 1:
    # Project on the simplex.
    p_estimate = probability_project_simplex(p_estimate)
  return p_estimate
