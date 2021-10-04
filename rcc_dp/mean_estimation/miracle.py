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
"""Minimal random coding (likelihood decoder) definitions.

This is the code to be used to simulate the minimum random coding algorithm. The
algorithm was introduced by Havasi et al. in "Minimal Random Code Learning:
Getting Bits Back from Compressed Model Parameters" -
https://arxiv.org/pdf/1810.00440.pdf.

For brevity, we may refer to it as 'MIRACLE', although technically this refers
to the complete model compression pipeline of Havasi et al.
"""

import numpy as np
from mean_estimation import privunit


def encoder(seed, x, number_candidates, c1, c2, gamma):
  """This is the encoder used by the miracle algorithm.

  Args:
    seed: The random seed to be used by the encoder.
    x: The 1-dimensional input data.
    number_candidates: The number of candidates to be sampled.
    c1: The larger constant that the privunit density is proportional to.
    c2: The smaller constant that the privunit density is proportional to.
    gamma: The gamma parameter of privunit.

  Returns:
    k: The index sampled by the encoder.
    z: The set of candidates sampled at the encoder.
    pi: The distribution over the candidates for the given input data x.
  """
  if x.ndim > 1:
    raise ValueError(f"x must be a vector, got shape {x.shape}.")
  d = x.shape[0]

  rs = np.random.RandomState(seed)
  # The proposal distribution is chosen to be uniform on surface of the sphere.
  z = rs.normal(0, 1, (d, number_candidates))
  z /= np.linalg.norm(z, axis=0)

  pi = np.where(np.dot(x, z) >= gamma, c1, c2)
  pi /= np.sum(pi)
  k = np.random.choice(number_candidates, 1, p=pi)[0]
  return k, z, pi


def decoder(seed, k, d, number_candidates):
  """This is the decoder used by the miracle algorithm.

  Args:
    seed: The random seed to be used by the decoder (this seed should be the
      same as the one used by the encoder).
    k: The index transmitted by the encoder.
    d: The dimension of the data.
    number_candidates: The number of candidates to be sampled.

  Returns:
    z_k: The candidate corresponding to the index k (This is the candidate that
    is distributed according to the conditional distribution of privunit).
  """
  rs = np.random.RandomState(seed)
  # The proposal distribution should be the same as the one used by the encoder.
  z = rs.normal(0, 1, (d, number_candidates))
  z /= np.linalg.norm(z, axis=0)
  z_k = z[:, k]
  return z_k