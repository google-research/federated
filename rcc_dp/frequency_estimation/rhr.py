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
"""RHR definitions.

This is the code to be used to simulate RHR.

RHR was introduced by Chen et al in "Breaking the Communication-Privacy-
Accuracy Trilemma" - https://arxiv.org/pdf/2007.11707.pdf
"""

import math
import numpy as np
from rcc_dp.frequency_estimation import ss


def encode_string(dim, epsilon, comm, x):
  """An implementation of the RHR protocol to privatize x."""
  n = len(x)

  # Pad dim to the power of 2.
  padded_d = int(math.pow(2, math.ceil(math.log(dim, 2))))
  # Calculate the effective communication cost
  # i.e., min(comm, log(e^\epsilon)).
  eff_comm = min(comm, math.ceil(epsilon*math.log(math.e, 2)),
    math.ceil(math.log(dim, 2))+1)
  # Set the block size.
  block_size = int(padded_d/math.pow(2, eff_comm-1))

  loc_sign = np.zeros(n)
  for idx in range(n):
    j = int(idx % block_size)
    x_individual = int(x[idx])
    loc_sign[idx] = 2*int(x_individual/block_size)
    if get_hadamard_entry(padded_d, j, x_individual) == -1:
      loc_sign[idx] = loc_sign[idx] + 1

  z = rr_encode_string(int(math.pow(2, eff_comm)), epsilon,
    np.array(loc_sign))
  return z


def rr_encode_string(alphabet_size, epsilon, samples):
  """An implementation of the RR protocol to privatize x."""
  n = len(samples)
  # Start by setting private_samples = samples.
  private_samples_rr = np.copy(samples)

  # Determine which samples need to be noised (i.e., flipped).
  flip = np.random.random_sample(n) < (alphabet_size
    - 1)/(math.exp(epsilon) + alphabet_size - 1)
  flip_samples = samples[flip]

  # Select new samples uniformly at random to replace the original ones.
  rand_samples = np.random.randint(0, alphabet_size - 1, len(flip_samples))

  # Shift the samples if needed to avoid sampling the orginal samples.
  rand_samples[rand_samples >= flip_samples] += 1

  # Replace the original samples by the randomly selected ones.
  private_samples_rr[flip] = rand_samples
  return private_samples_rr


def decode_string_fast(dim, epsilon, comm, z, normalization = 1):
  """Learn the original distribution from the privatized strings when
  the input is a matrix consisting of all the bit vectors.
  """
  l = len(z)
  # Pad dim to the power of 2.
  padded_d = int(math.pow(2, math.ceil(math.log(dim,2))))
  # Calculate the effective communication cost
  # i.e., min(comm, log(e^\epsilon)).
  eff_comm = min(comm, math.ceil(epsilon*math.log(math.e, 2)),
    math.ceil(math.log(dim, 2))+1)
  # Set the block size.
  block_size = int(padded_d/math.pow(2, eff_comm-1))
  n = int(l/block_size)*block_size

  group_list = np.array(z[:n]).reshape(int(n/block_size), block_size).T

  # Create histograms to specify the empirical distributions of each group.
  histograms = np.zeros((block_size, int(padded_d/block_size)))
  for g_idx in range(block_size):
    g_count, _ = np.histogram(group_list[g_idx],
      range(int(math.pow(2, eff_comm))+1))
    histograms[g_idx] = g_count[::2] - g_count[1::2]
    histograms[g_idx] = histograms[g_idx] * (math.exp(epsilon)
      + math.pow(2, eff_comm)-1)/(math.exp(epsilon)-1)*(block_size/n)

  # Obtain estimator of q.
  q = np.zeros((block_size, int(padded_d/block_size)))
  for j in range(block_size):
    q[j, :] = fast_inverse_hadamard_transform(int(padded_d/block_size),
      histograms[j, :])

  q = q.reshape((padded_d, ), order = 'F')

  # Perform inverse Hadamard transform to get p
  p_estimate = fast_inverse_hadamard_transform(padded_d, q)/padded_d
  p_estimate = p_estimate[:dim]

  if normalization == 0:
    # Clip and normalize.
    p_estimate = ss.probability_normalize(p_estimate)
  if normalization == 1:
    # Project on the simplex.
    p_estimate = ss.probability_project_simplex(p_estimate)
  return p_estimate


def get_hadamard_entry(d, x, y):
  """Get (H_d)_{x,y}."""
  z = x & y
  z_bit = bin(z)[2:].zfill(int(math.log(d, 2)))
  check = 0
  for i in range(0, int(math.log(d, 2))):
    check = check^int(z_bit[i])
  return (-1)**check


def fast_inverse_hadamard_transform(k, dist):
  """ Performs inverse Hadamard transform."""
  if k == 1:
    return dist
  dist1 = dist[0 : k//2]
  dist2 = dist[k//2 : k]
  trans1 = fast_inverse_hadamard_transform(k//2, dist1)
  trans2 = fast_inverse_hadamard_transform(k//2, dist2)
  trans = np.concatenate((trans1+ trans2, trans1 - trans2))
  return trans
