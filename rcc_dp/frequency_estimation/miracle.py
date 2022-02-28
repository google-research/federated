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

This is the code to be used to simulate the minimum random coding algorithm
(tailored to subset selection). The algorithm was introduced by Havasi et al.
in "Minimal Random Code Learning: Getting Bits Back from Compressed Model
Parameters" - https://arxiv.org/pdf/1810.00440.pdf.

For brevity, we may refer to it as 'MIRACLE', although technically this refers
to the complete model compression pipeline of Havasi et al.
"""

import numpy as np
import scipy.special as sc

def encoder(seed, x, k, epsilon, number_candidates):
  """This is the encoder used by the miracle algorithm.

  Args:
    seed: The random seed to be used by the encoder.
    x: The user input data.
    k: The dimension of the one-hot encoded x.
    epsilon: The privacy parameter epsilon.
    number_candidates: The number of candidates to be sampled.

  Returns:
    z: The set of candidates sampled at the encoder.
    index: The index sampled by the encoder.
    pi: The distribution over the candidates for the given input data x.
  """
  d = int(np.ceil(k/(np.exp(epsilon)+1)))

  c1 = np.exp(epsilon)/(sc.comb(k-1, d-1)* np.exp(epsilon) + sc.comb(k-1, d))
  c2 = 1/(sc.comb(k-1, d-1)* np.exp(epsilon) + sc.comb(k-1, d))

  # Proposal distribution is chosen to be uniform on all d-hot encoded vectors.
  z = np.zeros((k, number_candidates))
  z[:d, :] = 1
  rs = np.random.RandomState(seed)
  for i in range(number_candidates):
    rs.shuffle(z[:, i])

  pi = np.where(z[x, :] == 1, c1, c2)
  pi /= np.sum(pi)
  index = np.random.choice(number_candidates, 1, p=pi)[0]
  return z, pi, index


def decoder(seed, index, k, epsilon, number_candidates):
  """This is the decoder used by the miracle algorithm.

  Args:
    seed: The random seed to be used by the decoder (this seed should be the
      same as the one used by the encoder).
    index: The index transmitted by the encoder.
    k: The dimension of the data.
    epsilon: The privacy parameter epsilon.
    number_candidates: The number of candidates to be sampled.

  Returns:
    z_k: The candidate corresponding to the index (This is the candidate that
    is distributed according to the conditional distribution of privunit).
  """
  d = int(np.ceil(k/(np.exp(epsilon)+1)))

  # Proposal distribution should be the same as the one used by the encoder.
  z = np.zeros((k, number_candidates))
  z[:d, :] = 1
  rs = np.random.RandomState(seed)
  for i in range(number_candidates):
    rs.shuffle(z[:, i])
  z_k = z[:, index]
  return z_k


def get_approx_epsilon(epsilon_target, k, number_candidates, delta):
  """Get the effective epsilon for miracle with approxmiate-DP guarantees.

  Args:
    epsilon_target: The privacy guarantee we desire.
    k: The number of dimensions.
    number_candidates: The number of candidates.
    delta: The delta in the differential privacy guarantee.

  Returns:
    epsilon_approx: The resulting epsilon that this version of miracle ends
    up using.
  """
  epsilon_search_space = np.linspace(0,epsilon_target,100)
  epsilon_search_space = epsilon_search_space[:-1]
  epsilon_approx = 0
  # Find the largest epsilon for SS so that MIRACLE meets epsilon_target
  for _, epsilon in enumerate(epsilon_search_space):
    d = int(np.ceil(k/(np.exp(epsilon)+1)))
    cupper = (k*np.exp(epsilon))/(k-d+d*np.exp(epsilon))
    clower = k/(k-d+d*np.exp(epsilon))
    t = np.abs(cupper-clower)*np.sqrt((np.log(2/delta))/(2*number_candidates))
    if -1 < t < 1:
      if epsilon + np.log(1+t) - np.log(1-t) <= epsilon_target:
        epsilon_approx = epsilon
  return epsilon_approx


def encode_decode_miracle_fast(seed, x, k, epsilon, number_candidates):
  """A fast implementation of the miracle protocol -- instead of
  generating number_candidates samples, generate one sample with
  true expectation.

  Args:
    seed: The random seed to be used by the encoder.
    x: The user input data.
    k: The dimension of the one-hot encoded x.
    epsilon: The privacy parameter epsilon.
    number_candidates: The number of candidates to be sampled.

  Returns:
    z: The candidate sampled at the decoder.
  """
  d = int(np.ceil(k/(np.exp(epsilon)+1)))
  num_cand_in_cap = np.random.binomial(number_candidates, d/k, size=None)
  pi_in = np.exp(epsilon)/(num_cand_in_cap*np.exp(epsilon)
    + (number_candidates-num_cand_in_cap))
  prob_sample_from_cap = num_cand_in_cap*pi_in

  if np.random.uniform(0,1) <= prob_sample_from_cap:
    # Generate a sample uniformly from the cap
    z = [1]*(d-1)+[0]*(k-d)
    rs = np.random.RandomState(seed)
    rs.shuffle(z)
    z = z[:x] + [1] + z[x:]
  else:
    # Generate a sample uniformly from outside the cap
    z = [1]*(d)+[0]*(k-d-1)
    rs = np.random.RandomState(seed)
    rs.shuffle(z)
    z = z[:x] + [0] + z[x:]
  z = np.array(z)
  return z


def encode_decode_modified_miracle_fast(seed, x, k, epsilon, number_candidates):
  """A fast implementation of the modified miracle protocol -- instead of
  generating number_candidates samples, generate one sample with
  true expectation.

  Args:
    seed: The random seed to be used by the encoder.
    x: The user input data.
    k: The dimension of the one-hot encoded x.
    epsilon: The privacy parameter epsilon.
    number_candidates: The number of candidates to be sampled.

  Returns:
    z: The candidate sampled at the decoder.
  """
  d = int(np.ceil(k/(np.exp(epsilon)+1)))
  c1 = np.exp(epsilon)/(sc.comb(k-1,d-1)* np.exp(epsilon) + sc.comb(k-1,d))
  c2 = 1/(sc.comb(k-1,d-1)* np.exp(epsilon) + sc.comb(k-1,d))

  num_cand_in_cap = np.random.binomial(number_candidates, d/k, size=None)

  beta = num_cand_in_cap/number_candidates
  expected_beta = np.ceil(k/(np.exp(epsilon)+1))/k

  pi_in = c1/(num_cand_in_cap*c1+(number_candidates-num_cand_in_cap)*c2)
  tilde_pi_in = pi_in*(1+beta*(np.exp(epsilon)-1))/((1
    + expected_beta*(np.exp(epsilon)-1)))
  if beta > expected_beta:
    tilde_pi_in = tilde_pi_in*(beta+expected_beta*(np.exp(epsilon)
      - 1))/(beta*np.exp(epsilon))

  prob_sample_from_cap = num_cand_in_cap*tilde_pi_in

  if np.random.uniform(0,1) <= prob_sample_from_cap:
    # Generate a sample uniformly from the cap
    z = [1]*(d-1)+[0]*(k-d)
    rs = np.random.RandomState(seed)
    rs.shuffle(z)
    z = z[:x] + [1] + z[x:]
  else:
    # Generate a sample uniformly from outside the cap
    z = [1]*(d)+[0]*(k-d-1)
    rs = np.random.RandomState(seed)
    rs.shuffle(z)
    z = z[:x] + [0] + z[x:]
  z = np.array(z)
  return z
