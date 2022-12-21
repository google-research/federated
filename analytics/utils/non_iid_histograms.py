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
"""Generates histograms with different total counts and distributions."""

import numpy as np


def generate_non_iid_poisson_counts(
    num_users: int,
    iid_param: float,
    avg_count: float,
    rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
  """Randomly generates counts for a set of users from Poisson distributions.

  When `iid_param=0`, all counts are i.i.d. from Poisson(`avg_count`).
  When `iid_param>0`, counts are distributed according to
  Poi(avg_count*Dirichlet(1/iid_param)).
  Larger iid_param means that the distributions of counts are less identical.

  Args:
    num_users: An integer indicating the total number of users. Must be
      positive.
    iid_param: A float which controls the similarity of counts between users.
      Must be non-negative. Larger value means that the count distributions are
      less identical. 0 means that the counts are i.i.d.
    avg_count: A float indicating the expected average count. Must be
      non-negative.
    rng: A Numpy random generator.

  Returns:
    counts: A numpy array of size `num_users` containing the randomly generated
      counts.
  """
  if num_users <= 0:
    raise ValueError(f'num_users must be positive.'
                     f'Found num_users={num_users}.')
  if iid_param < 0:
    raise ValueError(f'iid_param must be non-negative.'
                     f'Found iid_param={iid_param}')
  if avg_count < 0:
    raise ValueError(f'avg_count must be non-negative.'
                     f'Found avg_count={avg_count}')

  if iid_param > 0:
    lambdas = rng.dirichlet(alpha=np.ones(num_users) / iid_param)
    counts = rng.poisson(lam=num_users * lambdas * avg_count)
  else:
    counts = rng.poisson(lam=avg_count, size=num_users)
  return counts


def generate_non_iid_distributions_dirichlet(
    num_users: int,
    ref_distribution: np.ndarray,
    distribution_delta: float,
    rng=np.random.default_rng()) -> np.ndarray:
  """Generate discrete distributions around a reference distribution.

  Args:
    num_users: An integer indicating the total number of users. Must be
      positive.
    ref_distribution: A 1-D `numpy` array representing the reference discrete
      distribution. Must be non-negative and sum to one.
    distribution_delta: A non-negative `float` indicating the level of
      perturbation around the reference distribution.
      distribution_delta=0: All distributions are identical.
    rng: A numpy random generator.

  Returns:
    distributions: A list of Numpy arrays with the same size as
    `ref_distribution`. Contains all the distributions.
    If distribution_delta=0, then all distributions are `ref_distribution`.
    Otherwise, the distributions are generated according to
    Dirichlet(`ref_distribution`/`distribution_delta`)
  """
  if num_users <= 0:
    raise ValueError('num_users must be positive. Found num_users={num_users}.')
  if distribution_delta < 0:
    raise ValueError(f'distribution_delta must be non-negative.'
                     f'Found distribution_delta={distribution_delta}')
  if ref_distribution.ndim != 1:
    raise ValueError(f'ref_distribution must be a 1-D array.'
                     f'Found dimension={ref_distribution.ndim}.')
  if (ref_distribution < 0).any() | (ref_distribution > 1).any():
    raise ValueError('Expecting elements in ref_distribution to be in [0, 1].')
  if abs(np.sum(ref_distribution) - 1) > 1e-8:
    raise ValueError(f'ref_distribution should sum up to 1.'
                     f'Found the sum to be {np.sum(ref_distribution)}.')
  if distribution_delta > 0:
    distributions = rng.dirichlet(
        alpha=ref_distribution / distribution_delta, size=num_users)
  else:
    distributions = np.tile(ref_distribution, (num_users, 1))
  return distributions


def generate_histograms(
    num_users: int,
    counts_iid_param: float,
    avg_count: float,
    ref_distribution: np.ndarray,
    hist_iid_param: float,
    rng=np.random.default_rng()) -> np.ndarray:
  """Generate histograms with different total counts and distributions.

  Args:
    num_users: An integer indicating the total number of users. Must be
      positive.
    counts_iid_param: A float which controls the similarity of total counts.
      Must be non-negative.
    avg_count: A float indicating the expected average total count. Must be at
      least 1.
    ref_distribution: reference distribution over the domain
    hist_iid_param: A non-negative float. Level of perturbation around the
      reference distribution.
    rng: A numpy random generator.

  Returns:
    histograms: list of numpy arrays which contains all histograms.
  """
  if num_users <= 0:
    raise ValueError(f'num_users must be positive.'
                     f'Found num_users={num_users}.')
  if counts_iid_param < 0:
    raise ValueError(f'counts_iid_param must be non-negative.'
                     f'Found counts_iid_param={counts_iid_param}')
  if avg_count < 1:
    raise ValueError(f'avg_count must be at least 1.'
                     f'Found avg_count={avg_count}')
  if hist_iid_param < 0:
    raise ValueError(f'hist_iid_param must be non-negative.'
                     f'Found hist_iid_param={hist_iid_param}')
  if ref_distribution.ndim != 1:
    raise ValueError(f'ref_distribution must be a 1-D array.'
                     f'Found dimension={ref_distribution.ndim}.')
  if (ref_distribution < 0).any() | (ref_distribution > 1).any():
    raise ValueError('Expecting elements in ref_distribution to be in [0, 1].')
  if abs(np.sum(ref_distribution) - 1) > 1e-8:
    raise ValueError(f'ref_distribution should sum up to 1.'
                     f'Found the sum to be {np.sum(ref_distribution)}.')
  # Make sure that each user has at least 1 item
  counts = generate_non_iid_poisson_counts(num_users, counts_iid_param,
                                           avg_count - 1, rng) + 1
  distributions = generate_non_iid_distributions_dirichlet(
      num_users, ref_distribution, hist_iid_param, rng)
  histograms = []
  for i in range(num_users):
    histograms.append(rng.multinomial(counts[i], distributions[i]))
  return np.array(histograms)
