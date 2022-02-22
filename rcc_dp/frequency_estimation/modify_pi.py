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
"""Code to modify the distribution pi to tilde_pi.

The distribution tilde_pi (which is equal to pi_all[-1]) is 2 * eta-DP.
"""

import numpy as np


def modify_pi(pi, eta, epsilon, multiplicative_factor):
  """This function modifies the distribution pi to make it 2eta-LDP.

  The function essentially ensures that the new distribution lies between the
  upper threshold `exp(eta) * multiplicative_factor * p` and the lower threshold
  `exp(-eta) * multiplicative_factor * p`. It first checks if the distribution
  already lies inside the thresholds. If not, it trades as mass beyond the upper
  threshold with mass beyond the lower threshold. In other words, it ensures
  that at least one of the constraint is satisfied (the one which is violated
  less severely). This is done with the help of the helper function
  `trade_mass`. Next, it iteratively enforces the constraint that is still
  violated and renormalizes the distribution. This is done with the help of the
  helper function `normalize`.

  Args:
    pi: The input distribution to be modified
    eta: The privacy parameter that is half times the desired privacy guarantee.
    epsilon: The privacy parameter epsilon.
    multiplicative_factor: The factor by which the uniform distribution over the
      candidates is scaled with.

  Returns:
    pi_all: The container containing how the distrbution pi evolved from pi to
      tilde_pi (which is equal to pi_all[-1]). Further, tilde_pi is 2eta-LDP.
  """
  if eta < epsilon / 2:
    raise ValueError('eta should be larger than epsilon/2.')

  number_candidates = len(pi)
  p = np.zeros(number_candidates) + 1.0 / number_candidates
  p = p * multiplicative_factor
  # The container used to track changes in the distribution.
  pi_all = [pi]

  # Calculate how much mass lies above the upper threshold.
  mass_above = np.sum(np.maximum(pi - np.exp(eta) * p, 0))
  # Calculate how much mass lies below the lower threshold.
  mass_below = np.sum(np.maximum(np.exp(-eta) * p - pi, 0))

  if mass_above == 0 and mass_below == 0:
    return pi_all
  elif mass_above > mass_below:
    # Since the lower threshold is violated less severely, correct all
    # probabilities which are too low.
    below = pi < np.exp(-eta) * p
    pi_new = pi.copy()
    pi_new[below] = np.exp(-eta) * p[below]

    # Sort by distance from upper threshold (biggest offenders first). The
    # indices above the upper threshold are first in order.
    indices = np.argsort(np.exp(eta) * p - pi_all[-1])

    # Trade mass above the upper threshold against mass below the lower
    # threshold as much as possible.
    budget = mass_below
    for i in indices:
      # Correct probability above threshold as much as possible.
      diff = pi_new[i] - np.exp(eta) * p[i]
      if diff > budget:
        pi_new[i] -= budget
        break
      elif diff > 0:
        pi_new[i] -= diff
        budget -= diff
    pi_all.append(pi_new.copy())

    # Now, pi_new disobeys at-most one constraint i.e., some mass is either
    # above the upper threshold or all the mass is between the thresholds.

    # Calculate the remaining mass above the upper threshold.
    mass_above = np.sum(np.maximum(pi_new - np.exp(eta) * p, 0))

    while mass_above > 0:
      # Correct probabilities above the upper threshold.
      above = pi_new >= np.exp(eta) * p
      pi_new[above] = np.exp(eta) * p[above]

      # Renormalize distribution.
      not_above = ~above
      pi_new[not_above] += mass_above * pi[not_above] / np.sum(pi[not_above])
      pi_all.append(pi_new.copy())

      # Calculate the remaining mass above the upper threshold.
      mass_above = np.sum(np.maximum(pi_new - np.exp(eta) * p, 0))

  else:
    # Since the upper threshold is violated less severely, correct all
    # probabilities which are too high.
    above = pi > np.exp(eta) * p
    pi_new = pi.copy()
    pi_new[above] = np.exp(eta) * p[above]

    # Sort by distance from lower threshold (biggest offenders first). The
    # indices below the lower threshold are first in order.
    indices = np.argsort(pi_all[-1] - np.exp(-eta) * p)

    # Trade mass below the lower threshold against mass above the upper
    # threshold as much as possible.
    budget = mass_above
    for i in indices:
      # Correct probability below threshold as much as possible.
      diff = np.exp(-eta) * p[i] - pi_new[i]
      if diff > budget:
        pi_new[i] += budget
        break
      elif diff > 0:
        pi_new[i] += diff
        budget -= diff
    pi_all.append(pi_new.copy())

    # Now, pi_new disobeys at-most one constraint i.e., some mass is either
    # below the lower threshold or all the mass is between the thresholds.

    # Calculate the remaining mass below the lower threshold.
    mass_below = np.sum(np.maximum(np.exp(-eta) * p - pi_new, 0))

    while mass_below > 0:
      # Correct probabilities below the lower threshold.
      below = pi_new <= np.exp(-eta) * p
      pi_new[below] = np.exp(-eta) * p[below]

      # Renormalize distribution.
      not_below = ~below
      pi_new[not_below] -= mass_below * pi[not_below] / np.sum(pi[not_below])
      pi_all.append(pi_new.copy())

      # Calculate the remaining mass below the lower threshold.
      mass_below = np.sum(np.maximum(np.exp(-eta) * p - pi_new, 0))

  return pi_all