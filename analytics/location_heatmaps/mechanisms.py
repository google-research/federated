# Copyright 2020, Google LLC.
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
"""Different noise classes."""

import abc

import numpy as np
from scipy import optimize
import tensorflow as tf


class NoiseAddition(abc.ABC):
  """Base class for different DP mechanisms."""

  @abc.abstractmethod
  def get_noise_tensor(self, input_shape):
    """Generates a noise tensor of the shape `input_tensor`.

    Args:
        input_shape: shape

    Returns:
        noised tensor
    """

    pass

  def apply_noise(self, input_tensor):
    """Applies noise to the provided tensor.

    Args:
        input_tensor: tensor to add noise

    Returns:
        noised tensor
    """

    noise = self.get_noise_tensor(input_tensor.shape)
    return input_tensor + noise


class ZeroNoise(NoiseAddition):
  """Implements no noise mechanism, when no DP guarantees needed."""

  def get_noise_tensor(self, input_shape):
    return 0


class GeometricNoise(NoiseAddition):
  """Geometric distributed DP noise.

  This approach implements the Distributed Geometric Mechanism.
  """

  def __init__(self, num_clients, differential_privacy_sensitivity,
               differential_privacy_epsilon):

    self.num_clients = num_clients
    self.differential_privacy_sensitivity = differential_privacy_sensitivity
    self.differential_privacy_epsilon = differential_privacy_epsilon
    self.r = np.exp(-self.differential_privacy_epsilon /
                    self.differential_privacy_sensitivity)

  def twosided_geometric_percentile(self, percentile):
    """Calculates percentiles of a Two-Sided Geometric Distribution (TSGD)."""

    # Convert TSGD percentile into a geometric distribution percentile.
    sign = np.sign(percentile - 50.0)
    one_sided_percentile = abs(percentile - 50.0) * 2

    return sign * np.round(
        np.log(1.0 - one_sided_percentile / 100.0) / np.log(self.r))

  def get_noise_tensor(self, input_shape):

    alpha = 1.0 / self.num_clients
    beta = (1 - self.r) / self.r

    gamma1 = tf.random.gamma(
        shape=input_shape, alpha=alpha, beta=beta, dtype=tf.dtypes.float16)
    gamma2 = tf.random.gamma(
        shape=input_shape, alpha=alpha, beta=beta, dtype=tf.dtypes.float16)
    polya1 = tf.random.poisson(shape=[1], lam=gamma1, dtype=tf.dtypes.int32)
    polya2 = tf.random.poisson(shape=[1], lam=gamma2, dtype=tf.dtypes.int32)
    client_noise = tf.reshape(tf.subtract(polya1, polya2), input_shape)

    return client_noise.numpy()


class RapporNoise(NoiseAddition):
  """RAPPOR implementation of LDP noise.

  This approach follows methods from the ESA++(https://arxiv.org/abs/2001.03618)
  specifically Lemma II.6.
  """

  def __init__(self,
               num_clients,
               sensitivity,
               epsilon,
               delta=1e-5):

    self.num_clients = num_clients
    self.sensitivity = sensitivity
    self.epsilon = epsilon
    self.delta = delta
    self.lam = self.rappor_central_to_local(self.epsilon,
                                            num_clients, self.delta)

  def sample_prob(self):
    return 1 - self.lam / (2 * self.num_clients)

  def sample_inverse_prob(self):
    return self.lam / (2 * self.num_clients)

  def eps_local(self):
    return np.log(2 * self.num_clients / self.lam - 1)

  def apply_noise(self, input_tensor):
    """Sample bool vector for zeros using RAPPOR.

    Sample bool vector for signal but zero everything except the signal.
    Use logical OR to combine two vectors and scale by 1 to return ints.

    Args:
        input_tensor: tensor to add noise

    Returns:
        noised tensor
    """
    sample_prob = self.sample_prob()
    inverse_prob = self.sample_inverse_prob()
    zeros_perturbed = np.random.choice([False, True],
                                       input_tensor.shape,
                                       p=[sample_prob, inverse_prob])
    signal_perturbed = input_tensor *\
     np.random.choice([True, False],
                      input_tensor.shape,
                      p=[sample_prob, inverse_prob])

    return np.logical_or(zeros_perturbed, signal_perturbed) * 1

  def rappor_central_eps(self, lam, n, delta=1e-5):
    """Compute central epsilon from provided lambda using Lemma II.6.

    Args:
      lam: lambda parameter
      n: total number of participants
      delta: delta parameter of DP

    Returns:
      Central epsilon
    """

    # NOTE(kairouz): original lemma uses binary vectors and a value 32, but
    # due to increase in sensitivity for one hot vectors
    # under removal DP we modify the scale from 32 to 64.
    enum = 64 * np.log(4 / delta)
    denom = lam - np.sqrt(2 * lam * np.log(2 / delta))

    right_enum = lam - np.sqrt(2 * lam * np.log(2 / delta))
    right_denom = n

    eps = np.sqrt(enum / denom) * (1 - right_enum / right_denom)
    return eps

  def rappor_central_to_local(self, eps, n, delta):
    sol = optimize.root(
        lambda lam: RapporNoise.rappor_central_eps(lam, n, delta) - eps,
        0.1 * n)
    return sol.x[0]


def get_eps_var(two_sigma, sens=1):
  """Utility function to retrieve geometric noise eps from provided error.

  Args:
    two_sigma: error bound of two-standard deviations.
    sens: sensitivity of the request

  Returns:
    epsilon that achieves desired error.
  """

  var = (two_sigma / 2)**2
  maybe_r = ((var + 1) - np.sqrt(2 * var + 1)) / var
  return -sens * np.log(maybe_r)


def std_geom(eps, sens):
  """Utility function to determine standard deviation from geometric noise.

  Args:
    eps: current budget
    sens: sensitivity of the request

  Returns:
    Std of the noise that achieves `eps`-DP with sensitivity `sens`.
  """
  r = np.exp(-eps / sens)
  variance = 2 * r / ((1 - r)**2)
  return np.sqrt(variance)
